// ============================================================================
//  expert_cache.hpp — bounded, lazy residency for Mixture-of-Experts weights.
//
//  Expert tensors (`blk.N.ffn_{gate,up,down}_exps.weight`) dominate the size of
//  an MoE model: Qwen3-235B-A22B carries ~235B parameters but only ~22B are
//  touched per token. Uploading every expert at load time would demand the whole
//  model resident and defeat the architecture.
//  So experts are never eagerly loaded. The loader keeps only the GGUF mapping;
//  the router decides per token which experts run, and ExpertCache materializes
  ////  exactly those three matrices on demand into a byte-budgeted residency set.
//  Evicted slots are released, so a machine with far less memory than the model
  ////  still decodes — just with more reload traffic. The engine sizes the default
  ////  budget to keep the *whole* routed set resident whenever it fits: top-k
  ////  routing touches every expert within a few dozen tokens, so a set that only
  ////  partly fits churns (evict + free + alloc + re-upload) forever, which costs
  ////  far more than the memory the cap saves.
//
//  The residency policy is LFU with aging, not LRU. Expert popularity is
//  extremely skewed and roughly stable across a conversation: a handful of
//  experts serve most tokens. Plain LRU evicts a long-serving hot expert the
//  first time a cold token brushes past it, which is precisely backwards. So:
//
//    * every hit increments the slot's counter; a miss that loads also counts
//    * a slot whose counter reaches the pin threshold becomes pinned — it is
//      never chosen for eviction while pinned (pins still count, so a pin is
//      always earned by traffic, never granted for free)
//    * the counters decay by DECAY every DECAY_PERIOD acquires, so a favourite
//      of the first thousand tokens cools off and releases its pin when the
//      conversation moves on
//
//  A victim is chosen by: unpinned first, then the lowest counter, then the
//  oldest load. Guarantee: if at least one slot can hold the request, a pinned
//  expert is never evicted; and the cache always makes progress (see acquire).
//
//  --- WARM: the read-through tier ---------------------------------------------
//  COLD (the GGUF mapping) is the immutable canonical store, WARM is pageable
//  host RAM, HOT is VRAM. Every expert read out of the file is admitted to WARM
//  before it is copied on to VRAM (always-admit; see warm_admit), so a file
//  read is never a one-shot: the next request for that expert is a promotion
//  out of RAM instead of another read of the file. A VRAM eviction then costs
//  nothing at all — the WARM copy is already there, so the slot simply stops
//  being resident on the device and no bytes move. WARM eviction is a free():
//  the file holds the canonical copy, and nothing is ever written back to it.
//
//  WARM is inclusive — promotion keeps the RAM copy — and that is exactly why
//  HOT -> WARM is free. It also means the WARM budget has to cover the hot set
//  plus the recently loaded experts, and it is why WARM is pageable rather than
//  page-locked: a tier sized in tens of GiB cannot come out of the pinned pool.
//  The DMA that feeds VRAM is staged through a small bounded pinned ring
//  instead (Backend::upload_paged), so the two allocations stay separate.
//
//  Host copies ignore pins: a pin is a promise about VRAM residency, and a WARM
//  copy is a cache of a cache — losing one costs a future file read, never
//  correctness — so the tier never lets a stale pin defend a copy that VRAM
//  traffic has moved on from.
//
//  Pointers returned by acquire() stay valid until the next acquire() call.
// ============================================================================
#ifndef KRK_EXPERT_CACHE_HPP
#define KRK_EXPERT_CACHE_HPP

#include <algorithm>
#include <list>
#include <unordered_map>
#include <utility>
#include <vector>

#include "krk/backend.hpp"
#include "krk/gguf.hpp"
#include "krk/moe_table.hpp"
#include "krk/quant.hpp"

namespace krk {

// A quantized weight matrix resident in backend memory. Defined here because the
// cache is what materializes these for experts; the model loader uses the same
// descriptor for its eagerly-loaded matrices.
struct QuantTensor {
    void *data = nullptr;        // device pointer (or host, on the CPU backend)
    DType type = DType::Unknown; // Unknown => tensor absent
    i64 n_in = 0;                // contiguous inner dimension
    i64 n_out = 1;               // outer dimension

    bool present() const { return data != nullptr && type != DType::Unknown; }
};

// One layer's three expert tensors, still pointing into the GGUF mapping. The
// loader records these instead of uploading, which is what makes loading lazy.
struct ExpertSource {
    const GgufTensor *gate = nullptr;
    const GgufTensor *up = nullptr;
    const GgufTensor *down = nullptr;

    // The per-tensor global scale GGUF stores BESIDE an NVFP4 weight, as
    // "<tensor>.scale". NVFP4 is encoded as a 4-bit code times a per-16-value
    // UE4M3 micro-scale, and that product is not the weight: the global factor
    // has to be applied on top. Measured on
    // NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4, layer 27 ffn_up_exps: the
    // codes and micro-scales alone give max|w| = 768, rms = 100, where the
    // global factor (~9e-5 per expert) brings the same matrix to max = 0.06,
    // rms = 0.0096. The raw magnitude is four orders out and overflows the f16
    // activation buffer, which is how it showed up: a handful of projection
    // cells stored as +inf, and then NaN where a second corruption multiplied
    // them by zero.
    //
    // One f32 per expert for a routed bank ([n_expert] long) and a single f32
    // for a dense one ([1]); scale_of reads whichever the shape says. A file
    // with no sidecar gets 1.0, which is what every non-NVFP4 expert tensor
    // means, so nothing changes for the formats that never carried one.
    const GgufTensor *gate_scale = nullptr;
    const GgufTensor *up_scale = nullptr;
    const GgufTensor *down_scale = nullptr;

    f32 scale_of(const GgufTensor *t, i32 e) const {
        if (!t || !t->data || t->n_elements < 1) return 1.0f;
        const f32 *s = reinterpret_cast<const f32 *>(t->data);
        const i64 want = (n_expert > 1 && t->ne[0] == static_cast<u64>(n_expert))
                             ? static_cast<i64>(e)
                             : 0;
        return s[want < t->n_elements ? want : 0];
    }
    i32 n_expert = 0;
    i64 n_embd = 0;   // inner dimension of the gate/up matrices
    i64 n_ff_exp = 0; // expert hidden width

    // up/down are the two tensors every MoE schema carries; nemotron_h_moe
    // routes with those alone (no ffn_gate_exps). Requiring the gate here made
    // every acquire() of a nemotron expert return null before touching a
    // counter — 0 acquires, 0 traffic, a silently expert-less model.
    bool present() const { return up != nullptr && down != nullptr; }
    // Bytes of one expert's slice in each of the three tensors.
    size_t per_expert(const GgufTensor *t) const {
        return t && n_expert > 0 ? t->n_bytes / static_cast<size_t>(n_expert) : 0;
    }
    size_t expert_bytes() const {
        return per_expert(gate) + per_expert(up) + per_expert(down);
    }
};

// The three resident matrices for one (layer, expert) pair.
struct ResidentExpert {
    QuantTensor gate, up, down;
};

class ExpertCache {
public:
    // Counters reaching this value pin a slot; every decay window
    // cools all counters, so a pin must be continuously re-earned.
    // The window is kPinDecay acquires per resident slot (see
    // decay_period): a fixed 64-acquire window would cool every
    // counter faster than any single expert is re-selected once a
    // model has hundreds of slots, and nothing could ever pin.
    static constexpr u32 kPinThreshold = 24;
    static constexpr u32 kPinDecay = 64;
    // Per-decay subtraction, applied when a counter exceeds it (keeps small
    // counts meaningful instead of driving everything to the same floor).
    static constexpr u32 kPinDecayStep = 8;

    // budget_bytes is a soft cap on device-resident expert memory. At least one
    // slot is always allowed, even if a single expert exceeds the budget.
    // host_budget_bytes caps the second tier; 0 (the default) disables it and
    // evicted experts are dropped to the mapping exactly as before.
    //
    // slot_budget is how many (layer, expert) slots the byte budget was sized
    // for, or 0 (the default) to leave that to the bytes. When it is set it is
    // a hard count cap AND the denominator capacity_slots() reports. It exists
    // because "how many slots fit in N bytes" has no single answer for a corpus
    // that mixes expert sizes, and the caller knows the answer exactly in the
    // two cases that matter: an explicit --expert-cache-slots N, and a budget
    // that covers the model's whole routed set.
    // Overrides configure()'s pin budget (which is KRK_HOST_PIN_MB). Called
    // after it, because configure() resets the tier it decorates.
    void set_host_pin_budget(size_t bytes);
    void configure(Backend *be, size_t budget_bytes, size_t host_budget_bytes = 0,
                   size_t slot_budget = 0);
    void clear();

    // Returns the resident matrices for (layer, expert), loading on a miss.
    // nullptr when the source has no expert tensors or the id is out of range.
    // The returned pointer is valid until the next acquire().
    const ResidentExpert *acquire(const ExpertSource &src, i32 layer, i32 expert);

    // Fills WARM from the source ahead of the run: the answer to *compulsory*
    // misses, which is what a short generation is made of. Opt-in, because
    // read-through already admits every expert the run touches and an eager
    // sweep pays the whole corpus up front. Staged slots carry an LFU count of
    // 0, so they are the first out of the tier when room is needed. Returns
    // bytes staged.
    size_t preload(const ExpertSource &src, i32 layer);
    // Makes one layer's WHOLE routed set device-resident and pins it.
    //
    // This is for the callers that page through one small, fixed bank rather
    // than a stack: the nextn (MTP) block is a single layer whose experts a
    // speculative round re-reads on every proposal, and against the stack's LRU
    // they are cold often enough to pay a promotion per proposal (measured on
    // Tiel: 154 of 1384 block acquires promoted, 179 ms of a 4.6 s run). The
    // set fits (845 MiB for 256 experts), so admit it once and hold it.
    //
    // Returns the bytes this layer now holds in the device tier (0 when it could
    // not be pinned) -- `why`, when given, says why, and the caller keeps the
    // normal policy instead of thrashing a pin it cannot defend. Pins are never
    // evicted while the budget holds them.
    size_t pin_layer(const ExpertSource &src, i32 layer, std::string *why);

    // ---- the PIN RING: the hot set in VRAM, outside the pool's byte budget ---
    //
    // A pin already stops eviction, but until now a pinned expert still spent
    // the pool's budget: `bytes_` counted it, so every pin shrank the LRU
    // window by its own size. At a budget the routed set does not fit that is
    // the whole problem -- Tiel-Coder-35B-A3B routes 22.1 GiB and the pool is
    // 12-13 GiB, so the pinned prefix ends up being most of the pool and the
    // traffic that is still *earning* residency thrashes in the remainder.
    //
    // The second ring is that separation: pinned bytes are charged to
    // `pin_budget_` and not to `budget_`, so the LRU window keeps its full
    // size for the experts that traffic still has to prove. It is one physical
    // card, so `pin_budget_` is a real cap -- admit_hot() stops at it rather
    // than overcommitting VRAM.
    // Re-derives the pool's byte budget from a LIVE free-VRAM reading: the
    // resident unpinned footprint plus whatever is free beside `floor_bytes`.
    // Only ever lowers, because a budget raised past what init measured is how
    // the pool grows into the card and turns every alloc into a driver
    // eviction; growth belongs to the ring, which names what it is for.
    // Returns the budget in force afterwards, so a caller can log the change.
    size_t reclamp_vram_budget(size_t free_bytes, size_t floor_bytes);
    void set_pin_budget(size_t bytes);
    size_t pin_budget() const { return pin_budget_; }
    size_t pin_used_bytes() const { return pin_bytes_; }
    // Admits the ranked prefix of `order` (hottest first) into the ring in one
    // pass, pinning each expert as it lands. This is the HOT half of a warmup:
    // WARM costs a file read and holds tens of GiB, VRAM holds the few
    // hundred MiB/ few GiB that the index says serve most tokens, and only the
    // ring makes those a guaranteed hit instead of a promotion race with the
    // rest of the layer's routing.
    //
    // Stops at the ring's byte cap, at a source that cannot be read, or after
    // `max_experts` admissions (0 = no limit). `src_of` maps a layer index to
    // its expert tensors; a null entry skips that layer (a dense one). Returns
    // how many experts this newly made resident-and-pinned.
    size_t admit_hot(const ExpertSource *const *src_of, i32 n_layer,
                     const std::vector<std::pair<i32, i32>> &order,
                     size_t max_experts = 0);
    // One expert, so the caller can choose the order: expert-major across
    // layers when the tier is smaller than the corpus.
    size_t preload_one(const ExpertSource &src, i32 layer, i32 expert);
    // Warms WARM for one layer's whole routed set in a SINGLE batched read.
    //
    // This is the fix for the shape of an expert miss. A layer routes k experts
    // and each is three slices, so the real request is a set of 0.5-2 MiB reads
    // -- but the tier issued them one at a time through one file handle, which
    // is queue depth 1. Measured on this machine (NVMe, 1.82 MiB reads): 593
    // MiB/s at one outstanding read, 1175 at two, 1446 at four, 3400 sequential,
    // and 593 MiB/s is exactly what the expert path reported while most of a
    // decode token's wall time was spent waiting for the file.
    //
    // Advisory by contract: a slot already in WARM is skipped, a slot that does
    // not fit the tier is counted and skipped, and a failed read leaves the
    // expert cold. It never blocks or fails a run -- acquire() still loads
    // whatever this did not warm.
    // Returns bytes read.
    size_t prefetch_layer(const ExpertSource &src, i32 layer, const i32 *experts,
                          int n);
    // One-step exact lookahead: only an already WARM next routed expert.
    // The current expert is leased during admission; too-small budgets skip it.
    bool stage_next(const ExpertSource &src, i32 layer, i32 expert, i32 current);
    // Multi-row lookahead: promote every WARM expert in `experts` that is not
    // resident yet, in one pass, before the rows that need them run.
    //
    // This exists for a BATCHED forward -- the verify pass of a speculative
    // round is the one that produces one. A layer's router is computed for all
    // d rows on the host before the first expert is acquired, so the union is
    // known a whole row early: staging it means rows 1..d-1 hit VRAM instead of
    // each promoting its own experts serially, and the promotions overlap the
    // first row's compute. A single-row decode has nothing to stage ahead of
    // (stage_next covers the ordering inside its own routing), so callers use
    // this only when n > 1.
    //
    // Advisory exactly like prefetch_layer: a slot that is not WARM is skipped
    // (acquire() still loads it), a slot the budget cannot hold is counted in
    // `lookahead_rejected()` and skipped, and a failed promotion is a miss, not
    // an error. Returns how many experts this promoted.
    size_t stage_batch(const ExpertSource &src, i32 layer, const i32 *experts, int n);
    // The same advisory staging for a WARM set, but the copies are issued as ONE
    // upload per source kind instead of one call per expert: a batch shares one
    // staging-ring cursor, so slice N+1 stages while slice N is in flight, and
    // the default stream is published once instead of 3k times. Room-making and
    // the cell allocations stay per expert (each slot has its own size class).
    // Returns the number of experts promoted.
    size_t promote_batch(const ExpertSource &src, i32 layer, const i32 *experts,
                         int n);
    // Stage `experts` of `layer` even when the byte budget is full, by retiring
    // slots the caller has finished with. Used one layer ahead: the fetch is
    // issued after this layer's expert loop, so it runs under the next layer's
    // mixer instead of in front of its own GEMMs. `leased` guards every slot the
    // pass touches, so the eviction walk cannot take back what it just staged.
    size_t stage_ahead(const ExpertSource &src, i32 layer, const i32 *experts,
                       int n);
    u64 lookahead_staged() const { return lookahead_staged_; }
    u64 lookahead_rejected() const { return lookahead_rejected_; }
    u64 active_staged() const { return active_staged_; }
    u64 active_used() const { return active_used_; }
    f64 active_wait_ms() const { return active_wait_ms_; }
    u64 bytes_staged() const { return bytes_staged_; }
    // Wall time spent copying promotions into VRAM, so the report can print
    // the one number that decides whether promotion is fast enough to hide
    // behind a token: MiB/s out of the host tier.
    f64 promote_ms() const { return promote_ms_; }

    // The expert path is 90% of a laguna-xs2 decode token, and promote_ms()
    // does not say which part of it is slow: the file read, the room-making,
    // the allocator, or the host->VRAM copy. These are wall time in each of
    // those four, accumulated over the whole run, so one run localises the
    // cost instead of a sequence of A/B arms.
    f64 read_ms() const { return read_ms_; }   // read_host / mapping memcpy
    f64 room_ms() const { return room_ms_; }   // make_vram_room: evict + retire
    // The two halves of the room row: the victim search (a scan of the slot
    // map) and the release of the cells it picked. They have different fixes,
    // and on a 14k-slot tier the scan is not obviously the cheap half.
    f64 scan_ms() const { return scan_ms_; }
    f64 retire_ms() const { return retire_ms_; }
    // Victim scans issued and slot visits they cost, so a windowed pick can
    // be compared against the global one on work done, not only on ms.
    u64 scans() const { return scans_; }
    size_t vram_index_keys() const { return vram_keys_.size(); }
    size_t vram_index_resident() const { return vram_slots_; }
    u64 scan_visits() const { return scan_visits_; }
    f64 alloc_ms() const { return alloc_ms_; } // alloc_pooled: pool wait or malloc
    f64 xfer_ms() const { return xfer_ms_; }   // upload_paged_batch, host wait in

    // The end-of-run split: how much of each stage ran inside PREFILL and how
    // much inside DECODE. The merged rows above cannot say which side a fix
    // belongs on, and the same DMA behind a 256-row prefill chunk is not the
    // same cost as one behind a single decode row. `PhaseOther` is load-time
    // warmup and prefetch, which is neither, so the three never double-count.
    enum Phase { PhaseOther = 0, PhasePrefill = 1, PhaseDecode = 2 };
    void set_phase(Phase p) { phase_ = p; }
    f64 prefill_promote_ms() const { return pf_promote_ms_; }
    f64 prefill_read_ms() const { return pf_read_ms_; }
    f64 prefill_room_ms() const { return pf_room_ms_; }
    f64 prefill_alloc_ms() const { return pf_alloc_ms_; }
    f64 prefill_xfer_ms() const { return pf_xfer_ms_; }
    f64 decode_promote_ms() const { return dc_promote_ms_; }
    f64 decode_read_ms() const { return dc_read_ms_; }
    f64 decode_room_ms() const { return dc_room_ms_; }
    f64 decode_alloc_ms() const { return dc_alloc_ms_; }
    f64 decode_xfer_ms() const { return dc_xfer_ms_; }

    // Zeroes the traffic counters and the stage timers WITHOUT touching
    // residency: the slots stay where they are, and so do their LFU counts.
    // The startup warmup earns promotions and file reads that belong to the
    // warmup, and a run's report has to describe the run -- otherwise the
    // 92.7%/7.3%/7.3% HOT/WARM/COLD split that decides the next experiment is
    // partly a measurement of the phase that was supposed to improve it. The
    // warmup prints its own bytes and milliseconds; nothing is hidden by this.
    void reset_counters();

    size_t budget_bytes() const { return budget_; }
    size_t resident_bytes() const { return bytes_; }
    // Device-resident slots, i.e. the ones whose weights are in VRAM right now.
    size_t resident_slots() const { return vram_slots_; }
    // How many (layer, expert) slots the device budget is sized for: the exact
    // count when the caller supplied one (slot_budget), otherwise the byte
    // budget divided by the expert footprint of the layer being acquired, which
    // is an estimate for a corpus whose layers differ in size.
    size_t capacity_slots() const { return capacity_; }
    size_t slot_budget() const { return slot_budget_; }
    size_t vram_budget() const { return budget_; }
    // Lower the device budget once the KV cache and the engine's own workspaces
    // exist. The loader decides this number BEFORE those allocations, so on a
    // card that is nearly full it can name every byte that was free at that
    // moment. Only ever lowers: a raise here would re-commit what the
    // measurement just said is not there.
    void set_vram_budget(size_t bytes);
    // WARM tier (the second tier). The warm_* names are the ones the reports
    // use; the host_* spellings are kept for existing callers.
    size_t warm_capacity_bytes() const { return host_budget_; }
    size_t warm_pinned_bytes() const { return host_pinned_bytes_; }
    size_t host_pin_budget() const { return host_pin_bytes_; }
    size_t warm_pin_budget_bytes() const { return host_pin_bytes_; }
    size_t warm_used_bytes() const { return host_bytes_; }
    size_t warm_slots() const;
    size_t host_budget_bytes() const { return host_budget_; }
    size_t host_resident_bytes() const { return host_bytes_; }
    size_t host_slots() const { return warm_slots(); }
    // Every slot the cache is tracking, in both tiers.
    size_t tracked_slots() const { return slots_.size(); }
    u64 loads() const { return loads_; }
    u64 evictions() const { return evictions_; }
    u64 hits() const { return hits_; }
    // Acquires served by promoting out of the host tier — a DMA instead of a
    // reload of the mapping. The load itself is not repeated, so this is
    // reported apart from hits() and loads().
    u64 host_hits() const { return host_hits_; }
    // Acquires served by promoting out of WARM: every one of these is a file
    // read that read-through turned into a RAM read.
    u64 warm_hits() const { return host_hits_; }
    // Experts admitted to WARM, and why admission ever failed: a tier that
    // could not fit the expert after eviction, or an expert larger than the
    // whole tier. Always-admit means rejects are rare by design.
    u64 warm_admissions() const { return warm_admissions_; }
    u64 warm_evictions() const { return warm_evictions_; }
    u64 warm_rejects() const { return warm_rejects_; }
    // Acquires that had to read the model file: the COLD tier's traffic.
    u64 cold_misses() const { return loads_; }
    u64 demotions() const { return demotions_; }
    u64 promotions() const { return promotions_; }
    // Batched-promotion evidence: how many batch calls were issued and
    // how many experts they carried, so a report can show the shape.
    u64 promote_batches() const { return promote_batches_; }
    u64 promote_batch_experts() const { return promote_batch_experts_; }
    // Lifetime acquire count (hits + loads) since the last clear().
    u64 acquires() const { return total_acquires_; }
    // Bytes that crossed a boundary, which is the traffic a residency policy
    // is actually trading against: loaded = read out of the model file (a
    // miss), demoted = written into the host tier, promoted = read back out
    // of it (a DMA instead of a file re-read).
    u64 bytes_loaded() const { return bytes_loaded_; }
    u64 bytes_demoted() const { return bytes_demoted_; }
    u64 bytes_promoted() const { return bytes_promoted_; }
    u64 decay_events() const { return decays_; }
    size_t pinned_slots() const;
    // Current LFU counter of one slot, or 0 when it is not resident. Because a
    // pin is granted at kPinThreshold and dropped below it, a count below the
    // threshold proves the slot is unpinned.
    u32 touch_count(i32 layer, i32 expert) const;

    // Residency queries, for a caller that has to decide what to do with an
    // expert BEFORE acquire() runs. acquire() promotes, and the hybrid
    // CPU+GPU expert path exists precisely to not promote the experts the host
    // is about to compute, so asking after the fact is asking too late.
    // Neither call changes residency, touches a counter or moves a byte.
    bool in_vram(i32 layer, i32 expert) const;
    bool in_host(i32 layer, i32 expert) const;

    // Seeds one resident slot's LFU counter from outside. The caller is the
    // ranked warmup, which measured which experts are hot BEFORE the run
    // started; that counter is the only thing eviction ranks on, so stamping the
    // offline order into it is how the two rankings are kept in agreement.
    //
    // Measured, and the reason this is OFF by default wherever it is called
    // (src/engine.cpp gates it on KRK_EXPERT_SEED=1): doing it HURTS. laguna-xs2,
    // 64 greedy tokens, ranked warmup, A B B A interleaved, stdout identical:
    // 1010 -> 1391 promotions (+38%) and 24.4 -> 21.3 tok/s (-13%, 4/4). The
    // mechanism ranks correctly (see test_expert_cache_policy, which pins a
    // seeded slot outliving an unseeded one on identical traffic); the ordering
    // it imposes does not. With 65% of the routed set resident, a frozen
    // 68-position scan is a worse predictor of the next acquire than the
    // run-time reuse the counters would otherwise accumulate, and freezing it
    // displaces the policy that was already holding 95.2% of acquires HOT.
    //
    // The value only ever rises, and it is clamped below kPinThreshold: a pin is
    // a promise VRAM traffic has to earn, so a seed cannot grant one. A slot
    // that is not resident is left alone -- a seed is not residency.
    void seed_count(i32 layer, i32 expert, u32 count);

    // The counter a ranked-warmup promotion seeds its slot with: linear from
    // kPinThreshold-1 at rank 0 down to 1 at the bottom of the warmed order, and
    // never at the threshold, so a seed ranks a slot without pinning it. Public
    // because the contract is testable on its own -- monotone in rank, always
    // inside [1, kPinThreshold) -- and because it is the one place the warmup's
    // measured order is translated into the policy's own units.
    static u32 warmup_rank_seed(size_t rank, size_t depth) {
        const i32 top = static_cast<i32>(kPinThreshold) - 1;
        if (depth <= 1) return static_cast<u32>(top);
        const i64 r = static_cast<i64>(rank);
        const i64 d = static_cast<i64>(depth) - 1;
        if (r > d) return 1u;
        i32 v = top - static_cast<i32>(r * (top - 1) / d);
        if (v < 1) v = 1;
        return static_cast<u32>(v);
    }

    // True when promoting one more expert of `src` would have to evict
    // something, i.e. the device tier is already full. The hybrid CPU+GPU path
    // uses it to take the OVERFLOW only: with a budget that holds the whole
    // routed set every expert keeps arriving through the normal path and the
    // host arm stays idle, which is the intent. Measured the other way first --
    // hybrid taking every miss -- and it read 6.4 tok/s against 99.3, because
    // an expert that is never promoted never becomes resident, so a budget that
    // could hold the entire set stayed cold forever.
    bool full_for(const ExpertSource &src) const;

    // ---- the DEVICE DISPATCH TABLE -----------------------------------------
    //
    // A dense `[n_layer * n_expert]` array of {three device pointers, their
    // shapes, their types} on the device, so an expert's GEMMs can be issued
    // from a kernel that reads the ids itself. Today the host scans the router
    // probabilities for the top-k and calls acquire() per expert to learn where
    // the matrices are, which is what puts a host round trip in the middle of
    // every MoE layer (measured: 0.523 ms/layer, 85% of it waiting for the
    // layer's own GPU work to retire, on Mellum2.1).
    //
    // The table is a promise about RESIDENCY, so it is only published when that
    // promise can be kept: `whole_set_resident` must hold, i.e. every
    // (layer, expert) of the corpus is in VRAM. Then no promotion and no
    // eviction can rewrite a pointer underneath the kernel. Anything else
    // publishes nothing and the caller keeps the host path -- a half-valid
    // table would be a wrong answer, not a slow one.
    bool whole_set_resident(i32 n_layer, i32 n_expert) const;
    // Builds and uploads the table; nullptr (with `why` filled) when the
    // residency promise does not hold or the allocator refuses. Re-publishing
    // releases the previous buffer, and *published_out is the number of entries
    // that carried all three matrices (a gate-less expert contributes two).
    void *publish_dispatch_table(i32 n_layer, i32 n_expert, i32 *published_out,
                                 std::string *why);
    const void *dispatch_table() const { return table_; }
    // A promotion or an eviction rewrote a pointer after the publish, so the
    // callers that already hold a table pointer must stop using it.
    bool dispatch_table_stale() const { return table_stale_; }
    void release_dispatch_table();

private:
    struct Slot {
        i32 layer = -1;
        i32 expert = -1;
        ResidentExpert m; // device copy
        ResidentExpert h; // host (L2) copy
        size_t bytes = 0;      // device footprint of this expert
        size_t host_bytes = 0; // host footprint, nonzero only while in the tier
        // Per-tensor slice size (gate, up, down). A property of the source
        // tensors, so it outlives residency and is what a demote/promote sizes
        // its copies from.
        size_t slice[3] = {};
        u32 count = 0;      // LFU counter, decayed
        u64 seq = 0;        // load order, the tie-break for victims
        bool pinned = false;
        // The WARM copy of this slot lives in PINNED host memory (see
        // KRK_HOST_PIN_MB). Distinct from `pinned`, which is the residency
        // promise the eviction scan honours: this one says which allocator
        // owns the bytes, so the free and the promotion copy both know.
        bool host_pinned = false;
        bool leased = false;
        bool staged = false;
        void *ready = nullptr;
        // WARM is inclusive: a slot may sit in both tiers at once, and a HOT
        // slot's WARM copy is what makes its eviction free. A slot in neither
        // tier is forgotten (see forget_if_dead).
        // Anchored on up/down -- the two tensors every MoE schema carries.
        // These used to test the *gate* copy, and nemotron_h_moe has no
        // ffn_gate_exps: every staged slot then reported itself as neither
        // warm nor resident. Measured on
        // NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4: "WARM 15758 MiB of
        // 15758 MiB in 0 slots", the warm evictor found no victims (5492
        // rejects), every acquire fell through to a cold file read (918
        // MiB/token) and leaked an untracked VRAM buffer per acquire until
        // the device reported 15.92 GiB in use with 0 evictions.
        bool in_vram() const { return m.up.present() && m.down.present(); }
        bool in_host() const { return h.up.present() && h.down.present(); }
        // Position in vram_keys_, or -1. A resident vector is what keeps the
        // victim search off the 14k-entry slot map: see pick_victim.
        i64 vram_idx = -1;
    };

    // One acquire on a resident slot: counters earn pins, pins are never free.
    void touch(Slot &s);
    // Frees the device copy. Leaves the footprint, counter and load order in
    // place — a slot that is only in the host tier still knows what it is.
    void drop(Slot &s);
    void drop_host(Slot &s);
    // VRAM eviction. The device copy is released; a WARM copy, when the tier
    // holds one, is what the slot falls back to — the bytes are already there,
    // so the move is a counter, not a copy. Without one the slot falls to COLD.
    void retire(Slot &s);
    // One expert's three slices, read out of the model file into the pageable
    // WARM buffers of `s` (read_host: a read(), not a mapping fault). A partial
    // copy is dropped by the caller.
    bool warm_fill(const ExpertSource &src, i32 expert, Slot &s);
    // COLD -> WARM: makes room, fills, and counts the admission. Always-admit,
    // except for an expert that cannot fit the entire tier.
    bool warm_admit(const ExpertSource &src, i32 expert, Slot &s);
    // Host -> device. Makes VRAM room first, so it can evict like any miss.
    // `prefetch` takes the pinned DMA ticket and leaves the default stream
    // unblocked (see Backend::upload_expert_prefetch); `staged` is the ring's
    // host-copy ticket. Promotion is the same arithmetic either way.
    bool promote_to_vram(Slot &s, bool staged = false, bool prefetch = false);
    void finish_stage(Slot &s);
    // Evicts until one more slot of `need` bytes fits. Bytes are the test, not a
    // slot count: experts are not all one size, so counting slots against a
    // per-expert estimate evicts with budget left unspent. Pins cannot override
    // the limit; only a single oversize expert gets the explicit minimum.
    void make_vram_room(size_t need);
    // True when admitting one more slot of `need` bytes would exceed what this
    // cache is allowed to hold: the byte budget, or the explicit slot count when
    // one was configured. The two rules disagree on purpose -- a Q4_K_M corpus
    // mixes Q4_K and Q6_K down-projections, so one layer's expert footprint is
    // not the next layer's, and budget_ / one_layer_expert is neither the memory
    // limit nor a count that holds the whole set.
    bool over_budget(size_t need) const {
        // The pool's byte test excludes the ring ONLY when a ring is
        // configured (set_pin_budget). With no ring every pinned expert still
        // spends the pool's budget, which is the old behaviour the policy tests
        // pin: a pin ranks victims, it never overrides the physical limit.
        // (No std::min here: <Windows.h>'s min macro has broken this header's
        // neighbours before, so the comparison is written out.)
        size_t held = 0;
        size_t ring_slots = 0;
        if (pin_budget_ > 0) {
            held = pin_bytes_ < bytes_ ? pin_bytes_ : bytes_;
            ring_slots = pin_slots_ < vram_slots_ ? pin_slots_ : vram_slots_;
        }
        const size_t pool = bytes_ - held;
        const size_t pool_slots = vram_slots_ - ring_slots;
        // A request with no bytes still takes a slot, so it falls back to the
        // count rule: unbounded residency for a degenerate source would grow the
        // slot map without limit.
        if (need == 0) return capacity_ > 0 && pool_slots >= capacity_;
        return pool + need > budget_ ||
               (slot_budget_ > 0 && pool_slots >= slot_budget_);
    }
    // The allocator's physical cap: the LRU pool's budget PLUS the ring's,
    // because both are device residency this cache owns. Handing the backend
    // only budget_ made the pool refuse the ring's own bytes -- measured on
    // Tiel-Coder-35B-A3B: "expert pool physical budget exhausted: live=5583 MB
    // need=860 KB budget=3396 MB" after the ring admitted 5325 MiB and the
    // clamp shrank the pool to 3239, and the run died on its first token.
    size_t owner_bytes(size_t need = 0) const {
        size_t total = budget_ + pin_budget_;
        const size_t live = bytes_ + need;
        if (live > total) total = live;
        return total;
    }
    // Evicts WARM copies until need bytes fit. Pins do not apply here.
    void make_warm_room(size_t need);
    // Drops a slot from the map once it lives in neither tier.
    void forget_if_dead(std::unordered_map<u64, Slot>::iterator it);
    // Halves counters past the step and releases pins that no longer qualify.
    void decay();
    // Acquires per decay window. Scales with the slot count so a pin
    // stays earnable at any model size: one window per ~kPinDecay
    // full sweeps of the resident set.
    u64 decay_period() const {
        const u64 slots = capacity_ > 0 ? static_cast<u64>(capacity_) : 1ULL;
        return static_cast<u64>(kPinDecay) * slots;
    }
    // Picks the slot to sacrifice from device: unpinned, then rarest, then
    // oldest. Host-tier copies are not candidates — they are not in the way.
    bool pick_victim(u64 *out);
    // Resident-slot index used by pick_victim: O(1) insert, O(1) swap-remove.
    void track_vram(const u64 key, Slot &s);
    void untrack_vram(Slot &s);
    // WARM victim: a copy that is not backing a HOT slot goes first, then the
    // same frequency ordering as the device tier. Pins do not protect a copy
    // here.
    bool pick_warm_victim(u64 *out);
    // The pool answers nullptr, not an abort, when the card has no room
    // beside its VRAM floor and it holds no free cell of the requested size
    // class. That is the steady state on the 16 GiB / 6-12 GiB targets this
    // engine is built for, so a refused allocation is answered by giving the
    // pool back cells this cache already owns -- the WARM copy survives, so
    // the reclaimed expert costs a copy instead of a file read.
    // False when there is nothing left to give back.
    // `want` is the slice size the caller needs a cell for: a victim of the
    // same size class hands its cells straight back (`alloc_pooled` looks
    // a cell up by rounded size), so it is preferred over a plain LRU
    // pick. 0 falls back to pick_victim().
    bool force_reclaim(size_t want = 0);

    Backend *be_ = nullptr;
    size_t budget_ = 0;
    size_t bytes_ = 0;
    size_t capacity_ = 0;
    size_t slot_budget_ = 0; // exact slot ceiling, 0 = derive from the bytes
    size_t vram_slots_ = 0;
    // Device allocations the pool refused on a full card, and the evictions
    // that answered them. Both are per-run evidence: refusals without
    // reclaims is the state that used to reach hipMemcpy with a null
    // destination.
    size_t pool_refusals_ = 0;
    size_t reclaims_ = 0;
    size_t host_budget_ = 0;
    size_t host_bytes_ = 0;
    // The ring: bytes of pinned device residency that the pool's budget does
    // not count, and how many slots they are, so over_budget() is O(1).
    size_t pin_budget_ = 0;
    size_t pin_bytes_ = 0;
    size_t pin_slots_ = 0;
    // The one place that flips `pinned`, so the two counters above can never
    // disagree with it.
    void set_pinned(Slot &s, bool on);
    // The pinned prefix of WARM: at most this many bytes of the tier are
    // admitted into pinned host memory, in admission order (rank order when the
    // model carries an expert index), so a promotion out of it is a DMA with no
    // host memcpy in front of it.
    size_t host_pin_bytes_ = 0;
    size_t host_pinned_bytes_ = 0;
    u64 acquires_ = 0;      // since the last decay (drives the window)
    u64 total_acquires_ = 0; // lifetime, for hit-rate reporting
    u64 loads_ = 0;
    u64 evictions_ = 0;
    u64 hits_ = 0;
    u64 host_hits_ = 0;
    u64 demotions_ = 0;
    u64 warm_admissions_ = 0;
    u64 warm_evictions_ = 0;
    u64 warm_rejects_ = 0;
    u64 bytes_loaded_ = 0;   // read out of the model file, via WARM
    u64 bytes_demoted_ = 0;  // VRAM evicted with a WARM copy already in place
    u64 bytes_promoted_ = 0; // WARM -> VRAM
    u64 bytes_staged_ = 0;   // file -> WARM ahead of the run (prefetch)
    u64 lookahead_staged_ = 0;   // stage_batch promotions
    u64 lookahead_rejected_ = 0; // stage_batch candidates the budget could not hold
    f64 promote_ms_ = 0;     // wall time in promote_to_vram
    f64 read_ms_ = 0;        // wall time in the WARM file read
    f64 room_ms_ = 0;        // wall time making VRAM room
    f64 scan_ms_ = 0;        // ... of which the victim search
    f64 retire_ms_ = 0;      // ... and the cell release
    u64 scans_ = 0, scan_visits_ = 0;
    // The resident keys, and the hand the windowed victim search starts from.
    // A scan of `slots_` is the wrong shape for this: the tier holds ~3.5k
    // resident entries inside ~14k WARM ones, so a full-map pick walked 3503
    // entries per eviction (measured: 9.19M visits / 2624 evictions, 505.9 ms
    // of a 1.7 s decode, 55 ns a visit).
    std::vector<u64> vram_keys_;
    size_t victim_cursor_ = 0;
    f64 alloc_ms_ = 0;       // wall time in alloc_pooled
    f64 xfer_ms_ = 0;        // wall time in the pageable -> VRAM batch copy
    // Per-phase twins of the five timers above. Only PhasePrefill and
    // PhaseDecode accumulate; PhaseOther (warmup, prefetch) touches neither.
    f64 pf_promote_ms_ = 0, pf_read_ms_ = 0, pf_room_ms_ = 0,
        pf_alloc_ms_ = 0, pf_xfer_ms_ = 0;
    f64 dc_promote_ms_ = 0, dc_read_ms_ = 0, dc_room_ms_ = 0,
        dc_alloc_ms_ = 0, dc_xfer_ms_ = 0;
    Phase phase_ = PhaseOther;
    // The phase bucket for a stage counter, or null outside prefill/decode so
    // warmup time is not silently attributed to whichever phase ran next.
    f64 *phase_sink(f64 *pf, f64 *dc) {
        return phase_ == PhasePrefill ? pf
             : phase_ == PhaseDecode  ? dc
                                      : nullptr;
    }
    u64 active_staged_ = 0, active_used_ = 0;
    f64 active_wait_ms_ = 0;
    u64 promotions_ = 0;
    u64 promote_batches_ = 0;
    u64 promote_batch_experts_ = 0;
    u64 decays_ = 0;
    u64 seq_ = 0;
    std::unordered_map<u64, Slot> slots_;
    // The published device dispatch table (see publish_dispatch_table) and the
    // flag that says a residency change invalidated it. `table_stale_` is set by
    // every write to a slot's device pointers -- drop() and the promotion that
    // assigns `s.m` -- so a caller holding the pointer re-reads it and falls
    // back instead of issuing GEMMs against freed addresses.
    void *table_ = nullptr;
    size_t table_bytes_ = 0;
    bool table_stale_ = false;
};

} // namespace krk

#endif // KRK_EXPERT_CACHE_HPP
