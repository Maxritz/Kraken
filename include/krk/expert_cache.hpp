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

#include <list>
#include <unordered_map>

#include "krk/backend.hpp"
#include "krk/gguf.hpp"
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
    i32 n_expert = 0;
    i64 n_embd = 0;   // inner dimension of the gate/up matrices
    i64 n_ff_exp = 0; // expert hidden width

    bool present() const { return gate != nullptr && up != nullptr && down != nullptr; }
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
    void configure(Backend *be, size_t budget_bytes, size_t host_budget_bytes = 0);
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
    f64 alloc_ms() const { return alloc_ms_; } // alloc_pooled: pool wait or malloc
    f64 xfer_ms() const { return xfer_ms_; }   // upload_paged_batch, host wait in

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
    size_t capacity_slots() const { return capacity_; }
    // WARM tier (the second tier). The warm_* names are the ones the reports
    // use; the host_* spellings are kept for existing callers.
    size_t warm_capacity_bytes() const { return host_budget_; }
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

    // True when promoting one more expert of `src` would have to evict
    // something, i.e. the device tier is already full. The hybrid CPU+GPU path
    // uses it to take the OVERFLOW only: with a budget that holds the whole
    // routed set every expert keeps arriving through the normal path and the
    // host arm stays idle, which is the intent. Measured the other way first --
    // hybrid taking every miss -- and it read 6.4 tok/s against 99.3, because
    // an expert that is never promoted never becomes resident, so a budget that
    // could hold the entire set stayed cold forever.
    bool full_for(const ExpertSource &src) const;

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
        // WARM is inclusive: a slot may sit in both tiers at once, and a HOT
        // slot's WARM copy is what makes its eviction free. A slot in neither
        // tier is forgotten (see forget_if_dead).
        bool in_vram() const { return m.gate.present(); }
        bool in_host() const { return h.gate.present(); }
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
    bool promote_to_vram(Slot &s);
    // Evicts until one more device slot fits. Over-resides by one rather than
    // stalling when every resident slot is pinned.
    void make_vram_room();
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
    // WARM victim: a copy that is not backing a HOT slot goes first, then the
    // same frequency ordering as the device tier. Pins do not protect a copy
    // here.
    bool pick_warm_victim(u64 *out);

    Backend *be_ = nullptr;
    size_t budget_ = 0;
    size_t bytes_ = 0;
    size_t capacity_ = 0;
    size_t vram_slots_ = 0;
    size_t host_budget_ = 0;
    size_t host_bytes_ = 0;
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
    f64 promote_ms_ = 0;     // wall time in promote_to_vram
    f64 read_ms_ = 0;        // wall time in the WARM file read
    f64 room_ms_ = 0;        // wall time making VRAM room
    f64 alloc_ms_ = 0;       // wall time in alloc_pooled
    f64 xfer_ms_ = 0;        // wall time in the pageable -> VRAM batch copy
    u64 promotions_ = 0;
    u64 decays_ = 0;
    u64 seq_ = 0;
    std::unordered_map<u64, Slot> slots_;
};

} // namespace krk

#endif // KRK_EXPERT_CACHE_HPP
