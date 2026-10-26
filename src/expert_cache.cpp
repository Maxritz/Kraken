// expert_cache.cpp — lazy, frequency-aware residency for MoE expert matrices.
//
// The policy is LFU with aging: hits earn counters, counters pin hot experts,
// and periodic decay makes pins expire when traffic moves on. See the header
// for the reasoning — expert popularity is skewed and sticky, which is exactly
// the workload LRU is worst at.
//
// Every cold read is read-through: the file read lands in WARM first and is
// copied on to VRAM from there, so a compulsory miss is paid once and the next
// request for that expert is a promotion. WARM is inclusive — a promotion keeps
// the RAM copy — which is what makes a VRAM eviction free (no download, no
// writeback), and a WARM eviction is a plain free(), because the file already
// holds the canonical copy. The tier can also be filled ahead of the run
// (preload, below): on a stack with hundreds of experts per layer a short
// generation is dominated by *first* touches, and those are a miss the policy
// above can never turn into a hit.
#include "krk/expert_cache.hpp"
#include "krk/host_time.hpp"
#include "krk/vram_guard.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <thread>

// ---------------------------------------------------------------------------
// Expert access trace (opt-in, KRK_TRACE_EXPERTS=1).
//
// The tier counters answer "how often did each tier win". They cannot answer
// the question the cache policy actually turns on: how far apart in the access
// stream two uses of the same expert are. That distance is what decides both
// how big HOT has to be and whether WARM earns its keep at all -- a reuse
// distance shorter than the HOT slot count is invisible to WARM, and a longer
// one makes WARM mandatory. Both cases look identical in the counters.
//
// This records one line per acquire so tools/expert_trace.py can build the
// distance histogram offline. It is off by default and costs one branch when
// off; the buffer is bounded so a long run cannot grow without limit.
// ---------------------------------------------------------------------------
namespace {

using krk::u32;
using krk::i32;
using krk::u8;

// Adds the time it was alive to *acc. Four of these carve the expert path into
// the parts that cost, so "promotion is 1.8 GB/s" can be answered with "the
// copy is 7 GB/s and the room-making is the rest". The optional second
// accumulator is the prefill/decode phase twin: a stage inside prefill counts
// into the merged row AND into the phase row, so the three phases sum to the
// merged figure exactly.
struct MsTimer {
    krk::f64 *acc;
    krk::f64 *acc2;
    std::chrono::steady_clock::time_point t0;
    explicit MsTimer(krk::f64 *a, krk::f64 *b = nullptr)
        : acc(a), acc2(b), t0(std::chrono::steady_clock::now()) {}
    MsTimer(const MsTimer &) = delete;
    MsTimer &operator=(const MsTimer &) = delete;
    ~MsTimer() {
        const krk::f64 ms = std::chrono::duration<krk::f64, std::milli>(
                               std::chrono::steady_clock::now() - t0)
                               .count();
        *acc += ms;
        if (acc2) *acc2 += ms;
    }
};

// HostTime twin of MsTimer: the tier counters say what the cache spent inside
// its own phases, but they never appeared in the host table, so a run whose
// step time was mostly *reading experts off the disk tier* reported that time
// under "unaccounted" and looked like it was somewhere else entirely. A read is
// the one host cost on a big MoE that no backend op can see (it is a pread, not
// a kernel), so it is measured here, in the same table as the phases it sits
// inside.
struct HostAdd {
    const char *name;
    bool concurrent;
    std::chrono::steady_clock::time_point t0;
    explicit HostAdd(const char *n, bool conc = false)
        : name(n), concurrent(conc), t0(std::chrono::steady_clock::now()) {}
    HostAdd(const HostAdd &) = delete;
    HostAdd &operator=(const HostAdd &) = delete;
    ~HostAdd() {
        if (!krk::HostTime::get().on()) return;
        const krk::f64 ms = std::chrono::duration<krk::f64, std::milli>(
                                std::chrono::steady_clock::now() - t0)
                                .count();
        krk::HostTime::get().add(name, ms, concurrent);
    }
    void bytes(krk::i64 n) {
        if (krk::HostTime::get().on()) krk::HostTime::get().add_bytes(name, n);
    }
};

struct XRec {
    u32 seq;
    i32 layer;
    i32 expert;
    u8 tier;  // 0 = HOT hit, 1 = WARM hit (promotion), 2 = COLD miss
};

std::vector<XRec> g_xrec;
bool g_xtrace_on = false;
const char *g_xtrace_path = "expert_trace.tsv";

struct XInit {
    XInit() {
        const char *v = std::getenv("KRK_TRACE_EXPERTS");
        g_xtrace_on = v && *v && std::string(v) != "0";
        const char *p = std::getenv("KRK_TRACE_PATH");
        if (p && *p) g_xtrace_path = p;
        if (g_xtrace_on) g_xrec.reserve(1u << 22);
    }
};

void xtrace_dump() {
    if (!g_xtrace_on || g_xrec.empty()) return;
    FILE *f = std::fopen(g_xtrace_path, "w");
    if (!f) return;
    std::fprintf(f, "seq\tlayer\texpert\ttier\n");
    for (const XRec &r : g_xrec)
        std::fprintf(f, "%u\t%d\t%d\t%u\n", r.seq, r.layer, r.expert,
                     static_cast<unsigned>(r.tier));
    std::fclose(f);
}

struct XAtExit {
    ~XAtExit() { xtrace_dump(); }
};

XInit g_xinit;
XAtExit g_xexit;

inline void xtrace(i32 layer, i32 expert, u8 tier) {
    if (!g_xtrace_on) return;
    if (g_xrec.size() >= g_xrec.capacity()) return;  // bounded; no realloc storm
    g_xrec.push_back(XRec{static_cast<u32>(g_xrec.size()), layer, expert, tier});
}

}  // namespace

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_set>

namespace krk {

namespace {

u64 slot_key(i32 layer, i32 expert) {
    return (static_cast<u64>(static_cast<u32>(layer)) << 32) |
           static_cast<u32>(expert);
}

// ---- one host block per expert ---------------------------------------------
//
// The WARM tier used to allocate each expert THREE times, once per slice, and
// that shape is what the load phase was paying for: on this box (RX 9070 XT,
// HIP, tools/probe_warmalloc.hip, one 17.2 GiB pinned tier) 30,720 slice-sized
// (0.573 MiB) allocations cost 4110 ms at 912 ns per page locked, against
// 2122 ms at 471 ns for 10,240 expert-sized (1.72 MiB) ones and 1810 ms at
// 401 ns for 16 experts per block. The per-CALL component is only ~10 us -- the
// same 30,720 calls at a tenth of the pages cost 324 ms -- so the win is the
// length of the page run the driver probes and locks, and three-to-one is where
// most of it is.
//
// Going further was measured and declined. A block shared by several experts
// cannot be handed back one expert at a time, so it needs a slab allocator's
// free list, and reusing a freed slot needs the promotion-completion event that
// the per-slice deferred free exists to honour: a slot handed to a new expert
// while a copy out of it is still in flight is the use-after-free that shows up
// as plausible text that differs every run. Sixteen experts per block buys
// another 310 ms of a 6 s phase; this shape buys 2 s for one pointer discipline.
//
// The churning tier is what settles it, because that is where slab granularity
// would have to pay off. Nemotron Bolt 74B, --expert-warm-mb 8192 (8 GiB over a
// 22 GiB file) and the ranked warmup on, 16 tokens: this shape runs 11.5 tok/s
// (87.3 ms/step) with the tier at 100% in use, 4245 admissions, 4253 evictions
// and ZERO rejects -- every eviction reclaimed exactly one expert's block, so
// there was no stranded capacity for a slab-sized victim to recover. The same
// run on the 16-slot slab build gave 5.8 tok/s: 6674 evictions for 2339
// admissions, the tier ending 5% resident with 5284 MiB committed free, because
// a 1-slot admission freed a 16-slot slab and refilled one slot.
//
// So: one allocation per expert, slices at aligned offsets inside it. Every
// release frees the FIRST slice's pointer, which is the block base -- that is
// the whole invariant, and warm_block_base is the only place that decides it.
constexpr size_t kWarmSliceAlign = 64;

inline size_t align_up(size_t v, size_t a) {
    return (v + a - 1) / a * a;
}

// Offsets of each slice inside one expert's block, and the block's size. A
// slice with no tensor behind it (nemotron_h_moe has no ffn_gate_exps) takes no
// space and keeps the offset it would have had.
struct WarmBlockLayout {
    size_t off[3];
    size_t total;
};

inline WarmBlockLayout warm_block_layout(const size_t slice[3]) {
    WarmBlockLayout b{{0, 0, 0}, 0};
    size_t at = 0;
    for (int i = 0; i < 3; i++) {
        if (slice[i] == 0) {
            b.off[i] = at;
            continue;
        }
        // Align a slice's start to the DMA / vector-load width, but never past
        // the slice's own size: a fixture with 8-byte tensors must not pay 64
        // bytes of padding each (no tensor in a real file is under 64 bytes, so
        // the real tier sees a full 64-byte start for every slice).
        const size_t a = slice[i] < kWarmSliceAlign ? slice[i] : kWarmSliceAlign;
        at = align_up(at, a);
        b.off[i] = at;
        at += slice[i];
    }
    b.total = at;
    return b;
}

// The pointer a block is released through: its lowest-offset slice.
inline void *warm_block_base(void *const dst[3]) {
    for (int i = 0; i < 3; i++)
        if (dst[i]) return dst[i];
    return nullptr;
}

// Releases one expert's host block. `deferred` keeps the bytes alive until a
// copy issued out of them has landed, which is the only reason the per-slice
// path had two deallocators to choose between; the block inherits both rules
// unchanged.
inline void warm_release(Backend *be, void *base, bool pinned, bool deferred) {
    if (!be || !base) return;
    if (deferred) {
        if (pinned) be->release_host_pinned_deferred(base);
        else be->release_host_deferred_budgeted(base);
    } else {
        if (pinned) be->release_host(base);
        else be->release_host_pageable_budgeted(base);
    }
}

} // namespace

void ExpertCache::configure(Backend *be, size_t budget_bytes, size_t host_budget_bytes,
                            size_t slot_budget) {
    clear();
    be_ = be;
    budget_ = budget_bytes;
    host_budget_ = host_budget_bytes;
    slot_budget_ = slot_budget;
    // KRK_EXPERT_WARM_MAP=1: the page cache is the second tier and nothing is
    // kept in host RAM (see the header). Read once per configure so the A/B is
    // one environment variable wide and the rest of the file asks a bool.
    if (const char *e = std::getenv("KRK_EXPERT_WARM_MAP")) warm_map_ = e[0] == '1';
    // KRK_HOST_PIN_MB: how much of WARM to admit into pinned host memory. Off by
    // default, because the arm has to exist before the measurement (and because
    // pinned pages are a scarce, non-swappable pool on Windows). Clamped to the
    // tier it decorates: a pin budget larger than WARM would just never bind.
    host_pin_bytes_ = 0;
    host_pinned_bytes_ = 0;
    if (const char *e = std::getenv("KRK_HOST_PIN_MB")) {
        const long long mib = std::atoll(e);
        if (mib > 0) {
            const size_t want = static_cast<size_t>(mib) * 1024u * 1024u;
            host_pin_bytes_ = host_budget_ ? std::min(want, host_budget_) : want;
        }
    }
    // An explicit slot budget IS the capacity. Without one this stays 0 until
    // acquire() fills it in from the expert size of the layer it is serving.
    capacity_ = slot_budget;
    if (be_) be_->set_expert_pool_owner(this, owner_bytes());
}

void ExpertCache::set_host_pin_budget(size_t bytes) {
    host_pin_bytes_ = host_budget_ ? std::min(bytes, host_budget_) : bytes;
    host_pinned_bytes_ = 0;
}

void ExpertCache::set_vram_budget(size_t bytes) {
    if (bytes >= budget_) return;
    budget_ = bytes;
    // Without an explicit slot budget the capacity is derived from the byte
    // budget on the next acquire(); clear it so the report cannot keep quoting
    // the count the old budget implied.
    if (slot_budget_ == 0) capacity_ = 0;
    if (be_) be_->set_expert_pool_owner(this, owner_bytes());
}

void ExpertCache::clear() {
    // The read-ahead worker holds buffers that are about to stop being reachable
    // from the slots below: stop it first, or a completed entry installs itself
    // into a generation that no longer exists.
    prefetch_warm_join();
    // The touch scratch is the only host buffer here that is not a slot's, and
    // it belongs to the backend that allocated it -- which is why this happens
    // before configure() reassigns be_.
    if (touch_buf_ && be_) be_->release_host_pageable(touch_buf_);
    touch_buf_ = nullptr;
    touch_buf_bytes_ = 0;
    {
        std::lock_guard<std::mutex> lk(pf_mu_);
        pf_stop_ = false; // a cleared cache may be used again
    }
    // The published table points at slots this call is about to release.
    release_dispatch_table();
    for (auto &kv : slots_) {
        drop(kv.second);
        drop_host(kv.second);
    }
    slots_.clear();
    vram_keys_.clear();
    victim_cursor_ = 0;
    if (be_) be_->set_expert_pool_owner(this, 0);
    bytes_ = 0;
    vram_slots_ = 0;
    pin_bytes_ = 0;
    pin_slots_ = 0;
    // A cleared cache starts a new residency generation, so the free-VRAM
    // reading the re-clamp measures against has to start over too.
    last_free_bytes_ = 0;
    have_free_bytes_ = false;
    host_bytes_ = 0;
    // Pinned bytes are a live figure like host_bytes_, and they are decremented
    // per slot in drop_host(): leaving the total behind makes every later
    // admission believe the pin prefix is already spent.
    host_pinned_bytes_ = 0;
    capacity_ = 0;
    acquires_ = 0;
    total_acquires_ = 0;
    seq_ = 0;
    // Statistics cover the current residency generation: clear() starts a new
    // one, so callers (and tests) read numbers about the policy they configured.
    loads_ = 0;
    evictions_ = 0;
    hits_ = 0;
    host_hits_ = 0;
    demotions_ = 0;
    promotions_ = 0;
    decays_ = 0;
    warm_admissions_ = 0;
    warm_evictions_ = 0;
    warm_rejects_ = 0;
    active_staged_ = active_used_ = 0;
    active_wait_ms_ = 0;
}

void ExpertCache::reset_counters() {
    // Deliberately NOT clear(): residency is the warmup's product and has to
    // survive. Only the traffic it generated is discarded, so the report that
    // follows describes the run. clear() also leaves the byte and millisecond
    // accumulators alone (they cover the whole process), which is why they are
    // reset here by hand rather than inherited.
    loads_ = 0;
    evictions_ = 0;
    hits_ = 0;
    host_hits_ = 0;
    demotions_ = 0;
    promotions_ = 0;
    decays_ = 0;
    acquires_ = 0;
    total_acquires_ = 0;
    warm_admissions_ = 0;
    warm_evictions_ = 0;
    warm_rejects_ = 0;
    bytes_loaded_ = 0;
    bytes_demoted_ = 0;
    bytes_promoted_ = 0;
    bytes_staged_ = 0;
    promote_ms_ = 0.0;
    read_ms_ = 0.0;
    room_ms_ = 0.0;
    scan_ms_ = 0.0;
    retire_ms_ = 0.0;
    scans_ = 0;
    scan_visits_ = 0;
    // vram_keys_ is NOT touched here. This function deliberately keeps
    // residency -- the warmup's product -- and an index cleared beside a
    // resident tier is worse than no index: pick_victim then sees one key for
    // thousands of resident slots, evicts by a corner of the set, and the
    // promotions rose 2589 -> 6754 on the REAP-320 file before this line was
    // removed (2.6x the traffic, xfer 578 -> 3537 ms).
    alloc_ms_ = 0.0;
    xfer_ms_ = 0.0;
    pf_promote_ms_ = pf_read_ms_ = pf_room_ms_ = pf_alloc_ms_ = pf_xfer_ms_ = 0.0;
    dc_promote_ms_ = dc_read_ms_ = dc_room_ms_ = dc_alloc_ms_ = dc_xfer_ms_ = 0.0;
    active_staged_ = active_used_ = 0;
    active_wait_ms_ = 0;
}

u32 ExpertCache::touch_count(i32 layer, i32 expert) const {
    const auto it = slots_.find(slot_key(layer, expert));
    return it == slots_.end() ? 0u : it->second.count;
}

// How many expert uses share one reader-done event. A decoder layer's uses are
// contiguous on the default stream, so n of them can share a single event at the
// cost of a slightly later mark -- and the mark is only what a REUSE waits for,
// which for a cell evicted a layer later is already long past. 8 is one record
// per layer at top-k 8: 12.7k records become 2.2k over 32 tokens on REAP-288 and
// the step is 0.7% faster (faster in 6 of 7 interleaved pairs). 1 restores one
// record per use, which is what section 12o measured.
// `KRK_POOL_MARK_BURST=<n>` is the arm; docs/PERF-ANALYSIS.md section 12q.
static u32 read_burst() {
    static const u32 v = [] {
        const char *e = std::getenv("KRK_POOL_MARK_BURST");
        const int n = e && *e ? std::atoi(e) : 8;
        return static_cast<u32>(n < 1 ? 1 : (n > 512 ? 512 : n));
    }();
    return v;
}

// Read-only residency probes. Deliberately not touch(): a caller that is
// deciding whether to *avoid* a promotion must be able to ask without changing
// the LFU counters or the load order that the eviction policy ranks on.
void ExpertCache::note_read(i32 layer, i32 expert) {
    if (!be_) return;
    const auto it = slots_.find(slot_key(layer, expert));
    if (it == slots_.end()) return;
    Slot &s = it->second;
    if (!s.in_vram()) return; // an in-place read touches no pool cell
    // The pending burst belongs to ONE layer: a mark recorded after the next
    // layer's first read would still be correct, but it would also cover that
    // read, and the point of the per-slot mark is to be as tight as possible.
    if (layer != rd_pending_layer_ && !rd_pending_.empty()) flush_read_notes();
    rd_pending_layer_ = layer;
    for (QuantTensor *q : {&s.m.gate, &s.m.up, &s.m.down})
        if (q->present() && q->data) rd_pending_.push_back(q->data);
    if (++rd_pending_experts_ >= read_burst()) flush_read_notes();
}

void ExpertCache::flush_read_notes() {
    if (!rd_pending_.empty() && be_)
        be_->pool_note_reader_done(rd_pending_.data(),
                                   static_cast<int>(rd_pending_.size()));
    rd_pending_.clear();
    rd_pending_experts_ = 0;
}

bool ExpertCache::in_vram(i32 layer, i32 expert) const {
    const auto it = slots_.find(slot_key(layer, expert));
    return it != slots_.end() && it->second.in_vram();
}

bool ExpertCache::in_host(i32 layer, i32 expert) const {
    const auto it = slots_.find(slot_key(layer, expert));
    return it != slots_.end() && it->second.in_host();
}

void ExpertCache::seed_count(i32 layer, i32 expert, u32 count) {
    const auto it = slots_.find(slot_key(layer, expert));
    if (it == slots_.end()) return;
    // Clamped below the pin threshold on purpose: see the header. A seed tells
    // the eviction ranking what the warmup measured, it does not promise VRAM.
    const u32 pin = pin_threshold();
    if (pin > 0 && count >= pin) count = pin - 1;
    Slot &s = it->second;
    if (s.count < count) s.count = count;
}

bool ExpertCache::full_for(const ExpertSource &src) const {
    const size_t need = src.expert_bytes();
    // A source with no bytes to promote is not "full", it is unusable: report
    // false so the caller takes the ordinary path and the acquire() below it
    // fails the same way it always did, rather than handing the host arm an
    // expert it also cannot compute.
    if (need == 0) return false;
    return bytes_ + need > budget_;
}

// ---------------------------------------------------------------------------
// The device dispatch table (see the header). Two facts make it publishable:
// every (layer, expert) of the corpus is in VRAM, and nothing in the run will
// move one. The first is checkable and the second follows from it -- a slot
// that is resident is only rewritten by a promotion or an eviction, and a
// corpus the budget already holds has neither. A budget that does not hold the
// set answers false and the caller keeps the host path.
// ---------------------------------------------------------------------------

bool ExpertCache::whole_set_resident(i32 n_layer, i32 n_expert) const {
    if (!be_ || n_layer <= 0 || n_expert <= 0) return false;
    for (i32 l = 0; l < n_layer; l++) {
        for (i32 e = 0; e < n_expert; e++) {
            auto it = slots_.find(slot_key(l, e));
            if (it == slots_.end() || !it->second.in_vram()) return false;
        }
    }
    return true;
}

void ExpertCache::release_dispatch_table() {
    if (table_ && be_) be_->release(table_);
    table_ = nullptr;
    table_bytes_ = 0;
    table_stale_ = false;
    table_host_cells_ = 0;
}

void *ExpertCache::publish_dispatch_table(i32 n_layer, i32 n_expert,
                                          i32 *published_out, std::string *why) {
    if (published_out) *published_out = 0;
    if (!be_) {
        if (why) *why = "no backend";
        return nullptr;
    }
    if (n_layer <= 0 || n_expert <= 0) {
        if (why) *why = "no expert geometry";
        return nullptr;
    }
    // The promise the table has to keep is ADDRESSABILITY, not residency.
    //
    // It used to be residency, and that made the whole arm unreachable on the
    // models this tier exists for: Tiel-Coder-35B-A3B is 22.3 GiB of routed
    // experts against a 9-11 GiB card budget, so `whole_set_resident` was false
    // for the entire run and the log said "0 of 10240 slots" from start to
    // finish. A kernel can read a slot's PINNED host copy directly -- measured
    // on this card at 24.3 GiB/s against 485 GiB/s for the same read out of
    // VRAM, with the bytes read back exactly (tools/probe_hostread.hip) -- so a
    // cell that lives in the pinned pool is as legal an operand as one in the
    // pool's VRAM cells, and the set that does not fit reads its overflow where
    // it lives instead of paying a host round trip per expert.
    //
    // What is NOT legal is a cell that is absent or pageable: the kernel would
    // have to skip it, and a skipped expert is a wrong answer, not a slow one.
    const size_t cells = static_cast<size_t>(n_layer) * static_cast<size_t>(n_expert);
    const bool host_ok = be_->host_load_supported();
    i32 in_vram = 0, in_pinned = 0, absent = 0, pageable = 0;
    for (i32 l = 0; l < n_layer; l++) {
        for (i32 e = 0; e < n_expert; e++) {
            auto it = slots_.find(slot_key(l, e));
            if (it == slots_.end()) {
                ++absent;
                continue;
            }
            const Slot &s = it->second;
            if (s.in_vram()) ++in_vram;
            else if (s.in_host() && s.host_pinned && host_ok) ++in_pinned;
            else ++pageable;
        }
    }
    if (absent || pageable) {
        if (why) {
            *why = "the routed set is not addressable from a kernel (";
            *why += std::to_string(in_vram);
            *why += " in VRAM, ";
            *why += std::to_string(in_pinned);
            *why += " read in place from pinned host, ";
            *why += std::to_string(absent);
            *why += " never loaded, ";
            *why += std::to_string(pageable);
            *why += " pageable of ";
            *why += std::to_string(static_cast<long long>(cells));
            *why += pageable && !host_ok
                        ? " -- this backend has not measured a kernel reading "
                          "pinned host memory)"
                        : " -- pin the whole routed set (--expert-pin-mb) or grow "
                          "the device budget)";
        }
        return nullptr;
    }

    std::vector<MoeDispatchEntry> host(cells);
    i32 filled = 0;
    for (i32 l = 0; l < n_layer; l++) {
        for (i32 e = 0; e < n_expert; e++) {
            auto it = slots_.find(slot_key(l, e));
            if (it == slots_.end()) continue;
            const Slot &slot = it->second;
            // A resident slot names its pool cells; a pinned host slot names the
            // WARM copy's slices, which the kernel dereferences the same way.
            const bool resident = slot.in_vram();
            const ResidentExpert &m = resident ? slot.m : slot.h;
            const QuantTensor *q[3] = {&m.gate, &m.up, &m.down};
            MoeDispatchEntry &d = host[static_cast<size_t>(l) *
                                       static_cast<size_t>(n_expert) +
                                       static_cast<size_t>(e)];
            bool all = true;
            for (int i = 0; i < 3; i++) {
                if (!q[i]->present()) continue;
                d.p[i] = q[i]->data;
                d.n_out[i] = q[i]->n_out;
                d.n_in[i] = q[i]->n_in;
                d.row_bytes[i] = static_cast<std::int64_t>(
                    dtype_row_bytes(q[i]->type, q[i]->n_in));
                d.type[i] = static_cast<std::int32_t>(q[i]->type);
                d.present[i] = 1;
            }
            // gate + up + down is the shape the dispatch needs. An expert with
            // only two of them is a different FFN and belongs to the host path.
            all = d.present[kMoeGate] && d.present[kMoeUp] && d.present[kMoeDown];
            if (all) filled++;
        }
    }
    if (filled != static_cast<i32>(cells)) {
        if (why) {
            *why = "only " + std::to_string(filled) + " of " +
                   std::to_string(static_cast<long long>(cells)) +
                   " experts carry all three matrices";
        }
        return nullptr;
    }

    release_dispatch_table();
    const size_t bytes = cells * sizeof(MoeDispatchEntry);
    table_ = be_->alloc(bytes);
    if (!table_) {
        if (why) *why = "the table allocator refused " + std::to_string(bytes) + " bytes";
        return nullptr;
    }
    table_bytes_ = bytes;
    be_->upload(table_, host.data(), bytes);
    table_stale_ = false;
    table_host_cells_ = in_pinned;
    if (published_out) *published_out = filled;
    return table_;
}

size_t ExpertCache::pinned_slots() const {
    size_t n = 0;
    for (const auto &kv : slots_)
        if (kv.second.pinned) n++;
    return n;
}

size_t ExpertCache::warm_slots() const {
    size_t n = 0;
    for (const auto &kv : slots_)
        if (kv.second.in_host()) n++;
    return n;
}

void ExpertCache::set_pinned(Slot &s, bool on) {
    if (s.pinned == on) return;
    s.pinned = on;
    if (on) {
        pin_bytes_ += s.bytes;
        pin_slots_++;
    } else {
        const size_t drop = std::min(pin_bytes_, s.bytes);
        pin_bytes_ -= drop;
        if (pin_slots_) pin_slots_--;
    }
}

size_t ExpertCache::reclamp_vram_budget(size_t free_bytes, size_t floor_bytes) {
    if (budget_ == 0 || bytes_ == 0) return budget_;
    const size_t room = free_bytes > floor_bytes ? free_bytes - floor_bytes : 0;
    const size_t live = bytes_ > pin_bytes_ ? bytes_ - pin_bytes_ : 0;
    const size_t cap = live + room;
    // A driver reading moves by a few MiB on its own; a budget that chases it
    // re-plans the cache every step for nothing. The tolerance is a FRACTION of
    // the budget (1/64, 86 MiB on a 5.5 GiB pool) rather than an absolute 32 MiB,
    // so the rule is testable at a fixture's 96 bytes instead of being dominated
    // by a constant that is larger than the whole test cache.
    const size_t margin = budget_ / 64;
    // Shrink only by what the CARD gave up since the previous reading. The
    // cache's own evictions do not appear in `free_bytes`: retire() hands the
    // cells back to the backend's pool (release_pooled_batch), not to the
    // driver, so hipMemGetInfo's free count does not move when a slot is
    // dropped. A cap computed from live residency alone therefore removed what
    // the cache had just evicted, at every clamp, and the budget walked
    // 5747 -> 4295 MiB over 93 tokens on the 288-expert file (and 4697 -> 359
    // on a longer chat) while free VRAM sat still. `taken` is the part of the
    // shortfall somebody else caused; room-making by the cache itself leaves it
    // at zero, which makes the clamp a no-op instead of a ratchet.
    size_t taken = 0;
    if (have_free_bytes_ && free_bytes < last_free_bytes_)
        taken = last_free_bytes_ - free_bytes;
    have_free_bytes_ = true;
    last_free_bytes_ = free_bytes;
    if (taken <= margin) return budget_;
    const size_t shortfall = cap < budget_ ? budget_ - cap : 0;
    if (shortfall <= margin) return budget_;
    const size_t cut = taken < shortfall ? taken : shortfall;
    if (cut <= margin) return budget_;
    set_vram_budget(budget_ - cut);
    return budget_;
}

void ExpertCache::set_pin_budget(size_t bytes) {
    pin_budget_ = bytes;
    if (be_) be_->set_expert_pool_owner(this, owner_bytes());
    // A ring smaller than what it already holds is not retroactively evicted:
    // admit_hot() refuses to grow it, and what is there was admitted against a
    // budget that was true when it was measured. Tightening it further is the
    // caller's business (clear() and a re-admit).
}

void ExpertCache::touch(Slot &s) {
    s.count++;
    if (!s.pinned && s.count >= pin_threshold()) set_pinned(s, true);
}

void ExpertCache::finish_stage(Slot &s) {
    if (!s.staged) return;
    MsTimer t(&active_wait_ms_);
    // A prefetch ticket is ordered on the DEVICE: the default stream waits for
    // the copy and the host never blocks, which is the whole point of issuing it
    // a layer early. The ring path has no such event and keeps its host wait.
    if (!be_->wait_marker_stream(s.ready)) be_->wait_marker(s.ready);
    be_->release_marker(s.ready);
    s.ready = nullptr;
    s.staged = false;
}

void ExpertCache::drop(Slot &s) {
    if (!be_) return;
    finish_stage(s);
    // An expert's three slices go back in ONE call so the pool issues one
    // release mark for the three instead of one per slice. An eviction is the
    // one place the decode loop pays this, once per promotion at a constrained
    // budget, and the three releases are the same default-stream point.
    void *cells[3] = {};
    size_t lens[3] = {};
    QuantTensor *q[3] = {&s.m.gate, &s.m.up, &s.m.down};
    int n = 0;
    for (int i = 0; i < 3; i++) {
        if (q[i]->present() && q[i]->data) {
            cells[n] = q[i]->data;
            lens[n] = s.slice[i];
            n++;
        }
        *q[i] = {};
    }
    s.m = {};
    table_stale_ = true;
    // A cell that is on its way back must not stay in the pending burst: the
    // flush would still record a mark for it, and the release that just happened
    // owns that slot's lifetime from here. Scrubbing is over at most
    // 3 * burst entries (the burst is one layer's worth of uses).
    if (!rd_pending_.empty()) {
        for (int i = 0; i < n; i++)
            rd_pending_.erase(std::remove(rd_pending_.begin(), rd_pending_.end(), cells[i]),
                              rd_pending_.end());
    }
    if (n) be_->release_pooled_batch(cells, lens, n);
}

const ResidentExpert *ExpertCache::acquire_inplace(const ExpertSource &src,
                                                   i32 layer, i32 expert) {
    if (!be_ || !src.present()) return nullptr;
    if (expert < 0 || expert >= src.n_expert) return nullptr;
    auto it = slots_.find(slot_key(layer, expert));
    if (it == slots_.end()) return nullptr;
    Slot &s = it->second;
    // Already on the card: use it there. Device memory reads at 485 GiB/s
    // against 24.3 GiB/s for pinned host memory on this card, so a resident
    // expert has nothing to gain from being read from RAM.
    if (s.in_vram() || s.staged) return nullptr;
    // Pageable WARM is not addressable from the device (only the pinned pool
    // is), so those slots keep the promotion path.
    if (!s.in_host() || !s.host_pinned) return nullptr;
    // An expert that has been asked for before goes where REUSE is fast. The
    // counter is the tier's own signal and the threshold is measured, not chosen:
    // gating on `pinned` (24 touches) instead kept every reused expert out of
    // VRAM for a whole run, and on Qwen3-30B-A3B -- where the routed set fits and
    // 84.6% of picks are repeats -- that cost 20% (42.9 ms/step against 33.7)
    // because 45.7% of acquires read 24.3 GiB/s host memory where a promotion
    // would have served them from the card forever. ONE in-place read pays for
    // itself immediately (it is the read the promotion would have copied); the
    // second use is where device memory wins by 20x, and that is `count >= 2`.
    if (s.count >= 2) return nullptr;
    ++total_acquires_;
    ++inplace_hits_;
    touch(s);
    xtrace(layer, expert, 1);
    if (++acquires_ >= decay_period()) decay();
    return &s.h;
}

void ExpertCache::drop_host(Slot &s) {
    if (!be_) return;
    void *base = nullptr;
    for (QuantTensor *q : {&s.h.gate, &s.h.up, &s.h.down}) {
        // The three slices are offsets inside ONE block, so the release below
        // happens once, through the lowest-offset pointer. Deferred: a
        // promotion copy of this expert may still be in flight on a side
        // stream, and freeing the source under it is a use-after-free. The
        // pinned arm needs its own deallocator -- hipFree on a hipHostMalloc
        // block is the other half of that same mistake.
        if (q->data && !base) base = q->data;
        *q = {};
    }
    warm_release(be_, base, s.host_pinned, /*deferred=*/true);
    s.h = {};
    host_bytes_ -= s.host_bytes;
    if (s.host_pinned) {
        const size_t drop = std::min(host_pinned_bytes_, s.host_bytes);
        host_pinned_bytes_ -= drop;
    }
    s.host_bytes = 0;
    s.host_pinned = false;
}

void ExpertCache::forget_if_dead(std::unordered_map<u64, Slot>::iterator it) {
    if (it == slots_.end()) return;
    if (!it->second.in_vram() && !it->second.in_host()) slots_.erase(it);
}

void ExpertCache::track_vram(const u64 key, Slot &s) {
    s.vram_idx = static_cast<i64>(vram_keys_.size());
    vram_keys_.push_back(key);
    vram_slots_++;
}

void ExpertCache::untrack_vram(Slot &s) {
    if (s.vram_idx < 0) {
        vram_slots_--;
        return;
    }
    const size_t at = static_cast<size_t>(s.vram_idx);
    if (at < vram_keys_.size()) {
        const u64 moved = vram_keys_.back();
        vram_keys_[at] = moved;
        vram_keys_.pop_back();
        // The slot that moved into `at` must know where it is now, or a later
        // untrack would clear somebody else's entry.
        if (moved != slot_key(s.layer, s.expert)) {
            auto mit = slots_.find(moved);
            if (mit != slots_.end()) mit->second.vram_idx = static_cast<i64>(at);
        }
    }
    s.vram_idx = -1;
    vram_slots_--;
}

void ExpertCache::retire(Slot &s) {
    untrack_vram(s);
    if (s.bytes <= bytes_) {
        bytes_ -= s.bytes;
    } else {
        bytes_ = 0;
    }
    // The ring's own accounting has to let go with the residency, or a
    // pin_bytes_ that outlives its bytes makes over_budget() believe the pool
    // has room it does not have.
    if (s.pinned) set_pinned(s, false);
    drop(s);
    // The device copy is gone. A WARM copy is exactly the destination the
    // lifecycle asks for, and it costs nothing: the bytes are already resident,
    // so HOT -> WARM is "stop counting VRAM", not a copy and not a write.
    // Without one the slot falls to COLD (the file), where it came from.
    if (s.in_host()) {
        demotions_++;
        bytes_demoted_ += s.host_bytes;
        // The pin promised VRAM residency and that just went away under duress
        // (every other device slot was pinned). Keep the counter, drop the pin:
        // WARM must not inherit protection that VRAM traffic revoked.
        if (s.pinned) s.pinned = false;
    }
}

void ExpertCache::make_vram_room(size_t need) {
    // Evict until this one fits, in BYTES. The slot count was the trigger here
    // and it does not work: experts are not one size. Qwen3-MoE-4x0.6B is 56
    // experts of 5.8359 MiB (Q6_K down_proj) and 56 of 5.0625 MiB (Q4_K), which
    // is 610.312 MiB -- and that total is exactly the budget the policy sizes for
    // it. A count derived as budget_ / need therefore read 104 at a Q6_K layer
    // and 126 at a Q4_K one, while the budget covers all 112, so the cache
    // evicted 23 experts per 64-token run with ~43 MiB of its own budget
    // unspent. bytes_ + need is the actual constraint; capacity_ is an estimate
    // for the report (see over_budget).
    //
    // Pins rank victims but never override the physical limit. A staging
    // lease protects only the current consumer; stage_next verifies that both
    // current and next fit before taking that lease.
    while (vram_slots_ > 0 && over_budget(need)) {
        u64 victim = 0;
        {
            MsTimer t(&scan_ms_);
            if (!pick_victim(&victim)) break;
        }
        auto vit = slots_.find(victim);
        if (vit == slots_.end()) break;
        {
            MsTimer t(&retire_ms_);
            retire(vit->second);
            forget_if_dead(vit);
        }
        evictions_++;
    }
}

// The answer to a refused device allocation. make_vram_room() decides on the
// byte budget, and that budget was planned at init -- before the KV cache, the
// workspaces and this run's own buffers existed -- so on a card whose free
// memory then shrank it can believe there is room while the pool's floor check
// says there is none (measured on the REAP-288 file: budget 7624 MiB, 7445 MiB
// resident, 3202 MiB free beside a 3261 MiB floor, i.e. negative room beside the
// floor with 179 MiB of "budget" left). The bytes that make the next allocation
// succeed are already on the device: they are this cache's own cells. Retiring
// one slot returns three cells to the pool's free list, which is a reuse and not
// a growth, so the floor is not crossed and the fit is exact rather than
// estimated. This is also the eviction the budget would have ordered itself had
// it been measured against the live card.
bool ExpertCache::force_reclaim(size_t want) {
    u64 victim = 0;
    bool found = false;
    // A victim of the SAME size class is worth more than the LRU: the pool hands
    // cells out by rounded size, so a released cell of another class does not
    // satisfy this allocation at all. Unpinned first, then pinned -- the ring's
    // promise is still weaker than a run that cannot proceed.
    //
    // It takes the first match on purpose, and three refinements of it are
    // measured: ordering the class like `pick_victim` (rarest, then oldest load),
    // ordering it by true least-recently-read, and preferring a slot whose
    // readers are already retired. The first two lose (a full scan per reclaim
    // costs more than it saves) and the third is neutral -- because the per-slot
    // reader-done mark already made the reuse wait land on that slot's own
    // readers, so there is no wait left for the victim choice to remove.
    // `docs/PERF-ANALYSIS.md` sections 12n and 12p.
    if (want) {
        for (int pass = 0; pass < 2 && !found; pass++) {
            for (const auto &kv : slots_) {
                const Slot &s = kv.second;
                if (!s.in_vram() || s.leased) continue;
                if (pass == 0 && s.pinned) continue;
                bool match = false;
                for (size_t slice : s.slice)
                    if (slice == want) { match = true; break; }
                if (!match) continue;
                victim = kv.first;
                found = true;
                break;
            }
        }
    }
    if (!found && !pick_victim(&victim)) return false;
    auto vit = slots_.find(victim);
    if (vit == slots_.end()) return false;
    retire(vit->second);
    forget_if_dead(vit);
    evictions_++;
    reclaims_++;
    return true;
}

void ExpertCache::make_warm_room(size_t need) {
    // WARM eviction is cheap on purpose: the file is the immutable canonical
    // copy, so a victim costs a free() and nothing else — no writeback, no
    // demotion, no device traffic. Nothing here writes to the model file.
    if (host_budget_ == 0) return;
    while (host_bytes_ + need > host_budget_) {
        u64 victim = 0;
        if (!pick_warm_victim(&victim)) return; // nothing left to evict
        auto vit = slots_.find(victim);
        if (vit == slots_.end()) return;
        drop_host(vit->second);
        warm_evictions_++;
        forget_if_dead(vit);
    }
}

bool ExpertCache::warm_fill(const ExpertSource &src, i32 expert, Slot &s) {
    if (!be_) return false;
    const GgufTensor *from[3] = {src.gate, src.up, src.down};
    QuantTensor *to[3] = {&s.h.gate, &s.h.up, &s.h.down};
    // Pin whole slots, never individual slices: the promotion hands all three
    // slices to one batch call, and a batch whose sources differ in kind has no
    // single copy path. Admission order is rank order when the model carries an
    // expert index, so this prefix is the hot set.
    size_t eff[3] = {0, 0, 0};
    for (int i = 0; i < 3; i++)
        if (from[i]) eff[i] = s.slice[i];
    const WarmBlockLayout blk = warm_block_layout(eff);
    const size_t host_need = blk.total;
    const bool pin = host_pin_bytes_ > 0 && host_need > 0 &&
                     host_pinned_bytes_ + host_need <= host_pin_bytes_;
    s.host_pinned = pin;
    // ---- allocate every slice, then read them as ONE batch -----------------
    //
    // This used to be three `read_host` calls, i.e. one file HANDLE at queue
    // depth 1 for the whole fill. Measured with tools/probe_readspeed.cpp on
    // this box, from the page cache at the 1792 KiB slice size this model uses:
    // 1 reader 7.37 GB/s, 2 readers 10.96, 4 readers 13.02, 8 no better. The
    // engine's own row read 3.9 GB/s -- half of what a single reader gets
    // standalone and a third of the pool -- because the three slices of one
    // expert are three megabytes apart in the file and were being serialised
    // through that one handle.
    //
    // `read_host_batch` is the backend's reader pool (KRK_WARM_READERS, default
    // 4), which is what prefetch_layer already uses for a whole layer's routed
    // set. Handing it the expert's own three slices is the same shape at the
    // granularity the miss actually arrives at, so a cold expert stops paying
    // the serial cap. It keeps read_host's fallback (the mapping memcpy) for a
    // file path that is unavailable.
    void *dst[3] = {nullptr, nullptr, nullptr};
    const void *srcp[3] = {nullptr, nullptr, nullptr};
    int ndst = 0;
    size_t bytes = 0;
    // ONE allocation for all three slices (see warm_block_layout). Each slice
    // used to reserve against the process-wide host ledger on its own; the
    // block now does it once, for the same bytes, so the ledger still holds one
    // entry per expert. A pinned slot still skips that ledger on purpose: its
    // bytes come out of the driver's pin pool, which host_pin_bytes_ bounds
    // instead, and charging both would count the same RAM twice.
    void *base = nullptr;
    if (blk.total) {
        base = pin ? be_->alloc_host(blk.total)
                   : be_->alloc_host_pageable_budgeted(blk.total);
        // Nothing is handed to the slot until the whole fill succeeds: a slot
        // holding some slices and not others is a state the promotion path
        // would have to defend against.
        if (!base) return false;
    }
    for (int i = 0; i < 3; i++) {
        const GgufTensor *t = from[i];
        const size_t per = eff[i];
        if (!t || per == 0) continue;
        dst[ndst] = static_cast<char *>(base) + blk.off[i];
        srcp[ndst] = t->data + static_cast<size_t>(expert) * per;
        ndst++;
        bytes += per;
    }
    if (ndst == 0) return false;
    {
        // A read() out of the page cache, not a memcpy of the mapping: the
        // mapping faults one 4 KiB page at a time and would leave a resident
        // page per staged byte, so the tier would cost the model's size in RAM
        // on top of itself.
        // One request per slice. Splitting each slice in half to reach a queue
        // depth of six was measured and REJECTED: the pool sorts the requests
        // by offset and gives each worker one contiguous run, so six requests
        // over four workers still lands as three streams (2+2+2, the fourth
        // worker idle) -- 6.1 GB/s against 6.3 for three, same text. The pool's
        // distribution, not the request count, is what shapes the read.
        Backend::ReadReq reqs[3];
        int nreq = 0;
        int k = 0;
        for (int i = 0; i < 3; i++) {
            if (!from[i] || s.slice[i] == 0) continue;
            reqs[nreq].dst = dst[k];
            reqs[nreq].mapped_src = srcp[k];
            reqs[nreq].bytes = s.slice[i];
            nreq++;
            k++;
        }
        try {
            MsTimer t(&read_ms_, phase_sink(&pf_read_ms_, &dc_read_ms_));
            // `warm.read` is the DISK-TIER read a cold expert pays: a pread
            // (or a mapping memcpy) on the calling thread, serial with the
            // layer that wants the expert. On a model whose routed set does not
            // fit, this row is the step.
            HostAdd h("expert.warm.read");
            be_->read_host_batch(reqs, nreq);
            h.bytes(static_cast<krk::i64>(bytes));
        } catch (...) {
            // The whole block goes back, not the slices that were filled: they
            // are offsets into it.
            warm_release(be_, base, pin, /*deferred=*/false);
            return false;
        }
    }
    for (int i = 0, k = 0; i < 3; i++) {
        const GgufTensor *t = from[i];
        const size_t per = s.slice[i];
        if (!t || per == 0) continue;
        to[i]->data = dst[k++];
        to[i]->type = t->type;
        to[i]->n_in = static_cast<i64>(t->ne[0]);
        to[i]->n_out = t->n_dims >= 2 ? static_cast<i64>(t->ne[1]) : 1;
    }
    s.host_bytes = blk.total;
    // The same number the release subtracts, or the pin counter drifts by the
    // padding between slices.
    if (pin) host_pinned_bytes_ += blk.total;
    return true;
}

bool ExpertCache::warm_admit(const ExpertSource &src, i32 expert, Slot &s) {
    // Under the mapping policy there is no retained copy to admit to, and the
    // caller's cold path reads the mapping instead -- which is the whole point
    // of the policy, so this is not a rejection and does not count as one.
    if (warm_map_) return false;
    const GgufTensor *from[3] = {src.gate, src.up, src.down};
    size_t eff[3] = {0, 0, 0};
    for (int i = 0; i < 3; i++)
        if (from[i]) eff[i] = s.slice[i];
    // The block, not the sum of the slices: the aligned offsets inside it are
    // what the allocator is asked for, and room has to be made for that.
    const size_t host_need = warm_block_layout(eff).total;
    if (!be_ || host_budget_ == 0 || host_need == 0) return false;
    make_warm_room(host_need);
    // An expert larger than the whole tier never fits: the alternative would be
    // evicting the entire tier for one copy that still does not fit. That, and
    // a failed allocation, are the only two rejects under always-admit.
    if (host_bytes_ + host_need > host_budget_) {
        warm_rejects_++;
        return false;
    }
    // Read-through admissions happen during inference too, not only in the two
    // startup sweeps. Preflight against the same ledger as KV, then reserve each
    // slice atomically in warm_fill(); a rejected admission falls back to a
    // cold direct load without retaining a partial host copy.
    if (!SharedHostRamBudget::can_admit(host_need, be_->host_available_bytes(),
                                         be_->host_total_bytes())) {
        warm_rejects_++;
        return false;
    }
    if (!warm_fill(src, expert, s)) {
        drop_host(s);
        warm_rejects_++;
        return false;
    }
    host_bytes_ += s.host_bytes;
    warm_admissions_++;
    return true;
}

bool ExpertCache::promote_to_vram(Slot &s, bool staged, bool prefetch) {
    const std::chrono::steady_clock::time_point t0 =
        std::chrono::steady_clock::now();
    struct ClockOut {
        f64 *ms;
        f64 *ms2; // prefill/decode phase twin, or null outside both
        std::chrono::steady_clock::time_point t;
        ~ClockOut() {
            const f64 took = std::chrono::duration<f64, std::milli>(
                                 std::chrono::steady_clock::now() - t)
                                 .count();
            *ms += took;
            if (ms2) *ms2 += took;
        }
    } clock_out{&promote_ms_, phase_sink(&pf_promote_ms_, &dc_promote_ms_),
                t0};
    if (!be_ || !s.in_host()) return false;
    size_t physical = 0;
    for (size_t slice : s.slice) if (slice) physical += be_->pooled_size(slice);
    s.bytes = physical;
    be_->set_expert_pool_owner(this, owner_bytes(physical));
    const u64 key = slot_key(s.layer, s.expert);
    // Snapshot before the scan: this slot is WARM, so retire() cannot take it
    // (it only drops device copies), but the reference is re-established below
    // anyway and reading `s` across a call that evicts is what the rest of this
    // file goes out of its way to avoid.
    const size_t need = s.bytes;
    {
        MsTimer t(&room_ms_, phase_sink(&pf_room_ms_, &dc_room_ms_));
        make_vram_room(need);
    }

    // Eviction never touches WARM copies (retire keeps them), so the slot's own
    // copy cannot be taken out from under this promotion. Re-establishing the
    // reference anyway turns any future aliasing into a plain miss instead of a
    // write into a freed node.
    {
        const auto it = slots_.find(key);
        if (it == slots_.end() || !it->second.in_host()) return false;
        s = it->second;
    }

    ResidentExpert m;
    const QuantTensor *from[3] = {&s.h.gate, &s.h.up, &s.h.down};
    QuantTensor *to[3] = {&m.gate, &m.up, &m.down};
    // The three slices as one batch: the source is pageable WARM memory, and a
    // batch lets the bounded pinned staging ring overlap the copies instead of
    // serialising them through one buffer.
    void *batch_dst[3] = {};
    const void *batch_src[3] = {};
    size_t batch_len[3] = {};
    int nb = 0;
    for (int i = 0; i < 3; i++) {
        if (!from[i]->present()) continue;
        void *p = nullptr;
        for (int tries = 0;; tries++) {
            {
                MsTimer t(&alloc_ms_, phase_sink(&pf_alloc_ms_, &dc_alloc_ms_));
                p = be_->alloc_pooled(s.slice[i]);
            }
            if (p) break;
            if (tries >= 63 || !force_reclaim(s.slice[i])) break;
        }
        if (!p) {
            ++pool_refusals_;
            // Same burst shape as drop(): the half-promoted slices go back at
            // one stream point, so they share one release mark.
            void *cells[3] = {};
            size_t lens[3] = {};
            int n = 0;
            for (int j = 0; j < 3; j++)
                if (to[j]->present() && to[j]->data) {
                    cells[n] = to[j]->data;
                    lens[n] = s.slice[j];
                    n++;
                }
            if (n) be_->release_pooled_batch(cells, lens, n);
            return false;
        }
        *to[i] = *from[i];
        to[i]->data = p;
        batch_dst[nb] = p;
        batch_src[nb] = from[i]->data;
        batch_len[nb] = s.slice[i];
        nb++;
    }
    {
        MsTimer t(&xfer_ms_, phase_sink(&pf_xfer_ms_, &dc_xfer_ms_));
        if (prefetch) {
            // One layer early: a DMA on the copy stream with a ticket, and the
            // default stream is deliberately NOT ordered behind it here. The
            // consumer (finish_stage, at the next layer's acquire) does that,
            // which is what lets the copy run under the layer in front of it.
            // merge=false: the three slices of this expert only LOOK contiguous
            // in the address space. A run the driver refuses costs a fallback,
            // not a wrong answer, but it also costs a warning line on every
            // promotion of the hot set -- measured on the 35B-A3B, whose three
            // 544 KiB slices merged into one 1632 KiB DMA and came back
            // hipErrorInvalidValue.
            s.ready = be_->upload_expert_prefetch(batch_dst, batch_src, batch_len, nb,
                                                  /*merge=*/false);
            s.staged = s.ready != nullptr;
        } else if (s.host_pinned) {
            // The hot arm: the source is already pinned, so this is the DMA and
            // nothing else -- no bounded ring, no host memcpy of the expert in
            // front of it. The backend publishes the copies to the default
            // stream with one event, which is why no marker travels back: the
            // consumer's own stream order already covers the bytes. merge=false
            // for the reason above: the entries are per-slice sources, and the
            // driver refused their merged span.
            be_->upload_pinned_batch(batch_dst, batch_src, batch_len, nb,
                                     /*merge=*/false);
        } else if (staged) {
            s.ready = be_->upload_expert_staged(batch_dst, batch_src, batch_len, nb);
            s.staged = true;
        } else {
            be_->upload_paged_batch(batch_dst, batch_src, batch_len, nb);
        }
    }
    s.m = m;
    table_stale_ = true;
    track_vram(key, s);
    bytes_ += s.bytes;
    // A slot that was pinned while WARM (pin_layer, admit_hot) enters the
    // device tier straight into the ring: the bytes are charged to
    // pin_budget_, not to the pool's budget_.
    if (s.pinned) {
        pin_bytes_ += s.bytes;
        pin_slots_++;
    }
    // The WARM copy stays — WARM is inclusive. It is what makes the next VRAM
    // eviction of this expert free, and it costs only the RAM the tier has
    // already budgeted for.
    bytes_promoted_ += s.bytes;
    promotions_++;
    return true;
}

void ExpertCache::decay() {
    for (auto &kv : slots_) {
        Slot &s = kv.second;
        if (s.count > kPinDecayStep) {
            s.count >>= 1; // geometric cooling; an 8-step pin survives ~2 decays
        } else {
            s.count = s.count > kPinDecayStep / 4 ? s.count - kPinDecayStep / 4 : 0;
        }
        if (s.pinned && s.count < pin_threshold()) set_pinned(s, false);
    }
    acquires_ = 0;
    decays_++;
}

// How many resident slots the victim search looks at per eviction. 0 means the
// whole set -- the behaviour before vram_keys_ existed, kept as the arm.
static size_t evict_window() {
    static const size_t v = [] {
        const char *e = std::getenv("KRK_EVICT_WINDOW");
        if (e) {
            const long k = std::strtol(e, nullptr, 10);
            if (k >= 0) return static_cast<size_t>(k);
        }
        // Measured on the REAP-320 file (16 greedy tokens): 1 -> 103.3 ms/step,
        // 2 -> 99.4, 4 -> 97.2, 8 -> 96.8, 16 -> 96.6, 32 -> 99.3, 64 -> 108.7,
        // whole set -> 107.9. Wider is NOT better: a window of 64+ reaches
        // freshly touched slots and evicts them, while a narrow window evicts
        // around the hand -- the region nothing has asked for in a while. 8 and
        // 16 tie; 16 keeps a little more choice for clustered routing.
        return static_cast<size_t>(16);
    }();
    return v;
}

// The victim search is the room row's expensive half, and it was expensive for a
// reason that has nothing to do with policy: it walked every slot in the cache
// (3.5k resident inside 14k WARM) for every eviction -- measured 9.19M visits
// for 2624 evictions, 505.9 ms of a 1.7 s REAP-320 decode, against 18.1 ms to
// release the cells it picked. The scan is 96% of the eviction cost and it is
// O(slots) per byte of room made, where the eviction itself is O(1).
//
// So the candidates come from vram_keys_ (resident slots only, O(1) insert and
// swap-remove) and the search looks at `evict_window()` of them from a moving
// hand, keeping the order the full scan used: unpinned before pinned, then
// rarest, then oldest. A window differs only in that it cannot see a lazy slot
// just outside it, and the hand is what covers that: consecutive evictions start
// where the last one stopped, so the whole set is swept within n/window
// evictions. KRK_EVICT_WINDOW=0 restores the global scan.
bool ExpertCache::pick_victim(u64 *out) {
    ++scans_;
    const size_t n = vram_keys_.size();
    if (n == 0) return false;
    const size_t window = evict_window();
    const size_t take = (window == 0 || window > n) ? n : window;
    if (victim_cursor_ >= n) victim_cursor_ = 0;
    u64 best = 0;
    size_t best_at = 0;
    bool have = false;
    for (size_t i = 0; i < take; i++) {
        const size_t at = (victim_cursor_ + i) % n;
        auto it = slots_.find(vram_keys_[at]);
        if (it == slots_.end()) continue;
        const Slot &s = it->second;
        ++scan_visits_;
        if (!s.in_vram() || s.leased) continue; // never evict a current consumer
        if (!have) {
            best = it->first;
            best_at = at;
            have = true;
            continue;
        }
        const Slot &cur = slots_.at(best);
        // 1. unpinned beats pinned -- a pin is a promise, keep it
        if (s.pinned != cur.pinned) {
            if (cur.pinned) {
                best = it->first;
                best_at = at;
            }
            continue;
        }
        // 2. the rarest expert goes first
        if (s.count < cur.count) {
            best = it->first;
            best_at = at;
            continue;
        }
        // 3. equal counts: the older load goes first
        if (s.count == cur.count && s.seq < cur.seq) {
            best = it->first;
            best_at = at;
        }
    }
    if (have) {
        // Start the next search after the slot just taken: the window then
        // sweeps the tier instead of re-reading the same 64 entries.
        victim_cursor_ = (best_at + 1) % n;
        *out = best;
        return true;
    }
    // Every slot in the window is leased. Rare, and refusing here is worse than
    // a wide search -- a lease belongs to a consumer that is nearly done, while
    // the room is needed now -- so fall back to the whole map once.
    for (const auto &kv : slots_) {
        const Slot &s = kv.second;
        ++scan_visits_;
        if (!s.in_vram() || s.leased) continue;
        if (!have) {
            best = kv.first;
            have = true;
            continue;
        }
        const Slot &cur = slots_.at(best);
        if (s.pinned != cur.pinned) {
            if (cur.pinned) best = kv.first;
            continue;
        }
        if (s.count < cur.count) {
            best = kv.first;
            continue;
        }
        if (s.count == cur.count && s.seq < cur.seq) best = kv.first;
    }
    if (!have) return false;
    victim_cursor_ = (victim_cursor_ + take) % n;
    *out = best;
    return true;
}

bool ExpertCache::pick_warm_victim(u64 *out) {
    // Same frequency ordering as the device tier, minus the pin: a WARM copy is
    // a cache of the file, so losing one costs a future read, not a reload
    // storm. Letting a stale pin defend it would make the tier grow into a
    // graveyard of experts VRAM traffic abandoned long ago.
    //
    // One addition: a copy that is not backing a VRAM-resident expert goes
    // before one that is. A WARM-only slot is a promise of a cheap promotion;
    // a WARM copy behind a HOT slot is a promise of a cheap eviction, so it is
    // worth more. Only when the tier is entirely HOT shadows does a victim come
    // from there.
    u64 best = 0;
    bool have = false;
    bool best_shadow = false; // the copy backs an expert that is in VRAM
    for (const auto &kv : slots_) {
        const Slot &s = kv.second;
        if (!s.in_host()) continue;
        const bool shadow = s.in_vram();
        if (!have) {
            best = kv.first;
            best_shadow = shadow;
            have = true;
            continue;
        }
        const Slot &cur = slots_.at(best);
        if (shadow != best_shadow) {
            if (best_shadow) {
                best = kv.first;
                best_shadow = shadow;
            }
            continue;
        }
        if (s.count < cur.count) {
            best = kv.first;
            continue;
        }
        if (s.count == cur.count && s.seq < cur.seq) best = kv.first;
    }
    if (!have) return false;
    *out = best;
    return true;
}

size_t ExpertCache::preload_one(const ExpertSource &src, i32 layer, i32 expert) {
    if (!be_ || !src.present()) return 0;
    if (expert < 0 || expert >= src.n_expert) return 0;
    // The mapping policy keeps nothing, so an eager sweep's only useful product
    // is a warm page cache -- read the bytes and drop them, and report them as
    // the traffic it was.
    if (warm_map_) return touch_layer(src, layer, &expert, 1);
    if (host_budget_ == 0) return 0;
    const u64 key = slot_key(layer, expert);
    if (slots_.count(key)) return 0; // already resident in some tier
    Slot s;
    s.layer = layer;
    s.expert = expert;
    s.bytes = src.per_expert(src.gate) + src.per_expert(src.up) +
              src.per_expert(src.down);
    s.slice[0] = src.per_expert(src.gate);
    s.slice[1] = src.per_expert(src.up);
    s.slice[2] = src.per_expert(src.down);
    s.seq = seq_++;
    // Count 0 on purpose: a prefetched slot has earned nothing, so it is the
    // first thing out of the tier when a run-time admission needs the room.
    if (!warm_admit(src, expert, s)) {
        drop_host(s);
        return 0; // the tier is the limit; stop rather than thrash
    }
    const size_t staged = s.host_bytes;
    bytes_staged_ += staged;
    slots_.emplace(key, std::move(s));
    return staged;
}

size_t ExpertCache::pin_layer(const ExpertSource &src, i32 layer,
                              std::string *why) {
    if (!be_ || !src.present() || src.n_expert <= 0) return 0;
    const size_t gate_b = src.per_expert(src.gate);
    const size_t up_b = src.per_expert(src.up);
    const size_t down_b = src.per_expert(src.down);
    const size_t per = be_->pooled_size(gate_b) + be_->pooled_size(up_b) +
                       be_->pooled_size(down_b);
    if (per == 0) return 0;
    // Only what is missing has to fit: a set that is already resident (or
    // partly resident from the warmup) is admitted for the difference.
    size_t need = 0;
    size_t already = 0;
    for (i32 e = 0; e < src.n_expert; e++) {
        const auto it = slots_.find(slot_key(layer, e));
        if (it != slots_.end() && it->second.in_vram())
            already++;
        else
            need += per;
    }
    // The test is whether the set can EVER fit, not whether the budget has room
    // this instant: the warmup fills the tier to its cap, so "no uncommitted
    // bytes" is the normal state and acquire()'s own make_vram_room is what
    // turns a full cache into room by evicting unpinned slots (Tiel: 3085
    // resident, 256 needed). Only a set larger than the whole budget is refused.
    if (need > budget_) {
        if (why) {
            char b[160];
            std::snprintf(b, sizeof(b),
                          "%d expert(s) need %.0f MiB and the whole device "
                          "budget is %.0f MiB",
                          src.n_expert, static_cast<f64>(need) / 1048576.0,
                          static_cast<f64>(budget_) / 1048576.0);
            *why = b;
        }
        return 0;
    }
    size_t pinned = 0;
    for (i32 e = 0; e < src.n_expert; e++) {
        acquire(src, layer, e);
        const auto it = slots_.find(slot_key(layer, e));
        if (it == slots_.end() || !it->second.in_vram()) continue;
        // A pin is the promise: the eviction scan keeps pinned slots last, and
        // the counter is raised to the threshold so the decay window cannot
        // drop it (this bank is re-acquired by every proposal, so it re-earns
        // the count anyway).
        set_pinned(it->second, true);
        if (it->second.count < pin_threshold())
            it->second.count = pin_threshold();
        pinned++;
    }
    // The footprint, not `bytes_` after the fact: admitting the set evicts other
    // slots, so the resident total can go DOWN while this set goes in. Reporting
    // the difference of the two totals underflowed to 1.7e13 MiB on the first
    // measured run.
    const size_t added = (pinned - already) * per;
    KRK_INFO("expert cache: pinned layer %d's routed set (%zu of %d expert(s), "
             "%.0f MiB added, %.0f MiB resident of %.0f MiB; %zu were already "
             "resident)",
             layer, pinned, src.n_expert,
             static_cast<f64>(added) / 1048576.0,
             static_cast<f64>(bytes_) / 1048576.0,
             static_cast<f64>(budget_) / 1048576.0, already);
    return pinned * per;
}

size_t ExpertCache::admit_hot(const ExpertSource *const *src_of, i32 n_layer,
                              const std::vector<std::pair<i32, i32>> &order,
                              size_t max_experts) {
    if (!be_ || pin_budget_ == 0 || order.empty()) return 0;
    size_t pinned = 0;
    size_t refused = 0;
    const std::chrono::steady_clock::time_point t0 =
        std::chrono::steady_clock::now();
    for (const auto &pr : order) {
        if (max_experts && pinned >= max_experts) break;
        if (pin_bytes_ >= pin_budget_) break;
        const i32 l = pr.first;
        const i32 e = pr.second;
        if (!src_of || l < 0 || l >= n_layer) continue;
        const ExpertSource &src = *src_of[l];
        if (!src.present() || e < 0 || e >= src.n_expert) continue;
        // Already resident: a pin it earns now is free, and counting it keeps
        // the report's "admitted" number honest about the ring's contents.
        auto it = slots_.find(slot_key(l, e));
        if (it != slots_.end() && it->second.in_vram()) {
            if (!it->second.pinned) {
                set_pinned(it->second, true);
                pinned++;
            }
            continue;
        }
        // What the expert costs on the device, so the ring's cap is tested in
        // the allocator's bytes (pooled cells are aligned, not the raw slice).
        const size_t per = (src.gate ? be_->pooled_size(src.per_expert(src.gate)) : 0) +
                           be_->pooled_size(src.per_expert(src.up)) +
                           be_->pooled_size(src.per_expert(src.down));
        if (per == 0) continue;
        if (pin_bytes_ + per > pin_budget_) break; // the ring is full
        if (!acquire(src, l, e)) {
            refused++;
            continue;
        }
        auto pit = slots_.find(slot_key(l, e));
        if (pit == slots_.end() || !pit->second.in_vram()) {
            // acquire() may evict rather than admit when every other slot is
            // pinned; that is a refusal, not a pin.
            refused++;
            continue;
        }
        // Set AFTER the promotion: promote_to_vram counts a slot pinned before
        // it, and this path's charge is this call.
        set_pinned(pit->second, true);
        pinned++;
    }
    const f64 took = std::chrono::duration<f64, std::milli>(
                         std::chrono::steady_clock::now() - t0)
                         .count();
    KRK_INFO("expert cache: hot ring admitted %zu expert(s) (%.0f MiB of a "
             "%.0f MiB ring, %zu refused) in %.0f ms; the pool's %.0f MiB "
             "budget no longer counts them",
             pinned, static_cast<f64>(pin_bytes_) / 1048576.0,
             static_cast<f64>(pin_budget_) / 1048576.0, refused, took,
             static_cast<f64>(budget_) / 1048576.0);
    return pinned;
}

size_t ExpertCache::preload(const ExpertSource &src, i32 layer) {
    size_t staged = 0;
    for (i32 e = 0; e < src.n_expert; e++) {
        const size_t one = preload_one(src, layer, e);
        if (one == 0) break;
        staged += one;
    }
    return staged;
}

const ResidentExpert *ExpertCache::acquire(const ExpertSource &src, i32 layer,
                                           i32 expert) {
    if (!be_ || !src.present()) return nullptr;
    if (expert < 0 || expert >= src.n_expert) return nullptr;
    ++total_acquires_;

    // A single expert is the minimum residency: a budget below that still works,
    // it just reloads on every step.
    const size_t gate_b = src.per_expert(src.gate);
    const size_t up_b = src.per_expert(src.up);
    const size_t down_b = src.per_expert(src.down);
    const size_t need = (gate_b ? be_->pooled_size(gate_b) : 0) +
                        (up_b ? be_->pooled_size(up_b) : 0) +
                        (down_b ? be_->pooled_size(down_b) : 0);
    // Preserve the documented minimum: one complete aligned expert can run.
    be_->set_expert_pool_owner(this, owner_bytes(need));
    // capacity_ is what the report says fits, not what the eviction decision is
    // made on (that is bytes_ + need against budget_, see over_budget). Without
    // an explicit slot budget it is the byte budget expressed in slots of THIS
    // layer's expert, which is an estimate and is documented as one: the corpus
    // can mix expert sizes, and the engine passes an exact slot_budget whenever
    // it knows one.
    if (slot_budget_ == 0) {
        const size_t cap = need > 0 ? (budget_ / need) : 0;
        capacity_ = cap > 0 ? cap : 1;
    }

    const u64 key = slot_key(layer, expert);
    auto it = slots_.find(key);
    if (it != slots_.end()) {
        Slot &s = it->second;
        if (s.in_vram()) {
            if (s.staged) {
                finish_stage(s);
                ++active_used_;
                touch(s);
                ++host_hits_;
                xtrace(layer, expert, 1);
                if (++acquires_ >= decay_period()) decay();
                return &s.m;
            }
            // Hit: the traffic that makes an expert hot happens here.
            touch(s);
            hits_++;
            xtrace(layer, expert, 0);
            if (++acquires_ >= decay_period()) decay();
            return &s.m;
        }
        // WARM hit: the weights are already in RAM, so this acquire costs a DMA
        // instead of a re-read of the file. It also earns the same counter —
        // an expert hot enough to be evicted is still hot.
        // Under the mapping policy this copy is dropped instead of promoted:
        // the bytes are in the file's page cache already, so a promotion copies
        // them twice (RAM -> staging ring -> VRAM) and holds a second copy of
        // the whole tier in RAM to do it. Falling through to the cold path
        // reads the mapping -- one copy, and no host allocation at all.
        if (!warm_map_ && promote_to_vram(s)) {
            touch(s);
            host_hits_++;
            xtrace(layer, expert, 1);
            if (++acquires_ >= decay_period()) decay();
            return &s.m;
        }
        // No promotion (device allocation failed). The cold path below rebuilds
        // this expert, and the slot MUST NOT survive into it: emplace() on an
        // existing key is a no-op, so the insert at the end of this function
        // would return the old entry -- whose ResidentExpert was never created,
        // because the promotion is exactly what failed. That hands the caller
        // null gate/up/down pointers to GEMM against, and leaks every cell the
        // cold path just allocated. Reachable whenever the pool refuses a DMA,
        // which is the ordinary state of a constrained card. The WARM bytes go
        // back with drop_host(); the cold path re-reads only if the tier can
        // take them again.
        drop_host(s);
        forget_if_dead(it);
    }

    Slot s;
    s.layer = layer;
    s.expert = expert;
    s.bytes = need;
    s.slice[0] = gate_b;
    s.slice[1] = up_b;
    s.slice[2] = down_b;
    s.count = 1; // a fresh load has already earned one touch
    s.seq = seq_++;

    // COLD -> WARM -> HOT, in that order. Read-through is the invariant: the
    // file read lands in WARM before it lands in VRAM, so this miss is paid
    // once and the next request for the expert is a promotion instead of
    // another read. (The earlier path read straight into VRAM and dropped the
    // bytes, which is why 41% of acquires stayed file loads forever.) Admission
    // is always-admit; when the tier refuses — disabled, or an expert larger
    // than the whole tier — the load still happens, from the mapping.
    warm_admit(src, expert, s);

    xtrace(layer, expert, 2);

    {
        MsTimer t(&room_ms_, phase_sink(&pf_room_ms_, &dc_room_ms_));
        make_vram_room(need);
    }

    const i32 e = expert;
    const GgufTensor *srcs[3] = {src.gate, src.up, src.down};
    QuantTensor *dsts[3] = {&s.m.gate, &s.m.up, &s.m.down};
    const QuantTensor *warms[3] = {&s.h.gate, &s.h.up, &s.h.down};
    const size_t lens[3] = {gate_b, up_b, down_b};
    void *batch_dst[3] = {};
    const void *batch_src[3] = {};
    size_t batch_len[3] = {};
    int nb = 0;
    bool loaded = true;
    for (int i = 0; i < 3 && loaded; i++) {
        const GgufTensor *t = srcs[i];
        const size_t per = lens[i];
        if (!t || per == 0) continue;
        // A refused allocation is NOT an error to ignore: alloc_pooled() answers
        // nullptr when the card has nothing left beside its floor and no free
        // cell of this size class, and that is the ordinary state of a
        // constrained run (see force_reclaim). Using the answer as a pointer is
        // what ended a REAP-288 decode with hipErrorInvalidValue raised by
        // hipMemcpy(nullptr, ...) five tokens in, reported against the upload
        // line rather than this one. Retry against the cache's own LRU, and if
        // that cannot produce a cell, skip the expert instead of the run.
        void *cell = nullptr;
        for (int tries = 0;; tries++) {
            {
                MsTimer tm(&alloc_ms_, phase_sink(&pf_alloc_ms_, &dc_alloc_ms_));
                cell = be_->alloc_pooled(per);
            }
            if (cell) break;
            // 64 is a size-class search, not a room search: experts differ by
            // layer, and a cell of another class is no use to this slice.
            if (tries >= 63 || !force_reclaim(per)) break;
        }
        if (!cell) {
            // Release the slices already taken for this slot, keep WARM (the
            // next acquire of this expert is then a promotion, not a file
            // read), and leave the slot map and the byte ledger untouched.
            void *cells[3] = {};
            size_t clens[3] = {};
            int n = 0;
            for (int j = 0; j < 3; j++)
                if (dsts[j]->data && lens[j]) {
                    cells[n] = dsts[j]->data;
                    clens[n] = lens[j];
                    n++;
                }
            if (n) be_->release_pooled_batch(cells, clens, n);
            for (int j = 0; j < 3; j++) dsts[j]->data = nullptr;
            ++pool_refusals_;
            if (pool_refusals_ == 1)
                KRK_WARN("expert cache: the device refused one expert of layer "
                         "%d (%.0f MiB) and no resident slot could give the cell "
                         "back; the expert is skipped until one is evicted. "
                         "Reduce --kv-mb/--ctx or --expert-cache-mb if this "
                         "repeats, or set KRK_VRAM_FLOOR_MB=0 to let the pool "
                         "take the card's last memory.",
                         layer, static_cast<f64>(need) / 1048576.0);
            loaded = false;
            break;
        }
        dsts[i]->data = cell;
        dsts[i]->type = t->type;
        dsts[i]->n_in = static_cast<i64>(t->ne[0]);
        dsts[i]->n_out = t->n_dims >= 2 ? static_cast<i64>(t->ne[1]) : 1;
        if (warms[i]->present() && warms[i]->data) {
            batch_dst[nb] = dsts[i]->data;
            batch_src[nb] = warms[i]->data;
            batch_len[nb] = per;
            nb++;
        } else {
            // Rejected or no tier: read it in the way the cache did before the
            // tier existed, straight from the mapping through the pull path.
            be_->upload(dsts[i]->data, t->data + static_cast<size_t>(e) * per, per);
        }
    }
    {
        MsTimer t(&xfer_ms_, phase_sink(&pf_xfer_ms_, &dc_xfer_ms_));
        // A COLD miss just read those bytes into WARM, and when that copy is
        // pinned the read left them in a buffer the device can be handed
        // directly -- so the promotion out of it is the same bare DMA the
        // warm-hit path pays, not a ring memcpy of bytes that were already
        // copied once on the way in.
        // merge=false: one DMA per slice. The merged span is the shape the
        // driver refused (hipErrorInvalidValue) on this card, and the split
        // copies of the same bytes from the same address all succeed.
        if (s.host_pinned)
            be_->upload_pinned_batch(batch_dst, batch_src, batch_len, nb,
                                     /*merge=*/false);
        else be_->upload_paged_batch(batch_dst, batch_src, batch_len, nb);
    }

    if (!loaded) {
        // The WARM copy warm_admit() just paid for is KEPT, as the refusal above
        // promises: the next acquire of this expert is then a promotion instead
        // of a file read. That promise only holds if the slot is in the map --
        // `s` is local, so returning here left the tier's host slices allocated
        // and unreachable, and no later clear() could free them. Measured: the
        // promotion-failure fixture left exactly one 24-byte expert in
        // SharedHostRamBudget, which is process-wide, so every later admission
        // in the same process ran against a ledger that never emptied.
        if (s.in_host()) {
            if (slots_.find(key) == slots_.end()) slots_.emplace(key, std::move(s));
            else drop_host(s); // invariant break: never leak what we cannot store
        }
        return nullptr;
    }
    bytes_ += need;
    track_vram(key, s);
    loads_++;
    bytes_loaded_ += need;
    auto ins = slots_.emplace(key, std::move(s));
    if (!ins.second) {
        // Defensive: a key that reappeared between the erase above and here is
        // an invariant break, not a policy choice. Hand the cells back rather
        // than returning an entry whose ResortExpert is somebody else's (see
        // the WARM-hit fall-through).
        void *cells[3] = {};
        size_t clens[3] = {};
        int nc = 0;
        for (int j = 0; j < 3; j++)
            if (dsts[j]->data && lens[j]) {
                cells[nc] = dsts[j]->data;
                clens[nc] = lens[j];
                nc++;
            }
        if (nc) be_->release_pooled_batch(cells, clens, nc);
        ++pool_refusals_;
        return nullptr;
    }
    if (++acquires_ >= decay_period()) decay();
    return &ins.first->second.m;
}

bool ExpertCache::stage_next(const ExpertSource &src, i32 layer, i32 expert,
                             i32 current) {
    if (!be_ || !src.present() || expert == current) return false;
    auto next = slots_.find(slot_key(layer, expert));
    auto cur = slots_.find(slot_key(layer, current));
    if (next == slots_.end() || cur == slots_.end() ||
        next->second.in_vram() || !next->second.in_host() || !cur->second.in_vram())
        return false;
    size_t need = 0;
    for (size_t slice : next->second.slice) if (slice) need += be_->pooled_size(slice);
    if (need > budget_ || cur->second.bytes > budget_ - need ||
        (slot_budget_ && slot_budget_ < 2)) return false;
    cur->second.leased = true;
    const bool ok = promote_to_vram(next->second, true);
    cur->second.leased = false;
    if (ok) ++active_staged_;
    return ok;
}

size_t ExpertCache::promote_batch(const ExpertSource &src, i32 layer,
                                  const i32 *experts, int n) {
    if (!be_ || !src.present() || n <= 0 || budget_ == 0) return 0;
    const std::chrono::steady_clock::time_point t0 =
        std::chrono::steady_clock::now();
    struct ClockOut {
        f64 *ms;
        f64 *ms2;
        std::chrono::steady_clock::time_point t;
        ~ClockOut() {
            const f64 took = std::chrono::duration<f64, std::milli>(
                                 std::chrono::steady_clock::now() - t)
                                 .count();
            *ms += took;
            if (ms2) *ms2 += took;
        }
    } clock_out{&promote_ms_, phase_sink(&pf_promote_ms_, &dc_promote_ms_),
                t0};

    struct Item {
        u64 key = 0;
        ResidentExpert m;
        size_t bytes = 0;
        size_t slice[3] = {0, 0, 0};
        // The WARM sources, captured before any room-making: WARM is inclusive,
        // so a promotion never takes these away (retire only drops device
        // copies), and a batch needs them without a second lookup.
        const void *hsrc[3] = {nullptr, nullptr, nullptr};
        bool pinned = false;
    };
    std::vector<Item> items;
    items.reserve(static_cast<size_t>(n));
    std::unordered_set<u64> seen;
    // Cells taken for this batch are not in bytes_ until the commit below, so
    // make_vram_room() would think the cache still has their room and stop
    // evicting -- and the pool's own growth check then aborts at its budget
    // (measured on the REAP-320 file: "expert pool physical budget exhausted"
    // on the first batched layer). Carrying the outstanding reservation into the
    // room request is what keeps the two ledgers agreeing while a batch is in
    // flight.
    size_t reserved = 0;

    // ---- phase 1: room and cells, no copy yet ---------------------------
    // The batch cannot be sized until every cell is known, so allocation stays
    // per expert. Only the copies are merged, which is where the per-call cost
    // of a decode step lives.
    for (int i = 0; i < n; i++) {
        const i32 e = experts[i];
        if (e < 0 || e >= src.n_expert) continue;
        const u64 key = slot_key(layer, e);
        if (!seen.insert(key).second) continue;
        auto it = slots_.find(key);
        if (it == slots_.end() || !it->second.in_host() ||
            it->second.in_vram())
            continue;
        size_t physical = 0;
        for (size_t sl : it->second.slice)
            if (sl) physical += be_->pooled_size(sl);
        if (physical == 0) continue;
        be_->set_expert_pool_owner(this, owner_bytes(physical));
        {
            MsTimer t(&room_ms_, phase_sink(&pf_room_ms_, &dc_room_ms_));
            make_vram_room(physical + reserved);
        }
        // Room-making erases entries, so re-establish this one rather than
        // reading across the call.
        it = slots_.find(key);
        if (it == slots_.end() || !it->second.in_host()) continue;
        Slot &s = it->second;
        Item item;
        item.key = key;
        item.bytes = physical;
        item.pinned = s.host_pinned;
        const QuantTensor *from[3] = {&s.h.gate, &s.h.up, &s.h.down};
        QuantTensor *to[3] = {&item.m.gate, &item.m.up, &item.m.down};
        bool ok = true;
        for (int j = 0; j < 3 && ok; j++) {
            if (!from[j]->present() || !from[j]->data) continue;
            void *cell = nullptr;
            for (int tries = 0;; tries++) {
                {
                    MsTimer t(&alloc_ms_, phase_sink(&pf_alloc_ms_, &dc_alloc_ms_));
                    cell = be_->alloc_pooled(s.slice[j]);
                }
                if (cell) break;
                if (tries >= 63 || !force_reclaim(s.slice[j])) break;
            }
            if (!cell) {
                ok = false;
                break;
            }
            *to[j] = *from[j];
            to[j]->data = cell;
            item.slice[j] = s.slice[j];
            item.hsrc[j] = from[j]->data;
        }
        if (!ok) {
            void *cells[3] = {};
            size_t clens[3] = {};
            int nc = 0;
            for (int j = 0; j < 3; j++)
                if (to[j]->data && item.slice[j]) {
                    cells[nc] = to[j]->data;
                    clens[nc] = item.slice[j];
                    nc++;
                }
            if (nc) be_->release_pooled_batch(cells, clens, nc);
            ++pool_refusals_;
            continue;
        }
        reserved += physical;
        items.push_back(std::move(item));
    }
    if (items.empty()) return 0;

    // ---- phase 2: one batch per source kind ------------------------------
    // Two at most, because a batch cannot mix a pinned source with a pageable
    // one (the copy paths differ, and mixing them makes a copy read the wrong
    // bytes). Everything else about the layer's routed set is one call.
    for (int pass = 0; pass < 2; pass++) {
        const bool want_pinned = pass == 0;
        std::vector<void *> dst;
        std::vector<const void *> srcp;
        std::vector<size_t> len;
        for (const Item &it2 : items) {
            if (it2.pinned != want_pinned) continue;
            const QuantTensor *q[3] = {&it2.m.gate, &it2.m.up, &it2.m.down};
            for (int j = 0; j < 3; j++) {
                if (!q[j]->present() || !q[j]->data || it2.slice[j] == 0)
                    continue;
                if (!it2.hsrc[j]) continue;
                dst.push_back(q[j]->data);
                srcp.push_back(it2.hsrc[j]);
                len.push_back(it2.slice[j]);
            }
        }
        if (dst.empty()) continue;
        {
            MsTimer t(&xfer_ms_, phase_sink(&pf_xfer_ms_, &dc_xfer_ms_));
            // merge=false: every expert in this batch is its own hipHostMalloc
            // block, and two of them can be neighbours in the address space --
            // which the coalescing loop reads as one contiguous source. The
            // driver refuses a copy that spans two registrations
            // (hipErrorInvalidValue), and the 35B-A3B died on the first batched
            // promotion with the default pinned prefix because of it. One copy
            // per entry is what this shape can have; the measured win the merge
            // exists for is the streamed-layer path, whose source IS one buffer.
            if (want_pinned)
                be_->upload_pinned_batch(dst.data(), srcp.data(), len.data(),
                                         static_cast<int>(dst.size()),
                                         /*merge=*/false);
            else
                be_->upload_paged_batch(dst.data(), srcp.data(), len.data(),
                                        static_cast<int>(dst.size()));
        }
        ++promote_batches_;
    }

    // ---- phase 3: commit -------------------------------------------------
    size_t done = 0;
    for (Item &it2 : items) {
        auto it = slots_.find(it2.key);
        if (it == slots_.end() || !it->second.in_host()) {
            // Evicted between the allocation and the copy: give the cells back
            // rather than leave them resident with no owner.
            void *cells[3] = {};
            size_t clens[3] = {};
            int nc = 0;
            QuantTensor *q[3] = {&it2.m.gate, &it2.m.up, &it2.m.down};
            for (int j = 0; j < 3; j++)
                if (q[j]->present() && q[j]->data && it2.slice[j]) {
                    cells[nc] = q[j]->data;
                    clens[nc] = it2.slice[j];
                    nc++;
                }
            if (nc) be_->release_pooled_batch(cells, clens, nc);
            continue;
        }
        Slot &s = it->second;
        s.bytes = it2.bytes;
        s.m = it2.m;
        table_stale_ = true;
        track_vram(it2.key, s);
        bytes_ += s.bytes;
        if (s.pinned) {
            pin_bytes_ += s.bytes;
            pin_slots_++;
        }
        // The WARM copy stays: that is what makes the next eviction of this
        // expert a promotion instead of a file read.
        bytes_promoted_ += s.bytes;
        promotions_++;
        ++done;
    }
    promote_batch_experts_ += done;
    return done;
}

size_t ExpertCache::stage_batch(const ExpertSource &src, i32 layer,
                                const i32 *experts, int n) {
    if (!be_ || !src.present() || n <= 0 || budget_ == 0) return 0;
    std::unordered_set<u64> seen;
    size_t staged = 0;
    for (int i = 0; i < n; i++) {
        const i32 e = experts[i];
        if (e < 0 || e >= src.n_expert) continue;
        const u64 key = slot_key(layer, e);
        if (!seen.insert(key).second) continue;
        auto it = slots_.find(key);
        // Not WARM yet: this is a miss stage_batch cannot help (acquire will
        // read it), and staging it here would only serialise the read ahead of
        // the rows that need it.
        if (it == slots_.end() || !it->second.in_host() || it->second.in_vram()) continue;
        // Budget: skip rather than evict-and-refill, because a promotion that
        // evicts the expert the current row is about to use is a loss, and the
        // acquire path makes the same call on its own.
        if (over_budget(it->second.bytes)) {
            ++lookahead_rejected_;
            continue;
        }
        if (promote_to_vram(it->second, /*staged=*/true)) {
            ++lookahead_staged_;
            ++staged;
        }
    }
    return staged;
}

size_t ExpertCache::stage_ahead(const ExpertSource &src, i32 layer,
                                const i32 *experts, int n) {
    if (!be_ || !src.present() || n <= 0 || budget_ == 0) return 0;
    if (host_pin_bytes_ == 0) return 0; // the DMA ticket needs pinned sources
    // Guard every slot this pass will need or create. `leased` is what
    // pick_victim already honours, and this pass is a consumer of them exactly
    // as a running row is: without it the LRU walk would take back the slot it
    // just staged (a prefetched slot is the FRESHEST thing in the cache, so by
    // count it is the first victim -- the one way this pass could thrash).
    std::vector<Slot *> guard;
    std::vector<std::pair<i32, i32>> want;
    want.reserve(static_cast<size_t>(n));
    // Every expert of the TARGET layer that is already resident is protected
    // too, not just the ones this pass stages. Same-size-class eviction (the
    // preference force_reclaim uses) would otherwise pick exactly those -- every
    // expert of one layer shares the layer's size classes -- and this pass would
    // evict the set it is trying to complete: measured +10-15% step time with
    // that bug in place, because it doubled the promotions instead of hiding
    // them.
    for (auto &kv : slots_) {
        Slot &g = kv.second;
        if (g.layer == layer && g.in_vram() && !g.leased) {
            g.leased = true;
            guard.push_back(&g);
        }
    }
    for (int i = 0; i < n; i++) {
        const i32 e = experts[i];
        if (e < 0 || e >= src.n_expert) continue;
        want.emplace_back(layer, e);
        const auto it = slots_.find(slot_key(layer, e));
        if (it != slots_.end() && it->second.in_vram() && !it->second.leased) {
            it->second.leased = true;
            guard.push_back(&it->second);
        }
    }
    size_t staged = 0;
    for (const auto &pr : want) {
        auto it = slots_.find(slot_key(pr.first, pr.second));
        if (it == slots_.end() || !it->second.in_host() ||
            it->second.in_vram())
            continue;
        if (!it->second.host_pinned) continue; // the ticket copies pinned bytes
        // The budget is full by construction on a constrained card (that is
        // what makes a miss expensive), so the admission has to retire
        // something the caller is done with rather than skip.
        for (int tries = 0; over_budget(it->second.bytes) && tries < 64; tries++)
            if (!force_reclaim(it->second.slice[0])) break;
        if (over_budget(it->second.bytes)) {
            ++lookahead_rejected_;
            continue;
        }
        if (promote_to_vram(it->second, /*staged=*/true, /*prefetch=*/true)) {
            ++lookahead_staged_;
            ++staged;
            auto pit = slots_.find(slot_key(pr.first, pr.second));
            if (pit != slots_.end() && !pit->second.leased) {
                pit->second.leased = true;
                guard.push_back(&pit->second);
            }
        }
    }
    for (Slot *g : guard) g->leased = false;
    return staged;
}

size_t ExpertCache::touch_layer(const ExpertSource &src, i32 layer,
                                const i32 *experts, int n) {
    if (!be_ || !src.present() || !experts || n <= 0) return 0;
    const GgufTensor *ts[3] = {src.gate, src.up, src.down};
    size_t slice[3] = {0, 0, 0};
    for (int i = 0; i < 3; i++)
        if (ts[i]) slice[i] = src.per_expert(ts[i]);
    const WarmBlockLayout blk = warm_block_layout(slice);
    if (blk.total == 0) return 0;
    // One bounded scratch block, reused for every chunk of the call. The bytes
    // have to land SOMEWHERE for the OS to fault the pages in, and eight
    // experts is enough to keep three reads each in flight, which is what the
    // reader pool needs to reach its ~7 GB/s. Nothing is retained, so this is
    // the whole of the tier's RAM cost under the mapping policy.
    const size_t chunk = 8;
    const size_t want = chunk * blk.total;
    if (touch_buf_bytes_ < want) {
        if (touch_buf_) be_->release_host_pageable(touch_buf_);
        touch_buf_ = be_->alloc_host_pageable(want);
        touch_buf_bytes_ = touch_buf_ ? want : 0;
        // No scratch is a touch that cannot happen, not a failure: the run's
        // first acquire of each expert still reads it.
        if (!touch_buf_) return 0;
    }
    size_t bytes = 0;
    {
        MsTimer t(&read_ms_, phase_sink(&pf_read_ms_, &dc_read_ms_));
        for (int b = 0; b < n; b += static_cast<int>(chunk)) {
            const int last = b + static_cast<int>(chunk) < n
                                 ? b + static_cast<int>(chunk)
                                 : n;
            std::vector<Backend::ReadReq> reqs;
            reqs.reserve(static_cast<size_t>(last - b) * 3);
            for (int k = b; k < last; k++) {
                const i32 id = experts[k];
                if (id < 0 || id >= src.n_expert) continue;
                char *base = static_cast<char *>(touch_buf_) +
                             static_cast<size_t>(k - b) * blk.total;
                for (int i = 0; i < 3; i++) {
                    if (!slice[i] || !ts[i] || !ts[i]->data) continue;
                    reqs.push_back(Backend::ReadReq{
                        base + blk.off[i],
                        ts[i]->data + static_cast<size_t>(id) * slice[i],
                        slice[i]});
                }
            }
            if (reqs.empty()) continue;
            try {
                be_->read_host_batch(reqs.data(), static_cast<int>(reqs.size()));
            } catch (...) {
                // Advisory by contract, exactly like prefetch_layer: a read
                // that threw costs the page-cache warmth, never the run.
                bytes_touched_ += bytes;
                return bytes;
            }
            for (const Backend::ReadReq &r : reqs) bytes += r.bytes;
        }
    }
    bytes_touched_ += bytes;
    return bytes;
}

size_t ExpertCache::prefetch_layer(const ExpertSource &src, i32 layer,
                                   const i32 *experts, int n) {
    // The mapping policy's version of this call: the same reads, kept nowhere.
    if (warm_map_) return touch_layer(src, layer, experts, n);
    if (!be_ || host_budget_ == 0 || !src.present() || n <= 0) return 0;
    const GgufTensor *ts[3] = {src.gate, src.up, src.down};
    size_t slice[3] = {0, 0, 0};
    for (int i = 0; i < 3; i++)
        if (ts[i]) slice[i] = src.per_expert(ts[i]);
    const size_t one = slice[0] + slice[1] + slice[2];
    if (one == 0) return 0;
    // One expert's host footprint is its whole BLOCK, not the sum of the slices:
    // the offsets inside it are aligned (warm_block_layout), so the allocator is
    // asked for `blk.total` bytes. Every budget test below has to use that or it
    // admits an expert the allocator cannot fit. `one` stays the number of bytes
    // actually read out of the file.
    const WarmBlockLayout blk = warm_block_layout(slice);

    // ---- phase 1: make room and reserve a host buffer per expert -----------
    // Keys, not Slot pointers, are carried across phases: make_warm_room below
    // can erase a slot, and a dangling Slot is exactly the kind of aliasing the
    // rest of this file re-establishes references to avoid.
    struct Job {
        u64 key;
        i32 expert;
        void *dst[3];
        // Whole slots only: the promotion batches all three slices and a batch
        // cannot mix a pinned source with a pageable one.
        bool pinned = false;
        // Allocated AND read. The allocation runs on the pipeline's helper
        // thread, so a job can fail there after phase 1 admitted it.
        bool ok = false;
    };
    std::vector<Job> jobs;
    jobs.reserve(static_cast<size_t>(n));
    std::unordered_set<u64> seen;
    size_t reserved = 0;
    // Pinned bytes are spoken for as each job is prepared but only counted once
    // the slot commits, so the budget has to hold both numbers at once.
    size_t pending_pin = 0;
    {
    MsTimer prep_clock(&prep_ms_);
    for (int j = 0; j < n; j++) {
        const i32 e = experts[j];
        if (e < 0 || e >= src.n_expert) continue;
        const u64 key = slot_key(layer, e);
        if (!seen.insert(key).second) continue;
        auto it = slots_.find(key);
        if (it != slots_.end() && it->second.in_host()) continue; // already warm
        // Pending reads own RAM too, but are not committed or evictable yet.
        if (blk.total > host_budget_ - reserved) {
            warm_rejects_++;
            continue;
        }
        {
            MsTimer room_clock(&room_ms_);
            make_warm_room(blk.total + reserved);
        }
        // Preflight this expert against the process-wide ledger. The actual
        // block reservation happens in phase 2, atomically, and accounts for
        // all earlier jobs plus any concurrent KV or expert staging.
        if (!SharedHostRamBudget::can_admit(blk.total, be_->host_available_bytes(),
                                            be_->host_total_bytes())) {
            warm_rejects_++;
            break; // every following expert costs at least this much RAM
        }
        if (reserved > host_budget_ || blk.total > host_budget_ - reserved ||
            host_bytes_ > host_budget_ - reserved - blk.total) {
            warm_rejects_++;
            continue;
        }
        // Room-making may erase entries; never carry its old iterator across it.
        it = slots_.find(key);
        if (it == slots_.end()) {
            Slot s;
            s.layer = layer;
            s.expert = e;
            for (int i = 0; i < 3; i++) s.slice[i] = slice[i];
            s.bytes = one;
            s.count = 0; // staged, not earned: first out of the tier
            s.seq = seq_++;
            it = slots_.emplace(key, std::move(s)).first;
        }
        Job job{key, e, {nullptr, nullptr, nullptr}, false, false};
        // The ranked prefix of the tier, in admission order: this is the hot set
        // the decode loop keeps coming back to, and it is the half whose
        // promotion is a bare DMA instead of a DMA behind a 12.6 MB memcpy.
        const bool pin = host_pin_bytes_ > 0 &&
                         host_pinned_bytes_ + pending_pin + blk.total <=
                             host_pin_bytes_;
        // The ALLOCATION is not made here. It is the expensive half of this
        // function (the driver page-locking an expert's block, ~390 us an
        // expert at the slice granularity it used to use) and doing it here is
        // exactly what kept it serial with the read; it moves to phase 2, where
        // a helper thread runs it ahead of the reader. The reservation is still
        // taken here, because that is what decides which experts are admitted at
        // all.
        job.pinned = pin;
        if (pin) pending_pin += blk.total;
        jobs.push_back(job);
        reserved += blk.total;
    }
    }
    if (jobs.empty()) return 0;

    // ---- phase 2: allocate a chunk ahead of the read -----------------------
    //
    // Two unrelated bottlenecks, previously issued back to back. The read is
    // page-cache I/O and already at this box's ceiling (7.1 GB/s through the
    // reader pool; 7.26 GB/s from 16 standalone threads, so a wider pool buys
    // nothing). The allocation is the driver page-locking an expert's host
    // block. Run in series the phase costs their SUM: measured on
    // Qwen3.5-35B-A3B with a 17.2 GiB tier, 2356 ms of read against 4027 ms of
    // allocation inside a 7214 ms phase.
    //
    // Nothing ties either one to the other, so one helper thread allocates
    // while this thread reads: chunk k's batch is issued as soon as its own
    // buffers exist, and the allocator runs on into the next chunks. The helper
    // takes no slot-map work -- `slots_`, the ledgers and the eviction scan stay
    // on this thread, the same rule prefetch_warm_next follows -- so the only
    // thing crossing a thread boundary is each job's own `dst`.
    auto free_job = [&](Job &jb) {
        // One block per expert: dst[] are offsets into it, so the release is one
        // call through the lowest-offset pointer (warm_block_base).
        warm_release(be_, warm_block_base(jb.dst), jb.pinned, /*deferred=*/false);
        for (int i = 0; i < 3; i++) jb.dst[i] = nullptr;
        jb.ok = false;
    };
    auto alloc_job = [&](Job &jb) {
        // A pinned block is not charged to the pageable LEDGER: it is pin pool,
        // bounded by host_pin_bytes_, and charging both would count the same RAM
        // twice. It IS charged to the tier budget -- a pin decorates WARM, it
        // does not extend it.
        if (!blk.total) {
            jb.ok = true;
            return true;
        }
        void *base = jb.pinned ? be_->alloc_host(blk.total)
                               : be_->alloc_host_pageable_budgeted(blk.total);
        if (!base) {
            free_job(jb);
            return false;
        }
        for (int i = 0; i < 3; i++)
            jb.dst[i] = slice[i] ? static_cast<char *>(base) + blk.off[i]
                                 : nullptr;
        jb.ok = true;
        return true;
    };
    // One chunk is one allocation batch and one read batch. 32 experts is 96
    // requests -- 24 per handle for the default 4-reader pool, still deep
    // enough to saturate the disk -- and about 8 ms of read against 13 ms of
    // allocation, so the allocator stays ahead.
    //
    // KRK_WARM_CHUNK overrides it, which is how the suite drives the multi-chunk
    // path on the small fixtures it can afford: its staging fixture holds four
    // experts, so the default is one chunk and would never cross a boundary.
    //
    // Load time only. A stage issued inside a token (phase_ != PhaseOther) keeps
    // the pre-pipeline shape EXACTLY -- allocate the set, then hand it to the
    // reader in one call -- because there the batch is one layer's routed set
    // and both the thread and the finer batching would cost more than the
    // overlap could return. This method is shared by the warmup and the decode
    // lookahead, so the distinction has to be made here rather than at the call
    // site.
    const bool load_time = phase_ == PhaseOther;
    size_t kChunk = jobs.size(); // one batch: the shape inference always had
    if (load_time) {
        kChunk = 32;
        if (const char *chunk_env = std::getenv("KRK_WARM_CHUNK")) {
            const long v = std::atol(chunk_env);
            if (v >= 1) kChunk = static_cast<size_t>(v);
        }
    }

    std::atomic<size_t> alloc_ready{0};
    f64 alloc_span = 0.0;
    // Only a batch with more than one chunk has anything to overlap; anything
    // smaller (every inference-time call above) takes the serial shape.
    std::thread allocator;
    if (jobs.size() > kChunk) {
        allocator = std::thread([&] {
            const std::chrono::steady_clock::time_point t0 =
                std::chrono::steady_clock::now();
            for (size_t b = 0; b < jobs.size(); b += kChunk) {
                const size_t e = std::min(jobs.size(), b + kChunk);
                for (size_t k = b; k < e; k++) alloc_job(jobs[k]);
                alloc_ready.store(e, std::memory_order_release);
            }
            alloc_span = std::chrono::duration<f64, std::milli>(
                             std::chrono::steady_clock::now() - t0)
                             .count();
        });
    } else {
        const std::chrono::steady_clock::time_point t0 =
            std::chrono::steady_clock::now();
        for (Job &jb : jobs) alloc_job(jb);
        alloc_span = std::chrono::duration<f64, std::milli>(
                         std::chrono::steady_clock::now() - t0)
                         .count();
        alloc_ready.store(jobs.size(), std::memory_order_release);
    }

    const f64 read_before = read_ms_;
    const std::chrono::steady_clock::time_point pip_t0 =
        std::chrono::steady_clock::now();
    for (size_t b = 0; b < jobs.size(); b += kChunk) {
        const size_t e = std::min(jobs.size(), b + kChunk);
        // Wait for THIS chunk's buffers, not the whole layer: the allocator is
        // free to run on ahead, which is what makes the two overlap instead of
        // alternate. Spin with a yield rather than park -- the wait is a few
        // milliseconds and this happens once per layer, at load.
        while (alloc_ready.load(std::memory_order_acquire) < e)
            std::this_thread::yield();

        std::vector<Backend::ReadReq> reqs;
        reqs.reserve((e - b) * 3);
        for (size_t k = b; k < e; k++) {
            const Job &jb = jobs[k];
            if (!jb.ok) continue;
            for (int i = 0; i < 3; i++) {
                if (!slice[i]) continue;
                reqs.push_back(Backend::ReadReq{
                    jb.dst[i],
                    ts[i]->data + static_cast<size_t>(jb.expert) * slice[i],
                    slice[i]});
            }
        }
        if (reqs.empty()) continue;
        MsTimer t(&read_ms_, phase_sink(&pf_read_ms_, &dc_read_ms_));
        try {
            be_->read_host_batch(reqs.data(), static_cast<int>(reqs.size()));
        } catch (...) {
            // A read that threw would hand back uninitialised bytes; dropping
            // these experts leaves them cold, which acquire() still covers.
            for (size_t k = b; k < e; k++)
                if (jobs[k].ok) free_job(jobs[k]);
        }
    }
    if (allocator.joinable()) allocator.join();
    alloc_ms_ += alloc_span;
    const f64 pip_ms = std::chrono::duration<f64, std::milli>(
                           std::chrono::steady_clock::now() - pip_t0)
                           .count();
    // What the pipeline actually bought: the two spans added together minus what
    // they cost as wall time. A phase that got faster and a read that merely
    // moved look identical without it.
    const f64 overlap = (alloc_span + (read_ms_ - read_before)) - pip_ms;
    if (overlap > 0.0) overlap_ms_ += overlap; // serial shape reports 0, not -0

    // A slot phase 1 made for a chunk that never landed holds nothing, and this
    // is the thread allowed to touch the map.
    for (Job &jb : jobs) {
        if (jb.ok) continue;
        warm_rejects_++;
        auto it = slots_.find(jb.key);
        if (it != slots_.end()) forget_if_dead(it);
    }

    // ---- phase 3: commit --------------------------------------------------
    const std::chrono::steady_clock::time_point commit_t0 =
        std::chrono::steady_clock::now();
    size_t total = 0;
    for (Job &jb : jobs) {
        if (!jb.ok) continue; // dropped in the pipeline above
        auto it = slots_.find(jb.key);
        if (it == slots_.end()) { // evicted while the batch was in flight
            // Nothing was ever copied out of this block, so it goes back now
            // rather than through the deferred path.
            warm_release(be_, warm_block_base(jb.dst), jb.pinned,
                         /*deferred=*/false);
            continue;
        }
        Slot &s = it->second;
        QuantTensor *h[3] = {&s.h.gate, &s.h.up, &s.h.down};
        for (int i = 0; i < 3; i++) {
            if (!slice[i]) continue;
            h[i]->data = jb.dst[i];
            h[i]->type = ts[i]->type;
            h[i]->n_in = static_cast<i64>(ts[i]->ne[0]);
            h[i]->n_out = ts[i]->n_dims >= 2 ? static_cast<i64>(ts[i]->ne[1]) : 1;
        }
        s.host_bytes = blk.total;
        s.host_pinned = jb.pinned;
        host_bytes_ += blk.total;
        if (jb.pinned) host_pinned_bytes_ += blk.total;
        warm_admissions_++;
        total += one;
        // The batch above was a real read out of the model file, so it is a
        // COLD miss and has to be counted as one. Without this the report
        // printed "COLD misses 0 (0.0%)" and "0 MiB read from the file" on a
        // run that had just pulled ten gigabytes off the disk, because the
        // prefetch path never touched loads_/bytes_loaded_ -- only the
        // acquire() path did, and acquire() sees a warm hit. The counters then
        // sum correctly against acquires() (a prefetched expert is charged one
        // load here and one promotion there, which is exactly the two events
        // that happened), and the residency report can no longer be used to
        // conclude that a model fits.
        loads_++;
        bytes_loaded_ += one;
    }
    commit_ms_ += std::chrono::duration<krk::f64, std::milli>(
                     std::chrono::steady_clock::now() - commit_t0)
                     .count();
    bytes_staged_ += total;
    return total;
}



// ---- the next layer's set, read while this layer computes -------------------
//
// Measured shape this exists for (REAP-288, --expert-cache-mb 8000
// --expert-warm-mb 20000): 274 MiB/token of file reads, 0.80 ms per layer, all
// of it on the host thread and serial with the layer's GPU work. The read
// depends on nothing the current layer computes, so it moves to a worker thread
// one layer ahead.
//
// Two rules keep this safe without locking the whole cache:
//
//  * the worker never touches `slots_`. It only allocates pinned buffers and
//    copies bytes into them, so the slot map, the ledgers and the eviction scan
//    stay single-threaded; `take_prefetched` (host thread) installs the result.
//  * the read is a memcpy out of the MAPPING, not Backend::read_host. The
//    backend's reader owns file handles and reader threads of its own, and a
//    second thread inside it is a race over those handles. The mapping is
//    read-only, so a memcpy from it is safe from any thread -- and it is what
//    the blocking path measured anyway (274 MiB in 38 ms = 7.2 GiB/s, which is
//    page-cache speed, not device speed).
bool ExpertCache::has_warm(i32 layer, i32 expert) const {
    const auto it = slots_.find(slot_key(layer, expert));
    return it != slots_.end() && it->second.in_host();
}

void ExpertCache::pf_free(PfEntry &e) {
    // One block per expert (warm_block_layout): t[i].p are offsets into it, so
    // the release goes through the base the worker recorded, once.
    if (e.base && be_) be_->release_host(e.base);
    e.base = nullptr;
    for (int i = 0; i < 3; i++) e.t[i].p = nullptr;
}

void ExpertCache::pf_worker() {
    for (;;) {
        ExpertSource src{};
        i32 layer = -1;
        std::vector<i32> ids;
        {
            std::unique_lock<std::mutex> lk(pf_mu_);
            pf_cv_.wait(lk, [this] { return pf_stop_ || !pf_ids_.empty(); });
            if (pf_stop_) return;
            src = pf_src_;
            layer = pf_layer_;
            ids.swap(pf_ids_);
        }
        const GgufTensor *from[3] = {src.gate, src.up, src.down};
        for (i32 e : ids) {
            {
                std::lock_guard<std::mutex> lk(pf_mu_);
                pf_at_layer_ = layer;
                pf_at_expert_ = e;
            }
            PfEntry ent;
            ent.layer = layer;
            ent.expert = e;
            bool ok = true;
            // One block per expert, the same shape the read-through fill uses.
            size_t eff[3] = {0, 0, 0};
            for (int i = 0; i < 3; i++)
                if (from[i]) eff[i] = src.per_expert(from[i]);
            const WarmBlockLayout blk = warm_block_layout(eff);
            if (blk.total) {
                ent.base = be_ ? be_->alloc_host(blk.total) : nullptr;
                ok = ent.base != nullptr;
            }
            for (int i = 0; i < 3 && ok; i++) {
                const GgufTensor *t = from[i];
                const size_t per = eff[i];
                if (!t || per == 0) continue;
                void *p = static_cast<char *>(ent.base) + blk.off[i];
                const void *mapped = t->data + static_cast<size_t>(e) * per;
                try {
                    // Concurrent: this runs while the main thread decodes, so
                    // the row is marked and is not additive with decode.step.
                    HostAdd h("expert.ahread", true);
                    std::memcpy(p, mapped, per);
                    h.bytes(static_cast<krk::i64>(per));
                } catch (...) {
                    // The whole block goes back through pf_free, which owns it.
                    ok = false;
                    break;
                }
                ent.t[i].p = p;
                ent.t[i].len = per;
                ent.t[i].type = t->type;
                ent.t[i].n_in = static_cast<i64>(t->ne[0]);
                ent.t[i].n_out = t->n_dims >= 2 ? static_cast<i64>(t->ne[1]) : 1;
            }
            ent.bytes = blk.total;
            if (!ok) {
                pf_free(ent);
                continue;
            }
            std::lock_guard<std::mutex> lk(pf_mu_);
            if (pf_stop_) {
                pf_free(ent);
                return;
            }
            if (pf_done_.size() >= kPfDoneMax) {
                ++pf_dropped_;
                pf_free(ent);
                continue;
            }
            pf_done_bytes_ += ent.bytes;
            pf_done_.push_back(std::move(ent));
            pf_at_layer_ = -1;
            pf_at_expert_ = -1;
            pf_cv_.notify_all(); // a waiter may be blocked on exactly this entry
        }
        {
            std::lock_guard<std::mutex> lk(pf_mu_);
            pf_at_layer_ = -1;
            pf_at_expert_ = -1;
        }
        pf_cv_.notify_all();
    }
}

void ExpertCache::prefetch_warm_next(const ExpertSource &src, i32 layer,
                                     const i32 *ids, int n) {
    // Nothing is retained under the mapping policy, so a read-ahead would copy
    // bytes into a host buffer the next acquire then drops -- strictly more
    // traffic than the acquire's own mapping read.
    if (warm_map_) return;
    if (!be_ || !src.present() || n <= 0) return;
    {
        std::lock_guard<std::mutex> lk(pf_mu_);
        // One job in flight: a request that arrives while the worker is still
        // reading the previous layer is already late, and queueing it would only
        // make the next layer's read start later than it already has.
        if (!pf_ids_.empty()) return;
        if (pf_done_.size() >= kPfDoneMax) return;
        pf_src_ = src;
        pf_layer_ = layer;
        pf_ids_.assign(ids, ids + n);
        ++pf_issued_;
        if (!pf_thread_.joinable())
            pf_thread_ = std::thread([this] { pf_worker(); });
    }
    pf_cv_.notify_one();
}

bool ExpertCache::take_prefetched(i32 layer, i32 expert) {
    PfEntry ent;
    {
        std::unique_lock<std::mutex> lk(pf_mu_);
        // In flight for THIS expert: wait for it rather than read the same bytes
        // twice. The wait is bounded by one expert's read (~0.1-0.3 ms) and the
        // GPU is running the earlier experts of the layer throughout.
        pf_cv_.wait(lk, [&] {
            return pf_stop_ ||
                   !(pf_at_layer_ == layer && pf_at_expert_ == expert);
        });
        if (pf_stop_ || pf_done_.empty()) return false;
        auto it = pf_done_.end();
        for (auto i = pf_done_.begin(); i != pf_done_.end(); ++i)
            if (i->layer == layer && i->expert == expert) {
                it = i;
                break;
            }
        if (it == pf_done_.end()) return false;
        ent = std::move(*it);
        pf_done_bytes_ -= it->bytes;
        pf_done_.erase(it);
    }
    // The layer got there first, or there is no tier to put it in: the bytes are
    // a duplicate and go back. Never a different answer, only a wasted read.
    // `warm_map_` belongs in the same test: the entry is a host copy, which is
    // exactly what the mapping policy keeps no copy of, so installing it would
    // leave WARM non-empty by a path that never asked the policy.
    auto sit = slots_.find(slot_key(layer, expert));
    if ((sit != slots_.end() && sit->second.in_host()) || !be_ ||
        host_budget_ == 0 || warm_map_) {
        pf_free(ent);
        return false;
    }
    make_warm_room(ent.bytes);
    if (host_bytes_ + ent.bytes > host_budget_) {
        ++warm_rejects_;
        pf_free(ent);
        return false;
    }
    QuantTensor *to[3] = {nullptr, nullptr, nullptr};
    Slot s;
    s.layer = layer;
    s.expert = expert;
    to[0] = &s.h.gate;
    to[1] = &s.h.up;
    to[2] = &s.h.down;
    for (int i = 0; i < 3; i++) {
        if (!ent.t[i].p) continue;
        s.slice[i] = ent.t[i].len;
        to[i]->data = ent.t[i].p;
        to[i]->type = ent.t[i].type;
        to[i]->n_in = ent.t[i].n_in;
        to[i]->n_out = ent.t[i].n_out;
    }
    s.host_bytes = ent.bytes;
    s.host_pinned = true; // alloc_host() is the pinned allotator
    s.count = 1;          // a read-ahead is not a hit: one touch, like a load
    s.seq = seq_++;
    host_bytes_ += ent.bytes;
    host_pinned_bytes_ += ent.bytes;
    ++warm_admissions_;
    ++pf_taken_;
    pf_taken_bytes_ += ent.bytes;
    if (sit != slots_.end()) {
        // A VRAM-only slot: attach the WARM copy without disturbing residency.
        Slot &old = sit->second;
        old.h = s.h;
        old.host_bytes = s.host_bytes;
        old.host_pinned = true;
        for (int i = 0; i < 3; i++)
            if (s.slice[i]) old.slice[i] = s.slice[i];
    } else {
        slots_.emplace(slot_key(layer, expert), std::move(s));
    }
    return true;
}

void ExpertCache::prefetch_warm_join() {
    {
        std::lock_guard<std::mutex> lk(pf_mu_);
        pf_stop_ = true;
        pf_ids_.clear();
    }
    pf_cv_.notify_all();
    if (pf_thread_.joinable()) pf_thread_.join();
    for (auto &e : pf_done_) pf_free(e);
    pf_done_.clear();
    pf_done_bytes_ = 0;
}

ExpertCache::~ExpertCache() {
    prefetch_warm_join();
    if (touch_buf_ && be_) be_->release_host_pageable(touch_buf_);
    touch_buf_ = nullptr;
    touch_buf_bytes_ = 0;
}

} // namespace krk
