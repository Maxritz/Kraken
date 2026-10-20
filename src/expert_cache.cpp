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
#include "krk/vram_guard.hpp"

#include <chrono>

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

} // namespace

void ExpertCache::configure(Backend *be, size_t budget_bytes, size_t host_budget_bytes,
                            size_t slot_budget) {
    clear();
    be_ = be;
    budget_ = budget_bytes;
    host_budget_ = host_budget_bytes;
    slot_budget_ = slot_budget;
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
    // The published table points at slots this call is about to release.
    release_dispatch_table();
    for (auto &kv : slots_) {
        drop(kv.second);
        drop_host(kv.second);
    }
    slots_.clear();
    if (be_) be_->set_expert_pool_owner(this, 0);
    bytes_ = 0;
    vram_slots_ = 0;
    pin_bytes_ = 0;
    pin_slots_ = 0;
    host_bytes_ = 0;
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

// Read-only residency probes. Deliberately not touch(): a caller that is
// deciding whether to *avoid* a promotion must be able to ask without changing
// the LFU counters or the load order that the eviction policy ranks on.
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
    if (count >= kPinThreshold) count = kPinThreshold - 1;
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
    if (!whole_set_resident(n_layer, n_expert)) {
        if (why) {
            *why = "the routed set is not wholly device-resident (";
            *why += std::to_string(vram_slots_);
            *why += " of ";
            *why += std::to_string(static_cast<long long>(n_layer) * n_expert);
            *why += " slots)";
        }
        return nullptr;
    }

    const size_t cells = static_cast<size_t>(n_layer) * static_cast<size_t>(n_expert);
    std::vector<MoeDispatchEntry> host(cells);
    i32 filled = 0;
    for (i32 l = 0; l < n_layer; l++) {
        for (i32 e = 0; e < n_expert; e++) {
            auto it = slots_.find(slot_key(l, e));
            if (it == slots_.end()) continue;
            const ResidentExpert &m = it->second.m;
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
    if (cap + margin >= budget_) return budget_;
    set_vram_budget(cap);
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
    if (!s.pinned && s.count >= kPinThreshold) set_pinned(s, true);
}

void ExpertCache::finish_stage(Slot &s) {
    if (!s.staged) return;
    MsTimer t(&active_wait_ms_);
    be_->wait_marker(s.ready);
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
    if (n) be_->release_pooled_batch(cells, lens, n);
}

void ExpertCache::drop_host(Slot &s) {
    if (!be_) return;
    for (QuantTensor *q : {&s.h.gate, &s.h.up, &s.h.down}) {
        // Deferred: a promotion copy of this expert may still be in flight on a
        // side stream, and freeing the source under it is a use-after-free.
        // The pinned arm needs its own deallocator -- hipFree on a hipHostMalloc
        // block is the other half of that same mistake.
        if (q->present() && q->data) {
            if (s.host_pinned) be_->release_host_pinned_deferred(q->data);
            else be_->release_host_deferred_budgeted(q->data);
        }
        *q = {};
    }
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

void ExpertCache::retire(Slot &s) {
    vram_slots_--;
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
        if (!pick_victim(&victim)) break;
        auto vit = slots_.find(victim);
        if (vit == slots_.end()) break;
        retire(vit->second);
        forget_if_dead(vit);
        evictions_++;
    }
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
    const size_t host_need = s.slice[0] + s.slice[1] + s.slice[2];
    const bool pin = host_pin_bytes_ > 0 && host_need > 0 &&
                     host_pinned_bytes_ + host_need <= host_pin_bytes_;
    s.host_pinned = pin;
    size_t bytes = 0;
    for (int i = 0; i < 3; i++) {
        const GgufTensor *t = from[i];
        const size_t per = s.slice[i];
        if (!t || per == 0) continue;
        // Each slice reserves against the process-wide host ledger atomically;
        // earlier slices and every other cache are already accounted for there.
        // A pinned slot skips the process-wide pageable ledger on purpose: its
        // bytes come out of the driver's pin pool, which host_pin_bytes_ bounds
        // instead, and charging both would double-count the same RAM.
        void *p = pin ? be_->alloc_host(per) : be_->alloc_host_pageable_budgeted(per);
        // Partial copies are dropped by the caller, which calls drop_host and
        // therefore frees whatever landed with the matching deallocator.
        if (!p) return false;
        // A read() out of the page cache, not a memcpy of the mapping: the
        // mapping faults one 4 KiB page at a time and would leave a resident
        // page per staged byte, so the tier would cost the model's size in RAM
        // on top of itself. read_host falls back to the mapping when the file
        // path is unavailable.
        const void *mapped = t->data + static_cast<size_t>(expert) * per;
        try {
            MsTimer t(&read_ms_, phase_sink(&pf_read_ms_, &dc_read_ms_));
            if (!be_->read_host(p, mapped, per)) std::memcpy(p, mapped, per);
        } catch (...) {
            if (pin) be_->release_host(p);
            else be_->release_host_pageable_budgeted(p);
            return false;
        }
        to[i]->data = p;
        to[i]->type = t->type;
        to[i]->n_in = static_cast<i64>(t->ne[0]);
        to[i]->n_out = t->n_dims >= 2 ? static_cast<i64>(t->ne[1]) : 1;
        bytes += per;
    }
    s.host_bytes = bytes;
    if (pin) host_pinned_bytes_ += bytes;
    return true;
}

bool ExpertCache::warm_admit(const ExpertSource &src, i32 expert, Slot &s) {
    const size_t host_need = s.slice[0] + s.slice[1] + s.slice[2];
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

bool ExpertCache::promote_to_vram(Slot &s, bool staged) {
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
        {
            MsTimer t(&alloc_ms_, phase_sink(&pf_alloc_ms_, &dc_alloc_ms_));
            p = be_->alloc_pooled(s.slice[i]);
        }
        if (!p) {
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
        if (s.host_pinned) {
            // The hot arm: the source is already pinned, so this is the DMA and
            // nothing else -- no bounded ring, no host memcpy of the expert in
            // front of it. The backend publishes the copies to the default
            // stream with one event, which is why no marker travels back: the
            // consumer's own stream order already covers the bytes.
            be_->upload_pinned_batch(batch_dst, batch_src, batch_len, nb);
        } else if (staged) {
            s.ready = be_->upload_expert_staged(batch_dst, batch_src, batch_len, nb);
            s.staged = true;
        } else {
            be_->upload_paged_batch(batch_dst, batch_src, batch_len, nb);
        }
    }
    s.m = m;
    table_stale_ = true;
    vram_slots_++;
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
        if (s.pinned && s.count < kPinThreshold) set_pinned(s, false);
    }
    acquires_ = 0;
    decays_++;
}

bool ExpertCache::pick_victim(u64 *out) {
    u64 best = 0;
    bool have = false;
    for (const auto &kv : slots_) {
        const Slot &s = kv.second;
        if (!s.in_vram() || s.leased) continue; // never evict a current consumer
        if (!have) {
            best = kv.first;
            have = true;
            continue;
        }
        const Slot &cur = slots_.at(best);
        // 1. unpinned beats pinned — a pin is a promise, keep it
        if (s.pinned != cur.pinned) {
            if (cur.pinned) {
                best = kv.first;
                continue;
            }
            continue;
        }
        // 2. the rarest expert goes first
        if (s.count < cur.count) {
            best = kv.first;
            continue;
        }
        // 3. equal counts: the older load goes first
        if (s.count == cur.count && s.seq < cur.seq) best = kv.first;
    }
    if (!have) return false;
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
    if (!be_ || !src.present() || host_budget_ == 0) return 0;
    if (expert < 0 || expert >= src.n_expert) return 0;
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
        if (it->second.count < kPinThreshold) it->second.count = kPinThreshold;
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
        if (promote_to_vram(s)) {
            touch(s);
            host_hits_++;
            xtrace(layer, expert, 1);
            if (++acquires_ >= decay_period()) decay();
            return &s.m;
        }
        // No promotion (device allocation failed). Fall through and reload from
        // the mapping; the host copy survives, so the next request is cheaper.
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
    for (int i = 0; i < 3; i++) {
        const GgufTensor *t = srcs[i];
        const size_t per = lens[i];
        if (!t || per == 0) continue;
        {
            MsTimer tm(&alloc_ms_, phase_sink(&pf_alloc_ms_, &dc_alloc_ms_));
            dsts[i]->data = be_->alloc_pooled(per);
        }
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
        if (s.host_pinned) be_->upload_pinned_batch(batch_dst, batch_src, batch_len, nb);
        else be_->upload_paged_batch(batch_dst, batch_src, batch_len, nb);
    }

    bytes_ += need;
    vram_slots_++;
    loads_++;
    bytes_loaded_ += need;
    auto ins = slots_.emplace(key, std::move(s));
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

size_t ExpertCache::prefetch_layer(const ExpertSource &src, i32 layer,
                                   const i32 *experts, int n) {
    if (!be_ || host_budget_ == 0 || !src.present() || n <= 0) return 0;
    const GgufTensor *ts[3] = {src.gate, src.up, src.down};
    size_t slice[3] = {0, 0, 0};
    for (int i = 0; i < 3; i++)
        if (ts[i]) slice[i] = src.per_expert(ts[i]);
    const size_t one = slice[0] + slice[1] + slice[2];
    if (one == 0) return 0;

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
    };
    std::vector<Job> jobs;
    jobs.reserve(static_cast<size_t>(n));
    std::unordered_set<u64> seen;
    size_t reserved = 0;
    // Pinned bytes are spoken for as each job is prepared but only counted once
    // the slot commits, so the budget has to hold both numbers at once.
    size_t pending_pin = 0;
    bool host_reserve_hit = false;
    for (int j = 0; j < n; j++) {
        const i32 e = experts[j];
        if (e < 0 || e >= src.n_expert) continue;
        const u64 key = slot_key(layer, e);
        if (!seen.insert(key).second) continue;
        auto it = slots_.find(key);
        if (it != slots_.end() && it->second.in_host()) continue; // already warm
        // Pending reads own RAM too, but are not committed or evictable yet.
        if (one > host_budget_ - reserved) {
            warm_rejects_++;
            continue;
        }
        make_warm_room(one + reserved);
        // Preflight this expert against the process-wide ledger. The actual
        // per-slice reservations below are atomic and account for all earlier
        // jobs plus any concurrent KV or expert staging.
        if (!SharedHostRamBudget::can_admit(one, be_->host_available_bytes(),
                                            be_->host_total_bytes())) {
            warm_rejects_++;
            host_reserve_hit = true;
            break; // every following expert costs at least this much RAM
        }
        if (reserved > host_budget_ || one > host_budget_ - reserved ||
            host_bytes_ > host_budget_ - reserved - one) {
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
        Job job{key, e, {nullptr, nullptr, nullptr}, false};
        // The ranked prefix of the tier, in admission order: this is the hot set
        // the decode loop keeps coming back to, and it is the half whose
        // promotion is a bare DMA instead of a DMA behind a 12.6 MB memcpy.
        const bool pin = host_pin_bytes_ > 0 &&
                         host_pinned_bytes_ + pending_pin + one <= host_pin_bytes_;
        bool ok = true;
        size_t job_reserved = 0;
        for (int i = 0; i < 3; i++) {
            if (!slice[i]) continue;
            // A pinned slice is not charged to the pageable LEDGER: it is pin
            // pool, bounded by host_pin_bytes_, and charging both would count
            // the same RAM twice. It IS charged to the tier budget below --
            // a pin decorates WARM, it does not extend it.
            job.dst[i] = pin ? be_->alloc_host(slice[i])
                             : be_->alloc_host_pageable_budgeted(slice[i]);
            if (!job.dst[i]) {
                ok = false;
                host_reserve_hit = !pin && !SharedHostRamBudget::can_admit(
                    slice[i], be_->host_available_bytes(), be_->host_total_bytes());
                break;
            }
            job_reserved += slice[i];
        }
        if (!ok) {
            for (int i = 0; i < 3; i++)
                if (job.dst[i]) {
                    if (pin) be_->release_host(job.dst[i]);
                    else be_->release_host_pageable_budgeted(job.dst[i]);
                }
            warm_rejects_++;
            forget_if_dead(it);
            if (host_reserve_hit) break;
            continue;
        }
        job.pinned = pin;
        if (pin) pending_pin += one;
        jobs.push_back(job);
        reserved += job_reserved;
    }
    if (jobs.empty()) return 0;

    // ---- phase 2: one batched read for the whole set ----------------------
    // 3k requests are handed to the backend at once so its reader pool can keep
    // several in flight. This is the entire point of the method.
    std::vector<Backend::ReadReq> reqs;
    reqs.reserve(jobs.size() * 3);
    for (const Job &jb : jobs) {
        for (int i = 0; i < 3; i++) {
            if (!slice[i]) continue;
            reqs.push_back(Backend::ReadReq{
                jb.dst[i],
                ts[i]->data + static_cast<size_t>(jb.expert) * slice[i],
                slice[i]});
        }
    }
    {
        MsTimer t(&read_ms_, phase_sink(&pf_read_ms_, &dc_read_ms_));
        try {
            be_->read_host_batch(reqs.data(), static_cast<int>(reqs.size()));
        } catch (...) {
            for (Job &job : jobs)
                for (void *p : job.dst)
                    if (p) {
                        if (job.pinned) be_->release_host(p);
                        else be_->release_host_pageable_budgeted(p);
                    }
            return 0;
        }
    }

    // ---- phase 3: commit --------------------------------------------------
    size_t total = 0;
    for (Job &jb : jobs) {
        auto it = slots_.find(jb.key);
        if (it == slots_.end()) { // evicted while the batch was in flight
            for (int i = 0; i < 3; i++)
                if (jb.dst[i]) {
                    if (jb.pinned) be_->release_host(jb.dst[i]);
                    else be_->release_host_pageable_budgeted(jb.dst[i]);
                }
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
        s.host_bytes = one;
        s.host_pinned = jb.pinned;
        host_bytes_ += one;
        if (jb.pinned) host_pinned_bytes_ += one;
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
    bytes_staged_ += total;
    return total;
}

} // namespace krk
