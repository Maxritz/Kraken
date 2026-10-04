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

#include <chrono>
#include <cstring>

namespace krk {

namespace {

u64 slot_key(i32 layer, i32 expert) {
    return (static_cast<u64>(static_cast<u32>(layer)) << 32) |
           static_cast<u32>(expert);
}

} // namespace

void ExpertCache::configure(Backend *be, size_t budget_bytes, size_t host_budget_bytes) {
    clear();
    be_ = be;
    budget_ = budget_bytes;
    host_budget_ = host_budget_bytes;
    capacity_ = 0;
}

void ExpertCache::clear() {
    for (auto &kv : slots_) {
        drop(kv.second);
        drop_host(kv.second);
    }
    slots_.clear();
    bytes_ = 0;
    vram_slots_ = 0;
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
}

u32 ExpertCache::touch_count(i32 layer, i32 expert) const {
    const auto it = slots_.find(slot_key(layer, expert));
    return it == slots_.end() ? 0u : it->second.count;
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

void ExpertCache::touch(Slot &s) {
    s.count++;
    if (!s.pinned && s.count >= kPinThreshold) s.pinned = true;
}

void ExpertCache::drop(Slot &s) {
    if (!be_) return;
    QuantTensor *q[3] = {&s.m.gate, &s.m.up, &s.m.down};
    for (int i = 0; i < 3; i++) {
        if (q[i]->present() && q[i]->data)
            be_->release_pooled(q[i]->data, s.slice[i]);
        *q[i] = {};
    }
    s.m = {};
}

void ExpertCache::drop_host(Slot &s) {
    if (!be_) return;
    for (QuantTensor *q : {&s.h.gate, &s.h.up, &s.h.down}) {
        // Deferred: a promotion copy of this expert may still be in flight on a
        // side stream, and freeing the source under it is a use-after-free.
        if (q->present() && q->data) be_->release_host_deferred(q->data);
        *q = {};
    }
    s.h = {};
    host_bytes_ -= s.host_bytes;
    s.host_bytes = 0;
}

void ExpertCache::forget_if_dead(std::unordered_map<u64, Slot>::iterator it) {
    if (it == slots_.end()) return;
    if (!it->second.in_vram() && !it->second.in_host()) slots_.erase(it);
}

void ExpertCache::retire(Slot &s) {
    vram_slots_--;
    bytes_ -= s.bytes;
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

void ExpertCache::make_vram_room() {
    // Evict until this one fits. The scan skips pinned slots; if every resident
    // slot is pinned and none can be dropped, the policy never gets stuck —
    // we simply over-reside by one slot and keep going. Correctness first:
    // the resident set is a performance hint, never an invariant.
    while (vram_slots_ >= capacity_ && vram_slots_ > 0) {
        u64 victim = 0;
        if (!pick_victim(&victim)) break;
        auto vit = slots_.find(victim);
        if (vit == slots_.end()) break;
        const bool all_pinned = vit->second.pinned;
        retire(vit->second);
        forget_if_dead(vit);
        evictions_++;
        if (all_pinned) break; // nothing unpinned existed; stop, do not pin-sweep
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
    size_t bytes = 0;
    for (int i = 0; i < 3; i++) {
        const GgufTensor *t = from[i];
        const size_t per = s.slice[i];
        if (!t || per == 0) continue;
        void *p = be_->alloc_host_pageable(per);
        if (!p) return false; // partial copies are dropped by the caller
        // A read() out of the page cache, not a memcpy of the mapping: the
        // mapping faults one 4 KiB page at a time and would leave a resident
        // page per staged byte, so the tier would cost the model's size in RAM
        // on top of itself. read_host falls back to the mapping when the file
        // path is unavailable.
        const void *mapped = t->data + static_cast<size_t>(expert) * per;
        if (!be_->read_host(p, mapped, per)) std::memcpy(p, mapped, per);
        to[i]->data = p;
        to[i]->type = t->type;
        to[i]->n_in = static_cast<i64>(t->ne[0]);
        to[i]->n_out = t->n_dims >= 2 ? static_cast<i64>(t->ne[1]) : 1;
        bytes += per;
    }
    s.host_bytes = bytes;
    return true;
}

bool ExpertCache::warm_admit(const ExpertSource &src, i32 expert, Slot &s) {
    if (!be_ || host_budget_ == 0 || s.bytes == 0) return false;
    make_warm_room(s.bytes);
    // An expert larger than the whole tier never fits: the alternative would be
    // evicting the entire tier for one copy that still does not fit. That, and
    // a failed allocation, are the only two rejects under always-admit.
    if (host_bytes_ + s.bytes > host_budget_) {
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

bool ExpertCache::promote_to_vram(Slot &s) {
    const std::chrono::steady_clock::time_point t0 =
        std::chrono::steady_clock::now();
    struct ClockOut {
        f64 *ms;
        std::chrono::steady_clock::time_point t;
        ~ClockOut() {
            *ms += std::chrono::duration<f64, std::milli>(
                       std::chrono::steady_clock::now() - t)
                       .count();
        }
    } clock_out{&promote_ms_, t0};
    if (!be_ || !s.in_host()) return false;
    const u64 key = slot_key(s.layer, s.expert);
    make_vram_room();

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
        void *p = be_->alloc_pooled(s.slice[i]);
        if (!p) {
            for (int j = 0; j < 3; j++)
                if (to[j]->present() && to[j]->data)
                    be_->release_pooled(to[j]->data, s.slice[j]);
            return false;
        }
        *to[i] = *from[i];
        to[i]->data = p;
        batch_dst[nb] = p;
        batch_src[nb] = from[i]->data;
        batch_len[nb] = s.slice[i];
        nb++;
    }
    be_->upload_paged_batch(batch_dst, batch_src, batch_len, nb);
    s.m = m;
    vram_slots_++;
    bytes_ += s.bytes;
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
        if (s.pinned && s.count < kPinThreshold) s.pinned = false;
    }
    acquires_ = 0;
    decays_++;
}

bool ExpertCache::pick_victim(u64 *out) {
    u64 best = 0;
    bool have = false;
    for (const auto &kv : slots_) {
        const Slot &s = kv.second;
        if (!s.in_vram()) continue; // a host copy is not occupying VRAM
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
    const size_t need = gate_b + up_b + down_b;
    const size_t cap = need > 0 ? (budget_ / need) : 0;
    capacity_ = cap > 0 ? cap : 1;

    const u64 key = slot_key(layer, expert);
    auto it = slots_.find(key);
    if (it != slots_.end()) {
        Slot &s = it->second;
        if (s.in_vram()) {
            // Hit: the traffic that makes an expert hot happens here.
            touch(s);
            hits_++;
            if (++acquires_ >= decay_period()) decay();
            return &s.m;
        }
        // WARM hit: the weights are already in RAM, so this acquire costs a DMA
        // instead of a re-read of the file. It also earns the same counter —
        // an expert hot enough to be evicted is still hot.
        if (promote_to_vram(s)) {
            touch(s);
            host_hits_++;
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

    make_vram_room();

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
        dsts[i]->data = be_->alloc_pooled(per);
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
    be_->upload_paged_batch(batch_dst, batch_src, batch_len, nb);

    bytes_ += need;
    vram_slots_++;
    loads_++;
    bytes_loaded_ += need;
    auto ins = slots_.emplace(key, std::move(s));
    if (++acquires_ >= decay_period()) decay();
    return &ins.first->second.m;
}

} // namespace krk
