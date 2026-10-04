// expert_cache.cpp — lazy, frequency-aware residency for MoE expert matrices.
//
// The policy is LFU with aging: hits earn counters, counters pin hot experts,
// and periodic decay makes pins expire when traffic moves on. See the header
// for the reasoning — expert popularity is skewed and sticky, which is exactly
// the workload LRU is worst at.
//
// Eviction has two destinations. Without a host budget a victim is released
// and the next request re-reads the GGUF mapping; with one it is demoted to
// page-locked host memory first, so the next request is a DMA. The tier can
// also be filled ahead of the run (preload, below): on a stack with hundreds
// of experts per layer a short generation is dominated by *first* touches,
// and those are a miss the policy above can never turn into a hit.
#include "krk/expert_cache.hpp"

#include <chrono>
#include <cstring>

namespace krk {

namespace {

u64 slot_key(i32 layer, i32 expert) {
    return (static_cast<u64>(static_cast<u32>(layer)) << 32) |
           static_cast<u32>(expert);
}

// Sentinel for "do not exclude any slot". Layer and expert ids are both
// non-negative, so this key cannot name a real slot.
constexpr u64 kNoSlot = ~static_cast<u64>(0);

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

size_t ExpertCache::host_slots() const {
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
        if (q->present() && q->data) be_->release_host(q->data);
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

void ExpertCache::retire(Slot &s, u64 keep_host_key) {
    vram_slots_--;
    bytes_ -= s.bytes;
    if (demote_to_host(s, keep_host_key)) {
        demotions_++;
        // The pin promised VRAM residency and that just went away under duress
        // (every other device slot was pinned). Keep the counter, drop the pin:
        // the host tier must not inherit protection that VRAM traffic revoked.
        if (s.pinned) s.pinned = false;
        return;
    }
    drop(s);
}

void ExpertCache::make_vram_room(u64 keep_host_key) {
    // Evict until this one fits. The scan skips pinned slots; if every resident
    // slot is pinned and none can be dropped, the policy never gets stuck —
    // we simply over-reside by one slot and keep going. Correctness first:
    // the resident set is a performance hint, never an invariant.
    //
    // keep_host_key names a slot that is mid-promotion: it has no device copy
    // yet, so it is invisible to pick_victim, but it still owns the host copy
    // the promotion is about to read. The key is threaded all the way down so
    // the demotions triggered here cannot evict it out from under us.
    while (vram_slots_ >= capacity_ && vram_slots_ > 0) {
        u64 victim = 0;
        if (!pick_victim(&victim)) break;
        auto vit = slots_.find(victim);
        if (vit == slots_.end()) break;
        const bool all_pinned = vit->second.pinned;
        retire(vit->second, keep_host_key);
        forget_if_dead(vit);
        evictions_++;
        if (all_pinned) break; // nothing unpinned existed; stop, do not pin-sweep
    }
}

void ExpertCache::make_host_room(size_t need, u64 keep_key) {
    if (host_budget_ == 0) return;
    while (host_bytes_ + need > host_budget_) {
        u64 victim = 0;
        // `keep_key` is never a candidate: the slot being promoted still has
        // its host copy at this point, and evicting it would free the very node
        // the caller is holding a reference to.
        if (!pick_host_victim(&victim, keep_key)) return; // nothing left to evict
        auto vit = slots_.find(victim);
        if (vit == slots_.end()) return;
        drop_host(vit->second);
        forget_if_dead(vit);
    }
}

bool ExpertCache::demote_to_host(Slot &s, u64 keep_host_key) {
    if (!be_ || host_budget_ == 0 || !s.in_vram()) return false;
    make_host_room(s.bytes, keep_host_key);
    // An expert larger than the whole tier never fits: the alternative would be
    // evicting the entire tier for one copy that still does not fit.
    if (host_bytes_ + s.bytes > host_budget_) return false;

    ResidentExpert h;
    const QuantTensor *from[3] = {&s.m.gate, &s.m.up, &s.m.down};
    QuantTensor *to[3] = {&h.gate, &h.up, &h.down};
    for (int i = 0; i < 3; i++) {
        if (!from[i]->present()) continue;
        void *p = be_->alloc_host(s.slice[i]);
        if (!p) {
            // Pinned pool exhausted. Undo the partial copy and let the caller
            // fall back to dropping the expert to the mapping.
            for (QuantTensor *q : to)
                if (q->present() && q->data) be_->release_host(q->data);
            return false;
        }
        be_->download(p, from[i]->data, s.slice[i]);
        *to[i] = *from[i];
        to[i]->data = p;
    }
    s.h = h;
    s.host_bytes = s.bytes;
    host_bytes_ += s.host_bytes;
    bytes_demoted_ += s.host_bytes;
    drop(s); // the weights live in the tier now
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
    make_vram_room(key);

    // Making room can demote (and therefore recurse through the tier), so
    // re-establish that this slot is still tracked and still holds its copy
    // before touching it. With the exclusion below it always is; this is the
    // belt to that braces, and it turns any future aliasing into a plain miss
    // instead of a write into a freed node.
    {
        const auto it = slots_.find(key);
        if (it == slots_.end() || !it->second.in_host()) return false;
        s = it->second;
    }

    ResidentExpert m;
    const QuantTensor *from[3] = {&s.h.gate, &s.h.up, &s.h.down};
    QuantTensor *to[3] = {&m.gate, &m.up, &m.down};
    for (int i = 0; i < 3; i++) {
        if (!from[i]->present()) continue;
        void *p = be_->alloc_pooled(s.slice[i]);
        if (!p) {
            for (int j = 0; j < 3; j++)
                if (to[j]->present() && to[j]->data)
                    be_->release_pooled(to[j]->data, s.slice[j]);
            return false;
        }
        be_->upload(p, from[i]->data, s.slice[i]);
        *to[i] = *from[i];
        to[i]->data = p;
    }
    s.m = m;
    vram_slots_++;
    bytes_ += s.bytes;
    // The RAM copy has done its job; keeping it would double-count the bytes
    // without making the next promotion any faster.
    drop_host(s);
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

bool ExpertCache::pick_host_victim(u64 *out, u64 keep_key) {
    // Same frequency ordering as the device tier, minus the pin: a host copy is
    // a cache of the mapping, so losing one costs a future DMA, not a reload
    // storm. Letting a stale pin defend it here would make the tier grow into a
    // graveyard of experts VRAM traffic abandoned long ago.
    u64 best = 0;
    bool have = false;
    for (const auto &kv : slots_) {
        const Slot &s = kv.second;
        if (kv.first == keep_key) continue; // the caller's own slot
        if (!s.in_host()) continue;
        if (!have) {
            best = kv.first;
            have = true;
            continue;
        }
        const Slot &cur = slots_.at(best);
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

bool ExpertCache::stage_host(const ExpertSource &src, i32 expert, Slot &s) {
    const GgufTensor *from[3] = {src.gate, src.up, src.down};
    QuantTensor *to[3] = {&s.h.gate, &s.h.up, &s.h.down};
    size_t bytes = 0;
    for (int i = 0; i < 3; i++) {
        const GgufTensor *t = from[i];
        const size_t per = s.slice[i];
        if (!t || per == 0) continue;
        void *p = be_->alloc_host(per);
        if (!p) return false; // partial copies are dropped by the caller
        // A host copy out of the mapping is a memcpy, not a device call. On a
        // cold mapping the pages fault in one at a time — once, here, instead
        // of on the token that first routes to this expert.
        std::memcpy(p, t->data + static_cast<size_t>(expert) * per, per);
        to[i]->data = p;
        to[i]->type = t->type;
        to[i]->n_in = static_cast<i64>(t->ne[0]);
        to[i]->n_out = t->n_dims >= 2 ? static_cast<i64>(t->ne[1]) : 1;
        bytes += per;
    }
    s.host_bytes = bytes;
    return true;
}

size_t ExpertCache::preload_one(const ExpertSource &src, i32 layer, i32 expert) {
    if (!be_ || !src.present() || host_budget_ == 0) return 0;
    if (expert < 0 || expert >= src.n_expert) return 0;
    const size_t need = src.per_expert(src.gate) + src.per_expert(src.up) +
                        src.per_expert(src.down);
    if (need == 0 || host_bytes_ + need > host_budget_) return 0;
    const u64 key = slot_key(layer, expert);
    if (slots_.count(key)) return 0; // already resident in some tier
    Slot s;
    s.layer = layer;
    s.expert = expert;
    s.bytes = need;
    s.slice[0] = src.per_expert(src.gate);
    s.slice[1] = src.per_expert(src.up);
    s.slice[2] = src.per_expert(src.down);
    s.seq = seq_++;
    if (!stage_host(src, expert, s)) {
        drop_host(s);
        return 0; // the pinned pool is the limit; stop rather than thrash
    }
    host_bytes_ += s.host_bytes;
    bytes_staged_ += s.host_bytes;
    slots_.emplace(key, std::move(s));
    return need;
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
        // L2 hit: the weights are already in RAM, so this acquire costs a DMA
        // instead of a re-read of the mapping. It also earns the same counter —
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

    make_vram_room(kNoSlot);

    Slot s;
    s.layer = layer;
    s.expert = expert;
    s.bytes = need;
    s.slice[0] = gate_b;
    s.slice[1] = up_b;
    s.slice[2] = down_b;
    s.count = 1; // a fresh load has already earned one touch
    s.seq = seq_++;

    const i32 e = expert;
    auto load = [&](const GgufTensor *t, size_t per, QuantTensor *out) {
        if (!t || per == 0) return;
        out->data = be_->alloc_pooled(per);
        be_->upload(out->data, t->data + static_cast<size_t>(e) * per, per);
        out->type = t->type;
        out->n_in = static_cast<i64>(t->ne[0]);
        out->n_out = t->n_dims >= 2 ? static_cast<i64>(t->ne[1]) : 1;
    };
    load(src.gate, gate_b, &s.m.gate);
    load(src.up, up_b, &s.m.up);
    load(src.down, down_b, &s.m.down);

    bytes_ += need;
    vram_slots_++;
    loads_++;
    bytes_loaded_ += need;
    auto ins = slots_.emplace(key, std::move(s));
    if (++acquires_ >= decay_period()) decay();
    return &ins.first->second.m;
}

} // namespace krk
