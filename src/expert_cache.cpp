// expert_cache.cpp — lazy, frequency-aware residency for MoE expert matrices.
//
// The policy is LFU with aging: hits earn counters, counters pin hot experts,
// and periodic decay makes pins expire when traffic moves on. See the header
// for the reasoning — expert popularity is skewed and sticky, which is exactly
// the workload LRU is worst at.
#include "krk/expert_cache.hpp"

namespace krk {

namespace {

u64 slot_key(i32 layer, i32 expert) {
    return (static_cast<u64>(static_cast<u32>(layer)) << 32) |
           static_cast<u32>(expert);
}

} // namespace

void ExpertCache::configure(Backend *be, size_t budget_bytes) {
    clear();
    be_ = be;
    budget_ = budget_bytes;
    capacity_ = 0;
}

void ExpertCache::clear() {
    for (auto &kv : slots_) drop(kv.second);
    slots_.clear();
    bytes_ = 0;
    capacity_ = 0;
    acquires_ = 0;
    seq_ = 0;
    // Statistics cover the current residency generation: clear() starts a new
    // one, so callers (and tests) read numbers about the policy they configured.
    loads_ = 0;
    evictions_ = 0;
    hits_ = 0;
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

void ExpertCache::drop(Slot &s) {
    if (!be_) return;
    for (QuantTensor *q : {&s.m.gate, &s.m.up, &s.m.down}) {
        if (q->present() && q->data) be_->release(q->data);
        *q = {};
    }
    s.m = {};
    s.bytes = 0;
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

u64 ExpertCache::pick_victim() {
    u64 best = 0;
    bool have = false;
    for (const auto &kv : slots_) {
        const Slot &s = kv.second;
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
    return best;
}

const ResidentExpert *ExpertCache::acquire(const ExpertSource &src, i32 layer,
                                           i32 expert) {
    if (!be_ || !src.present()) return nullptr;
    if (expert < 0 || expert >= src.n_expert) return nullptr;

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
        // Hit: the traffic that makes an expert hot happens here.
        Slot &s = it->second;
        s.count++;
        if (!s.pinned && s.count >= kPinThreshold) s.pinned = true;
        hits_++;
        if (++acquires_ >= kPinDecay) decay();
        return &s.m;
    }

    // Evict until this one fits. The scan skips pinned slots; if every resident
    // slot is pinned and none can be dropped, the policy never gets stuck —
    // we simply over-reside by one slot and keep going. Correctness first:
    // the resident set is a performance hint, never an invariant.
    while (static_cast<size_t>(slots_.size()) >= capacity_ && !slots_.empty()) {
        const u64 victim = pick_victim();
        auto vit = slots_.find(victim);
        if (vit == slots_.end()) break;
        const bool all_pinned = vit->second.pinned;
        bytes_ -= vit->second.bytes;
        drop(vit->second);
        slots_.erase(vit);
        evictions_++;
        if (all_pinned) break; // nothing unpinned existed; stop, do not pin-sweep
    }

    Slot s;
    s.layer = layer;
    s.expert = expert;
    s.bytes = need;
    s.count = 1; // a fresh load has already earned one touch
    s.seq = seq_++;

    const i32 e = expert;
    auto load = [&](const GgufTensor *t, size_t per, QuantTensor *out) {
        if (!t || per == 0) return;
        out->data = be_->alloc(per);
        be_->upload(out->data, t->data + static_cast<size_t>(e) * per, per);
        out->type = t->type;
        out->n_in = static_cast<i64>(t->ne[0]);
        out->n_out = t->n_dims >= 2 ? static_cast<i64>(t->ne[1]) : 1;
    };
    load(src.gate, gate_b, &s.m.gate);
    load(src.up, up_b, &s.m.up);
    load(src.down, down_b, &s.m.down);

    bytes_ += need;
    loads_++;
    auto ins = slots_.emplace(key, std::move(s));
    if (++acquires_ >= kPinDecay) decay();
    return &ins.first->second.m;
}

} // namespace krk
