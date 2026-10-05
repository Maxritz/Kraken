// kv_tier.cpp — HOT / WARM / COLD residency for the KV cache
#include "krk/kv_tier.hpp"

#include <cstdio>
#include <cstring>
#include <chrono>
#include <cstdlib>
#include <thread>

namespace krk {

// ---- persistent trace: survives a reboot, pinpoints a hang ----
//
// KRK_TIER_TRACE=<path> opens an UNBUFFERED log. Every tier-cache operation
// writes one line and it hits the OS immediately, so if the GPU wedges or
// the box reboots, the last line names the exact op that was in flight.
// KRK_TIER_WAIT_S bounds every slot wait (default 30 s); a wait that exceeds
// it logs STUCK and exits 42 instead of hanging into a driver watchdog.
// KRK_TIER_FENCE=1 forces a full device sync after every upload/download/
// fill0 -- slow, but it turns any remaining ordering question into a
// non-issue while the trace is being read.

namespace {
std::FILE *g_trace = nullptr;
long long g_t0 = 0;
int g_seq = 0;
bool g_fence = false;
int g_wait_s = 30;

long long tier_now_ms() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}
void trace_open() {
    if (g_trace) return;
    const char *p = std::getenv("KRK_TIER_TRACE");
    if (!p || !p[0]) return;
    g_trace = std::fopen(p, "w");
    if (!g_trace) return;
    std::setvbuf(g_trace, nullptr, _IONBF, 0); // unbuffered: every line hits the OS
    g_t0 = tier_now_ms();
    if (const char *f = std::getenv("KRK_TIER_FENCE"))
        g_fence = f[0] == '1';
    if (const char *w = std::getenv("KRK_TIER_WAIT_S")) {
        const int v = std::atoi(w);
        if (v > 0) g_wait_s = v;
    }
    std::fprintf(g_trace, "[%lld] +%lldms trace open fence=%d wait_s=%d\n",
                 (long long)0, (long long)0, (int)g_fence, g_wait_s);
}
void trace_line(const char *op, long long a, long long b, const char *extra) {
    if (!g_trace) return;
    std::fprintf(g_trace, "[%d] +%lldms %-14s a=%lld b=%lld %s\n",
                 g_seq++, tier_now_ms() - g_t0, op, a, b, extra ? extra : "");
}
} // namespace

const char *kv_tier_name(KvTier t) {
    switch (t) {
    case KvTier::kHot: return "HOT";
    case KvTier::kWarm: return "WARM";
    case KvTier::kCold: return "COLD";
    default: return "absent";
    }
}

KvTierCache::~KvTierCache() { detach(); }

void KvTierCache::detach() {
    if (!be_) return;
    for (void *p : slot_mem_)
        if (p) be_->release(p);
    for (void *m : slot_marker_)
        if (m) be_->release_marker(m);
    for (void *p : warm_mem_)
        if (p) be_->release_host_pageable(p);
    if (flat_) be_->release(flat_);
    // Clear everything so a second call -- from the destructor after the
    // engine has already called this -- is a no-op rather than a double free.
    slot_mem_.clear();
    slot_marker_.clear();
    warm_mem_.clear();
    layer_slot_.clear();
    slot_layer_.clear();
    lru_.clear();
    tier_.clear();
    flat_ = nullptr;
    be_ = nullptr;
}

bool KvTierCache::init(Backend *be, i64 n_layer, size_t layer_bytes,
                       size_t hot_budget, size_t warm_budget,
                       const std::string &cold_dir, std::string *err) {
    detach();
    trace_open();
    trace_line("init", n_layer, (long long)layer_bytes, cold_dir.c_str());
    be_ = be;
    n_layer_ = n_layer > 0 ? n_layer : 0;
    layer_bytes_ = layer_bytes;
    hot_budget_ = hot_budget;
    warm_budget_ = warm_budget;
    cold_dir_ = cold_dir;

    st_.layers = n_layer_;
    st_.layer_bytes = layer_bytes;

    if (!be_ || n_layer_ <= 0 || layer_bytes_ == 0) return true;

    const size_t total = static_cast<size_t>(n_layer_) * layer_bytes_;

    // Everything fits: one flat allocation, byte-identical to the untiered
    // engine. No host buffer is touched, so a model that already fit cannot
    // regress here.
    if (hot_budget_ >= total || n_layer_ == 1) {
        flat_ = be_->alloc(total);
        if (!flat_) {
            if (err) *err = "kv: device allocation failed";
            be_ = nullptr;
            return false;
        }
        return true;
    }

    n_slots_ = static_cast<int>(hot_budget_ / layer_bytes_);
    if (n_slots_ < 1) n_slots_ = 1;
    if (n_slots_ > n_layer_) n_slots_ = static_cast<int>(n_layer_);

    const size_t n = static_cast<size_t>(n_layer_);
    slot_mem_.assign(static_cast<size_t>(n_slots_), nullptr);
    slot_marker_.assign(static_cast<size_t>(n_slots_), nullptr);
    slot_layer_.assign(static_cast<size_t>(n_slots_), -1);
    layer_slot_.assign(n, -1);
    lru_.assign(n, 0);
    tier_.assign(n, KvTier::kAbsent);
    warm_mem_.assign(n, nullptr);

    for (int i = 0; i < n_slots_; i++) {
        slot_mem_[static_cast<size_t>(i)] = be_->alloc(layer_bytes_);
        slot_marker_[static_cast<size_t>(i)] = be_->make_marker();
        if (!slot_mem_[static_cast<size_t>(i)]) {
            if (err)
                *err = "kv: could not allocate a HOT slot; lower the KV VRAM "
                       "budget or the context";
            be_ = nullptr;
            return false;
        }
    }

    n_warm_slots_ = static_cast<i64>(warm_budget_ / layer_bytes_);
    if (n_warm_slots_ < 0) n_warm_slots_ = 0;

    tiered_ = true;
    st_.tiered = true;
    st_.slots = n_slots_;
    st_.hot_bytes = static_cast<size_t>(n_slots_) * layer_bytes_;
    return true;
}

std::string KvTierCache::cold_path(i64 layer) const {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "/kv_layer_%lld.bin",
                  static_cast<long long>(layer));
    return cold_dir_ + buf;
}

// Demote whatever currently owns `slot`. WARM first; COLD only if WARM is
// full or unavailable; otherwise the page is dropped and recomputed.
void KvTierCache::fence_if_on(const char *op) {
    if (g_fence && be_) {
        trace_line("fence", 0, 0, op);
        be_->sync();
        trace_line("fence-ok", 0, 0, op);
    }
}

// Wait for a slot's previous occupant, bounded. An unbounded wait turns a
// stuck GPU into a hung process and then a driver watchdog reset with no
// evidence. This turns it into a named STUCK line and a clean exit 42, with
// the trace file showing exactly which slot and layer were in flight.
bool KvTierCache::wait_slot(int slot, const char *why, i64 layer) {
    void *m = slot < 0 || slot >= n_slots_
                  ? nullptr
                  : slot_marker_[static_cast<size_t>(slot)];
    trace_line("wait", layer, slot, why);
    const long long dl = tier_now_ms() + (long long)g_wait_s * 1000LL;
    long long last_log = 0;
    while (!be_->poll_marker(m)) {
        const long long t = tier_now_ms();
        if (t - last_log > 5000) {
            trace_line("wait-hold", layer, slot, why);
            last_log = t;
        }
        if (t > dl) {
            trace_line("STUCK", layer, slot, why);
            std::fflush(nullptr);
            std::_Exit(42);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    trace_line("wait-ok", layer, slot, why);
    return true;
}

void KvTierCache::evict_slot(int slot, i64 keep) {
    const i64 owner = slot_layer_[static_cast<size_t>(slot)];
    if (owner < 0 || owner == keep) return;
    wait_slot(slot, "evict", owner);
    if (to_warm(owner, slot)) return;
    if (to_cold(owner, slot)) return;
    tier_[static_cast<size_t>(owner)] = KvTier::kAbsent;
    layer_slot_[static_cast<size_t>(owner)] = -1;
    slot_layer_[static_cast<size_t>(slot)] = -1;
    st_.evictions_to_cold++; // counted as a drop; there is nowhere left to put it
}

bool KvTierCache::to_warm(i64 layer, int slot) {
    trace_line("to-warm", layer, slot, "");
    i64 live = 0;
    for (KvTier t : tier_)
        if (t == KvTier::kWarm) live++;
    if (live >= n_warm_slots_) return false;

    void *host = be_->alloc_host_pageable(layer_bytes_);
    if (!host) return false;
    be_->download(host, slot_mem_[static_cast<size_t>(slot)], layer_bytes_);
    fence_if_on("download");
    warm_mem_[static_cast<size_t>(layer)] = host;
    tier_[static_cast<size_t>(layer)] = KvTier::kWarm;
    layer_slot_[static_cast<size_t>(layer)] = -1;
    slot_layer_[static_cast<size_t>(slot)] = -1;
    st_.evictions_to_warm++;
    st_.migrate_bytes += static_cast<i64>(layer_bytes_);
    return true;
}

bool KvTierCache::to_cold(i64 layer, int slot) {
    trace_line("to-cold", layer, slot, "");
    if (cold_dir_.empty()) return false;
    void *host = be_->alloc_host_pageable(layer_bytes_);
    if (!host) return false;
    be_->download(host, slot_mem_[static_cast<size_t>(slot)], layer_bytes_);
    fence_if_on("download");
    std::FILE *f = std::fopen(cold_path(layer).c_str(), "wb");
    if (!f) {
        be_->release_host_pageable(host);
        return false;
    }
    const size_t w = std::fwrite(host, 1, layer_bytes_, f);
    std::fclose(f);
    be_->release_host_pageable(host);
    if (w != layer_bytes_) return false;
    tier_[static_cast<size_t>(layer)] = KvTier::kCold;
    layer_slot_[static_cast<size_t>(layer)] = -1;
    slot_layer_[static_cast<size_t>(slot)] = -1;
    st_.evictions_to_cold++;
    st_.migrate_bytes += static_cast<i64>(layer_bytes_);
    return true;
}

bool KvTierCache::from_warm(i64 layer, int slot) {
    trace_line("from-warm", layer, slot, "");
    void *host = warm_mem_[static_cast<size_t>(layer)];
    if (!host) return false;
    be_->upload(slot_mem_[static_cast<size_t>(slot)], host, layer_bytes_);
    fence_if_on("upload");
    be_->release_host_pageable(host);
    warm_mem_[static_cast<size_t>(layer)] = nullptr;
    tier_[static_cast<size_t>(layer)] = KvTier::kHot;
    layer_slot_[static_cast<size_t>(layer)] = slot;
    slot_layer_[static_cast<size_t>(slot)] = layer;
    lru_[static_cast<size_t>(layer)] = ++clock_;
    st_.promotions_from_warm++;
    st_.migrate_bytes += static_cast<i64>(layer_bytes_);
    return true;
}

bool KvTierCache::from_cold(i64 layer, int slot) {
    trace_line("from-cold", layer, slot, "");
    if (cold_dir_.empty()) return false;
    std::FILE *f = std::fopen(cold_path(layer).c_str(), "rb");
    if (!f) return false;
    void *host = be_->alloc_host_pageable(layer_bytes_);
    if (!host) {
        std::fclose(f);
        return false;
    }
    const size_t r = std::fread(host, 1, layer_bytes_, f);
    std::fclose(f);
    if (r != layer_bytes_) {
        be_->release_host_pageable(host);
        return false;
    }
    be_->upload(slot_mem_[static_cast<size_t>(slot)], host, layer_bytes_);
    fence_if_on("upload");
    be_->release_host_pageable(host);
    tier_[static_cast<size_t>(layer)] = KvTier::kHot;
    layer_slot_[static_cast<size_t>(layer)] = slot;
    slot_layer_[static_cast<size_t>(slot)] = layer;
    lru_[static_cast<size_t>(layer)] = ++clock_;
    st_.promotions_from_cold++;
    st_.migrate_bytes += static_cast<i64>(layer_bytes_);
    return true;
}

// A free slot if there is one, else the least-recently-used one that is not
// the layer we are about to make resident.
int KvTierCache::take_slot(i64 keep) {
    for (int i = 0; i < n_slots_; i++)
        if (slot_layer_[static_cast<size_t>(i)] < 0) return i;

    int victim = -1;
    i64 best = 0;
    for (int i = 0; i < n_slots_; i++) {
        const i64 owner = slot_layer_[static_cast<size_t>(i)];
        if (owner < 0 || owner == keep) continue;
        if (victim < 0 || lru_[static_cast<size_t>(owner)] < best) {
            victim = i;
            best = lru_[static_cast<size_t>(owner)];
        }
    }
    if (victim < 0) return -1; // every slot holds the layer we want
    evict_slot(victim, keep);
    return victim;
}

int KvTierCache::ensure_hot(i64 layer) {
    trace_line("ensure", layer, 0, "");
    if (!tiered_) return 0;
    if (layer < 0 || layer >= n_layer_) return -1;

    const int cur = layer_slot_[static_cast<size_t>(layer)];
    if (cur >= 0 && tier_[static_cast<size_t>(layer)] == KvTier::kHot) {
        lru_[static_cast<size_t>(layer)] = ++clock_;
        return cur;
    }

    const int slot = take_slot(layer);
    if (slot < 0) return -1;

    wait_slot(slot, "promote", layer);

    const KvTier from = tier_[static_cast<size_t>(layer)];
    if (from == KvTier::kWarm && from_warm(layer, slot)) return slot;
    if (from == KvTier::kCold && from_cold(layer, slot)) return slot;

    // Absent, or a tier we could not read back. The engine writes this layer
    // on the step that reads it, but only for the positions of that step, so
    // the rest must not be left as whatever the previous occupant staged.
    trace_line("fill0", layer, slot, "");
    be_->fill0(slot_mem_[static_cast<size_t>(slot)], layer_bytes_);
    fence_if_on("fill0");
    tier_[static_cast<size_t>(layer)] = KvTier::kHot;
    layer_slot_[static_cast<size_t>(layer)] = slot;
    slot_layer_[static_cast<size_t>(slot)] = layer;
    lru_[static_cast<size_t>(layer)] = ++clock_;
    return slot;
}

void *KvTierCache::base(i64 layer) {
    if (!tiered_) {
        return flat_ ? static_cast<u8 *>(flat_) +
                           static_cast<size_t>(layer) * layer_bytes_
                     : nullptr;
    }
    const int slot = ensure_hot(layer);
    if (slot < 0) return nullptr;
    return slot_mem_[static_cast<size_t>(slot)];
}

void KvTierCache::end_layer(i64 layer) {
    trace_line("end", layer, 0, "");
    if (!tiered_) return;
    if (layer < 0 || layer >= n_layer_) return;
    const int slot = layer_slot_[static_cast<size_t>(layer)];
    if (slot < 0) return;
    // Recorded on the default stream, so it completes only after every kernel
    // submitted so far -- which is exactly this layer's attention.
    be_->record_marker(slot_marker_[static_cast<size_t>(slot)]);
}

void KvTierCache::stats(KvTierStats *out) const {
    if (!out) return;
    *out = st_;
    out->hot = out->warm = out->cold = 0;
    for (KvTier t : tier_) {
        if (t == KvTier::kHot) out->hot++;
        else if (t == KvTier::kWarm) out->warm++;
        else if (t == KvTier::kCold) out->cold++;
    }
}

} // namespace krk