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

// Can this directory hold a COLD spill? One page is written with fopen/fwrite
// per eviction and a failure there is a DROP, so the answer decides whether the
// disk tier exists at all rather than how fast it is. Probe with one byte and
// remove it: cheaper than discovering a typo'd path 1500 dropped pages in.
bool cold_dir_writable(const std::string &dir) {
    const std::string probe = dir + "/.krk_kv_probe";
    std::FILE *f = std::fopen(probe.c_str(), "wb");
    if (!f) return false;
    const int w = std::fputc('\n', f);
    const int c = std::fclose(f);
    std::remove(probe.c_str());
    return w != EOF && c == 0;
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

bool KvTierCache::init(Backend *be, i64 n_layer,
                       const std::vector<size_t> &layer_bytes, size_t hot_budget,
                       size_t warm_budget, const std::string &cold_dir,
                       const char *plane_tag, std::string *err) {
    detach();
    be_ = be;
    n_layer_ = n_layer > 0 ? n_layer : 0;
    const size_t n = static_cast<size_t>(n_layer_);

    // Per-layer lengths, and the prefix sums that make the flat layout pack
    // them. A layer with no KV contributes nothing: nemotron_h_moe keeps KV on
    // 6 of its 52 layers, and the old arithmetic multiplied by 52 anyway.
    len_.assign(n, 0);
    const size_t have = layer_bytes.size() < n ? layer_bytes.size() : n;
    for (size_t l = 0; l < have; l++) len_[l] = layer_bytes[l];
    off_.assign(n, 0);
    total_ = 0;
    n_kv_layers_ = 0;
    slot_bytes_ = 0;
    for (size_t l = 0; l < n; l++) {
        off_[l] = total_;
        const size_t b = len_[l];
        total_ += b;
        if (b == 0) continue;
        n_kv_layers_++;
        if (b > slot_bytes_) slot_bytes_ = b;
    }

    trace_open();
    trace_line("init", n_layer, (long long)total_, cold_dir.c_str());
    hot_budget_ = hot_budget;
    warm_budget_ = warm_budget;
    cold_dir_ = cold_dir;
    plane_tag_ = plane_tag ? plane_tag : "";

    // A cold dir that cannot be written is not a cold dir. to_cold() fails on
    // fopen/fwrite and the caller then DROPS the page, so a typo'd path would
    // turn every eviction into a lost history while the run reports a disk
    // tier. Probe it once here, and treat a failure as "no disk tier" so the
    // geometry floor below applies and the engine can say what happened.
    if (!cold_dir_.empty() && !cold_dir_writable(cold_dir_)) {
        cold_dir_.clear();
        cold_dir_rejected_ = true;
    }

    // The KV-carrying layer count, not n_layer: with the flat layout packed,
    // a layer that holds no KV is charged nothing, so it must not be counted
    // either. nemotron_h_moe reports 6 here where it used to report 52.
    st_.layers = n_kv_layers_;
    st_.layer_bytes = slot_bytes_;

    if (!be_ || n_layer_ <= 0 || total_ == 0) return true;

    // Everything fits: one flat allocation over the KV-carrying layers only,
    // byte-identical to the untiered engine for a model that keeps KV on every
    // layer -- off_[l] is then exactly l * len. No host buffer is touched, so a
    // model that already fit cannot regress here.
    if (hot_budget_ >= total_ || n_kv_layers_ <= 1) {
        flat_ = be_->alloc(total_);
        if (!flat_) {
            if (err) *err = "kv: device allocation failed";
            be_ = nullptr;
            return false;
        }
        return true;
    }

    // One slot is sized for the LARGEST layer, which is what makes the tier
    // indifferent to the widths: a layer is promoted into a slot big enough for
    // it and writes its own len_ bytes.
    n_slots_ = static_cast<int>(hot_budget_ / slot_bytes_);
    if (n_slots_ < 1) n_slots_ = 1;
    if (n_slots_ > n_kv_layers_) n_slots_ = static_cast<int>(n_kv_layers_);

    // A HOT tier that cannot hold one pass over the model migrates every
    // layer per decode step with no reuse (measured up to 206x slower than
    // fitting flat). The warning for that lives in the engine, which knows
    // the model's geometry and prints it once; here the cache only records
    // the counts.

    slot_mem_.assign(static_cast<size_t>(n_slots_), nullptr);
    slot_marker_.assign(static_cast<size_t>(n_slots_), nullptr);
    slot_layer_.assign(static_cast<size_t>(n_slots_), -1);
    layer_slot_.assign(n, -1);
    lru_.assign(n, 0);
    tier_.assign(n, KvTier::kAbsent);
    warm_mem_.assign(n, nullptr);

    for (int i = 0; i < n_slots_; i++) {
        slot_mem_[static_cast<size_t>(i)] = be_->alloc(slot_bytes_);
        slot_marker_[static_cast<size_t>(i)] = be_->make_marker();
        if (!slot_mem_[static_cast<size_t>(i)]) {
            if (err)
                *err = "kv: could not allocate a HOT slot; lower the KV VRAM "
                       "budget or the context";
            be_ = nullptr;
            return false;
        }
    }

    // WARM is counted in the same slots HOT is. A budget that covers the whole
    // plane covers every layer, so the count is the layer count and not
    // `budget / slot_bytes_` floored one division at a time: with the tiers one
    // slot short of the working set, every eviction loses a page -- the page
    // goes nowhere, the layer is zero-filled next time it is needed, and the
    // run keeps producing fluent text (see the header). Rounding the auto
    // budget down by two slots was enough to do that on a 30-layer model.
    warm_budget_slots_ = warm_budget_ >= total_
                             ? n_kv_layers_
                             : static_cast<i64>(warm_budget_ / slot_bytes_);
    if (warm_budget_slots_ < 0) warm_budget_slots_ = 0;

    // The capacity that makes an admission failure impossible, derived from the
    // GEOMETRY in the only unit the geometry has -- layers:
    //
    //   at any admission the layer being promoted is still counted in WARM,
    //   and HOT holds at least the page being evicted, so |WARM| <=
    //   n_kv_layers_ - 1 and one more admission needs n_kv_layers_.
    //
    // n_slots_ does not appear in it because of that "at least 1 in HOT": the
    // bound already accounts for the page in flight. This is what makes the
    // invariant hold for every flag combination and not just the auto one, so
    // `--kv-hot-mb 1 --kv-warm-mb 0` on a 30-layer model ends up with 2 HOT + 30
    // WARM rather than 2 + 0 and a page dropped per step.
    //
    // Only when there is nowhere to spill: with a COLD dir the budget is a real
    // budget, and a page that cannot reach WARM is on disk instead of lost.
    const i64 slots_needed = cold_dir_.empty() ? n_kv_layers_ : 0;
    n_warm_slots_ = warm_budget_slots_ > slots_needed ? warm_budget_slots_
                                                      : slots_needed;
    if (n_warm_slots_ > n_kv_layers_) n_warm_slots_ = n_kv_layers_;
    if (n_warm_slots_ < 0) n_warm_slots_ = 0;
    warm_from_geometry_ = n_warm_slots_ > warm_budget_slots_;

    tiered_ = true;
    st_.tiered = true;
    st_.slots = n_slots_;
    // hot_bytes is the resident sum, filled in by stats(): the slots are
    // allocated at slot_bytes_ each, and how much of that is live depends on
    // which layers are hot.
    st_.hot_bytes = 0;
    return true;
}

// One file per (plane, layer). The plane MUST be in the name: the engine owns
// two caches that share a --kv-cold-dir, and a name built from the layer index
// alone makes K's spill and V's spill the same file. The second writer wins,
// a restore hands one plane the other's bytes, and nothing in the counters
// says so -- a faithful-looking 1916 ->COLD / 1860 COLD->HOT with the text
// silently diverged from the flat cache. Measured; see the header.
std::string KvTierCache::cold_path(i64 layer) const {
    char buf[80];
    std::snprintf(buf, sizeof(buf), "/kv_%s_layer_%lld.bin",
                  plane_tag_.empty() ? "x" : plane_tag_.c_str(),
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
    // Counted apart from a COLD spill: this page's KV is GONE. Nothing
    // recomputes it -- ensure_hot() zero-fills the plane, and the model then
    // attends over a history that is not its own. The engine reports this
    // number and says what it means; a run with a non-zero value is wrong.
    st_.evictions_dropped++;
}

bool KvTierCache::to_warm(i64 layer, int slot) {
    trace_line("to-warm", layer, slot, "");
    const size_t len = len_[static_cast<size_t>(layer)];
    if (len == 0) return false; // this layer holds no KV to move
    i64 live = 0;
    for (KvTier t : tier_)
        if (t == KvTier::kWarm) live++;
    if (live >= n_warm_slots_) return false;

    void *host = be_->alloc_host_pageable(len);
    if (!host) return false;
    be_->download(host, slot_mem_[static_cast<size_t>(slot)], len);
    fence_if_on("download");
    warm_mem_[static_cast<size_t>(layer)] = host;
    tier_[static_cast<size_t>(layer)] = KvTier::kWarm;
    layer_slot_[static_cast<size_t>(layer)] = -1;
    slot_layer_[static_cast<size_t>(slot)] = -1;
    st_.evictions_to_warm++;
    st_.migrate_bytes += static_cast<i64>(len);
    return true;
}

bool KvTierCache::to_cold(i64 layer, int slot) {
    trace_line("to-cold", layer, slot, "");
    if (cold_dir_.empty()) return false;
    const size_t len = len_[static_cast<size_t>(layer)];
    if (len == 0) return false;
    void *host = be_->alloc_host_pageable(len);
    if (!host) return false;
    be_->download(host, slot_mem_[static_cast<size_t>(slot)], len);
    fence_if_on("download");
    std::FILE *f = std::fopen(cold_path(layer).c_str(), "wb");
    if (!f) {
        be_->release_host_pageable(host);
        return false;
    }
    const size_t w = std::fwrite(host, 1, len, f);
    std::fclose(f);
    be_->release_host_pageable(host);
    if (w != len) return false;
    tier_[static_cast<size_t>(layer)] = KvTier::kCold;
    layer_slot_[static_cast<size_t>(layer)] = -1;
    slot_layer_[static_cast<size_t>(slot)] = -1;
    st_.evictions_to_cold++;
    st_.migrate_bytes += static_cast<i64>(len);
    return true;
}

bool KvTierCache::from_warm(i64 layer, int slot) {
    trace_line("from-warm", layer, slot, "");
    void *host = warm_mem_[static_cast<size_t>(layer)];
    if (!host) return false;
    const size_t len = len_[static_cast<size_t>(layer)];
    if (len == 0) return false;
    be_->upload(slot_mem_[static_cast<size_t>(slot)], host, len);
    fence_if_on("upload");
    be_->release_host_pageable(host);
    warm_mem_[static_cast<size_t>(layer)] = nullptr;
    tier_[static_cast<size_t>(layer)] = KvTier::kHot;
    layer_slot_[static_cast<size_t>(layer)] = slot;
    slot_layer_[static_cast<size_t>(slot)] = layer;
    lru_[static_cast<size_t>(layer)] = ++clock_;
    st_.promotions_from_warm++;
    st_.migrate_bytes += static_cast<i64>(len);
    return true;
}

bool KvTierCache::from_cold(i64 layer, int slot) {
    trace_line("from-cold", layer, slot, "");
    if (cold_dir_.empty()) return false;
    std::FILE *f = std::fopen(cold_path(layer).c_str(), "rb");
    if (!f) return false;
    const size_t len = len_[static_cast<size_t>(layer)];
    if (len == 0) {
        std::fclose(f);
        return false;
    }
    void *host = be_->alloc_host_pageable(len);
    if (!host) {
        std::fclose(f);
        return false;
    }
    const size_t r = std::fread(host, 1, len, f);
    std::fclose(f);
    if (r != len) {
        be_->release_host_pageable(host);
        return false;
    }
    be_->upload(slot_mem_[static_cast<size_t>(slot)], host, len);
    fence_if_on("upload");
    be_->release_host_pageable(host);
    tier_[static_cast<size_t>(layer)] = KvTier::kHot;
    layer_slot_[static_cast<size_t>(layer)] = slot;
    slot_layer_[static_cast<size_t>(slot)] = layer;
    lru_[static_cast<size_t>(layer)] = ++clock_;
    st_.promotions_from_cold++;
    st_.migrate_bytes += static_cast<i64>(len);
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
    // A layer with no KV has no plane to make resident. base() answers nullptr
    // for it; this is the belt to that braces.
    if (len_[static_cast<size_t>(layer)] == 0) return -1;

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
    be_->fill0(slot_mem_[static_cast<size_t>(slot)],
               len_[static_cast<size_t>(layer)]);
    fence_if_on("fill0");
    tier_[static_cast<size_t>(layer)] = KvTier::kHot;
    layer_slot_[static_cast<size_t>(layer)] = slot;
    slot_layer_[static_cast<size_t>(slot)] = layer;
    lru_[static_cast<size_t>(layer)] = ++clock_;
    return slot;
}

void *KvTierCache::base(i64 layer) {
    if (layer < 0 || layer >= n_layer_) return nullptr;
    // No KV on this layer: no plane in the flat allocation (the layout packs
    // the KV-carrying layers) and no slot to make resident. A caller walking
    // every layer has to skip it rather than treat this as an error.
    if (len_[static_cast<size_t>(layer)] == 0) return nullptr;
    if (!tiered_) {
        return flat_ ? static_cast<u8 *>(flat_) +
                           off_[static_cast<size_t>(layer)]
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
    for (size_t l = 0; l < tier_.size(); l++) {
        const KvTier t = tier_[l];
        if (t == KvTier::kHot) out->hot++;
        else if (t == KvTier::kWarm) out->warm++;
        else if (t == KvTier::kCold) out->cold++;
    }
    // hot_bytes is the DEVICE MEMORY this tier holds, i.e. the slots, not the
    // sum of the layers that happen to be hot at the instant of the call: a
    // report taken after a run would otherwise print "2 HOT slots of 0 MiB",
    // because the last thing every decode step does is evict the previous
    // layer. Which layers are hot is the three counts above.
    out->hot_bytes = static_cast<size_t>(n_slots_) * slot_bytes_;
}

} // namespace krk