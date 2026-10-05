// kv_tier.hpp — HOT / WARM / COLD residency for the KV cache
//
// Unit of management is ONE LAYER'S KV for the whole context. The design
// note (docs/ARCHITECTURE-TIERED.md section 6) asks for (layer,
// position-block) pages; layer granularity is what this implements, and the
// reason is arithmetic rather than convenience:
//
//   attention for layer l is the only reader of layer l's KV, and layer l is
//   finished before layer l+1 starts. So a layer is a natural residency unit
//   with no kernel change at all -- the engine passes a per-layer base and
//   layer_stride = 0, and every existing kv_append/attention/fused-chain
//   call site keeps working unchanged. Position-block pages would require a
//   page-table indirection inside the attention kernels.
//
//   On this hardware the coarser unit is not a real loss. A layer's KV for
//   Qwen3-8B at a 32k context is 4.6 GiB / 36 = 128 MiB, so a 15.9 GiB card
//   can still hold more than a hundred layers' worth resident. It would only
//   start to matter if a single layer's KV exceeded the whole device, which
//   needs a context far longer than the model trains for.
//
// When everything fits, the cache is a single flat allocation and every code
// path is byte-identical to the untiered engine. Tiering only engages once
// the KV genuinely does not fit, so this cannot regress a model that fits.
#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "krk/backend.hpp"

namespace krk {

enum class KvTier : u8 { kAbsent = 0, kHot = 1, kWarm = 2, kCold = 3 };

const char *kv_tier_name(KvTier t);

struct KvTierStats {
    i64 layers = 0;
    i64 hot = 0;
    i64 warm = 0;
    i64 cold = 0;
    i64 slots = 0;
    i64 promotions_from_warm = 0;
    i64 promotions_from_cold = 0;
    i64 evictions_to_warm = 0;
    i64 evictions_to_cold = 0;
    i64 migrate_bytes = 0;
    size_t hot_bytes = 0;
    size_t layer_bytes = 0;
    bool tiered = false;
};

// One plane (K or V) of the KV cache. The engine owns two of these.
class KvTierCache {
public:
    ~KvTierCache();

    // hot_budget / warm_budget are byte budgets for the device and host
    // planes. cold_dir empty disables the disk tier: WARM eviction then keeps
    // the page in host RAM and lets the oldest be overwritten rather than
    // writing anything, which is the right default because a KV page we
    // evicted is cheaper to recompute than to rewrite.
    bool init(Backend *be, i64 n_layer, size_t layer_bytes, size_t hot_budget,
              size_t warm_budget, const std::string &cold_dir,
              std::string *err);

    // Device pointer holding this layer's KV, promoted into HOT if needed.
    //
    // Blocks until the slot's PREVIOUS occupant has finished on the device --
    // that wait is the whole reason this class cannot corrupt the GPU, and it
    // is a per-slot event wait rather than a device sync (see
    // Backend::make_marker). The caller must then call end_layer() once the
    // work using this layer has been submitted.
    void *base(i64 layer);

    // Mark the point in the stream after the kernels using `layer` have been
    // submitted. MUST be called after the attention/kv_append launches for
    // that layer, or the next occupant of the slot is not actually ordered
    // behind this one.
    void end_layer(i64 layer);

    // Flat base for the untiered case: callers pass this as the cache pointer
    // and keep layer_stride = kv_cap * kv_dim, exactly as before.
    void *flat() const { return flat_; }

    // Release every buffer NOW, while the backend is known alive, and forget
    // the backend so the destructor cannot touch it again.
    //
    // This exists because of member destruction order. Engine::shutdown()
    // releases its buffers and then the backend is deleted, but a member's
    // destructor runs AFTER ~Engine's body -- so a cache that only frees in
    // its destructor calls release() on a dangling pointer. The engine calls
    // this from shutdown() instead, and it is idempotent.
    void detach();

    void stats(KvTierStats *out) const;

private:
    int ensure_hot(i64 layer);
    // Bounded, traced wait for a slot's previous occupant. Exits 42 on
    // timeout instead of hanging into a driver watchdog.
    bool wait_slot(int slot, const char *why, i64 layer);
    void fence_if_on(const char *op);
    int take_slot(i64 keep); // evict LRU if every slot is resident
    void evict_slot(int slot, i64 keep);
    bool to_warm(i64 layer, int slot);
    bool to_cold(i64 layer, int slot);
    bool from_warm(i64 layer, int slot);
    bool from_cold(i64 layer, int slot);
    std::string cold_path(i64 layer) const;

    Backend *be_ = nullptr;
    i64 n_layer_ = 0;
    size_t layer_bytes_ = 0;
    size_t hot_budget_ = 0;
    size_t warm_budget_ = 0;
    std::string cold_dir_;

    bool tiered_ = false;
    void *flat_ = nullptr;

    // HOT plane. slot_marker_[i] is the completion marker for the work that
    // most recently used slot i; it is waited on before the slot is handed to
    // anyone else.
    int n_slots_ = 0;
    std::vector<void *> slot_marker_;
    std::vector<void *> slot_mem_;    // device, n_slots_ entries
    std::vector<int> layer_slot_;     // layer -> slot, or -1
    std::vector<int> slot_layer_;     // slot -> layer, or -1 (the inverse)
    std::vector<i64> lru_;            // layer -> last-touch stamp
    i64 clock_ = 0;

    // WARM plane: one host buffer per layer, allocated lazily
    std::vector<void *> warm_mem_;
    std::vector<KvTier> tier_;
    i64 n_warm_slots_ = 0;

    KvTierStats st_;
};

} // namespace krk