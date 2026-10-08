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
    // Spilled to a COLD file: the page is on disk and comes back on the next
    // promotion. evictions_dropped is the OTHER case and is not the same
    // event: the page went nowhere, its KV is gone, and the layer is
    // zero-filled when it is next needed. They used to share a counter, so a
    // run that lost a page every step reported the loss as a working disk
    // tier -- and kept printing fluent text.
    i64 evictions_to_cold = 0;
    i64 evictions_dropped = 0;
    i64 migrate_bytes = 0;
    size_t hot_bytes = 0;
    size_t layer_bytes = 0; // the largest layer's plane (a HOT slot's size)
    bool tiered = false;
};

// One plane (K or V) of the KV cache. The engine owns two of these.
class KvTierCache {
public:
    ~KvTierCache();

    // hot_budget / warm_budget are byte budgets for the device and host
    // planes. cold_dir empty disables the disk tier, and then WARM is the LAST
    // home a page has: an eviction that cannot reach WARM is DROPPED -- the
    // KV is lost and the layer is zero-filled next time it is needed. Six
    // lines of comment here used to say the oldest page was "overwritten"
    // instead, and that pages are "cheaper to recompute than to rewrite";
    // nothing recomputes them. MEASURED, SmolLM2-135M --ctx 512 -n 32: with
    // HOT+WARM one slot short of the 30 KV-carrying layers the run drops a
    // page per step and its text diverges from the flat cache at the first
    // dropped page; with WARM covering every layer it is byte-identical
    // (md5 2456dfa5ba9e both arms).
    //
    // A warm_budget >= the plane total therefore means "cover every layer"
    // rather than being floored into slots one at a time, because covering the
    // cache is a correctness requirement here and not an optimisation. WARM
    // buffers are allocated lazily, so this costs the count and nothing else.
    //
    // THE CAPACITY IS DERIVED FROM THE GEOMETRY, NOT FROM THE BUDGET. With no
    // cold dir, init pins the WARM capacity at n_kv_layers_ -- the count
    // `layer_bytes` implies -- whatever warm_budget asks for, so HOT + WARM >=
    // the KV-carrying layers holds for every flag combination and no
    // configuration can be short of the working set. The count is sufficient
    // and not merely plausible: at any admission the layer being promoted is
    // still counted in WARM and HOT holds at least the page being evicted, so
    // |WARM| <= n_kv_layers_ - 1 and one more admission needs n_kv_layers_.
    // warm_from_geometry() says whether the budget was overridden, and the
    // engine reports the asked and applied values when it was.
    //
    // The budget still decides when a COLD dir is set: a page that cannot reach
    // WARM has a home on disk there, so a small WARM is a working choice rather
    // than a lost history. A cold dir that cannot be WRITTEN is treated as
    // absent (and reported), because a spill that fails on fopen/fwrite is a
    // drop, and the geometry floor is what keeps that from being silent.
    //
    // layer_bytes is per LAYER and 0 means "this layer holds no KV at all":
    // nemotron_h_moe keeps it on 6 of its 52 layers, and a mamba or pure-MoE
    // block has nothing to append. The flat layout PACKS the non-zero entries,
    // so the bytes allocated and charged are the KV the model actually has, and
    // the tier sizes a slot by the largest layer rather than assuming every
    // layer is one width. One number for every layer is what made this model's
    // charge 8.67x the KV that can exist (208 MiB reserved and reported at
    // --ctx 4096 for 24 MiB); the packing is also what removes the layout's
    // dependence on every layer being the same size.
    //
    // plane_tag namespaces the COLD spill files. Two caches share one
    // --kv-cold-dir, and when both built the path from the layer index alone
    // they wrote ONE SET OF FILES for two planes: V's spill overwrote K's, a
    // restore handed back the other plane's bytes, and the run diverged from
    // the flat cache while every counter said the disk tier was working.
    // MEASURED, SmolLM2-135M --ctx 512 -n 32 --kv-hot-mb 1 --kv-warm-mb 0
    // --kv-cold-dir: 1916 ->COLD, 1860 COLD->HOT, 0 dropped, md5 a2e087cbeada
    // against the flat cache's 2456dfa5ba9e. Use "k" and "v".
    bool init(Backend *be, i64 n_layer, const std::vector<size_t> &layer_bytes,
              size_t hot_budget, size_t warm_budget, const std::string &cold_dir,
              const char *plane_tag, std::string *err);

    // Device pointer holding this layer's KV, promoted into HOT if needed.
    // nullptr for a layer that holds no KV (len_ == 0) or a bad index, so a
    // caller walking every layer -- kv_rollback_zero does -- has to skip rather
    // than assume a plane exists.
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

    // Geometry, so the engine can report it once instead of each plane
    // printing its own identical warning.
    i64 hot_slots() const { return n_slots_; }
    i64 warm_slots() const { return n_warm_slots_; }
    i64 kv_layers() const { return n_kv_layers_; }
    size_t plane_bytes() const { return total_; }
    size_t slot_bytes() const { return slot_bytes_; }
    bool tiered() const { return tiered_; }
    // True when the WARM capacity was set by the geometry floor rather than by
    // warm_budget -- i.e. the caller asked for a tier that cannot hold one pass
    // and got one that can.
    bool warm_from_geometry() const { return warm_from_geometry_; }
    // What the budget alone would have allowed, in slots (0 when n/a).
    i64 warm_slots_from_budget() const { return warm_budget_slots_; }
    // True when --kv-cold-dir was set and is not writable, so it was ignored.
    bool cold_dir_rejected() const { return cold_dir_rejected_; }
    // Public so callers can verify the per-plane file naming without running a
    // full eviction: the test asserts K and V cold paths differ and neither is
    // the bare layer-index name (the defect that made two planes share one file).
    std::string cold_path(i64 layer) const;

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

    Backend *be_ = nullptr;
    i64 n_layer_ = 0;
    std::vector<size_t> len_;  // per layer: bytes of one plane, 0 = no KV
    std::vector<size_t> off_;  // per layer: byte offset into the flat plane
    size_t total_ = 0;         // sum of len_: what the flat layout holds
    size_t slot_bytes_ = 0;    // largest len_: what one HOT slot holds
    i64 n_kv_layers_ = 0;      // layers with len_ > 0
    size_t hot_budget_ = 0;
    size_t warm_budget_ = 0;
    std::string cold_dir_;
    std::string plane_tag_; // "k"/"v": kept out of the other plane's spills

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

    // WARM plane: one host buffer per layer, allocated lazily.
    // n_warm_slots_ caps how many layers may be resident in WARM at once. A
    // budget that covers the plane sets it to n_kv_layers_, i.e. no cap in
    // practice: a layer is only in WARM when it is not in HOT.
    std::vector<void *> warm_mem_;
    std::vector<KvTier> tier_;
    i64 n_warm_slots_ = 0;
    i64 warm_budget_slots_ = 0;       // what warm_budget_ alone allowed
    bool warm_from_geometry_ = false; // the geometry floor raised it
    bool cold_dir_rejected_ = false;  // --kv-cold-dir is not writable

    KvTierStats st_;
};

} // namespace krk