// moe_table.hpp — the residency table the expert cache PUBLISHES to the device.
//
// Why this exists. The decode step's MoE routing ends with a host round trip:
// the router's probabilities come back, the host scans them for the top-k, and
// the cache resolves each selected expert to a device pointer so the per-expert
// loop can issue its three GEMMs. The scan is cheap; the *dependency* is not.
// Measured on Mellum2.1 (28 layers, 23.7 ms/step): `moe.route.fetch` is
// 0.523 ms/layer of which only 0.077 is a ready copy -- the rest is the layer
// waiting for its own GPU work to retire before the host may look at it.
//
// A device pointer table is what removes the dependency instead of shortening
// it: with `table[layer * n_expert + e]` on the device, a kernel can pick the
// matrices itself from the ids the router produced, so no per-expert host
// pointer and no `acquire()` sits between the routing and the GEMMs.
//
// The table is only valid while residency is STABLE. A promotion or an
// eviction rewrites a slot's device pointers, so the cache marks the table
// stale on both; the engine publishes it once (after the warm-up) and uses it
// only when the whole routed set is device-resident, where no such event can
// happen mid-run. A cache that overflows its budget simply never publishes.
//
// This header is shared with the HIP translation unit, so it holds plain POD
// types only: no HIP headers and no std:: members, and every field is a fixed
// width so host and device agree on the layout.
#pragma once

#include <cstdint>

namespace krk {

// Per-expert matrices in the order the dispatch kernel consumes them.
enum : int {
    kMoeGate = 0,
    kMoeUp = 1,
    kMoeDown = 2,
};

// One (layer, expert) bank as the device dispatch sees it. Everything a dot
// product needs is here, read as a value: a kernel that had to ask the host
// for `n_in` or the row stride would have a host dependency of its own, which
// is the whole thing this removes.
struct MoeDispatchEntry {
    void *p[3] = {nullptr, nullptr, nullptr};  // device weight pointers
    std::int64_t n_out[3] = {0, 0, 0};         // rows of each matrix
    std::int64_t n_in[3] = {0, 0, 0};          // columns (chunked by 32)
    std::int64_t row_bytes[3] = {0, 0, 0};     // bytes per quantized row
    std::int32_t type[3] = {0, 0, 0};          // DType as an int
    std::int32_t present[3] = {0, 0, 0};       // 0 => matrix absent (gate-less)
    std::int32_t _pad = 0;                     // keeps the struct 8-byte multiple
};

}  // namespace krk
