// Register, scratch and LDS usage of the prefill attention instantiations.
//
// Compiled by hand with -Rpass-analysis=kernel-resource-usage, not by the
// build: the question it answers is whether the staged K/V pair or the
// accumulators are what hold the occupancy of `attention_bexact_kernel` down
// per CU, which is the difference between a tile-depth change being tuning or
// the fix. Nothing here runs.
//
//   hipcc -c --offload-arch=gfx1201 -O3 -Iinclude -Isrc \
//     -Rpass-analysis=kernel-resource-usage tools/attn_res.hip -o /dev/null
#include "hip/kernels/attention.hpp"

using namespace krk;

// One per (VPT, QT) the dispatch can reach, in hd order: the lane slice is
// hd/32, so hd 64 is VPT 2, hd 128 is VPT 4, hd 256 is VPT 8. The tile depth
// is a runtime argument, so its effect on LDS is what the audit computes from
// `attn_bexact_smem_bytes` -- what the pass reports here is the VGPR side,
// which no tile depth can change.
void *krk_attn_resource_instances[] = {
    reinterpret_cast<void *>(&attention_bexact_kernel<2, 4>),   // hd 64  QT 4
    reinterpret_cast<void *>(&attention_bexact_kernel<2, 8>),   // hd 64  QT 8
    reinterpret_cast<void *>(&attention_bexact_kernel<2, 16>),  // hd 64  QT 16
    reinterpret_cast<void *>(&attention_bexact_kernel<3, 8>),   // hd 96  QT 8
    reinterpret_cast<void *>(&attention_bexact_kernel<4, 4>),   // hd 128 QT 4
    reinterpret_cast<void *>(&attention_bexact_kernel<4, 8>),   // hd 128 QT 8
    reinterpret_cast<void *>(&attention_bexact_kernel<4, 16>),  // hd 128 QT 16
    reinterpret_cast<void *>(&attention_bexact_kernel<8, 4>),   // hd 256 QT 4
    reinterpret_cast<void *>(&attention_bexact_kernel<8, 8>),   // hd 256 QT 8
    reinterpret_cast<void *>(&attention_wmma_kernel),           // hd 256 matrix cores
    reinterpret_cast<void *>(&attention_decode_kernel<4>),      // decode row, same chain
};
