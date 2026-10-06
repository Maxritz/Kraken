// ============================================================================
//  hadamard.hpp — the prism.hadamard activation transform (Backend::hadamard_act).
//
//  PrismML's llama.cpp fork stores a handful of matmul weights in a rotated
//  basis: each weight's INPUT dimension was multiplied by a per-width +-1
//  sign vector and then by a normalized Sylvester Walsh-Hadamard over
//  consecutive blocks of `block` elements (entries +-1/sqrt(block), so the
//  transform is its own inverse). The activation has to be transformed the
//  same way immediately before such a matmul — and, for the one table stored
//  rotated (`token_embd.weight`), once after the lookup, with the sign stage
//  on the other side of the FWHT.
//
//  Two kernels carry the whole op:
//
//    hadamard_gather  one thread per element. Stage 1 materializes the row
//                     into the destination applying, in order, the ssm_out
//                     grouped-V perm (when the weight says so) and the sign
//                     vector (when the fold wants signs first). Stage 3 is
//                     the same kernel in place with only the signs, for the
//                     inverse-after-lookup order. With no perm and no signs
//                     it degenerates to a copy, which is what makes
//                     out != in work in every case.
//
//    hadamard_fwht    one workgroup per (row, block). The block is loaded
//                     into LDS as f32, butterfly-staged (len = 1, 2, 4, ...
//                     with pairs (i, i+len) summed/differenced), then scaled
//                     by 1/sqrt(block) and narrowed to f16 once at the
//                     store. Widening first and narrowing once is what keeps
//                     the device within one rounding of the CPU reference,
//                     whose transform is pure f32.
//
//  Dtype contract: activation buffers are act_type() (f16 here); `signs` is
//  an f32 device vector of n values of +-1 uploaded by the model loader, or
//  null when the file declares sign_mode=identity. Rows never interact, so
//  out may alias in.
// ============================================================================
#ifndef KRK_KERNEL_HADAMARD_HPP
#define KRK_KERNEL_HADAMARD_HPP

#include <algorithm>

#include "../krk_hip.hpp"

namespace krk {

// ---------------------------------------------------------------------------
// Stage 1 and stage 3: the optional grouped-V perm and/or the sign multiply.
//   out[r, oi] = in[r, perm(oi)] * signs[oi]
// The inverse mapping of the perm is computed rather than tabulated:
//   oi = i + j*hd + k*hd*rep   =>   i = oi % hd, t = oi / hd,
//   j = t % rep, k = t / rep   =>   in index = i + k*hd + j*hd*nk
// ---------------------------------------------------------------------------

__global__ void hadamard_gather_kernel(_Float16 *__restrict__ out,
                                       const _Float16 *__restrict__ in,
                                       i64 n_elem, i64 width,
                                       const f32 *__restrict__ signs, i64 hd,
                                       i64 nk, i64 rep) {
    const i64 i = blockIdx.x * static_cast<i64>(blockDim.x) + threadIdx.x;
    if (i >= n_elem) return;
    i64 src = i;
    if (hd > 0) {
        const i64 n_row = hd * nk * rep;
        const i64 row = i / n_row;
        const i64 oi = i - row * n_row;
        const i64 ii = oi % hd;
        const i64 t = oi / hd;
        const i64 j = t % rep;
        const i64 k = t / rep;
        src = row * n_row + ii + k * hd + j * hd * nk;
    }
    f32 v = static_cast<f32>(in[src]);
    if (signs) v *= signs[i % width];
    out[i] = static_cast<_Float16>(v);
}

// ---------------------------------------------------------------------------
// The normalized Sylvester FWHT, one workgroup per (row, block).
// grid.x = rows * (width / block); threads = min(block, 256).
// ---------------------------------------------------------------------------

__global__ void hadamard_fwht_kernel(_Float16 *__restrict__ x, i64 n_elem,
                                     i64 block) {
    extern __shared__ f32 sm[];
    const i64 cell = blockIdx.x;
    const i64 off = cell * block;
    if (off + block > n_elem) return;
    const int t = static_cast<int>(threadIdx.x);
    const int bp = static_cast<int>(blockDim.x);
    const f32 scale = 1.0f / sqrtf(static_cast<f32>(block));

    for (i64 e = t; e < block; e += bp) sm[e] = static_cast<f32>(x[off + e]);
    __syncthreads();

    // Each stage halves the pair stride; the pairs of one stage are disjoint,
    // so the only barrier needed is between stages.
    for (i64 len = 1; len < block; len <<= 1) {
        const i64 npairs = block / (2 * len);
        for (i64 g = t; g < npairs; g += bp) {
            const i64 base = 2 * g * len;
            for (i64 j = 0; j < len; j++) {
                const f32 a = sm[base + j];
                const f32 b = sm[base + len + j];
                sm[base + j] = a + b;
                sm[base + len + j] = a - b;
            }
        }
        __syncthreads();
    }

    for (i64 e = t; e < block; e += bp)
        x[off + e] = static_cast<_Float16>(sm[e] * scale);
}

} // namespace krk

#endif // KRK_KERNEL_HADAMARD_HPP
