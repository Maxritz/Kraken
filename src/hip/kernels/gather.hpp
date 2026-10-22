// ============================================================================
//  gather.hpp — row gather out of the KV cache (QSA's cell compaction).
//
//  attention() takes no mask, and the QSA indexer gives every QUERY ROW its own
//  cell list. The way to express that without a second attention kernel is to
//  compact the row's chosen cells and run the ordinary causal kernel over the
//  compact buffer. This kernel is the compaction: a pure byte move, so it does
//  not care whether the cache is f16 or f32, and it is 16-byte vectorised
//  because a KV row (512 f16 on the REAP-288 file) divides evenly.
// ============================================================================
#ifndef KRK_KERNEL_GATHER_HPP
#define KRK_KERNEL_GATHER_HPP

#include "../krk_hip.hpp"

namespace krk {

// dst[r * row_bytes + b] = src[idx[r] * row_bytes + b]
// One block per row; the block's threads walk the row in 16-byte steps and the
// remainder byte by byte. A negative index writes nothing (the caller pads its
// list and does not care about the cell it lands on).
template <int THREADS>
__global__ void __launch_bounds__(THREADS)
    gather_rows_kernel(u8 *__restrict__ dst, const u8 *__restrict__ src,
                       const i64 *__restrict__ idx, i64 row_bytes) {
    const i64 r = static_cast<i64>(blockIdx.x);
    const i64 cell = idx[r];
    if (cell < 0) return;
    u8 *o = dst + r * row_bytes;
    const u8 *s = src + cell * row_bytes;
    const i64 n16 = row_bytes & ~static_cast<i64>(15);
    for (i64 b = static_cast<i64>(threadIdx.x) * 16; b < n16;
         b += static_cast<i64>(THREADS) * 16) {
        const uint4 v = *reinterpret_cast<const uint4 *>(s + b);
        *reinterpret_cast<uint4 *>(o + b) = v;
    }
    for (i64 b = n16 + threadIdx.x; b < row_bytes; b += THREADS) o[b] = s[b];
}

} // namespace krk

#endif // KRK_KERNEL_GATHER_HPP
