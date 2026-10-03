// ============================================================================
//  topk.hpp — device-side top-K of a logits row.
//
//  WHY THIS EXISTS. The decode loop used to end every token with
//  be_->download_f32(host, ws_logits_, n_vocab): a whole vocab row (248320
//  f32 = 993 kB for Qwen3.5) across PCIe, then a host partial_sort over
//  248320 elements. KRK_TIME's idle column priced that at 3.4 ms per token
//  on the 8B and 2.7 ms of it was pure *device* idle — the GPU had finished
//  and sat waiting for the host to stop thinking. The sampler only ever uses
//  the top min(top_k, ...) candidates, so the row never has to leave the
//  device.
//
//  ALGORITHM — radix select on a sortable key, four launches, no sort of the
//  vocabulary. The key packs (value, id):
//
//      key = (u64)sortable_f32(v) << 32 | ~id
//
//  The id is complemented on the way in and uncomplemented on the way out, so
//  that a *larger* key means a *smaller* id: equal values rank the lower id
//  first, which is what the host argmax (`if (v > best)`) does. Ids are
//  unique, so keys are distinct and "top K" is a total order either way --
//  but the convention has to be right, because it is not a cosmetic choice.
//  Gemma's final logit softcap clips a whole shelf of logits to exactly
//  softcap, so its top row is a *massive* tie; ranking ties by the higher id
//  would pick a different token than every reference implementation.
//
//    1. histogram  512 bins on the top 9 key bits, one atomicAdd per element,
//                  plus a block max reduced into a global key
//    2. scan       one block walks bins from the top until the running count
//                  reaches k; that bin is the cut
//    3. collect    everything at or above the cut into a candidate buffer,
//                  and sum exp((v - max)/temp) for the exact softmax mass
//    4. final      k rounds of block-wide max over the candidates with a
//                  strictly decreasing cut — no sort, no LDS for the set
//
//  Steps 1 and 3 each stream the row once (993 kB); step 4 touches only the
//  candidates. Bins span the whole f32 range, so the cut bin is narrow where
//  the candidates actually are. The candidate buffer is sized for a
//  degenerate vocab (a huge tie exactly at the cut) and the collected count
//  comes back, so an overflow is detectable rather than silent.
//
//  The repetition penalty is applied *here*, before the ranking, because it
//  can raise a token: applying it to the returned candidates instead can miss
//  one that should have been in the set. It arrives as a bitmask of token ids
//  (the sampler keeps a 64-token ring; one bit test per element).
// ============================================================================
#ifndef KRK_KERNEL_TOPK_HPP
#define KRK_KERNEL_TOPK_HPP

#include <cfloat>

#include "../krk_hip.hpp"

namespace krk {

// f32 -> u32 with the same ordering as the floats.
__device__ __forceinline__ u32 f2key(f32 v) {
    const u32 b = __float_as_uint(v);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

__device__ __forceinline__ unsigned long long pack_kv(f32 v, i32 id) {
    return (static_cast<unsigned long long>(f2key(v)) << 32) |
           static_cast<unsigned long long>(~static_cast<u32>(id));
}

__device__ __forceinline__ f32 key2f(unsigned long long k) {
    const u32 b = static_cast<u32>(k >> 32);
    return __uint_as_float((b & 0x80000000u) ? (b & 0x7fffffffu) : ~b);
}

__device__ __forceinline__ i32 key2id(unsigned long long k) {
    return static_cast<i32>(~static_cast<u32>(k));
}

// softcap then repetition penalty — the order fetch_logits / Sampler use.
// The penalty direction is Sampler's, not a guess: a penalty is >= 1, it
// *divides* a positive logit and *multiplies* a negative one (llama.cpp's
// rule). Getting this backwards would raise the very tokens the penalty is
// meant to suppress, and would silently disagree with the host path.
__device__ __forceinline__ f32 logits_transform(f32 v, i32 i,
                                                const u32 *__restrict__ pmask,
                                                f32 rep_pen, f32 softcap) {
    if (softcap > 0.0f)
        v = v > softcap ? softcap : (v < -softcap ? -softcap : v);
    if (pmask && rep_pen > 0.0f && rep_pen != 1.0f &&
        ((pmask[static_cast<u32>(i) >> 5] >> (static_cast<u32>(i) & 31)) & 1u))
        v = v > 0.0f ? v / rep_pen : v * rep_pen;
    return v;
}

// --- 1. histogram -----------------------------------------------------------
// Each block owns a contiguous span [lo, hi) so the row streams coalesced.
//
// T is the *activation* type, not f32. The caller hands us the head's output
// buffer, which is f16 on the fast path, and reading it as f32 would pair
// every logit with its neighbour's exponent bits: no NaNs, no crash, a
// plausible-looking ranking that is simply wrong. Templating the load costs
// nothing and makes that mistake impossible.
template <typename T, int THREADS>
__global__ void __launch_bounds__(THREADS)
    topk_hist_kernel(const T *__restrict__ logits, const u32 *__restrict__ pmask,
                     f32 rep_pen, f32 softcap, u32 *__restrict__ hist,
                     unsigned long long *__restrict__ gmax, i32 n, i32 span) {
    __shared__ unsigned long long smax[THREADS / 32];
    const i32 lo = static_cast<i32>(blockIdx.x) * span;
    const i32 hi = min(n, lo + span);
    unsigned long long best = 0;
    for (i32 i = lo + static_cast<i32>(threadIdx.x); i < hi; i += THREADS) {
        const unsigned long long k = pack_kv(
            logits_transform(static_cast<f32>(logits[i]), i, pmask, rep_pen, softcap),
            i);
        if (k > best) best = k;
        atomicAdd(&hist[static_cast<u32>(k >> 55)], 1u);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const unsigned long long o = __shfl_xor(best, off);
        if (o > best) best = o;
    }
    const int w = static_cast<int>(threadIdx.x) >> 5;
    if ((static_cast<int>(threadIdx.x) & 31) == 0) smax[w] = best;
    __syncthreads();
    if (w == 0) {
        unsigned long long m = smax[0];
#pragma unroll
        for (int i = 1; i < THREADS / 32; i++)
            if (smax[i] > m) m = smax[i];
        if (threadIdx.x == 0) atomicMax(gmax, m);
    }
}

// --- 1b. the k == 1 shortcut ------------------------------------------------
// An argmax does not need a radix select, and this is the measurement that
// says so: the four-stage path took 262 ms over 64 decode tokens where the
// 993 kB download plus the host argmax loop it replaced took 258 ms. Three
// things cost more than the PCIe transfer they saved -- four launches, four
// memsets, and one __expf per vocab entry for a mass the caller passed null
// for. The packed key is *already* a total order (value, then lower id), so
// the answer is a single block-wide max and nothing else: no histogram, no
// candidate buffer, and no way to overflow one.
//
// The mass is deliberately not computed here. It is defined against the
// *global* max, which is only known once every block has retired, so asking
// for it would cost exactly the second pass this shortcut exists to avoid.
// A caller that wants both k == 1 and a mass gets the four-stage path.
template <typename T, int THREADS>
__global__ void __launch_bounds__(THREADS)
    topk_argmax_kernel(const T *__restrict__ logits, const u32 *__restrict__ pmask,
                       f32 rep_pen, f32 softcap, i32 n, i32 span,
                       unsigned long long *__restrict__ gmax) {
    __shared__ unsigned long long smax[THREADS / 32];
    const i32 lo = static_cast<i32>(blockIdx.x) * span;
    const i32 hi = min(n, lo + span);
    unsigned long long best = 0;
    for (i32 i = lo + static_cast<i32>(threadIdx.x); i < hi; i += THREADS) {
        const unsigned long long kk = pack_kv(
            logits_transform(static_cast<f32>(logits[i]), i, pmask, rep_pen, softcap),
            i);
        if (kk > best) best = kk;
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const unsigned long long o = __shfl_xor(best, off);
        if (o > best) best = o;
    }
    const int w = static_cast<int>(threadIdx.x) >> 5;
    if ((static_cast<int>(threadIdx.x) & 31) == 0) smax[w] = best;
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned long long m = smax[0];
#pragma unroll
        for (int i = 1; i < THREADS / 32; i++)
            if (smax[i] > m) m = smax[i];
        atomicMax(gmax, m);
    }
}

// Unpacks the winner and stamps the count. One thread: the global max is a
// single u64 and the work is two shifts. It exists so the caller's ids/vals
// and count_out are written by the op itself, as the contract says, instead
// of by the host reading a different buffer.
__global__ void topk_unpack_kernel(const unsigned long long *__restrict__ gmax,
                                   unsigned long long *__restrict__ count_out,
                                   i32 *__restrict__ ids, f32 *__restrict__ vals) {
    if (threadIdx.x != 0) return;
    const unsigned long long m = *gmax;
    *ids = key2id(m);
    *vals = key2f(m);
    // Exact by construction: a single winner cannot be truncated.
    *count_out = 1ull;
}

// --- 2. scan ----------------------------------------------------------------
// One block. Walks bins from the top until the running count reaches k.
__global__ void topk_scan_kernel(const u32 *__restrict__ hist, i32 k,
                                 u32 *__restrict__ cut_out) {
    if (threadIdx.x != 0) return;
    u32 run = 0;
    for (int b = 511; b >= 0; b--) {
        run += hist[b];
        if (run >= static_cast<u32>(k)) {
            *cut_out = static_cast<u32>(b);
            return;
        }
    }
    *cut_out = 0;
}

// --- 3. collect -------------------------------------------------------------
// Reads the cut and the max back from device memory rather than taking them as
// arguments: that keeps the whole chain async. Nothing between stage 1 and
// stage 4 touches the host, so a token's worth of ranking costs four launches
// and never a readback.
template <typename T, int THREADS>
__global__ void __launch_bounds__(THREADS)
    topk_collect_kernel(const T *__restrict__ logits, const u32 *__restrict__ pmask,
                        f32 rep_pen, f32 softcap, const u32 *__restrict__ cut,
                        const unsigned long long *__restrict__ gmax,
                        unsigned long long *__restrict__ cand, i32 cap,
                        unsigned long long *__restrict__ count,
                        float *__restrict__ mass, f32 inv_temp, i32 n, i32 span) {
    __shared__ float smass[THREADS / 32];
    const unsigned long long cutkey = static_cast<unsigned long long>(*cut) << 55;
    const f32 maxv = key2f(*gmax);
    const i32 lo = static_cast<i32>(blockIdx.x) * span;
    const i32 hi = min(n, lo + span);
    float m = 0.0f;
    for (i32 i = lo + static_cast<i32>(threadIdx.x); i < hi; i += THREADS) {
        const f32 v = logits_transform(static_cast<f32>(logits[i]), i, pmask,
                                       rep_pen, softcap);
        m += __expf((v - maxv) * inv_temp);
        const unsigned long long k = pack_kv(v, i);
        if (k >= cutkey) {
            const unsigned long long pos = atomicAdd(count, 1ull);
            if (pos < static_cast<unsigned long long>(cap)) cand[pos] = k;
        }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) m += __shfl_xor(m, off);
    const int w = static_cast<int>(threadIdx.x) >> 5;
    if ((static_cast<int>(threadIdx.x) & 31) == 0) smass[w] = m;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = smass[0];
#pragma unroll
        for (int i = 1; i < THREADS / 32; i++) t += smass[i];
        atomicAdd(mass, t);
    }
}

// --- 4. final ---------------------------------------------------------------
// k rounds of block-wide selection with a strictly decreasing bound. Keys are
// unique, so the loop always makes progress and no sort is needed.
//
// The selection runs in ~key space, where "largest key so far, descending"
// becomes a single min-scan with a rising exclusive lower bound. Two wrong
// versions came before it, both caught by the gate: searching for keys ABOVE
// the previous winner (which finds nothing after round 0, because the
// second-best key is smaller by definition, and returns ids = -1 for every
// rank past the first), and searching for the largest key below an upper
// bound of ~0 (which returns the *smallest* key on round 0 instead of the
// largest). Inverting the key makes both impossible: one comparison, one
// bound, and it can only ever walk up ~key, i.e. down key.
//
// The loop bound is the *collected* count, read back from device memory, not
// cand_cap: the candidate buffer is reused every token and its tail still
// holds the previous token's keys -- stale, but well-formed keys that stage 4
// would happily rank as if they were real. When fewer than k elements are in
// range the remaining slots come back ids = -1 / vals = -FLT_MAX, which is
// exactly what the CPU twin writes for the same case.
template <int THREADS>
__global__ void __launch_bounds__(THREADS)
    topk_final_kernel(const unsigned long long *__restrict__ cand,
                      const unsigned long long *__restrict__ count, i32 cap, i32 k,
                      i32 *__restrict__ ids, f32 *__restrict__ vals,
                      unsigned long long *__restrict__ count_out) {
    __shared__ unsigned long long smin[THREADS / 32];
    // The bound is shared state, not a register: it is *read* by every thread
    // on the next round while being *written* by thread 0, and a per-thread
    // copy leaves threads 1..N-1 scanning with the stale bound -- which shows
    // up as every rank after the first repeating rank 0's answer.
    __shared__ unsigned long long s_lower;
    const unsigned long long got = *count;
    const i64 have = static_cast<i64>(
        got < static_cast<unsigned long long>(cap)
            ? got
            : static_cast<unsigned long long>(cap));
    // Nothing is excluded at first, so round 0 takes the global maximum.
    if (threadIdx.x == 0) s_lower = 0ull;
    __syncthreads();

    for (int r = 0; r < k; r++) {
        const unsigned long long lower = s_lower;
        unsigned long long best = ~0ull;
        for (i64 i = static_cast<i64>(threadIdx.x); i < have; i += THREADS) {
            const unsigned long long mk = ~cand[i];
            if (mk > lower && mk < best) best = mk;
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const unsigned long long o = __shfl_xor(best, off);
            if (o < best) best = o;
        }
        const int w = static_cast<int>(threadIdx.x) >> 5;
        if ((static_cast<int>(threadIdx.x) & 31) == 0) smin[w] = best;
        __syncthreads();
        if (threadIdx.x == 0) {
            unsigned long long m = smin[0];
#pragma unroll
            for (int i = 1; i < THREADS / 32; i++)
                if (smin[i] < m) m = smin[i];
            // best is still the sentinel when nothing is left above `lower`,
            // i.e. fewer than k candidates were collected.
            if (m != ~0ull) {
                const unsigned long long kk = ~m;
                s_lower = m;
                ids[r] = key2id(kk);
                vals[r] = key2f(kk);
            } else {
                ids[r] = -1;
                vals[r] = -FLT_MAX;
            }
        }
        // Before the next round overwrites smin, thread 0 is still reading it.
        __syncthreads();
    }
    // The collected count is reported *raw*, not clamped to the capacity: the
    // contract is that a value above cand_cap tells the caller the set was
    // truncated and it must fall back. Clamping here would hide exactly the
    // case the caller needs to know about. It was previously written into the
    // scratch instead of to the caller, which left the caller reading an
    // uninitialised word and falling back on every single token.
    if (threadIdx.x == 0) *count_out = *count;
}
} // namespace krk

#endif // KRK_KERNEL_TOPK_HPP