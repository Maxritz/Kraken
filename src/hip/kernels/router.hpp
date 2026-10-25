//  router.hpp — the MoE router's activation, on the device.
//
//  The router GEMM already runs here. What does not is the softmax that turns its
//  row into the mixing probabilities: `Engine::moe_ffn` downloads the row and
//  computes exp/max/sum on the host, because the top-k that follows needs the
//  expert ids on the host anyway (the expert cache resolves a pointer from them).
//
//  This kernel exists to answer one question with a number instead of an
//  argument: how far does the *text* move when that softmax runs here instead?
//  Device `expf` and a tree-reduction sum are not `std::exp` and a sequential
//  sum, so the probabilities differ in the last bits, and the last bits are what
//  a greedy decoder's argmax turns into different tokens after enough steps.
//  `KRK_DEVICE_SOFTMAX=1` in the engine uses this and reports the drift against
//  the host result; without that number, "the rewrite cannot be bit-identical"
//  is a worry rather than a cost.
//
//  The reduction order is fixed by the launch shape on purpose: warp lanes take
//  every 32nd element, the warp sums by shfl-down, and the warps combine in
//  shared memory in warp order. A measurement that changed the order between
//  runs would report noise as drift.
#ifndef KRK_KERNEL_ROUTER_HPP
#define KRK_KERNEL_ROUTER_HPP

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

namespace krk {

// One block per row. `lg` is the router GEMM's row in the activation type (f16
// on this backend), `out` is f32 [n_rows, n_expert].
template <int kThreads>
__global__ void router_softmax_kernel(const _Float16 *__restrict__ lg,
                                      f32 *__restrict__ out, i32 ne) {
    const i32 row = blockIdx.x;
    const _Float16 *r = lg + static_cast<size_t>(row) * static_cast<size_t>(ne);
    f32 *o = out + static_cast<size_t>(row) * static_cast<size_t>(ne);
    __shared__ f32 sh[kThreads / 32];

    // 1. the row maximum
    f32 mx = -3.0e38f;
    for (i32 i = threadIdx.x; i < ne; i += kThreads) {
        const f32 v = static_cast<f32>(r[i]);
        mx = v > mx ? v : mx;
    }
    for (int off = 16; off > 0; off >>= 1) {
        const f32 o2 = __shfl_down(mx, off);
        mx = o2 > mx ? o2 : mx;
    }
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = mx;
    __syncthreads();
    if (threadIdx.x < kThreads / 32) {
        f32 v = sh[threadIdx.x];
        for (int off = (kThreads / 32) / 2; off > 0; off >>= 1) {
            const f32 o2 = __shfl_down(v, off);
            v = o2 > v ? o2 : v;
        }
        if (threadIdx.x == 0) sh[0] = v;
    }
    __syncthreads();
    mx = sh[0];

    // 2. exp(x - max), accumulating the sum in the same walk, stored in place
    f32 sum = 0.0f;
    for (i32 i = threadIdx.x; i < ne; i += kThreads) {
        const f32 e = expf(static_cast<f32>(r[i]) - mx);
        o[i] = e;
        sum += e;
    }
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_down(sum, off);
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = sum;
    __syncthreads();
    if (threadIdx.x < kThreads / 32) {
        f32 v = sh[threadIdx.x];
        for (int off = (kThreads / 32) / 2; off > 0; off >>= 1)
            v += __shfl_down(v, off);
        if (threadIdx.x == 0) sh[0] = v;
    }
    __syncthreads();

    // 3. normalize
    const f32 inv = sh[0] > 0.0f ? 1.0f / sh[0] : 0.0f;
    for (i32 i = threadIdx.x; i < ne; i += kThreads) o[i] *= inv;
}

// Top-k over the router's probabilities, on the device.
//
// One block, one row: the selection rule is the host's, exactly -- highest
// probability first, and a tie keeps the LOWER expert id. That rule is why this
// is a k-pass scan rather than a sort: a sort's tie order is a property of the
// sort, and every gate that compares against the host plan would then depend on
// it. `taken` marks the slots already chosen; k is small (8-10).
//
// The scan is split across the block: thread `e % kThreads` owns experts
// `e, e + kThreads, ...`, the block reduces (value, id) warp by warp, and the
// tie rule is carried through the reduction as `ov == best && oi < bid` --
// which is the same answer as the ascending scan with a strict `>` for the same
// reason a maximum over a total order is order-independent: the key is the pair
// (value, -id), and the reduction compares that key.
//
// The first version of this kernel walked the whole expert range on EVERY
// thread and let thread 0 keep its answer, so a block of 128 threads did one
// thread's work with a full block barrier per slot. Measured on
// Qwen3.5-35B-A3B, K=8 over 256 experts, per layer: **0.385 ms**, which over 40
// layers was 15.4 ms of a 35 ms decode step -- 44% of the step spent in a
// latency-bound serial scan of a 256-entry array. The parallel form is the same
// arithmetic on the same data with 128 threads' worth of memory-level
// parallelism behind it.
//
// `wts` is left as the raw probabilities of the selected slots. The caller's
// normalization (`expert_w_norm`) and scale are applied by the host today and
// are applied here by `router_weights_kernel`, so the two halves can be checked
// independently against the reference.
template <int kThreads>
__global__ void router_topk_kernel(const f32 *__restrict__ prob, i32 n_expert,
                                   i32 k, i32 *__restrict__ ids,
                                   f32 *__restrict__ wts) {
    constexpr int kWarps = kThreads / 32;
    const i32 row = blockIdx.x;
    const f32 *p = prob + static_cast<size_t>(row) * static_cast<size_t>(n_expert);
    i32 *id = ids + static_cast<size_t>(row) * static_cast<size_t>(k);
    f32 *w = wts + static_cast<size_t>(row) * static_cast<size_t>(k);
    __shared__ i32 taken[64]; // k is at most this in practice; bounds the scan
    __shared__ f32 shv[kWarps];
    __shared__ i32 shi[kWarps];
    for (i32 i = threadIdx.x; i < k && i < 64; i += kThreads) taken[i] = -1;
    __syncthreads();

    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    for (i32 s = 0; s < k; s++) {
        f32 best = -3.0e38f;
        i32 bid = -1;
        for (i32 e = static_cast<i32>(threadIdx.x); e < n_expert; e += kThreads) {
            bool used = false;
            for (i32 q = 0; q < s && q < 64; q++)
                if (taken[q] == e) { used = true; break; }
            if (used) continue;
            const f32 v = p[e];
            if (v > best || (v == best && bid >= 0 && e < bid)) {
                best = v;
                bid = e;
            }
        }
        // Warp reduce, carrying the tie rule.
        for (int off = 16; off > 0; off >>= 1) {
            const f32 ov = __shfl_down(best, off);
            const i32 oi = __shfl_down(bid, off);
            if (ov > best || (ov == best && oi >= 0 && (bid < 0 || oi < bid))) {
                best = ov;
                bid = oi;
            }
        }
        if (lane == 0) {
            shv[warp] = best;
            shi[warp] = bid;
        }
        __syncthreads();
        // Block reduce over the warp results, in warp order.
        if (warp == 0) {
            f32 v = lane < kWarps ? shv[lane] : -3.0e38f;
            i32 e0 = lane < kWarps ? shi[lane] : -1;
            for (int off = kWarps / 2; off > 0; off >>= 1) {
                const f32 ov = __shfl_down(v, off);
                const i32 oi = __shfl_down(e0, off);
                if (ov > v || (ov == v && oi >= 0 && (e0 < 0 || oi < e0))) {
                    v = ov;
                    e0 = oi;
                }
            }
            if (lane == 0) {
                id[s] = e0;
                w[s] = e0 >= 0 ? v : 0.0f;
                if (s < 64) taken[s] = e0;
            }
        }
        __syncthreads();
    }
}

// The post-selection weights: sum-normalize over the k selected slots (the
// Qwen convention, `expert_w_norm`) and apply the file's scale. Split from the
// selection so a file that declares neither still arrives here and divides by
// its own sum once, which is what the host does unconditionally when the flag
// is absent.
template <int kThreads>
__global__ void router_weights_kernel(f32 *__restrict__ wts, i32 k,
                                      i32 normalize, f32 scale) {
    const i32 row = blockIdx.x;
    f32 *w = wts + static_cast<size_t>(row) * static_cast<size_t>(k);
    __shared__ f32 ssum;
    if (threadIdx.x == 0) {
        f32 sum = 0.0f;
        for (i32 s = 0; s < k; s++) sum += w[s];
        ssum = sum;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        const f32 inv = normalize && ssum > 0.0f ? 1.0f / ssum : 1.0f;
        for (i32 s = 0; s < k; s++) {
            w[s] *= inv;
            if (scale != 0.0f && scale != 1.0f) w[s] *= scale;
        }
    }
}

// ---- routing statistics, accumulated on the device -------------------------
//
// What a tracking run (`--expert-scan`, `--expert-adapt`) needs from a routed
// layer is two numbers per expert: how much probability mass it received, and
// how often it was picked. Both are already on the device when the dispatch is
// on -- `ws_routerp_` holds the probabilities and `ws_ids_` the choice -- so
// collecting them on the host means a blocking readback per layer that exists
// only to fill a counter. On a 40-layer stack that is 40 drains per token, and
// measured on Qwen3.5-35B-A3B it was 8 ms of a 22 ms step: a tracking run was
// slower than the path it was measuring.
//
// This kernel folds one layer's call into the caller's histograms, which are
// cumulative and never cleared here. `mass` uses atomics, so its low bits depend
// on the order rows land in -- it ranks experts, it does not decide text, and
// the index it writes is a prior for residency rather than anything exact.
template <int THREADS>
__global__ void scan_accum_kernel(const f32 *__restrict__ probs,
                                  const i32 *__restrict__ ids, i32 n, i32 ne,
                                  i32 k, i32 layer,
                                  unsigned int *__restrict__ hits,
                                  f32 *__restrict__ mass) {
    const i64 base = static_cast<i64>(layer) * static_cast<i64>(ne);
    for (i32 e = static_cast<i32>(threadIdx.x); e < ne; e += THREADS) {
        f32 s = 0.0f;
        for (i32 t = 0; t < n; t++)
            s += probs[static_cast<i64>(t) * ne + e];
        if (s != 0.0f) atomicAdd(&mass[base + e], s);
    }
    const i32 pairs = n * k;
    for (i32 i = static_cast<i32>(threadIdx.x); i < pairs; i += THREADS) {
        const i32 id = ids[i];
        if (id >= 0 && id < ne) atomicAdd(&hits[base + id], 1u);
    }
}

// The layer's expert plan, built on the device from the selected ids.
//
// `off` is [n_expert + 1] prefix offsets, `rows` and `wt` are [n * k] in
// EXPERT-MAJOR order -- exactly the host's `plan_off_`/`plan_dev_`/`alpha_dev_`.
// The fill is serial on purpose: the host iterates tokens ascending then slots
// ascending, and an atomic fill would put the pairs of one expert in whatever
// order the warps happened to finish. A differential check against the host
// plan has to compare two orderings that are meant to be the same, so the
// ordering is prescribed here rather than left to the scheduler.
//   grid 1, block 1: nk <= 408 and n_expert <= 256 in this engine, so the
//   whole plan is 700 iterations of integer work per layer.
__global__ void moe_plan_kernel(const i32 *__restrict__ ids, i32 n, i32 k,
                                i32 n_expert, i32 *__restrict__ off,
                                i32 *__restrict__ rows,
                                f32 *__restrict__ wt,
                                const i32 *__restrict__ row_of,
                                const f32 *__restrict__ wts) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    for (i32 e = 0; e <= n_expert; e++) off[e] = 0;
    const i32 nk = n * k;
    for (i32 i = 0; i < nk; i++) {
        const i32 e = ids[i];
        if (e >= 0 && e < n_expert) off[e + 1]++;
    }
    for (i32 e = 0; e < n_expert; e++) off[e + 1] += off[e];
    // Stable fill: walk the pairs in (token, slot) order and append per expert.
    i32 cursor[256];
    for (i32 e = 0; e < n_expert && e < 256; e++) cursor[e] = off[e];
    for (i32 i = 0; i < nk; i++) {
        const i32 e = ids[i];
        if (e < 0 || e >= n_expert || e >= 256) continue;
        const i32 at = cursor[e]++;
        rows[at] = row_of ? row_of[i] : (i / k);
        wt[at] = wts ? wts[i] : 0.0f;
    }
}

// The EXPERT-MAJOR group index, for the tiled dispatch's batch form.
//
// moe_plan_kernel orders the pairs expert-major too, but it stores the TOKEN for
// each entry, which is enough for a loop that already knows the slot it is on. A
// grouped kernel starts from the expert and needs the whole flat (token, slot)
// index, because that is what addresses `mid` and the per-slot partial. So this
// emits `off[n_expert + 1]` and `idx[n * k]` in the same prescribed order
// (tokens ascending, then slots ascending) -- the order is what keeps a grouped
// run comparable with the token-major one, so the fill stays serial.
//   grid 1, block 1: nk <= a few thousand and n_expert <= 256, so this is a few
//   thousand integer steps per layer.
__global__ void moe_group_plan_kernel(const i32 *__restrict__ ids, i32 n, i32 k,
                                      i32 n_expert, i32 *__restrict__ off,
                                      i32 *__restrict__ idx) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    for (i32 e = 0; e <= n_expert; e++) off[e] = 0;
    const i32 nk = n * k;
    for (i32 i = 0; i < nk; i++) {
        const i32 e = ids[i];
        if (e >= 0 && e < n_expert) off[e + 1]++;
    }
    for (i32 e = 0; e < n_expert; e++) off[e + 1] += off[e];
    i32 cursor[256];
    for (i32 e = 0; e < n_expert && e < 256; e++) cursor[e] = off[e];
    for (i32 i = 0; i < nk; i++) {
        const i32 e = ids[i];
        if (e < 0 || e >= n_expert || e >= 256) continue;
        idx[cursor[e]++] = i;
    }
}

} // namespace krk

#endif // KRK_KERNEL_ROUTER_HPP
