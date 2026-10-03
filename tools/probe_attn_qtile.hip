// A/B probe: attention_qtile_kernel vs the reference attention_kernel on
// identical inputs. The prefill rewrite produced garbage, and reading the two
// side by side kept not showing it, so this compares them numerically stage by
// stage: same q, same KV, same descriptor, and it reports the worst offending
// (token, head, channel) so the failing element can be looked up directly.
//
// It also sweeps pos0, because the query-tiled version has to handle a token
// block whose KV rows sit at absolute positions above the chunk origin.
//
// Build:
//   hipcc -O3 --offload-arch=gfx1201 -I include -I src -I src/hip \
//       tools/probe_attn_qtile.hip -o probe_attn_qtile
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#include "krk_hip.hpp"
#include "kernels/attention.hpp"

using namespace krk;

// The library's hip_check lives in backend_hip.hip; a standalone probe has to
// supply it so KRK_HIP aborts with the failing call instead of continuing.
namespace krk {
const char *hip_err_name(hipError_t e) { return hipGetErrorName(e); }
void hip_check(hipError_t e, const char *what) {
    if (e != hipSuccess) {
        std::fprintf(stderr, "FATAL %s: %s\n", what, hipGetErrorName(e));
        std::exit(1);
    }
}
} // namespace krk

static f32 frand(unsigned &s) {
    s = s * 1664525u + 1013904223u;
    return static_cast<f32>((s >> 8) & 0xffff) / 32768.0f - 1.0f;
}

template <int QT>
static void run_case(i64 n_head, i64 n_kv, i64 hd, i64 n_tok, i64 pos0,
                     i64 ctx, int tile_k, bool causal, unsigned seed,
                     const char *label) {
    const i64 pos_stride = ctx;
    const size_t qn = static_cast<size_t>(n_tok) * n_head * hd;
    const size_t kn = static_cast<size_t>(ctx) * n_kv * hd;

    std::vector<_Float16> hq(qn), hk(kn), hv(kn), href(qn);
    unsigned s = seed;
    for (auto &x : hq) x = static_cast<_Float16>(frand(s));
    for (auto &x : hk) x = static_cast<_Float16>(frand(s) * 0.5f);
    for (auto &x : hv) x = static_cast<_Float16>(frand(s) * 0.5f);

    _Float16 *q, *kc, *vc, *ref, *got;
    KRK_HIP(hipMalloc(&q, qn * 2));
    KRK_HIP(hipMalloc(&kc, kn * 2));
    KRK_HIP(hipMalloc(&vc, kn * 2));
    KRK_HIP(hipMalloc(&ref, qn * 2));
    KRK_HIP(hipMalloc(&got, qn * 2));
    KRK_HIP(hipMemcpy(q, hq.data(), qn * 2, hipMemcpyHostToDevice));
    KRK_HIP(hipMemcpy(kc, hk.data(), kn * 2, hipMemcpyHostToDevice));
    KRK_HIP(hipMemcpy(vc, hv.data(), kn * 2, hipMemcpyHostToDevice));
    KRK_HIP(hipMemset(ref, 0, qn * 2));
    KRK_HIP(hipMemset(got, 0, qn * 2));

    const f32 scale = 1.0f / std::sqrt(static_cast<f32>(hd));

    // The dispatch only selects this kernel when its LDS request fits, so the
    // probe has to apply the same gate instead of asking for an allocation the
    // device would reject.
    int lds = 0;
    KRK_HIP(hipDeviceGetAttribute(&lds, hipDeviceAttributeMaxSharedMemoryPerBlock, 0));
    if (attn_qtile_smem_bytes<QT>(tile_k, hd) + 512 > static_cast<size_t>(lds)) {
        std::printf("%-34s hd=%-3lld tok=%-3lld pos0=%-3lld tile=%-3d  SKIP (needs %zu B LDS, device has %d)\n",
                    label, hd, n_tok, pos0, tile_k,
                    attn_qtile_smem_bytes<QT>(tile_k, hd), lds);
        hipFree(q); hipFree(kc); hipFree(vc); hipFree(ref); hipFree(got);
        return;
    }

    const size_t rsmem = attn_smem_bytes(tile_k, hd);
    KRK_HIP(hipFuncSetAttribute(reinterpret_cast<const void *>(&attention_kernel),
                                hipFuncAttributeMaxDynamicSharedMemorySize,
                                static_cast<int>(rsmem + 512)));
    const dim3 rgrid(static_cast<unsigned>(n_head), static_cast<unsigned>(n_tok));
    attention_kernel<<<rgrid, kAttnBlock, rsmem>>>(
        ref, q, kc, vc, n_head, n_kv, hd, n_tok, pos0, pos_stride, causal ? 1 : 0,
        scale, tile_k);
    KRK_HIP(hipDeviceSynchronize());

    const size_t qsmem = attn_qtile_smem_bytes<QT>(tile_k, hd);
    KRK_HIP(hipFuncSetAttribute(
        reinterpret_cast<const void *>(&attention_qtile_kernel<QT>),
        hipFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(qsmem + 512)));
    const dim3 qgrid(static_cast<unsigned>(n_head),
                     static_cast<unsigned>((n_tok + QT - 1) / QT));
    attention_qtile_kernel<QT><<<qgrid, kAttnBlock, qsmem>>>(
        got, q, kc, vc, n_head, n_kv, hd, n_tok, pos0, pos_stride, causal ? 1 : 0,
        scale, tile_k);
    KRK_HIP(hipDeviceSynchronize());
    KRK_HIP(hipGetLastError());

    href.resize(qn);
    std::vector<_Float16> hgot(qn);
    KRK_HIP(hipMemcpy(href.data(), ref, qn * 2, hipMemcpyDeviceToHost));
    KRK_HIP(hipMemcpy(hgot.data(), got, qn * 2, hipMemcpyDeviceToHost));

    f32 worst = 0.0f, scale_ref = 0.0f;
    i64 wt = -1, wh = -1, wd = -1, nbad = 0;
    for (i64 t = 0; t < n_tok; t++)
        for (i64 h = 0; h < n_head; h++)
            for (i64 d = 0; d < hd; d++) {
                const size_t i = static_cast<size_t>((t * n_head + h) * hd + d);
                const f32 a = static_cast<f32>(href[i]);
                const f32 b = static_cast<f32>(hgot[i]);
                scale_ref = fmaxf(scale_ref, fabsf(a));
                const f32 e = fabsf(a - b);
                if (e > 1e-2f) nbad++;
                if (e > worst) { worst = e; wt = t; wh = h; wd = d; }
            }
    const f32 rel = scale_ref > 0 ? worst / scale_ref : worst;
    std::printf("%-34s hd=%-3lld tok=%-3lld pos0=%-3lld tile=%-3d  worst=%.3e rel=%.3e bad=%lld/%lld",
                label, hd, n_tok, pos0, tile_k, worst, rel, nbad,
                static_cast<long long>(n_tok * n_head * hd));
    if (wt >= 0) std::printf("  worst@t=%lld h=%lld d=%lld ref=%+.4f got=%+.4f",
                             wt, wh, wd,
                             static_cast<f32>(href[static_cast<size_t>((wt * n_head + wh) * hd + wd)]),
                             static_cast<f32>(hgot[static_cast<size_t>((wt * n_head + wh) * hd + wd)]));
    std::printf("\n");

    hipFree(q); hipFree(kc); hipFree(vc); hipFree(ref); hipFree(got);
}

// Ground truth in double, plus a per-element dump for a case small enough to
// read. When the two GPU kernels disagree it matters which one is wrong, so the
// CPU result is the arbiter -- and a single-element view is what makes an
// indexing slip visible instead of looking like "accumulated error".
template <int QT>
static void run_tiny(i64 n_head, i64 hd, i64 n_tok, i64 pos0, i64 ctx,
                     int tile_k, unsigned seed, const char *label) {
    const i64 pos_stride = ctx;
    const size_t qn = static_cast<size_t>(n_tok) * n_head * hd;
    const size_t kn = static_cast<size_t>(ctx) * n_head * hd;

    std::vector<_Float16> hq(qn), hk(kn), hv(kn);
    unsigned s = seed;
    for (auto &x : hq) x = static_cast<_Float16>(frand(s));
    for (auto &x : hk) x = static_cast<_Float16>(frand(s) * 0.5f);
    for (auto &x : hv) x = static_cast<_Float16>(frand(s) * 0.5f);

    _Float16 *q, *kc, *vc, *ref, *got;
    KRK_HIP(hipMalloc(&q, qn * 2));
    KRK_HIP(hipMalloc(&kc, kn * 2));
    KRK_HIP(hipMalloc(&vc, kn * 2));
    KRK_HIP(hipMalloc(&ref, qn * 2));
    KRK_HIP(hipMalloc(&got, qn * 2));
    KRK_HIP(hipMemcpy(q, hq.data(), qn * 2, hipMemcpyHostToDevice));
    KRK_HIP(hipMemcpy(kc, hk.data(), kn * 2, hipMemcpyHostToDevice));
    KRK_HIP(hipMemcpy(vc, hv.data(), kn * 2, hipMemcpyHostToDevice));

    const f32 scale = 1.0f / std::sqrt(static_cast<f32>(hd));
    const size_t rsmem = attn_smem_bytes(tile_k, hd);
    KRK_HIP(hipFuncSetAttribute(reinterpret_cast<const void *>(&attention_kernel),
                                hipFuncAttributeMaxDynamicSharedMemorySize,
                                static_cast<int>(rsmem + 512)));
    attention_kernel<<<dim3(static_cast<unsigned>(n_head), static_cast<unsigned>(n_tok)),
                       kAttnBlock, rsmem>>>(ref, q, kc, vc, n_head, n_head, hd, n_tok,
                                            pos0, pos_stride, 1, scale, tile_k);
    const size_t qsmem = attn_qtile_smem_bytes<QT>(tile_k, hd);
    KRK_HIP(hipFuncSetAttribute(
        reinterpret_cast<const void *>(&attention_qtile_kernel<QT>),
        hipFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(qsmem + 512)));
    attention_qtile_kernel<QT><<<dim3(static_cast<unsigned>(n_head),
                                      static_cast<unsigned>((n_tok + QT - 1) / QT)),
                             kAttnBlock, qsmem>>>(got, q, kc, vc, n_head, n_head, hd,
                                                  n_tok, pos0, pos_stride, 1, scale,
                                                  tile_k);
    KRK_HIP(hipDeviceSynchronize());

    std::vector<_Float16> href(qn), hgot(qn);
    KRK_HIP(hipMemcpy(href.data(), ref, qn * 2, hipMemcpyDeviceToHost));
    KRK_HIP(hipMemcpy(hgot.data(), got, qn * 2, hipMemcpyDeviceToHost));

    // double-precision causal attention
    std::vector<double> cpu(qn, 0.0);
    for (i64 t = 0; t < n_tok; t++) {
        const i64 my = pos0 + t;
        for (i64 h = 0; h < n_head; h++) {
            const i64 nk = my + 1;
            std::vector<double> sc(static_cast<size_t>(nk));
            double mx = -1e300;
            for (i64 kk = 0; kk < nk; kk++) {
                double dot = 0;
                for (i64 d = 0; d < hd; d++)
                    dot += static_cast<double>(static_cast<f32>(hq[static_cast<size_t>((t * n_head + h) * hd + d)])) *
                           static_cast<double>(static_cast<f32>(hk[static_cast<size_t>(h * hd + kk * pos_stride + d)]));
                sc[static_cast<size_t>(kk)] = dot * static_cast<double>(scale);
                mx = sc[static_cast<size_t>(kk)] > mx ? sc[static_cast<size_t>(kk)] : mx;
            }
            double sum = 0;
            std::vector<double> pv(static_cast<size_t>(nk));
            for (i64 kk = 0; kk < nk; kk++) {
                pv[static_cast<size_t>(kk)] = std::exp(sc[static_cast<size_t>(kk)] - mx);
                sum += pv[static_cast<size_t>(kk)];
            }
            for (i64 d = 0; d < hd; d++) {
                double a = 0;
                for (i64 kk = 0; kk < nk; kk++)
                    a += pv[static_cast<size_t>(kk)] *
                         static_cast<double>(static_cast<f32>(hv[static_cast<size_t>(h * hd + kk * pos_stride + d)]));
                cpu[static_cast<size_t>((t * n_head + h) * hd + d)] = a / sum;
            }
        }
    }

    double er = 0, eg = 0;
    for (size_t i = 0; i < qn; i++) {
        er = std::max(er, std::fabs(static_cast<double>(static_cast<f32>(href[i])) - cpu[i]));
        eg = std::max(eg, std::fabs(static_cast<double>(static_cast<f32>(hgot[i])) - cpu[i]));
    }
    std::printf("\n== %s == tiled_vs_cpu=%.3e qtile_vs_cpu=%.3e\n", label, er, eg);
    const bool full = qn <= 512;
    for (i64 t = 0; t < n_tok; t += (full ? 1 : (n_tok > 8 ? n_tok / 4 : 1)))
        for (i64 d = 0; d < hd && (full || d < 6); d++) {
            const size_t i = static_cast<size_t>((t * n_head + 0) * hd + d);
            std::printf("  t=%-3lld d=%-2lld cpu=%+.6f tiled=%+.6f qtile=%+.6f\n",
                        pos0 + t, d, cpu[i], static_cast<f32>(href[i]),
                        static_cast<f32>(hgot[i]));
        }
    hipFree(q); hipFree(kc); hipFree(vc); hipFree(ref); hipFree(got);
}

// Compare the kernel's per-key scores and its final (m, l) against values
// computed on the host from the same inputs. This separates "the score matrix
// is wrong" from "the online softmax over the scores is wrong".
static void run_scores(i64 n_head, i64 hd, i64 n_tok, i64 pos0, i64 ctx,
                       int tile_k, unsigned seed) {
    const i64 pos_stride = ctx;
    const size_t qn = static_cast<size_t>(n_tok) * n_head * hd;
    const size_t kn = static_cast<size_t>(ctx) * n_head * hd;
    std::vector<_Float16> hq(qn), hk(kn), hv(kn);
    unsigned s = seed;
    for (auto &x : hq) x = static_cast<_Float16>(frand(s));
    for (auto &x : hk) x = static_cast<_Float16>(frand(s) * 0.5f);
    for (auto &x : hv) x = static_cast<_Float16>(frand(s) * 0.5f);

    constexpr int QT = 8;
    const i64 tmax = pos0 + n_tok - 1;
    const i64 n_keys = tmax + 1;
    const i64 ntiles = (n_keys + tile_k - 1) / tile_k;
    const size_t dsz = static_cast<size_t>((ntiles * QT * tile_k) + QT * 2);

    _Float16 *q, *kc, *vc, *out;
    f32 *dbg;
    KRK_HIP(hipMalloc(&q, qn * 2));
    KRK_HIP(hipMalloc(&kc, kn * 2));
    KRK_HIP(hipMalloc(&vc, kn * 2));
    KRK_HIP(hipMalloc(&out, qn * 2));
    KRK_HIP(hipMalloc(&dbg, dsz * 4));
    KRK_HIP(hipMemcpy(q, hq.data(), qn * 2, hipMemcpyHostToDevice));
    KRK_HIP(hipMemcpy(kc, hk.data(), kn * 2, hipMemcpyHostToDevice));
    KRK_HIP(hipMemcpy(vc, hv.data(), kn * 2, hipMemcpyHostToDevice));
    KRK_HIP(hipMemset(dbg, 0xff, dsz * 4));

    const f32 scale = 1.0f / std::sqrt(static_cast<f32>(hd));
    const size_t qsmem = attn_qtile_smem_bytes<QT>(tile_k, hd);
    KRK_HIP(hipFuncSetAttribute(
        reinterpret_cast<const void *>(&attention_qtile_kernel<QT>),
        hipFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(qsmem + 512)));
    attention_qtile_kernel<QT><<<dim3(static_cast<unsigned>(n_head),
                                      static_cast<unsigned>((n_tok + QT - 1) / QT)),
                             kAttnBlock, qsmem>>>(out, q, kc, vc, n_head, n_head, hd,
                                                  n_tok, pos0, pos_stride, 1, scale,
                                                  tile_k);
    KRK_HIP(hipDeviceSynchronize());
    std::vector<f32> hd32(dsz);
    KRK_HIP(hipMemcpy(hd32.data(), dbg, dsz * 4, hipMemcpyDeviceToHost));

    std::printf("\n== scores hd=%lld tok=%lld pos0=%lld tile_k=%d ntiles=%lld ==\n",
                hd, n_tok, pos0, tile_k, ntiles);
    // Host score for (qi, key) -- independent of the kernel's tiling.
    // KV element (h, kk, d) lives at h*hd + kk*pos_stride + d -- the head
    // stride is hd but the position stride is pos_stride, and conflating the
    // two makes the reference disagree with BOTH kernels.
    auto hscore = [&](i64 t, i64 h, i64 kk) {
        double dot = 0;
        for (i64 d = 0; d < hd; d++)
            dot += static_cast<double>(static_cast<f32>(hq[static_cast<size_t>((t * n_head + h) * hd + d)])) *
                   static_cast<double>(static_cast<f32>(hk[static_cast<size_t>(h * hd + kk * pos_stride + d)]));
        return dot * static_cast<double>(scale);
    };
    int bad = 0;
    for (i64 t0 = 0; t0 < n_tok; t0 += QT) {
        const i64 nq = (n_tok - t0 < QT) ? (n_tok - t0) : QT;
        for (i64 qi = 0; qi < nq; qi++)
            for (i64 tb = 0; tb < ntiles; tb++)
                for (int j = 0; j < tile_k; j++) {
                    const i64 kk = tb * tile_k + j;
                    if (kk >= n_keys) continue;
                    const f32 got = hd32[static_cast<size_t>(tb * QT * tile_k + qi * tile_k + j)];
                    const double want = hscore(t0 + qi, 0, kk);
                    if (std::fabs(static_cast<double>(got) - want) > 1e-3) {
                        if (bad < 8)
                            std::printf("  SCORE t=%lld kk=%lld got=%+.6f want=%+.6f\n",
                                        pos0 + t0 + qi, kk, got, want);
                        bad++;
                    }
                }
    }
    std::printf("  score mismatches: %d\n", bad);

    for (i64 t0 = 0; t0 < n_tok; t0 += QT) {
        const i64 nq = (n_tok - t0 < QT) ? (n_tok - t0) : QT;
        for (i64 qi = 0; qi < nq; qi++) {
            const i64 my = pos0 + t0 + qi;
            double mx = -1e300, sum = 0;
            std::vector<double> pv(static_cast<size_t>(my + 1));
            for (i64 kk = 0; kk <= my; kk++) {
                const double sc = hscore(t0 + qi, 0, kk);
                mx = sc > mx ? sc : mx;
            }
            for (i64 kk = 0; kk <= my; kk++) {
                pv[static_cast<size_t>(kk)] =
                    std::exp(hscore(t0 + qi, 0, kk) - mx);
                sum += pv[static_cast<size_t>(kk)];
            }
            const f32 gm = hd32[static_cast<size_t>(ntiles * QT * tile_k + qi * 2 + 0)];
            const f32 gl = hd32[static_cast<size_t>(ntiles * QT * tile_k + qi * 2 + 1)];
            const f32 okm = std::fabs(static_cast<double>(gm) - mx) < 1e-3 ? 1.0f : 0.0f;
            const f32 okl = std::fabs(static_cast<double>(gl) - sum) < 1e-3 * sum ? 1.0f : 0.0f;
            if (!okm || !okl)
                std::printf("  ML t=%lld got_m=%+.6f want_m=%+.6f | got_l=%+.6f want_l=%+.6f %s\n",
                            my, gm, mx, gl, sum, (!okm && !okl) ? "BOTH" : (!okm ? "M" : "L"));
        }
    }
    hipFree(q); hipFree(kc); hipFree(vc); hipFree(out); hipFree(dbg);
}

int main() {
    unsigned seed = 12345;
    // hd = 64 is the SmolLM2 head width: the fastest failing case.
    // tile_k = 1 puts every key in its own tile, so a failure that survives
    // here is in the cross-tile rescaling, not in the intra-tile indexing.
    run_case<8>(8, 8, 64, 32, 0, 64, 1, true, seed, "tile_k=1 (no intra-tile)");
    run_case<8>(8, 8, 64, 32, 0, 64, 2, true, seed, "tile_k=2");
    run_case<8>(8, 8, 64, 32, 0, 64, 4, true, seed, "tile_k=4");
    run_case<8>(8, 8, 64, 32, 0, 64, 8, true, seed, "tile_k=8");
    run_case<8>(8, 8, 64, 32, 0, 64, 16, true, seed, "tile_k=16");
    run_case<8>(8, 8, 64, 32, 0, 64, 32, true, seed, "tile_k=32");
    run_case<8>(8, 8, 64, 32, 0, 64, 32, true, seed, "single chunk pos0=0");
    run_case<8>(8, 8, 64, 32, 0, 64, 32, false, seed, "single chunk pos0=0 nocausal");
    run_case<8>(8, 8, 64, 256, 0, 512, 32, true, seed, "hd64 pos0=0 4 tiles");
    run_case<8>(8, 8, 64, 256, 0, 512, 64, true, seed, "hd64 pos0=0 1 tile");
    run_case<8>(8, 8, 64, 256, 128, 512, 32, true, seed, "hd64 pos0=128");
    run_case<8>(8, 8, 256, 64, 0, 128, 16, true, seed, "hd256 pos0=0");
    run_case<8>(8, 2, 128, 48, 16, 256, 32, true, seed, "hd128 gqa pos0=16");
    run_case<8>(8, 8, 64, 9, 0, 64, 32, true, seed, "ragged 9 tokens");
    run_case<8>(8, 8, 64, 8, 0, 64, 32, true, seed, "exact one tile of QT");
    // n_head must be 1: every block writes the same dbg buffer, so a multi-head
    // grid leaves the dump as whichever block finished last.
    run_scores(1, 8, 4, 0, 8, 8, seed);
    run_scores(1, 8, 12, 0, 32, 4, seed);
    run_scores(1, 64, 32, 0, 64, 8, seed);
    run_scores(1, 16, 20, 4, 64, 8, seed);
    return 0;
    return 0;
}