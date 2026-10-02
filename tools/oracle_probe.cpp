// tools/oracle_probe.cpp — op-by-op GPU vs CPU oracle comparison.
//
// Mirrors Engine::forward_core on two backends (CPU f32 reference and
// HIP f16) with the same model, weights and token ids, downloading and
// comparing after every stage. The first stage whose max-abs diff
// explodes pinpoints the faulty HIP kernel.
#include <krk/backend.hpp>
#include <krk/model.hpp>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace krk;

struct StageStat {
    const char *name;
    double maxabs = 0;
    double maxrel = 0;
    size_t idx = 0;
    f32 refv = 0, gotv = 0;
};

static StageStat compare(const char *name, const std::vector<f32> &ref,
                         const std::vector<f32> &got) {
    StageStat s;
    s.name = name;
    for (size_t i = 0; i < ref.size(); i++) {
        const double d = std::fabs((double)ref[i] - (double)got[i]);
        const double rel = d / (std::fabs((double)ref[i]) + 1e-6);
        if (d > s.maxabs) {
            s.maxabs = d;
            s.idx = i;
            s.refv = ref[i];
            s.gotv = got[i];
        }
        if (rel > s.maxrel) s.maxrel = rel;
    }
    return s;
}

static std::vector<f32> grab(Backend *be, void *p, size_t n) {
    std::vector<f32> v(n);
    if (be->act_type() == DType::F16)
        be->download_f32(v.data(), p, (i64)n);
    else
        be->download(v.data(), p, n * 4);
    return v;
}

static void print_stat(const StageStat &s) {
    std::printf("   %-14s maxabs=%.4g  maxrel=%.3g  (idx %zu: ref=%.5f got=%.5f)\n",
                s.name, s.maxabs, s.maxrel, s.idx,
                (double)s.refv, (double)s.gotv);
}

// ============================================================================
//  gated delta net: the CPU f32 reference against the HIP f16 kernels
//
//  Self-contained — no checkpoint needed, because the ops are the thing under
//  test. Every buffer is filled with the same deterministic values, narrowed to
//  f16 for the GPU exactly the way the engine's activations arrive, and the two
//  backends run the identical op. The recurrent ops (conv, delta rule) are run
//  as a SEQUENCE of calls and the carried state is compared as well, because a
//  single call cannot tell a correct state update from a plausible one.
// ============================================================================

namespace {

int gdn_bad = 0;
unsigned gdn_seed = 2463534242u;

f32 gdn_rand(f32 amp) {
    gdn_seed = gdn_seed * 1664525u + 1013904223u;
    const f32 u =
        static_cast<f32>((gdn_seed >> 8) & 0xFFFFu) / 32768.0f - 1.0f;
    return u * amp;
}

std::vector<f32> gdn_data(i64 n, f32 amp) {
    std::vector<f32> v(static_cast<size_t>(n));
    for (i64 i = 0; i < n; i++) v[static_cast<size_t>(i)] = gdn_rand(amp);
    return v;
}

void gdn_report(const char *name, const std::vector<f32> &ref,
                const std::vector<f32> &got, f64 atol, f64 rtol) {
    const StageStat s = compare(name, ref, got);
    // Absolute tolerances are meaningless for the recurrent ops, whose outputs
    // grow with the state: the yardstick is the reference's own magnitude.
    f64 peak = 0;
    for (f32 v : ref) peak = std::max(peak, (f64)std::fabs(v));
    const f64 rel_peak = s.maxabs / (peak + 1e-30);
    const bool bad = rel_peak > rtol;
    if (bad) gdn_bad++;
    std::printf("  %-26s maxabs=%-11.4g /peak=%-11.3g =%-10.3g at %-8zu %s\n",
                name, s.maxabs, peak, rel_peak, s.idx,
                bad ? "<-- FAIL" : "ok");
    if (bad)
        std::printf("      worst element %zu: ref=%.6g got=%.6g\n", s.idx,
                    (double)s.refv, (double)s.gotv);
    std::fflush(stdout);
}

// An activation buffer, activation-typed on each backend: f32 on the reference,
// f16 on the device. set() fills the CPU side in f32 and uploads the f16
// narrowing, which is precisely the rounding the engine's own activations
// carry, so the diff reported is the real one and not an artifact of the test.
struct ActBuf {
    Backend *cpu, *gpu;
    i64 n;
    void *c = nullptr, *g = nullptr;
    ActBuf(Backend *a, Backend *b, i64 n_) : cpu(a), gpu(b), n(n_) {
        c = cpu->alloc(static_cast<size_t>(n) * 4);
        g = gpu->alloc(static_cast<size_t>(n) * 2);
    }
    ~ActBuf() {
        cpu->release(c);
        gpu->release(g);
    }
    void set(const std::vector<f32> &v) {
        std::memcpy(c, v.data(), static_cast<size_t>(n) * 4);
        std::vector<u16> h(static_cast<size_t>(n));
        for (i64 i = 0; i < n; i++)
            h[static_cast<size_t>(i)] = fp32_to_fp16(v[static_cast<size_t>(i)]);
        gpu->upload(g, h.data(), static_cast<size_t>(n) * 2);
    }
    std::vector<f32> ref() const {
        const f32 *p = static_cast<const f32 *>(c);
        return std::vector<f32>(p, p + n);
    }
    std::vector<f32> got() const {
        std::vector<f32> v(static_cast<size_t>(n));
        gpu->download_f32(v.data(), g, n);
        return v;
    }
};

// An f32 buffer both backends read directly: the small per-head vectors the
// model loader uploads (ssm_dt, ssm_a, the norm weights).
struct F32Vec {
    Backend *cpu, *gpu;
    i64 n;
    f32 *c = nullptr, *g = nullptr;
    F32Vec(Backend *a, Backend *b, i64 n_) : cpu(a), gpu(b), n(n_) {
        c = static_cast<f32 *>(a->alloc(static_cast<size_t>(n) * 4));
        g = static_cast<f32 *>(b->alloc(static_cast<size_t>(n) * 4));
    }
    ~F32Vec() {
        cpu->release(c);
        gpu->release(g);
    }
    void set(const std::vector<f32> &v) {
        std::memcpy(c, v.data(), static_cast<size_t>(n) * 4);
        gpu->upload(g, v.data(), static_cast<size_t>(n) * 4);
    }
    std::vector<f32> ref() const { return std::vector<f32>(c, c + n); }
    std::vector<f32> got() const {
        std::vector<f32> v(static_cast<size_t>(n));
        gpu->download(v.data(), g, static_cast<size_t>(n) * 4);
        return v;
    }
};

int gdn_suite(Backend *cpu, Backend *gpu) {
    std::printf("== gated delta net ops: CPU f32 vs HIP f16 ==\n");
    // The Qwen3.5 geometry: 16 key heads over 32 value heads, 128-wide heads,
    // one fused post-conv row of 2*2048 + 4096.
    const i64 d_state = 128, hd = 128, n_k = 16, n_v = 32, ksize = 4;
    const i64 key_dim = n_k * d_state, val_dim = n_v * hd;
    const i64 cdim = key_dim * 2 + val_dim;
    // f16 activations carry ~3 decimal digits, so the reference diff for a
    // single op is set by that; the recurrent ops accumulate a little more.
    // The bound is checked against the reference's peak, so it means the same
    // thing for an O(1) activation and for an O(100) state.
    const f64 kA = 3e-3, kR = 6e-2, kAc = 8e-3, kRc = 1e-1;
    (void)kA; (void)kR; (void)kAc; (void)kRc;
    const i64 tok = 5;

    // ---- l2norm (strided: a region of one fused row, and in place) --------
    {
        ActBuf x(cpu, gpu, tok * cdim), o(cpu, gpu, tok * cdim);
        const std::vector<f32> v = gdn_data(tok * cdim, 1.0f);
        x.set(v);
        o.set(std::vector<f32>(static_cast<size_t>(tok * cdim), 0.0f));
        cpu->l2norm(o.c, x.c, tok, n_k, d_state, cdim, 1e-6f);
        gpu->l2norm(o.g, x.g, tok, n_k, d_state, cdim, 1e-6f);
        gdn_report("l2norm", o.ref(), o.got(), kA, kR);
        // in place on the k region instead
        x.set(v);
        cpu->l2norm(x.c, x.c, tok, n_k, d_state, cdim, 1e-6f);
        gpu->l2norm(x.g, x.g, tok, n_k, d_state, cdim, 1e-6f);
        gdn_report("l2norm in place", x.ref(), x.got(), kA, kR);
    }

    // ---- scale_act (the 1/sqrt(state) query scale, strided) --------------
    {
        ActBuf x(cpu, gpu, tok * cdim);
        const std::vector<f32> v = gdn_data(tok * cdim, 1.0f);
        x.set(v);
        const f32 alpha = static_cast<f32>(1.0 / std::sqrt(static_cast<f64>(d_state)));
        cpu->scale_act(x.c, alpha, tok, key_dim, cdim);
        gpu->scale_act(x.g, alpha, tok, key_dim, cdim);
        gdn_report("scale_act", x.ref(), x.got(), kA, kR);
    }

    // ---- activations on the per-head gate/forget row ---------------------
    {
        ActBuf x(cpu, gpu, tok * n_v);
        x.set(gdn_data(tok * n_v, 2.0f));
        cpu->sigmoid_act(x.c, tok * n_v);
        gpu->sigmoid_act(x.g, tok * n_v);
        gdn_report("sigmoid_act", x.ref(), x.got(), kA, kR);

        x.set(gdn_data(tok * n_v, 2.0f));
        cpu->softplus_act(x.c, tok * n_v);
        gpu->softplus_act(x.g, tok * n_v);
        gdn_report("softplus_act", x.ref(), x.got(), kA, kR);
    }

    // ---- mul_act ----------------------------------------------------------
    {
        ActBuf a(cpu, gpu, tok * n_v), b(cpu, gpu, tok * n_v);
        a.set(gdn_data(tok * n_v, 1.0f));
        b.set(gdn_data(tok * n_v, 1.0f));
        cpu->mul_act(a.c, b.c, tok * n_v);
        gpu->mul_act(a.g, b.g, tok * n_v);
        gdn_report("mul_act", a.ref(), a.got(), kA, kR);
    }

    // ---- the per-column bias and decay scale -----------------------------
    {
        ActBuf x(cpu, gpu, tok * n_v);
        F32Vec bias(cpu, gpu, n_v);
        x.set(gdn_data(tok * n_v, 1.0f));
        bias.set(gdn_data(n_v, 3.0f));
        cpu->add_bias_cols(x.c, bias.c, tok, n_v);
        gpu->add_bias_cols(x.g, bias.g, tok, n_v);
        gdn_report("add_bias_cols", x.ref(), x.got(), kA, kR);

        x.set(gdn_data(tok * n_v, 1.0f));
        bias.set(gdn_data(n_v, 3.0f));
        cpu->scale_cols(x.c, bias.c, tok, n_v);
        gpu->scale_cols(x.g, bias.g, tok, n_v);
        gdn_report("scale_cols", x.ref(), x.got(), kA, kR);
    }

    // ---- scale_rows (the shared-expert gate) -----------------------------
    {
        ActBuf x(cpu, gpu, tok * val_dim), al(cpu, gpu, tok);
        x.set(gdn_data(tok * val_dim, 1.0f));
        al.set(gdn_data(tok, 1.0f));
        cpu->scale_rows(x.c, al.c, tok, val_dim);
        gpu->scale_rows(x.g, al.g, tok, val_dim);
        gdn_report("scale_rows", x.ref(), x.got(), kA, kR);
    }

    // ---- the packed attention query --------------------------------------
    {
        const i64 np = tok * n_k * 2 * hd, nq = tok * n_k * hd;
        ActBuf packed(cpu, gpu, np), q(cpu, gpu, nq), gate(cpu, gpu, nq);
        const std::vector<f32> v = gdn_data(np, 1.0f);
        packed.set(v);
        q.set(std::vector<f32>(static_cast<size_t>(nq), 0.0f));
        gate.set(std::vector<f32>(static_cast<size_t>(nq), 0.0f));
        cpu->qwen3_next_split(q.c, gate.c, packed.c, tok, n_k, hd);
        gpu->qwen3_next_split(q.g, gate.g, packed.g, tok, n_k, hd);
        gdn_report("qwen3_next_split q", q.ref(), q.got(), kA, kR);
        gdn_report("qwen3_next_split gate", gate.ref(), gate.got(), kA, kR);

        // in place with no gate: the unpack the engine actually runs
        packed.set(v);
        cpu->qwen3_next_split(packed.c, nullptr, packed.c, tok, n_k, hd);
        gpu->qwen3_next_split(packed.g, nullptr, packed.g, tok, n_k, hd);
        gdn_report("qwen3_next_split in place", packed.ref(), packed.got(), kA, kR);
    }

    // ---- the convolution by hand, one value per (tap, channel) ----------
    // Distinctive magnitudes so any index confusion is visible at a glance
    // rather than hidden in an aggregate: tap[j][c] = 1 + 10j + 100c,
    // in[t][c] = 1 + t + c, state[i] = 1 + i. For t=0, c=0 that gives
    // 1*1 + 11*2 + 21*3 + 31*1 = 117, and silu(117) is 117.
    {
        const i64 cch = 8, nt = 3, ks = 4;
        std::vector<f32> taps(static_cast<size_t>(ks * cch));
        for (i64 j = 0; j < ks; j++)
            for (i64 c = 0; c < cch; c++)
                taps[static_cast<size_t>(j * cch + c)] =
                    static_cast<f32>(1 + j * 10 + c * 100);
        std::vector<f32> inp(static_cast<size_t>(nt * cch));
        for (i64 t = 0; t < nt; t++)
            for (i64 c = 0; c < cch; c++)
                inp[static_cast<size_t>(t * cch + c)] =
                    static_cast<f32>(1 + t + c);
        std::vector<f32> st0(static_cast<size_t>((ks - 1) * cch));
        for (size_t i = 0; i < st0.size(); i++)
            st0[i] = static_cast<f32>(1 + i);

        f32 *kc = static_cast<f32 *>(cpu->alloc(taps.size() * 4));
        f32 *kg = static_cast<f32 *>(gpu->alloc(taps.size() * 4));
        std::memcpy(kc, taps.data(), taps.size() * 4);
        gpu->upload(kg, taps.data(), taps.size() * 4);
        ActBuf x(cpu, gpu, nt * cch);
        x.set(inp);
        f32 *sc = static_cast<f32 *>(cpu->alloc(st0.size() * 4));
        f32 *sg = static_cast<f32 *>(gpu->alloc(st0.size() * 4));
        std::memcpy(sc, st0.data(), st0.size() * 4);
        gpu->upload(sg, st0.data(), st0.size() * 4);
        cpu->conv1d_silu(x.c, x.c, sc, kc, DType::F32, nt, cch, ks);
        gpu->conv1d_silu(x.g, x.g, sg, kg, DType::F32, nt, cch, ks);
        const std::vector<f32> r = x.ref(), o = x.got();
        std::printf("  -- tiny conv (ref | got) --\n");
        for (i64 t = 0; t < nt; t++) {
            std::printf("     t%lld ", (long long)t);
            for (i64 c = 0; c < cch; c++)
                std::printf(" %8.0f/%-8.0f", (double)r[static_cast<size_t>(t * cch + c)],
                            (double)o[static_cast<size_t>(t * cch + c)]);
            std::printf("\n");
        }
        std::printf("     carried state row0  ref=");
        for (i64 c = 0; c < cch; c++)
            std::printf(" %.0f", (double)sc[c]);
        std::printf("  got=");
        std::vector<f32> gs(static_cast<size_t>(st0.size()));
        gpu->download(gs.data(), sg, gs.size() * 4);
        for (i64 c = 0; c < cch; c++)
            std::printf(" %.0f", (double)gs[static_cast<size_t>(c)]);
        std::printf("\n");
        bool tiny_ok = true;
        for (i64 c = 0; c < cch; c++)
            tiny_ok = tiny_ok && gs[static_cast<size_t>(c)] ==
                                       inp[static_cast<size_t>(c)];
        std::printf("  %-26s %s\n", "tiny conv (carried state)",
                    tiny_ok ? "ok" : " <-- FAIL");
        if (!tiny_ok) gdn_bad++;
        cpu->release(kc);
        gpu->release(kg);
        cpu->release(sc);
        gpu->release(sg);
    }

    // ---- the short convolution, in place, over a call SEQUENCE -----------
    // Two calls share one state, so the returned state is compared too: a
    // kernel that keeps its window right but stores it wrong is the failure
    // this catches, and a single call cannot see it.
    {
        const i64 st_n = (ksize - 1) * cdim;
        const std::vector<f32> taps = gdn_data(ksize * cdim, 0.4f);
        const size_t tap_bytes = taps.size() * 4;
        // Weights live on the device for the GPU and in host memory for the
        // reference; both sides must be handed the same bytes in the same
        // format, which is why the f32 taps are uploaded too.
        f32 *kc = static_cast<f32 *>(cpu->alloc(tap_bytes));
        f32 *kg = static_cast<f32 *>(gpu->alloc(tap_bytes));
        std::memcpy(kc, taps.data(), tap_bytes);
        gpu->upload(kg, taps.data(), tap_bytes);
        // The same taps in Q8_0 (fp16 scale + 32 int8 per block) to exercise the
        // device dequant path rather than only the f32 one.
        const i64 blocks = static_cast<i64>(taps.size()) / 32;
        std::vector<u8> q8(static_cast<size_t>(blocks) * 34);
        for (i64 b = 0; b < blocks; b++) {
            u8 *p = q8.data() + b * 34;
            const u16 d = fp32_to_fp16(0.02f);
            std::memcpy(p, &d, 2);
            for (int i = 0; i < 32; i++)
                p[2 + i] = static_cast<u8>(static_cast<int8_t>(gdn_rand(1.0f) * 120.0f));
        }
        u8 *q8c = static_cast<u8 *>(cpu->alloc(q8.size()));
        std::memcpy(q8c, q8.data(), q8.size());
        u8 *q8g = static_cast<u8 *>(gpu->alloc(q8.size()));
        gpu->upload(q8g, q8.data(), q8.size());

        f32 *sc = static_cast<f32 *>(cpu->alloc(static_cast<size_t>(st_n) * 4));
        f32 *sg = static_cast<f32 *>(gpu->alloc(static_cast<size_t>(st_n) * 4));
        const std::vector<f32> init = gdn_data(st_n, 0.8f);
        std::memcpy(sc, init.data(), init.size() * 4);
        gpu->upload(sg, init.data(), init.size() * 4);

        for (int variant = 0; variant < 2; variant++) {
            const DType wt = variant == 0 ? DType::F32 : DType::Q8_0;
            const void *kw = variant == 0 ? static_cast<const void *>(kc)
                                          : static_cast<const void *>(q8c);
            const void *kgp = variant == 0 ? static_cast<const void *>(kg)
                                           : static_cast<const void *>(q8g);
            // the carried state starts over for the second variant
            std::memcpy(sc, init.data(), init.size() * 4);
            gpu->upload(sg, init.data(), init.size() * 4);

            const i64 steps[2] = {3, 2};
            std::vector<f32> in0, st0v;
            for (int s = 0; s < 2; s++) {
                ActBuf x(cpu, gpu, steps[s] * cdim);
                x.set(gdn_data(steps[s] * cdim, 1.0f));
                cpu->conv1d_silu(x.c, x.c, sc, kw, wt, steps[s], cdim, ksize);
                gpu->conv1d_silu(x.g, x.g, sg, kgp, wt, steps[s], cdim, ksize);
                char nm[64];
                std::snprintf(nm, sizeof(nm), "conv1d_silu %s n=%lld%s",
                              variant == 0 ? "f32" : "q8_0", (long long)steps[s],
                              s == 1 ? " (carried)" : "");
                const std::vector<f32> r = x.ref(), o = x.got();
                gdn_report(nm, r, o, kA, kR);
                // how widespread is it, and what does the worst channel see?
                size_t over = 0;
                for (size_t i = 0; i < r.size(); i++)
                    if (std::fabs((double)r[i] - (double)o[i]) > 1e-2) over++;
                if (over)
                    std::printf("      %zu/%zu elements over 1e-2\n", over,
                                r.size());
            }
            std::vector<f32> got_st(static_cast<size_t>(st_n));
            gpu->download(got_st.data(), sg, got_st.size() * 4);
            gdn_report("  (conv state carried)",
                       std::vector<f32>(sc, sc + st_n), got_st, kA, kR);
        }
        cpu->release(kc);
        gpu->release(kg);
        cpu->release(q8c);
        gpu->release(q8g);
        cpu->release(sc);
        gpu->release(sg);
    }

    // ---- the delta rule, over a call SEQUENCE ----------------------------
    {
        const i64 state_n = n_v * d_state * hd;
        f32 *stc = static_cast<f32 *>(cpu->alloc(static_cast<size_t>(state_n) * 4));
        f32 *stg = static_cast<f32 *>(gpu->alloc(static_cast<size_t>(state_n) * 4));
        const std::vector<f32> init = gdn_data(state_n, 0.3f);
        std::memcpy(stc, init.data(), init.size() * 4);
        gpu->upload(stg, init.data(), init.size() * 4);

        const i64 steps[3] = {1, 4, 1};
        for (int s = 0; s < 3; s++) {
            // q | k | v are three regions of ONE row, which is what makes
            // row_stride the conv width rather than a head-packed width.
            ActBuf x(cpu, gpu, steps[s] * cdim), out(cpu, gpu, steps[s] * val_dim);
            ActBuf g(cpu, gpu, steps[s] * n_v), beta(cpu, gpu, steps[s] * n_v);
            x.set(gdn_data(steps[s] * cdim, 1.0f));
            g.set(gdn_data(steps[s] * n_v, 0.5f));
            beta.set(gdn_data(steps[s] * n_v, 1.0f));
            out.set(std::vector<f32>(static_cast<size_t>(steps[s] * val_dim), 0.0f));
            const f32 *xp = static_cast<const f32 *>(x.c);
            cpu->delta_rule(out.c, stc, xp, xp + key_dim, xp + 2 * key_dim, g.c,
                            beta.c, steps[s], n_k, n_v, d_state, hd, cdim);
            const _Float16 *xg = static_cast<const _Float16 *>(x.g);
            gpu->delta_rule(out.g, stg, xg, xg + key_dim, xg + 2 * key_dim, g.g,
                            beta.g, steps[s], n_k, n_v, d_state, hd, cdim);
            char nm[64];
            std::snprintf(nm, sizeof(nm), "delta_rule n=%lld%s", (long long)steps[s],
                          s == 2 ? " (carried)" : "");
            gdn_report(nm, out.ref(), out.got(), kAc, kRc);
        }
        std::vector<f32> got_st(static_cast<size_t>(state_n));
        gpu->download(got_st.data(), stg, got_st.size() * 4);
        gdn_report("  (delta state carried)", std::vector<f32>(stc, stc + state_n),
                   got_st, kAc, kRc);
        cpu->release(stc);
        gpu->release(stg);
    }

    std::printf("== %s ==\n", gdn_bad == 0 ? "all gated delta net ops agree"
                                           : "GATED DELTA NET MISMATCH");
    return gdn_bad == 0 ? 0 : 1;
}

} // namespace

int main(int argc, char **argv) {
    if (argc > 1 && std::string(argv[1]) == "--gdn") {
        std::string err;
        Backend *c = make_cpu_backend();
        Backend *g = make_hip_backend(0, &err);
        if (!g) {
            std::printf("no HIP device: %s\n", err.c_str());
            return 1;
        }
        const int rc = gdn_suite(c, g);
        delete g;
        delete c;
        return rc;
    }
    const char *path =
        argc > 1 ? argv[1] : "models/SmolLM2-135M-Instruct.Q4_K_M.gguf";
    std::string err;
    Backend *cpu = make_cpu_backend();
    Backend *gpu = make_hip_backend(0, &err);
    if (!gpu) {
        std::printf("no HIP device: %s\n", err.c_str());
        return 1;
    }

    Model mc, mg;
    if (!mc.load(*cpu, path, &err)) {
        std::printf("cpu load failed: %s\n", err.c_str());
        return 1;
    }
    if (!mg.load(*gpu, path, &err)) {
        std::printf("gpu load failed: %s\n", err.c_str());
        return 1;
    }

    const ModelConfig &c = mc.cfg();
    const i64 N = 4, POS0 = 0, KV_CAP = 512;
    const i64 n_embd = c.n_embd, n_ff = c.n_ff;
    const i64 q_dim = mc.q_dim(), kv_dim = mc.kv_dim();
    const f32 scale =
        static_cast<f32>(1.0 / std::sqrt(static_cast<f64>(c.head_dim))) *
        c.rope_attn_scale;

    std::vector<i32> toks = {1, 4821, 13, 278};

    // CPU (f32) workspaces.
    void *xc = cpu->alloc((size_t)N * n_embd * 4);
    void *xn_c = cpu->alloc((size_t)N * n_embd * 4);
    void *x2_c = cpu->alloc((size_t)N * n_embd * 4);
    void *qc = cpu->alloc((size_t)N * q_dim * 4);
    void *kc = cpu->alloc((size_t)N * kv_dim * 4);
    void *vsc = cpu->alloc((size_t)N * kv_dim * 4);
    void *ac = cpu->alloc((size_t)N * q_dim * 4);
    void *gc = cpu->alloc((size_t)N * n_ff * 4);
    void *uc = cpu->alloc((size_t)N * n_ff * 4);
    void *lc = cpu->alloc((size_t)c.n_vocab * 4);
    // GPU (f16) workspaces.
    void *xg = gpu->alloc((size_t)N * n_embd * 2);
    void *xn_g = gpu->alloc((size_t)N * n_embd * 2);
    void *x2_g = gpu->alloc((size_t)N * n_embd * 2);
    void *qg = gpu->alloc((size_t)N * q_dim * 2);
    void *kg = gpu->alloc((size_t)N * kv_dim * 2);
    void *vsg = gpu->alloc((size_t)N * kv_dim * 2);
    void *ag = gpu->alloc((size_t)N * q_dim * 2);
    void *gg = gpu->alloc((size_t)N * n_ff * 2);
    void *ug = gpu->alloc((size_t)N * n_ff * 2);
    void *lg = gpu->alloc((size_t)c.n_vocab * 2);
    // KV caches.
    const size_t layer_span_c = (size_t)KV_CAP * (size_t)kv_dim * 4;
    const size_t layer_span_g = (size_t)KV_CAP * (size_t)kv_dim * 2;
    void *kch = cpu->alloc((size_t)c.n_layer * layer_span_c);
    void *vch = cpu->alloc((size_t)c.n_layer * layer_span_c);
    void *kgh = gpu->alloc((size_t)c.n_layer * layer_span_g);
    void *vgh = gpu->alloc((size_t)c.n_layer * layer_span_g);
    cpu->fill0(kch, (size_t)c.n_layer * layer_span_c);
    cpu->fill0(vch, (size_t)c.n_layer * layer_span_c);
    gpu->fill0(kgh, (size_t)c.n_layer * layer_span_g);
    gpu->fill0(vgh, (size_t)c.n_layer * layer_span_g);

    auto row_span_c = [&]() { return (size_t)kv_dim * 4; };

    // ---- embed ----------------------------------------------------------
    cpu->embed(xc, mc.tok_embd().data, mc.tok_embd().type, c.n_vocab,
               n_embd, toks.data(), N);
    gpu->embed(xg, mg.tok_embd().data, mg.tok_embd().type, c.n_vocab,
               n_embd, toks.data(), N);
    StageStat e = compare("embed", grab(cpu, xc, (size_t)N * n_embd),
                          grab(gpu, xg, (size_t)N * n_embd));
    std::printf("== embed ==\n");
    print_stat(e);

    for (i32 l = 0; l < c.n_layer; l++) {
        const LayerWeights &Lc = mc.layers()[(size_t)l];
        const LayerWeights &Lg = mg.layers()[(size_t)l];
        StageStat worst{"", 0, 0, 0, 0, 0};
        auto note = [&](StageStat s) {
            if (s.maxabs > worst.maxabs) worst = s;
            if (s.maxabs > 0.5) print_stat(s); // suspicious stage detail
        };

        cpu->rmsnorm(xn_c, xc, Lc.attn_norm, N, n_embd, c.rms_eps);
        gpu->rmsnorm(xn_g, xg, Lg.attn_norm, N, n_embd, c.rms_eps);
        note(compare("rmsnorm", grab(cpu, xn_c, (size_t)N * n_embd),
                     grab(gpu, xn_g, (size_t)N * n_embd)));

        cpu->gemm(qc, xn_c, Lc.wq.data, Lc.wq.type, q_dim, n_embd, N);
        gpu->gemm(qg, xn_g, Lg.wq.data, Lg.wq.type, q_dim, n_embd, N);
        note(compare("gemm q", grab(cpu, qc, (size_t)N * q_dim),
                     grab(gpu, qg, (size_t)N * q_dim)));

        cpu->gemm(kc, xn_c, Lc.wk.data, Lc.wk.type, kv_dim, n_embd, N);
        gpu->gemm(kg, xn_g, Lg.wk.data, Lg.wk.type, kv_dim, n_embd, N);
        note(compare("gemm k", grab(cpu, kc, (size_t)N * kv_dim),
                     grab(gpu, kg, (size_t)N * kv_dim)));

        cpu->gemm(vsc, xn_c, Lc.wv.data, Lc.wv.type, kv_dim, n_embd, N);
        gpu->gemm(vsg, xn_g, Lg.wv.data, Lg.wv.type, kv_dim, n_embd, N);
        note(compare("gemm v", grab(cpu, vsc, (size_t)N * kv_dim),
                     grab(gpu, vsg, (size_t)N * kv_dim)));

        if (c.qk_norm) {
            cpu->qk_norm(qc, kc, Lc.q_norm, Lc.k_norm, c.n_head, c.n_head_kv,
                         c.head_dim, N, c.rms_eps);
            gpu->qk_norm(qg, kg, Lg.q_norm, Lg.k_norm, c.n_head,
                         c.n_head_kv, c.head_dim, N, c.rms_eps);
            note(compare("qk_norm q", grab(cpu, qc, (size_t)N * q_dim),
                         grab(gpu, qg, (size_t)N * q_dim)));
            note(compare("qk_norm k", grab(cpu, kc, (size_t)N * kv_dim),
                         grab(gpu, kg, (size_t)N * kv_dim)));
        }

        cpu->rope(qc, kc, c.n_head, c.n_head_kv, c.head_dim, N, POS0,
                  mc.inv_freq().data(), c.rope_scale, c.rope_frac);
        gpu->rope(qg, kg, c.n_head, c.n_head_kv, c.head_dim, N, POS0,
                  mg.inv_freq().data(), c.rope_scale, c.rope_frac);
        note(compare("rope q", grab(cpu, qc, (size_t)N * q_dim),
                     grab(gpu, qg, (size_t)N * q_dim)));
        note(compare("rope k", grab(cpu, kc, (size_t)N * kv_dim),
                     grab(gpu, kg, (size_t)N * kv_dim)));

        AttnDesc d;
        d.n_head = c.n_head;
        d.n_kv = c.n_head_kv;
        d.hd = c.head_dim;
        d.n_tok = N;
        d.pos0 = POS0;
        d.layer_stride = KV_CAP * kv_dim;
        d.pos_stride = kv_dim;
        d.layer = l;
        d.scale = scale;
        cpu->kv_append(kch, vch, kc, vsc, d);
        gpu->kv_append(kgh, vgh, kg, vsg, d);
        {
            const size_t off = (size_t)l * KV_CAP * (size_t)kv_dim +
                               (size_t)POS0 * (size_t)kv_dim;
            std::vector<f32> ref((size_t)N * (size_t)kv_dim),
                got((size_t)N * (size_t)kv_dim);
            std::memcpy(ref.data(),
                        static_cast<const f32 *>(kch) + off,
                        ref.size() * 4);
            gpu->download_f32(got.data(),
                              static_cast<const char *>(kgh) +
                                  off * 2,
                              (i64)ref.size());
            note(compare("kv_append k", ref, got));
        }

        cpu->attention(ac, qc, kch, vch, d);
        gpu->attention(ag, qg, kgh, vgh, d);
        note(compare("attention", grab(cpu, ac, (size_t)N * q_dim),
                     grab(gpu, ag, (size_t)N * q_dim)));

        cpu->gemm(x2_c, ac, Lc.wo.data, Lc.wo.type, n_embd, q_dim, N);
        gpu->gemm(x2_g, ag, Lg.wo.data, Lg.wo.type, n_embd, q_dim, N);
        note(compare("gemm wo", grab(cpu, x2_c, (size_t)N * n_embd),
                     grab(gpu, x2_g, (size_t)N * n_embd)));

        cpu->add_inplace(xc, x2_c, N * n_embd);
        gpu->add_inplace(xg, x2_g, N * n_embd);
        note(compare("add_inplace", grab(cpu, xc, (size_t)N * n_embd),
                     grab(gpu, xg, (size_t)N * n_embd)));

        cpu->rmsnorm(xn_c, xc, Lc.ffn_norm, N, n_embd, c.rms_eps);
        gpu->rmsnorm(xn_g, xg, Lg.ffn_norm, N, n_embd, c.rms_eps);
        note(compare("rmsnorm ffn", grab(cpu, xn_c, (size_t)N * n_embd),
                     grab(gpu, xn_g, (size_t)N * n_embd)));

        cpu->gemm(gc, xn_c, Lc.wgate.data, Lc.wgate.type, n_ff, n_embd, N);
        gpu->gemm(gg, xn_g, Lg.wgate.data, Lg.wgate.type, n_ff, n_embd, N);
        note(compare("gemm gate", grab(cpu, gc, (size_t)N * n_ff),
                     grab(gpu, gg, (size_t)N * n_ff)));

        cpu->gemm(uc, xn_c, Lc.wup.data, Lc.wup.type, n_ff, n_embd, N);
        gpu->gemm(ug, xn_g, Lg.wup.data, Lg.wup.type, n_ff, n_embd, N);
        note(compare("gemm up", grab(cpu, uc, (size_t)N * n_ff),
                     grab(gpu, ug, (size_t)N * n_ff)));

        cpu->silu_mul(gc, gc, uc, N * n_ff);
        gpu->silu_mul(gg, gg, ug, N * n_ff);
        note(compare("silu_mul", grab(cpu, gc, (size_t)N * n_ff),
                     grab(gpu, gg, (size_t)N * n_ff)));

        cpu->gemm(x2_c, gc, Lc.wdown.data, Lc.wdown.type, n_embd, n_ff, N);
        gpu->gemm(x2_g, gg, Lg.wdown.data, Lg.wdown.type, n_embd, n_ff, N);
        note(compare("gemm down", grab(cpu, x2_c, (size_t)N * n_embd),
                     grab(gpu, x2_g, (size_t)N * n_embd)));

        cpu->add_inplace(xc, x2_c, N * n_embd);
        gpu->add_inplace(xg, x2_g, N * n_embd);
        note(compare("add_inplace2", grab(cpu, xc, (size_t)N * n_embd),
                     grab(gpu, xg, (size_t)N * n_embd)));

        std::printf("L%02d worst: ", l);
        print_stat(worst);
        std::fflush(stdout);
        if (worst.maxabs > 1.0) {
            std::printf(">>> divergence exploded at layer %d — stopping\n", l);
            break;
        }
    }

    // ---- head -----------------------------------------------------------
    cpu->rmsnorm(xn_c, xc, mc.out_norm(), 1, n_embd, c.rms_eps);
    gpu->rmsnorm(xn_g, xg, mg.out_norm(), 1, n_embd, c.rms_eps);
    StageStat h = compare("head rmsnorm", grab(cpu, xn_c, (size_t)n_embd),
                          grab(gpu, xn_g, (size_t)n_embd));
    std::printf("== head ==\n");
    print_stat(h);
    cpu->gemm(lc, xn_c, mc.out_head().data, mc.out_head().type, c.n_vocab,
              n_embd, 1);
    gpu->gemm(lg, xn_g, mg.out_head().data, mg.out_head().type, c.n_vocab,
              n_embd, 1);
    std::vector<f32> lr = grab(cpu, lc, (size_t)c.n_vocab);
    std::vector<f32> lg2 = grab(gpu, lg, (size_t)c.n_vocab);
    StageStat lo = compare("logits", lr, lg2);
    print_stat(lo);
    i32 ac_max = 0, ag_max = 0;
    for (i32 i = 1; i < c.n_vocab; i++) {
        if (lr[(size_t)i] > lr[(size_t)ac_max]) ac_max = i;
        if (lg2[(size_t)i] > lg2[(size_t)ag_max]) ag_max = i;
    }
    std::printf("argmax cpu=%d (%.4f)  gpu=%d (%.4f)\n", ac_max,
                (double)lr[(size_t)ac_max], ag_max,
                (double)lg2[(size_t)ag_max]);

    (void)row_span_c;
    return 0;
}
