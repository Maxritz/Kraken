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

int main(int argc, char **argv) {
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
