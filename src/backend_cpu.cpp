// backend_cpu.cpp — the portable reference backend.
//
// This is the oracle: it is simple and obviously correct. Ops run
// across INDEPENDENT units (output columns, attention heads, token
// rows) on the worker pool, but no reduction is ever split, so
// results stay bit-identical to the serial path — the GPU backends
// are tested against exactly these numbers. Activations are f32
// here, which is why the engine asks the backend for its activation
// type instead of assuming f16.
#include "krk/backend.hpp"
#include "par_pool.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

namespace krk {

namespace {

// Below this many estimated flops, the pool round trip costs more
// than the computation, so ops stay serial. Decode's elementwise ops
// (RMSNorm over one 576-wide row, RoPE over 9 heads, silu_mul over
// 1536 lanes) fall under the floor; every GEMM, the head GEMV
// (~57 MFLOP) and attention at context length do not. The pool's
// spin-then-park wake policy makes the round trip nearly free when
// ops arrive back to back, so the floor can stay low.
constexpr f64 kParFlopFloor = 1.0e5;

// Parallel-for over disjoint index ranges. `est_flops` gates pool use;
// the serial fallback skips std::function construction entirely.
template <class F>
inline void par_for(i64 n, i64 grain, f64 est_flops, F &&body) {
    if (n <= 0) return;
    if (est_flops < kParFlopFloor || grain < 1) {
        body(static_cast<i64>(0), n);
        return;
    }
    Pool::get().run(n, grain, std::forward<F>(body));
}

void rmsnorm_rows(f32 *out, const f32 *x, const f32 *w, i64 rows, i64 n, f32 eps) {
    // Rows are independent and each row keeps its own reduction.
    par_for(rows, 1, 4.0 * static_cast<f64>(rows) * n,
            [&](i64 b, i64 e) {
                for (i64 r = b; r < e; r++) {
                    const f32 *xr = x + r * n;
                    f32 *orow = out + r * n;
                    f32 ss = 0;
                    for (i64 i = 0; i < n; i++) ss += xr[i] * xr[i];
                    const f32 inv = 1.0f / std::sqrt(ss / static_cast<f32>(n) + eps);
                    for (i64 i = 0; i < n; i++) orow[i] = xr[i] * inv * w[i];
                }
            });
}

} // namespace

class CpuBackend final : public Backend {
public:
    explicit CpuBackend(const DeviceCaps &c) : caps_(c) {}

    BackendKind kind() const override { return BackendKind::CPU; }
    const DeviceCaps &caps() const override { return caps_; }
    DType act_type() const override { return DType::F32; }
    size_t act_size() const override { return 4; }

    void *alloc(size_t bytes) override { return host_alloc(bytes); }
    void release(void *p) override { host_free(p); }
    void upload(void *dst, const void *src, size_t bytes, size_t off) override {
        std::memcpy(static_cast<char *>(dst) + off, src, bytes);
    }
    void download(void *dst, const void *src, size_t bytes, size_t off) override {
        std::memcpy(dst, static_cast<const char *>(src) + off, bytes);
    }
    void fill0(void *dst, size_t bytes) override { std::memset(dst, 0, bytes); }
    void sync() override {}
    void download_f32(f32 *dst, const void *src, i64 n) override {
        std::memcpy(dst, src, static_cast<size_t>(n) * 4);
    }
    std::string describe() const override {
        return format("CPU (scalar reference), %s", caps_.gfx.c_str());
    }

    void embed(void *out, const void *tok_embd, DType tt, i64 n_vocab, i64 n_embd,
               const i32 *tokens, i64 n_tok) override {
        (void)n_vocab;
        const size_t row_bytes = dtype_row_bytes(tt, n_embd);
        f32 *o = static_cast<f32 *>(out);
        // Token rows are independent; each dequantizes into its
        // own output slice.
        par_for(n_tok, 1, 2.0 * static_cast<f64>(n_tok) * n_embd,
                [&](i64 b, i64 e) {
                    for (i64 t = b; t < e; t++) {
                        const i32 id = tokens[t];
                        dequant_row(tt, static_cast<const u8 *>(tok_embd) +
                                           static_cast<size_t>(id) * row_bytes,
                                    o + t * n_embd, n_embd);
                    }
                });
    }

    void rmsnorm(void *out, const void *x, const f32 *w, i64 rows, i64 n,
                 f32 eps) override {
        rmsnorm_rows(static_cast<f32 *>(out), static_cast<const f32 *>(x), w, rows,
                     n, eps);
    }

    void gemm(void *out, const void *x, const void *w, DType wt, i64 n_out,
              i64 n_in, i64 rows) override {
        const size_t row_bytes = dtype_row_bytes(wt, n_in);
        const f32 *xf = static_cast<const f32 *>(x);
        f32 *of = static_cast<f32 *>(out);
        // Every output element is an independent dot product, so the
        // flat (row, column) index space splits without sharing any
        // reduction — results are bit-identical to the serial loop.
        // This is the decode hot path: the head GEMV (n_vocab x n_embd)
        // is ~91% of decode FLOPs and parallelizes across all cores.
        const i64 total = rows * n_out;
        const i64 grain = std::max<i64>(32, std::min<i64>(2048, n_out / 32));
        par_for(total, grain, 2.0 * static_cast<f64>(total) * n_in,
                [&](i64 b, i64 e) {
                    i64 r = b / n_out;
                    i64 o = b - r * n_out;
                    for (i64 f = b; f < e; f++) {
                        const f32 *xr = xf + r * n_in;
                        const u8 *wrow = static_cast<const u8 *>(w) +
                                         static_cast<size_t>(o) * row_bytes;
                        of[f] = vec_dot(wt, wrow, xr, n_in);
                        if (++o >= n_out) {
                            o = 0;
                            ++r;
                        }
                    }
                });
    }

    void rope(void *q, void *k, i64 n_head, i64 n_kv, i64 hd, i64 n_tok, i64 pos0,
              const f32 *inv_freq, f32 scale, f32 frac) override {
        rope_apply(static_cast<f32 *>(q), n_head, hd, n_tok, pos0, inv_freq, scale, frac);
        rope_apply(static_cast<f32 *>(k), n_kv, hd, n_tok, pos0, inv_freq, scale, frac);
    }

    void qk_norm(void *q, void *k, const f32 *wq, const f32 *wk, i64 n_head,
                 i64 n_kv, i64 hd, i64 n_tok, f32 eps) override {
        if (wq) head_norm(static_cast<f32 *>(q), wq, n_head, hd, n_tok, eps);
        if (wk) head_norm(static_cast<f32 *>(k), wk, n_kv, hd, n_tok, eps);
    }

    void kv_append(void *kcache, void *vcache, const void *k, const void *v,
                   const AttnDesc &d) override {
        f32 *kc = static_cast<f32 *>(kcache) + d.layer * d.layer_stride;
        f32 *vc = static_cast<f32 *>(vcache) + d.layer * d.layer_stride;
        const f32 *ks = static_cast<const f32 *>(k);
        const f32 *vs = static_cast<const f32 *>(v);
        // Tokens land at disjoint cache slots.
        par_for(d.n_tok, 4, 2.0 * static_cast<f64>(d.n_tok) * d.pos_stride,
                [&](i64 b, i64 e) {
                    for (i64 t = b; t < e; t++) {
                        const i64 dst = (d.pos0 + t) * d.pos_stride;
                        std::memcpy(kc + dst, ks + t * d.pos_stride,
                                    static_cast<size_t>(d.pos_stride) * 4);
                        std::memcpy(vc + dst, vs + t * d.pos_stride,
                                    static_cast<size_t>(d.pos_stride) * 4);
                    }
                });
    }

    void attention(void *out, const void *q, const void *kcache, const void *vcache,
                   const AttnDesc &d) override {
        const i64 n_rep = d.n_kv > 0 ? d.n_head / d.n_kv : 1;
        const f32 *qf = static_cast<const f32 *>(q);
        const f32 *kc = static_cast<const f32 *>(kcache) + d.layer * d.layer_stride;
        const f32 *vc = static_cast<const f32 *>(vcache) + d.layer * d.layer_stride;
        f32 *of = static_cast<f32 *>(out);
        // (token, head) pairs are independent: each owns its
        // output rows and keeps a PRIVATE score scratch, so
        // nothing is shared across blocks and the softmax
        // max/sum stay inside one head — bit-identical to the
        // serial loop (which used one shared scratch).
        const i64 units = d.n_tok * d.n_head;
        par_for(units, 1,
                4.0 * static_cast<f64>(units) * (d.pos0 + d.n_tok) * d.hd,
                [&](i64 b, i64 e) {
                    std::vector<f32> scores;
                    for (i64 u = b; u < e; u++) {
                        const i64 t = u / d.n_head;
                        const i64 h = u - t * d.n_head;
                        const i64 pos = d.pos0 + t;
                        const i64 n_keys = d.causal ? pos + 1 : d.pos0 + d.n_tok;
                        scores.resize(static_cast<size_t>(n_keys));
                        const i64 hkv = n_rep > 0 ? h / n_rep : 0;
                        const f32 *qh = qf + t * (d.n_head * d.hd) + h * d.hd;
                        for (i64 j = 0; j < n_keys; j++) {
                            const f32 *kj = kc + j * d.pos_stride + hkv * d.hd;
                            f32 dot = 0;
                            for (i64 i = 0; i < d.hd; i++) dot += qh[i] * kj[i];
                            scores[static_cast<size_t>(j)] = dot * d.scale;
                        }
                        f32 mx = scores[0];
                        for (i64 j = 1; j < n_keys; j++)
                            mx = std::max(mx, scores[static_cast<size_t>(j)]);
                        f32 sum = 0;
                        for (i64 j = 0; j < n_keys; j++) {
                            const f32 ex = std::exp(scores[static_cast<size_t>(j)] - mx);
                            scores[static_cast<size_t>(j)] = ex;
                            sum += ex;
                        }
                        const f32 inv = sum > 0 ? 1.0f / sum : 0.0f;
                        f32 *oh = of + t * (d.n_head * d.hd) + h * d.hd;
                        for (i64 i = 0; i < d.hd; i++) oh[i] = 0;
                        for (i64 j = 0; j < n_keys; j++) {
                            const f32 p = scores[static_cast<size_t>(j)] * inv;
                            const f32 *vj = vc + j * d.pos_stride + hkv * d.hd;
                            for (i64 i = 0; i < d.hd; i++) oh[i] += p * vj[i];
                        }
                    }
                });
    }

    void silu_mul(void *out, const void *gate, const void *up, i64 n) override {
        f32 *o = static_cast<f32 *>(out);
        const f32 *g = static_cast<const f32 *>(gate);
        const f32 *u = static_cast<const f32 *>(up);
        par_for(n, 8192, 2.0 * static_cast<f64>(n),
                [&](i64 b, i64 e) {
                    for (i64 i = b; i < e; i++) {
                        const f32 x = g[i];
                        o[i] = (x / (1.0f + std::exp(-x))) * u[i];
                    }
                });
    }

    void add_bias_rows(void *out, const f32 *bias, i64 n, i64 rows) override {
        f32 *o = static_cast<f32 *>(out);
        par_for(rows, 1, 2.0 * static_cast<f64>(rows) * n,
                [&](i64 b, i64 e) {
                    for (i64 r = b; r < e; r++)
                        for (i64 i = 0; i < n; i++) o[r * n + i] += bias[i];
                });
    }

    void add_inplace(void *a, const void *b, i64 n) override {
        f32 *af = static_cast<f32 *>(a);
        const f32 *bf = static_cast<const f32 *>(b);
        par_for(n, 8192, 2.0 * static_cast<f64>(n),
                [&](i64 b, i64 e) {
                    for (i64 i = b; i < e; i++) af[i] += bf[i];
                });
    }

    void copy_act(void *dst, const void *src, i64 n) override {
        std::memcpy(dst, src, static_cast<size_t>(n) * 4);
    }

    void axpy(void *dst, const void *src, f32 alpha, i64 n) override {
        f32 *d = static_cast<f32 *>(dst);
        const f32 *s = static_cast<const f32 *>(src);
        par_for(n, 8192, 2.0 * static_cast<f64>(n),
                [&](i64 b, i64 e) {
                    if (alpha == 1.0f) {
                        for (i64 i = b; i < e; i++) d[i] += s[i];
                    } else {
                        for (i64 i = b; i < e; i++) d[i] += alpha * s[i];
                    }
                });
    }

    // The reference backend shares one address space, so "staging" is a copy.
    void upload_i32(i32 *dst, const i32 *src, i64 n) override {
        std::memcpy(dst, src, static_cast<size_t>(n) * sizeof(i32));
    }

    void gather_rows(void *dst, const void *src, const i32 *rows, i64 n_rows,
                     i64 n) override {
        const f32 *s = static_cast<const f32 *>(src);
        f32 *d = static_cast<f32 *>(dst);
        par_for(n_rows, 1, 2.0 * static_cast<f64>(n_rows) * n,
                [&](i64 b, i64 e) {
                    for (i64 i = b; i < e; i++) {
                        const i64 r = rows[i];
                        std::memcpy(d + i * n, s + r * n,
                                    static_cast<size_t>(n) * sizeof(f32));
                    }
                });
    }

    void scatter_axpy_rows(void *dst, const void *src, const i32 *rows,
                           const f32 *alpha, i64 n_rows, i64 n) override {
        f32 *d = static_cast<f32 *>(dst);
        const f32 *s = static_cast<const f32 *>(src);
        // Each row writes a distinct destination slice.
        par_for(n_rows, 1, 2.0 * static_cast<f64>(n_rows) * n,
                [&](i64 b, i64 e) {
                    for (i64 i = b; i < e; i++) {
                        f32 *dr = d + static_cast<i64>(rows[i]) * n;
                        const f32 *sr = s + i * n;
                        const f32 a = alpha[i];
                        if (a == 1.0f) {
                            for (i64 j = 0; j < n; j++) dr[j] += sr[j];
                        } else {
                            for (i64 j = 0; j < n; j++) dr[j] += a * sr[j];
                        }
                    }
                });
    }

private:
    // RoPE in the interleaved (original LLaMA / llama.cpp) convention: pair
    // channels (2i, 2i+1). GGUF q/k weights for Llama-family models are
    // stored in this layout (the HF exporter permutes the half-split rows),
    // so pairing adjacent channels reproduces HF's rotate_half semantics.
    // `frac` limits rotation to a prefix of the hd/2 pairs (Phi-2/Gemma
    // partial RoPE); `scale` divides the position (linear scaling).
    static void rope_apply(f32 *x, i64 n_heads, i64 hd, i64 n_tok, i64 pos0,
                           const f32 *inv_freq, f32 scale, f32 frac) {
        const i64 half = hd / 2;
        const i64 rot = std::max<i64>(1, static_cast<i64>(static_cast<f32>(half) * frac));
        const f32 s = scale > 0 ? scale : 1.0f;
        // (token, head) pairs write disjoint lanes.
        const i64 units = n_tok * n_heads;
        par_for(units, 1, 6.0 * static_cast<f64>(units) * rot,
                [&](i64 b, i64 e) {
                    for (i64 u = b; u < e; u++) {
                        const i64 t = u / n_heads;
                        const i64 h = u - t * n_heads;
                        const f32 p = static_cast<f32>(pos0 + t) / s;
                        f32 *hx = x + (t * n_heads + h) * hd;
                        for (i64 i = 0; i < rot && i < half; i++) {
                            const f32 theta = p * inv_freq[i];
                            const f32 c = std::cos(theta), sn = std::sin(theta);
                            const f32 a = hx[2 * i], bb = hx[2 * i + 1];
                            hx[2 * i] = a * c - bb * sn;
                            hx[2 * i + 1] = a * sn + bb * c;
                        }
                    }
                });
    }

    static void head_norm(f32 *x, const f32 *w, i64 n_heads, i64 hd, i64 n_tok,
                          f32 eps) {
        // (token, head) pairs are independent; each
        // keeps its own reduction.
        const i64 units = n_tok * n_heads;
        par_for(units, 1, 4.0 * static_cast<f64>(units) * hd,
                [&](i64 b, i64 e) {
                    for (i64 u = b; u < e; u++) {
                        const i64 t = u / n_heads;
                        const i64 h = u - t * n_heads;
                        f32 *hx = x + (t * n_heads + h) * hd;
                        f32 ss = 0;
                        for (i64 i = 0; i < hd; i++) ss += hx[i] * hx[i];
                        const f32 inv = 1.0f / std::sqrt(ss / static_cast<f32>(hd) + eps);
                        for (i64 i = 0; i < hd; i++) hx[i] = hx[i] * inv * w[i];
                    }
                });
    }

    DeviceCaps caps_;
};

Backend *make_cpu_backend() {
    DeviceCaps c;
    c.name = "CPU";
    c.gfx = "cpu";
    c.family = GfxFamily::Unknown;
    c.wave_size = 64;
    c.cu_count = 1;
    return new CpuBackend(c);
}

} // namespace krk
