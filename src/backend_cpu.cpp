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
#include <cfloat> // FLT_MAX bounding the SIMT/CPU rsqrt and the clamp paths
#include <cmath>
#include <utility>
#include <vector>

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

// The gated delta net's short convolution is a fixed 4-tap FIR in the shipped
// checkpoints; 16 is a generous ceiling that still keeps the rolling window on
// the stack.
constexpr i64 kMaxConvKernel = 16;

inline f32 silu_scalar(f32 x) { return x / (1.0f + std::exp(-x)); }

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

    void rmsnorm_grouped(void *o, const void *x, const f32 *w, i64 n_tok,
                         i64 n_group, i64 width, i64 stride,
                         f32 eps) override {
        const f32 *s = static_cast<const f32 *>(x);
        f32 *d = static_cast<f32 *>(o);
        // Salient difference from rmsnorm_rows above: the reduction and the
        // weight both belong to the (token, group) pair, so the weight index is
        // the row's group, not a row-invariant w[i]. Safe in place: the
        // reduction completes before the first write to that group's channels,
        // and no other row touches them.
        const i64 rows = n_tok * n_group;
        par_for(rows, 1, 4.0 * static_cast<f64>(rows) * width,
                [&](i64 b, i64 e) {
                    for (i64 r = b; r < e; r++) {
                        const i64 t = r / n_group;
                        const i64 g = r % n_group;
                        const f32 *sr = s + t * stride + g * width;
                        f32 *dr = d + t * stride + g * width;
                        const f32 *wr = w + g * width;
                        f32 ss = 0;
                        for (i64 i = 0; i < width; i++) ss += sr[i] * sr[i];
                        const f32 inv = 1.0f /
                            std::sqrt(ss / static_cast<f32>(width) + eps);
                        for (i64 i = 0; i < width; i++)
                            dr[i] = sr[i] * inv * wr[i];
                    }
                });
    }
    // The scalar twin of the device radix select (kernels/topk.hpp). Same
    // contract and same order -- transform first, then rank by (value, id)
    // descending -- so the two backends can be compared token for token. This
    // one has no download to save, so it exists for comparability, not speed.
    void logits_topk(const void *logits, i64 n, i32 k, const void *pen_mask,
                     f32 rep_pen, f32 softcap, f32 temp, i32 *ids, f32 *vals,
                     f32 *mass_out, void *scratch, i32 cand_cap,
                     i64 *count_out) override {
        (void)scratch;
        (void)cand_cap;
        const f32 *l = static_cast<const f32 *>(logits);
        const u32 *pm = static_cast<const u32 *>(pen_mask);
        auto key = [&](i32 i) {
            f32 v = l[i];
            if (softcap > 0.0f)
                v = v > softcap ? softcap : (v < -softcap ? -softcap : v);
            if (pm && rep_pen > 0.0f && rep_pen != 1.0f &&
                ((pm[static_cast<u32>(i) >> 5] >> (static_cast<u32>(i) & 31)) & 1u))
                v = v > 0.0f ? v / rep_pen : v * rep_pen;
            return v;
        };
        const i64 keep = std::min<i64>(n, k);
        std::vector<i32> idx(static_cast<size_t>(n));
        for (i64 i = 0; i < n; i++) idx[static_cast<size_t>(i)] = static_cast<i32>(i);
        // This O(n log k) sort is exactly the host work the device kernels
        // exist to avoid; here it is what makes the scalar backend a usable
        // oracle for the fast path.
        std::partial_sort(idx.begin(), idx.begin() + keep, idx.end(),
                          [&](i32 a, i32 b) {
                              const f32 ka = key(a), kb = key(b);
                              return ka != kb ? ka > kb : a < b;
                          });
        const f32 maxv = key(idx[0]);
        for (i32 r = 0; r < k; r++) {
            if (r < keep) {
                ids[r] = idx[static_cast<size_t>(r)];
                vals[r] = key(ids[r]);
            } else {
                // Fewer than k elements exist. The device kernel writes the
                // same marker, so the two backends stay interchangeable.
                ids[r] = -1;
                vals[r] = -FLT_MAX;
            }
        }
        if (mass_out) {
            const f32 inv = temp > 0.0f ? 1.0f / temp : 1.0f;
            f32 m = 0.0f;
            for (i64 i = 0; i < n; i++)
                m += std::exp((key(static_cast<i32>(i)) - maxv) * inv);
            *mass_out = m;
        }
        if (count_out) *count_out = keep;
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
              const f32 *inv_freq, f32 scale, f32 frac, bool neox) override {
        rope_apply(static_cast<f32 *>(q), n_head, hd, n_tok, pos0, inv_freq, scale, frac, neox);
        rope_apply(static_cast<f32 *>(k), n_kv, hd, n_tok, pos0, inv_freq, scale, frac, neox);
    }

    void qk_norm(void *q, void *k, const f32 *wq, const f32 *wk, i64 n_head,
                 i64 n_kv, i64 hd, i64 n_tok, f32 eps, bool wide) override {
        if (wq) {
            if (wide)
                row_norm(static_cast<f32 *>(q), wq, n_tok, n_head * hd, eps);
            else
                head_norm(static_cast<f32 *>(q), wq, n_head, hd, n_tok, eps);
        }
        if (wk) {
            if (wide)
                row_norm(static_cast<f32 *>(k), wk, n_tok, n_kv * hd, eps);
            else
                head_norm(static_cast<f32 *>(k), wk, n_kv, hd, n_tok, eps);
        }
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
                        // Sliding window (laguna's hybrid layers): the query
                        // sees only the `window` most recent keys, so the ones
                        // below the bound are skipped outright -- they never
                        // enter the max or the denominator.
                        const i64 j_lo = d.window > 0
                                             ? std::max<i64>(0, n_keys - d.window)
                                             : 0;
                        const i64 n_vis = n_keys - j_lo;
                        scores.resize(static_cast<size_t>(n_vis));
                        const i64 hkv = n_rep > 0 ? h / n_rep : 0;
                        const f32 *qh = qf + t * (d.n_head * d.hd) + h * d.hd;
                        for (i64 j = 0; j < n_vis; j++) {
                            const f32 *kj = kc + (j_lo + j) * d.pos_stride + hkv * d.hd;
                            f32 dot = 0;
                            for (i64 i = 0; i < d.hd; i++) dot += qh[i] * kj[i];
                            scores[static_cast<size_t>(j)] = dot * d.scale;
                        }
                        f32 mx = scores[0];
                        for (i64 j = 1; j < n_vis; j++)
                            mx = std::max(mx, scores[static_cast<size_t>(j)]);
                        f32 sum = 0;
                        for (i64 j = 0; j < n_vis; j++) {
                            const f32 ex = std::exp(scores[static_cast<size_t>(j)] - mx);
                            scores[static_cast<size_t>(j)] = ex;
                            sum += ex;
                        }
                        const f32 inv = sum > 0 ? 1.0f / sum : 0.0f;
                        f32 *oh = of + t * (d.n_head * d.hd) + h * d.hd;
                        for (i64 i = 0; i < d.hd; i++) oh[i] = 0;
                        for (i64 j = 0; j < n_vis; j++) {
                            const f32 p = scores[static_cast<size_t>(j)] * inv;
                            const f32 *vj = vc + (j_lo + j) * d.pos_stride + hkv * d.hd;
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

    // ---- prism.hadamard activation transform (Backend::hadamard_act) ----
    //
    // The reference definition, and deliberately the plain one: per row, an
    // optional grouped-V perm, an optional +-1 sign multiply, then a
    // normalized Sylvester FWHT over consecutive blocks of `block`. The FWHT
    // is the textbook butterfly (H[ i ][ j ] = (-1)^popcount(i & j)), scaled
    // by 1/sqrt(block) at the end so H . H = I -- which is what makes the
    // same routine both the fold before a rotated matmul and the inverse
    // after a rotated embedding lookup; only the sign order differs.
    // Everything stays in f32, exactly like the fork's graph tensors.
    void hadamard_act(void *out, const void *in, i64 rows,
                      const Backend::HadDesc &d) override {
        if (rows <= 0 || d.n <= 0) return;
        const i64 n = d.n;
        const i64 block = d.block;
        if (block <= 0 || (block & (block - 1)) != 0 || n % block != 0) {
            KRK_ERROR("hadamard_act: width %lld is not a whole number of "
                      "%lld-element blocks", (long long)n, (long long)block);
            std::abort();
        }
        const bool perm = d.perm_hd > 0;
        const f32 *signs = d.signs;
        const f32 scale = 1.0f / std::sqrt(static_cast<f32>(block));
        f32 *o = static_cast<f32 *>(out);
        const f32 *x = static_cast<const f32 *>(in);
        // A row is transformed into a scratch row before anything is written
        // back, so out == in is safe: the whole input row is read first.
        par_for(rows, 1, 8.0 * static_cast<f64>(rows) * static_cast<f64>(n),
                [&](i64 b, i64 e) {
                    std::vector<f32> cur(static_cast<size_t>(n));
                    std::vector<f32> tmp(static_cast<size_t>(n));
                    for (i64 r = b; r < e; r++) {
                        f32 *a = cur.data();
                        const f32 *src = x + r * n;
                        for (i64 i = 0; i < n; i++) a[i] = src[i];
                        if (perm) {
                            const i64 H = d.perm_hd;
                            const i64 Nk = d.perm_nk;
                            const i64 R = d.perm_rep;
                            f32 *b2 = tmp.data();
                            for (i64 j = 0; j < R; j++)
                                for (i64 k = 0; k < Nk; k++)
                                    for (i64 i = 0; i < H; i++)
                                        b2[i + j * H + k * H * R] =
                                            a[i + k * H + j * H * Nk];
                            a = b2;
                        }
                        if (d.signs_first && signs)
                            for (i64 i = 0; i < n; i++) a[i] *= signs[i];
                        for (i64 off = 0; off < n; off += block) {
                            f32 *y = a + off;
                            for (i64 len = 1; len < block; len <<= 1)
                                for (i64 g = 0; g < block; g += 2 * len)
                                    for (i64 j = 0; j < len; j++) {
                                        const f32 u = y[g + j];
                                        const f32 v = y[g + len + j];
                                        y[g + j] = u + v;
                                        y[g + len + j] = u - v;
                                    }
                            for (i64 i = 0; i < block; i++) y[i] *= scale;
                        }
                        if (!d.signs_first && signs)
                            for (i64 i = 0; i < n; i++) a[i] *= signs[i];
                        f32 *dst = o + r * n;
                        for (i64 i = 0; i < n; i++) dst[i] = a[i];
                    }
                });
    }

    // ---- gated delta net (the reference oracle) -------------------------
    bool gdn_supported() const override { return true; }

    void l2norm(void *o, const void *x, i64 n_tok, i64 n_head, i64 hd, i64 stride,
                f32 eps) override {
        const f32 *s = static_cast<const f32 *>(x);
        f32 *d = static_cast<f32 *>(o);
        // Each head carries its own reduction, so (token, head) pairs are the
        // units. Safe in place: the reduction finishes before the write.
        par_for(n_tok, 1, 3.0 * static_cast<f64>(n_tok) * n_head * hd,
                [&](i64 b, i64 e) {
                    for (i64 t = b; t < e; t++) {
                        for (i64 h = 0; h < n_head; h++) {
                            const f32 *sh = s + t * stride + h * hd;
                            f32 *dh = d + t * stride + h * hd;
                            f32 ss = 0;
                            for (i64 i = 0; i < hd; i++) ss += sh[i] * sh[i];
                            const f32 inv = 1.0f / std::sqrt(ss + eps);
                            for (i64 i = 0; i < hd; i++) dh[i] = sh[i] * inv;
                        }
                    }
                });
    }

    void sigmoid_act(void *x, i64 n) override {
        f32 *p = static_cast<f32 *>(x);
        par_for(n, 8192, 4.0 * static_cast<f64>(n),
                [&](i64 b, i64 e) {
                    for (i64 i = b; i < e; i++) p[i] = 1.0f / (1.0f + std::exp(-p[i]));
                });
    }

    void relu_sqr_act(void *x, i64 n) override {
        f32 *p = static_cast<f32 *>(x);
        par_for(n, 8192, 4.0 * static_cast<f64>(n),
                [&](i64 b, i64 e) {
                    for (i64 i = b; i < e; i++) {
                        const f32 v = p[i];
                        const f32 r = v > 0.0f ? v : 0.0f;
                        p[i] = r * r;
                    }
                });
    }

    void softplus_act(void *x, i64 n) override {
        f32 *p = static_cast<f32 *>(x);
        par_for(n, 8192, 4.0 * static_cast<f64>(n),
                [&](i64 b, i64 e) {
                    for (i64 i = b; i < e; i++) {
                        // log1p(exp(x)) overflows well before the value matters;
                        // above the threshold softplus(x) == x to f32 accuracy.
                        const f32 v = p[i];
                        p[i] = v > 20.0f ? v : std::log1p(std::exp(v));
                    }
                });
    }

    void scale_act(void *x, f32 alpha, i64 n_tok, i64 width, i64 stride) override {
        f32 *p = static_cast<f32 *>(x);
        par_for(n_tok, 1, static_cast<f64>(n_tok) * width,
                [&](i64 b, i64 e) {
                    for (i64 t = b; t < e; t++) {
                        f32 *row = p + t * stride;
                        for (i64 i = 0; i < width; i++) row[i] *= alpha;
                    }
                });
    }

    void mul_act(void *a, const void *b, i64 n) override {
        f32 *p = static_cast<f32 *>(a);
        const f32 *s = static_cast<const f32 *>(b);
        par_for(n, 8192, 2.0 * static_cast<f64>(n),
                [&](i64 lo, i64 e) {
                    for (i64 i = lo; i < e; i++) p[i] *= s[i];
                });
    }

    void mul_head_broadcast(void *x, const void *g, i64 n_tok, i64 n_head,
                            i64 hd) override {
        f32 *xf = static_cast<f32 *>(x);
        const f32 *gf = static_cast<const f32 *>(g);
        par_for(n_tok, 1, 2.0 * static_cast<f64>(n_tok) * n_head * hd,
                [&](i64 b, i64 e) {
                    for (i64 t = b; t < e; t++) {
                        for (i64 h = 0; h < n_head; h++) {
                            const f32 w = gf[t * n_head + h];
                            f32 *row = xf + (t * n_head + h) * hd;
                            for (i64 i = 0; i < hd; i++) row[i] *= w;
                        }
                    }
                });
    }

    void add_bias_cols(void *x, const f32 *bias, i64 n_row, i64 n_col) override {
        f32 *p = static_cast<f32 *>(x);
        par_for(n_row, 1, static_cast<f64>(n_row) * n_col,
                [&](i64 b, i64 e) {
                    for (i64 i = b; i < e; i++) {
                        f32 *row = p + i * n_col;
                        for (i64 j = 0; j < n_col; j++) row[j] += bias[j];
                    }
                });
    }

    void scale_cols(void *x, const f32 *col, i64 n_row, i64 n_col) override {
        f32 *p = static_cast<f32 *>(x);
        par_for(n_row, 1, static_cast<f64>(n_row) * n_col,
                [&](i64 b, i64 e) {
                    for (i64 i = b; i < e; i++) {
                        f32 *row = p + i * n_col;
                        for (i64 j = 0; j < n_col; j++) row[j] *= col[j];
                    }
                });
    }

    void scale_rows(void *x, const void *alpha, i64 n_row, i64 n) override {
        f32 *p = static_cast<f32 *>(x);
        const f32 *a = static_cast<const f32 *>(alpha);
        par_for(n_row, 1, 2.0 * static_cast<f64>(n_row) * n,
                [&](i64 b, i64 e) {
                    for (i64 i = b; i < e; i++) {
                        f32 *row = p + i * n;
                        const f32 s = a[i];
                        for (i64 j = 0; j < n; j++) row[j] *= s;
                    }
                });
    }

    void qwen3_next_split(void *q, void *gate, const void *packed, i64 n_tok,
                          i64 n_head, i64 hd) override {
        const f32 *p = static_cast<const f32 *>(packed);
        f32 *qd = static_cast<f32 *>(q);
        f32 *gd = static_cast<f32 *>(gate);
        auto one = [&](i64 t) {
            for (i64 h = 0; h < n_head; h++) {
                const f32 *src = p + (t * n_head + h) * 2 * hd;
                f32 *dst = qd + (t * n_head + h) * hd;
                std::memcpy(dst, src, static_cast<size_t>(hd) * sizeof(f32));
                if (gd)
                    std::memcpy(gd + (t * n_head + h) * hd, src + hd,
                                static_cast<size_t>(hd) * sizeof(f32));
            }
        };
        // Callers must not alias `packed` (see the contract in backend.hpp):
        // the row being written is half the width of the row being read, so a
        // token's output lands inside another token's input. The scalar
        // backend can still be exact where the device cannot — a token's
        // writes only ever hit the sources of the tokens before it, so
        // walking tokens in order is safe — and it is the reference, so it
        // should not be the one that quietly races.
        if (qd == p) {
            for (i64 t = 0; t < n_tok; t++) one(t);
            return;
        }
        par_for(n_tok, 1, 2.0 * static_cast<f64>(n_tok) * n_head * hd,
                [&](i64 b, i64 e) {
                    for (i64 t = b; t < e; t++) one(t);
                });
    }

    void conv1d_silu(void *out, const void *in, void *state, const void *kern,
                     DType wt, i64 n_tok, i64 chan, i64 ksize,
                     const void *bias = nullptr) override {
        if (ksize <= 0 || ksize > kMaxConvKernel) return;
        const f32 *xp = static_cast<const f32 *>(in);
        f32 *st = static_cast<f32 *>(state);
        f32 *o = static_cast<f32 *>(out);
        const i64 keep = ksize - 1;
        // One tap value per (tap, channel), dequantized once per call — the
        // kernel is tiny and every channel reads all of it.
        const i64 nk = ksize * chan;
        std::vector<f32> w(static_cast<size_t>(nk));
        if (wt == DType::F32) {
            std::memcpy(w.data(), kern, static_cast<size_t>(nk) * sizeof(f32));
        } else {
            dequant_row(wt, kern, w.data(), nk);
        }
        // Channels are independent down the whole time axis, and each worker
        // touches only its own column of `state`, so the pool can split here.
        par_for(chan, 32, 4.0 * static_cast<f64>(n_tok) * chan * ksize,
                [&](i64 cb, i64 ce) {
                    for (i64 c = cb; c < ce; c++) {
                        // The ksize-1 stored steps, then this call's rows, held
                        // in registers rather than re-read: the engine convolves
                        // IN PLACE (out == in), and the receptive field of row t
                        // reaches back ksize-1 rows that earlier outputs have
                        // already overwritten. A window that slides forward is
                        // the only way to read those rows before they die.
                        f32 win[kMaxConvKernel];
                        for (i64 i = 1; i < ksize; i++)
                            win[i] = st[(ksize - 1 - i) * chan + c];
                        win[0] = xp[c];
                        const f32 b = bias ? static_cast<const f32 *>(bias)[c] : 0.0f;
                        for (i64 t = 0; t < n_tok; t++) {
                            f32 acc = b;
                            for (i64 j = 0; j < ksize; j++)
                                acc += w[static_cast<size_t>(j * chan + c)] * win[keep - j];
                            o[t * chan + c] = silu_scalar(acc);
                            for (i64 i = ksize - 1; i > 0; i--) win[i] = win[i - 1];
                            if (t + 1 < n_tok) win[0] = xp[(t + 1) * chan + c];
                        }
                        // The window now spans X(n_tok + keep - i), so these are
                        // the rows the next call continues from.
                        for (i64 j = 0; j < keep; j++)
                            st[j * chan + c] = win[keep - j];
                    }
                });
    }

    // The scalar Mamba-2 SSD scan — the reference oracle the device kernel is
    // checked against. One thread-per-slot recurrence per block-methodology
    // claim in the header above: slot (h, p, s) owns an f32 cell of `state`
    // (layout [head][p][s]), walks the token loop in order, and writes nothing
    // but out[t, pair] — so the token sequence is the only sequential part.
    //
    // Inputs are regions of one fused post-conv post-SiLU row of conv_dim
    // elements per token; a/b/dp are per-head vectors; `out` is dense
    // [n_tok, inner]. The GDN path keeps f32 state at higher precision than
    // its f16 activations; this path keeps it too, for the same reason.
    void ssd_scan(void *out, void *state, const void *xbc_v, const void *dt_v,
                   const void *a_v, const void *db_v, const void *dp_v,
                   i64 n_tok, i64 n_head, i64 head_dim, i64 d_state,
                   i64 n_group, i64 inner, i64 conv_dim, bool has_d) override {
        f32 *o = static_cast<f32 *>(out);
        f32 *S = static_cast<f32 *>(state);
        const f32 *xbc = static_cast<const f32 *>(xbc_v);
        const f32 *dtv = static_cast<const f32 *>(dt_v);
        const f32 *a = static_cast<const f32 *>(a_v);
        const f32 *db = static_cast<const f32 *>(db_v);
        const f32 *dp = has_d ? static_cast<const f32 *>(dp_v) : nullptr;

        // dt = softplus(dt_raw + db[h]), the step size. The recurrence is
        //   h' = exp(a[h]*dt) * h + dt * u * B[g,s]
        // -- the DECAY takes A, the input term does NOT. ggml's reference:
        //   dA = expf(dt_soft_plus * A[h]);
        //   x_dt = x[ii] * dt_soft_plus;
        //   s    = s*dA + B*x_dt;
        // Both backends used to scale the input term by a[h]*dt as well, which
        // is invisible to a device-vs-CPU test because they shared it.
        auto dt_step = [&](i64 t, i64 h) {
            const f32 x = dtv[t * n_head + h] + db[h];
            // Same tail behaviour as the device d_softplus: above 20 it is v
            // to f32, below -20 it is exp(v), else the real logarithm.
            return x > 20.0f ? x : (x < -20.0f ? std::exp(x)
                                               : std::log1p(std::exp(x)));
        };

        // Per-slot recurrences. Slots partition over (h, p): one state cell +
        // its own output accumulator, so the pool can split over heads with
        // whole-head ownership. The token loop inside a head is sequential.
        par_for(n_head, 1, 8.0 * static_cast<f64>(n_tok) * n_head * head_dim *
                               d_state,
                [&](i64 hb, i64 he) {
                    std::vector<f32> y(n_tok);
                    for (i64 h = hb; h < he; h++) {
                        f32 *Sh = S + static_cast<size_t>(h) * head_dim * d_state;
                        const i64 g = h * n_group / n_head; // grouped state
                        for (i64 t = 0; t < n_tok; t++) {
                            const f32 dtv_ = dt_step(t, h);
                            const f32 decay = std::exp(a[h] * dtv_);
                            const f32 *Bt = xbc + t * conv_dim + inner + g * d_state;
                            const f32 *Ct = Bt + n_group * d_state;
                            f32 *ot = o + (t * n_head + h) * head_dim;
                            for (i64 p = 0; p < head_dim; p++) {
                                const f32 up = xbc[t * conv_dim + h * head_dim + p];
                                f32 *Sp = Sh + p * d_state;
                                f32 acc = 0;
                                for (i64 s = 0; s < d_state; s++) {
                                    f32 cell = Sp[s] * decay + dtv_ * up * Bt[s];
                                    Sp[s] = cell;
                                    acc += cell * Ct[s];
                                }
                                // D is per HEAD and broadcast over the
                                // head's channels: the reference adds
                                // ggml_mul(x, ssm_d) with x [head_dim, n_head]
                                // and ssm_d [1, n_head], so the head index
                                // selects the scalar. n_head == head_dim on the
                                // 30B file, which is exactly why a per-position
                                // read here survives every test that uses that
                                // one shape.
                                ot[p] = acc + (dp ? dp[h] * up : 0.0f);
                            }
                        }
                    }
                });
    }

    void delta_rule(void *out, void *state, const void *q, const void *k,
                    const void *v, const void *g, const void *beta, i64 n_tok,
                    i64 n_k_head, i64 n_v_head, i64 d_state, i64 hd,
                    i64 row_stride) override {
        f32 *S = static_cast<f32 *>(state);
        const f32 *qp = static_cast<const f32 *>(q);
        const f32 *kp = static_cast<const f32 *>(k);
        const f32 *vp = static_cast<const f32 *>(v);
        const f32 *gp = static_cast<const f32 *>(g);
        const f32 *bp = static_cast<const f32 *>(beta);
        f32 *op = static_cast<f32 *>(out);
        // Value heads own disjoint state, so they are the parallel units; the
        // token loop inside one head is sequential by definition.
        par_for(n_v_head, 1, 8.0 * static_cast<f64>(n_tok) * n_v_head * d_state * hd,
                [&](i64 hb, i64 he) {
                    std::vector<f32> kv(static_cast<size_t>(hd));
                    for (i64 h = hb; h < he; h++) {
                        f32 *Sh = S + h * d_state * hd;
                        const i64 kh = h % n_k_head; // tiled key head
                        for (i64 t = 0; t < n_tok; t++) {
                            const f32 decay = std::exp(gp[t * n_v_head + h]);
                            const f32 bta = bp[t * n_v_head + h];
                            const f32 *qt = qp + t * row_stride + kh * d_state;
                            const f32 *kt = kp + t * row_stride + kh * d_state;
                            const f32 *vt = vp + t * row_stride + h * hd;

                            // S <- S * exp(g)
                            for (i64 i = 0; i < d_state; i++) {
                                f32 *row = Sh + i * hd;
                                for (i64 j = 0; j < hd; j++) row[j] *= decay;
                            }
                            // kv = S^T k  (kv[j] = sum_i S[i, j] k[i])
                            for (i64 j = 0; j < hd; j++) {
                                f32 acc = 0;
                                for (i64 i = 0; i < d_state; i++)
                                    acc += Sh[i * hd + j] * kt[i];
                                kv[static_cast<size_t>(j)] = acc;
                            }
                            // S += k (x) d, d = (v - kv) * beta
                            for (i64 i = 0; i < d_state; i++) {
                                f32 *row = Sh + i * hd;
                                const f32 ki = kt[i];
                                for (i64 j = 0; j < hd; j++)
                                    row[j] += ki * (vt[j] - kv[static_cast<size_t>(j)]) * bta;
                            }
                            // out = S^T q
                            f32 *ot = op + (t * n_v_head + h) * hd;
                            for (i64 j = 0; j < hd; j++) {
                                f32 acc = 0;
                                for (i64 i = 0; i < d_state; i++)
                                    acc += Sh[i * hd + j] * qt[i];
                                ot[j] = acc;
                            }
                        }
                    }
                });
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
    // partial RoPE); `scale` divides the position (linear scaling). `neox`
    // switches to the half-split convention the NeoX family is exported with:
    // pair i is channel i with channel i + rot, where rot is the number of
    // rotated pairs. Both conventions leave every other channel untouched, so
    // the only difference is *which* channels a pair is made of — and since
    // only the relative rotation between q and k reaches the score, that is
    // the difference between the reference answer and a plausible wrong one.
    static void rope_apply(f32 *x, i64 n_heads, i64 hd, i64 n_tok, i64 pos0,
                           const f32 *inv_freq, f32 scale, f32 frac, bool neox) {
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
                            const i64 ia = neox ? i : 2 * i;
                            const i64 ib = neox ? i + rot : 2 * i + 1;
                            const f32 a = hx[ia], bb = hx[ib];
                            hx[ia] = a * c - bb * sn;
                            hx[ib] = a * sn + bb * c;
                        }
                    }
                });
    }

    // OLMoE's convention: one RMS over the whole projected row (n = n_head *
    // hd), the weight spanning that row. One row per token, so the unit of
    // parallelism is the token, not the (token, head) pair.
    static void row_norm(f32 *x, const f32 *w, i64 n_rows, i64 n, f32 eps) {
        par_for(n_rows, 1, 4.0 * static_cast<f64>(n_rows) * n,
                [&](i64 b, i64 e) {
                    for (i64 r = b; r < e; r++) {
                        f32 *xr = x + r * n;
                        f32 ss = 0;
                        for (i64 i = 0; i < n; i++) ss += xr[i] * xr[i];
                        const f32 inv = 1.0f / std::sqrt(ss / static_cast<f32>(n) + eps);
                        for (i64 i = 0; i < n; i++) xr[i] = xr[i] * inv * w[i];
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
