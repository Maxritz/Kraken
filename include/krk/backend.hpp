// ============================================================================
//  backend.hpp — the compute abstraction every runtime implements.
//
//  Two implementations ship:
//    * HipBackend  (src/hip/backend_hip.hip) — the real one, WMMA/packed math
//    * CpuBackend  (src/backend_cpu.cpp)     — portable oracle used by tests
//
//  Activations live in whichever element type the backend advertises
//  (F16 on the GPU, F32 on the CPU). Weights are never materialized: they stay
//  in their GGUF block format and the GEMM/GEMV kernels dequantize them inline.
// ============================================================================
#ifndef KRK_BACKEND_HPP
#define KRK_BACKEND_HPP

#include "krk/common.hpp"
#include "krk/quant.hpp"

namespace krk {

enum class BackendKind { CPU, HIP };

// RDNA family generation, used to pick the compute path.
enum class GfxFamily : int {
    Unknown = 0,
    Rdna1 = 10,
    Rdna2 = 102, // gfx1030-1032: no WMMA, packed math + v_dot2
    Rdna3 = 110, // gfx1100-1103 / gfx1150-1153: WMMA 16x16x16 f16/bf16/int8
    Rdna4 = 120, // gfx1200/1201: WMMA gfx12 variant + fp8/bf16, K-split layout
};

struct DeviceCaps {
    std::string name;   // "AMD Radeon RX 9070 XT"
    std::string gfx;    // "gfx1201"
    GfxFamily family = GfxFamily::Unknown;
    int wave_size = 32;
    int cu_count = 0;
    int lds_bytes = 65536;
    size_t vram_total = 0;
    size_t vram_free = 0;
    bool has_wmma = false;      // gfx11+ dense WMMA
    bool has_wmma_gfx12 = false; // gfx12 K-split layout + fp8
    bool has_dot2 = false;      // v_dot2_f32_f16 / packed fp16 rate
    bool unified_mem = false;   // APU (Ryzen AI) — affects tiling choices

    const char *family_name() const {
        switch (family) {
            case GfxFamily::Rdna1: return "RDNA1";
            case GfxFamily::Rdna2: return "RDNA2";
            case GfxFamily::Rdna3: return "RDNA3";
            case GfxFamily::Rdna4: return "RDNA4";
            default: return "unknown";
        }
    }
};

// ---------------------------------------------------------------------------
// geometry passed to the attention kernel
// ---------------------------------------------------------------------------

struct AttnDesc {
    i64 n_head = 0;        // query heads
    i64 n_kv = 0;          // key/value heads (GQA)
    i64 hd = 0;            // head dimension
    i64 n_tok = 0;         // query rows in this call
    i64 pos0 = 0;          // position of query row 0
    i64 layer = 0;
    i64 layer_stride = 0;  // elements between layers in the KV cache
    i64 pos_stride = 0;    // elements between positions in the KV cache
    f32 scale = 1.0f;      // 1/sqrt(hd)
    bool causal = true;
};

// ---------------------------------------------------------------------------

class Backend {
public:
    virtual ~Backend() = default;

    virtual BackendKind kind() const = 0;
    virtual const DeviceCaps &caps() const = 0;

    // Element type used for activations (F16 on GPU, F32 on CPU).
    virtual DType act_type() const = 0;
    virtual size_t act_size() const = 0; // bytes per activation element

    // ---- memory ----------------------------------------------------------
    virtual void *alloc(size_t bytes) = 0;
    virtual void release(void *p) = 0;
    // Host-side allocation for the expert cache's second tier. The GPU backend
    // returns page-locked memory, so moving an evicted expert between VRAM and
    // the tier is a DMA in either direction instead of a staged copy through a
    // pageable bounce buffer. Distinct from alloc() because pinned pages are a
    // scarce, non-swappable resource: spending them on activations would trade
    // a 12 GiB VRAM problem for a host-RAM one. The default is ordinary aligned
    // host memory, which is all the CPU reference backend needs.
    virtual void *alloc_host(size_t bytes) { return host_alloc(bytes ? bytes : 1); }
    virtual void release_host(void *p) { host_free(p); }
    virtual void upload(void *dst, const void *src, size_t bytes, size_t off = 0) = 0;
    virtual void download(void *dst, const void *src, size_t bytes, size_t off = 0) = 0;
    virtual void fill0(void *dst, size_t bytes) = 0;
    virtual void sync() = 0;
    // Copy n activation elements to host f32 (handles F16/F32 conversion).
    virtual void download_f32(f32 *dst, const void *src, i64 n) = 0;
    // Public device name for logging.
    virtual std::string describe() const = 0;

    // ---- ops -------------------------------------------------------------

    // out[n_tok, n_embd] <- rows of tok_embd selected by tokens.
    virtual void embed(void *out, const void *tok_embd, DType tt, i64 n_vocab,
                       i64 n_embd, const i32 *tokens, i64 n_tok) = 0;

    // out[rows, n] = rmsnorm(x[rows, n]) * w[n], computed in f32.
    virtual void rmsnorm(void *out, const void *x, const f32 *w, i64 rows, i64 n,
                         f32 eps) = 0;

    // out[rows, n_out] = x[rows, n_in] * W[n_out, n_in]^T ; W is quantized.
    virtual void gemm(void *out, const void *x, const void *w, DType wt,
                      i64 n_out, i64 n_in, i64 rows) = 0;

    // A group of n_mat projections that all read the same activation
    // x and have no dependency between them (a decode layer's q/k/v,
    // or its gate/up), issued as one launch. Only the M=1 decode row
    // is fused; any backend, dtype mix or prefill shape falls back to
    // one gemm() per matrix. All matrices must share `wt`. Returns
    // true when the fused path ran.
    virtual bool gemm_group(void *const out[], const void *x,
                            const void *const w[], DType wt,
                            const i64 n_out[], i64 n_in, int n_mat,
                            i64 rows) {
        for (int m = 0; m < n_mat; m++)
            gemm(out[m], x, w[m], wt, n_out[m], n_in, rows);
        return false;
    }

    // NeoX-style rotary embedding applied in place to q and k.
    virtual void rope(void *q, void *k, i64 n_head, i64 n_kv, i64 hd, i64 n_tok,
                      i64 pos0, const f32 *inv_freq, f32 scale, f32 frac) = 0;

    // Per-head RMSNorm on q/k (Qwen3, Gemma3). No-op when w == nullptr.
    virtual void qk_norm(void *q, void *k, const f32 *wq, const f32 *wk,
                         i64 n_head, i64 n_kv, i64 hd, i64 n_tok, f32 eps) {
        (void)q; (void)k; (void)wq; (void)wk; (void)n_head; (void)n_kv;
        (void)hd; (void)n_tok; (void)eps;
    }

    // Append activations to the KV cache at positions pos0..pos0+n_tok-1.
    virtual void kv_append(void *kcache, void *vcache, const void *k,
                           const void *v, const AttnDesc &d) = 0;

    // out[n_tok, n_head*hd] = softmax(q K^T / sqrt(hd)) V
    virtual void attention(void *out, const void *q, const void *kcache,
                           const void *vcache, const AttnDesc &d) = 0;

    // RoPE + KV append + decode attention as ONE launch (the
    // single-query-token decode row only): rotates q in registers,
    // ropes k straight into its cache slot, copies v beside it and
    // attends over the cache including the fresh row, so the k/v
    // rows never round-trip through global memory between the
    // three launches. `inv_freq` is host memory; the backend
    // stages it. rope_scale divides the position (LLaMA-style RoPE
    // scaling) and rope_frac is the rotation fraction. Returns
    // true when the fused chain ran; anything it cannot express
    // (multi-token rows, head widths the packed lane slice cannot
    // cover) falls back to the separate rope() + kv_append() +
    // attention() calls the engine then runs instead.
    virtual bool attn_fused_chain(void *out, void *q, void *kcache,
                                    void *vcache, const void *k,
                                    const void *v, const AttnDesc &d,
                                    const f32 *inv_freq,
                                    f32 rope_scale, f32 rope_frac) {
        (void)out; (void)q; (void)kcache; (void)vcache; (void)k;
        (void)v; (void)d; (void)inv_freq; (void)rope_scale;
        (void)rope_frac;
        return false;
    }

    // out = silu(gate) * up
    virtual void silu_mul(void *out, const void *gate, const void *up, i64 n) = 0;

    // out += bias  for each of `rows` output rows of width n (Qwen2/Phi QKV biases)
    virtual void add_bias_rows(void *out, const f32 *bias, i64 n, i64 rows) = 0;

    // a += b  (activation elementwise)
    virtual void add_inplace(void *a, const void *b, i64 n) = 0;

    // dst <- src, n activation elements
    virtual void copy_act(void *dst, const void *src, i64 n) = 0;

    // dst[i] += alpha * src[i]. The MoE path accumulates routed-expert outputs
    // with their softmax gate weight through this.
    virtual void axpy(void *dst, const void *src, f32 alpha, i64 n) = 0;

    // ---- MoE batching helpers --------------------------------------------
    //
    // The batched expert path permutes the tokens that picked an expert into a
    // contiguous block so one GEMM serves all of them:
    //
    //   dst[i * n + j] = src[rows[i] * n + j]   for i in [0, n_rows)
    //
    // rows points at n_rows device-visible i32 row indices (see upload_i32).
    virtual void gather_rows(void *dst, const void *src, const i32 *rows,
                             i64 n_rows, i64 n) = 0;

    // The un-permute plus weighted accumulation, fused so nothing is read back:
    //
    //   dst[rows[i] * n + j] += alpha[i] * src[i * n + j]
    //
    // alpha points at n_rows host f32 weights; they are staged to the device
    // once per call. Every expert writes distinct rows of dst, so no atomics
    // are needed.
    virtual void scatter_axpy_rows(void *dst, const void *src, const i32 *rows,
                                   const f32 *alpha, i64 n_rows, i64 n) = 0;

    // Stages n i32 values to device memory. The batched path builds its row
    // plan on the host and ships it through this; a no-op on the CPU backend.
    virtual void upload_i32(i32 *dst, const i32 *src, i64 n) = 0;

    // ---- gated delta net (Qwen3.5 / Qwen3-Next linear attention) ---------
    //
    // A recurrent layer keeps state across tokens, so unlike every op above
    // these are only correct for a backend that can address the state buffers
    // it is handed. The default bodies therefore do nothing and the flag below
    // is what the model loader consults: a backend that reports false is
    // refused a recurrent model at load time, loudly, rather than being handed
    // a checkpoint it would silently mis-decode. The CPU backend implements
    // all of them and is the reference oracle.
    virtual bool gdn_supported() const { return false; }

    // Per-head L2 normalization, in place or via a distinct output:
    //   o[t, h, i] = x[t, h, i] / sqrt(sum_j x[t, h, j]^2 + eps)
    // `stride` is the element distance between consecutive tokens, so a region
    // carved out of a wider fused row (the gated delta net's post-conv buffer)
    // is handled without a copy. o and x may be the same pointer.
    virtual void l2norm(void *o, const void *x, i64 n_tok, i64 n_head, i64 hd,
                        i64 stride, f32 eps) {
        (void)o; (void)x; (void)n_tok; (void)n_head; (void)hd; (void)stride;
        (void)eps;
    }

    // x <- sigmoid(x), and x <- softplus(x) = log1p(exp(x)). Both in place.
    virtual void sigmoid_act(void *x, i64 n) { (void)x; (void)n; }
    virtual void softplus_act(void *x, i64 n) { (void)x; (void)n; }

    // x <- alpha * x over n_tok rows of `width` elements spaced `stride` apart,
    // and the elementwise product a[i] *= b[i] (both dense).
    virtual void scale_act(void *x, f32 alpha, i64 n_tok, i64 width, i64 stride) {
        (void)x; (void)alpha; (void)n_tok; (void)width; (void)stride;
    }
    virtual void mul_act(void *a, const void *b, i64 n) { (void)a; (void)b; (void)n; }

    // Per-column (per-head) vectors broadcast down the rows:
    //   x[i, j] += bias[j]     and    x[i, j] *= col[j]
    // Used for the ssm_dt bias and the A_log decay scale.
    virtual void add_bias_cols(void *x, const f32 *bias, i64 n_row, i64 n_col) {
        (void)x; (void)bias; (void)n_row; (void)n_col;
    }
    virtual void scale_cols(void *x, const f32 *col, i64 n_row, i64 n_col) {
        (void)x; (void)col; (void)n_row; (void)n_col;
    }
    // Per-row scalars from an activation buffer: x[i, :] *= alpha[i].
    virtual void scale_rows(void *x, const void *alpha, i64 n_row, i64 n) {
        (void)x; (void)alpha; (void)n_row; (void)n;
    }

    // Splits Qwen3.5's packed attention query. The projection interleaves each
    // head's query and output gate inside one 2*hd block, so neither is
    // contiguous and attention cannot be fed the packed buffer:
    //   q[t, h, i]    <- packed[t, h*2*hd + i]
    //   gate[t, h, i] <- packed[t, h*2*hd + hd + i]
    // `gate` may be null when only the query part is wanted. `q` may alias
    // `packed`: every write lands at a lower offset than the read it comes
    // from, so an in-place unpack is safe.
    virtual void qwen3_next_split(void *q, void *gate, const void *packed,
                                  i64 n_tok, i64 n_head, i64 hd) {
        (void)q; (void)gate; (void)packed; (void)n_tok; (void)n_head; (void)hd;
    }

    // The gated-delta-net short convolution, depthwise over `chan` channels:
    //   out[t, c] = silu( sum_{j<ksize} kern[j, c] * X[t + j, c] )
    // where X is `state` (the ksize-1 previous steps) followed by the n_tok new
    // rows in `in`, and kern is [ksize, chan] with the newest tap last, stored
    // in the given (possibly quantized) weight format. On return `state` holds
    // the last ksize-1 rows of that window, which is what makes the next call
    // continue the same sequence. ksize <= 16.
    virtual void conv1d_silu(void *out, const void *in, void *state,
                             const void *kern, DType wt, i64 n_tok, i64 chan,
                             i64 ksize) {
        (void)out; (void)in; (void)state; (void)kern; (void)wt;
        (void)n_tok; (void)chan; (void)ksize;
    }

    // The delta rule, run over n_tok consecutive tokens of one sequence:
    //   S *= exp(g[t]);  kv = S^T k;  d = (v - kv) * beta;  S += k (x) d
    //   out[t] = S^T q
    // `state` is f32 [n_v_head, d_state, hd] and persists across calls. q, k
    // and v are the three regions of one fused post-conv row — token t starts
    // at t * row_stride, and each holds n_k_head/n_k_head/n_v_head heads of
    // d_state/hd — while `out`, `g` and `beta` are dense. Qwen3-Next runs 16
    // key heads over 32 value heads by tiling, so value head h reads key head
    // h % n_k_head.
    virtual void delta_rule(void *out, void *state, const void *q, const void *k,
                            const void *v, const void *g, const void *beta,
                            i64 n_tok, i64 n_k_head, i64 n_v_head, i64 d_state,
                            i64 hd, i64 row_stride) {
        (void)out; (void)state; (void)q; (void)k; (void)v; (void)g;
        (void)beta; (void)n_tok; (void)n_k_head; (void)n_v_head;
        (void)d_state; (void)hd; (void)row_stride;
    }
};

// ---------------------------------------------------------------------------
// factories
// ---------------------------------------------------------------------------

// Always available: the scalar reference backend.
Backend *make_cpu_backend();

// Compiled only when KRK_ENABLE_HIP is defined (see CMakeLists.txt).
Backend *make_hip_backend(int device_index, std::string *err);
// Number of visible HIP devices, or 0 when this build has no HIP support.
int hip_device_count();
// Human-readable runtime/version string for logging, or "unavailable".
std::string hip_runtime_version();

} // namespace krk

#endif // KRK_BACKEND_HPP
