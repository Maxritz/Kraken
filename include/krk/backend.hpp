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
