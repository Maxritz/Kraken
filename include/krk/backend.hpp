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

#include <cstdlib> // std::abort in hadamard_act's default body
#include <cstring>
#include <functional>
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
    size_t vram_free = 0; // as of device enumeration, not a live figure
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
    // Sliding-window bound: when > 0 a query attends only to the `window`
    // most recent keys, i.e. key j is visible when j >= n_keys - window.
    // 0 means no bound, the whole (causal) prefix. laguna's hybrid stacks
    // need this: the windowed layers carry a 512-key window while the full
    // layers see everything, so the bound is per layer, not per model.
    i64 window = 0;
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
    // Live device memory, when the backend can ask the driver. caps().vram_free
    // is a fixed snapshot taken at enumeration -- on the HIP backend it is
    // total * 0.9, not free at all -- so anything that sizes itself against
    // free VRAM (the expert budget does) has to ask, or it over-commits a
    // card something else is already using.
    virtual size_t device_free_bytes() const { return caps().vram_free; }
    virtual size_t device_total_bytes() const { return caps().vram_total; }

    virtual void release_host(void *p) { host_free(p); }

    // Host-side allocation for the expert cache's WARM tier: ordinary pageable
    // RAM, the largest tier in the hierarchy and the one sized in tens of GiB.
    // Deliberately NOT alloc_host(): pinned pages are non-swappable and drawn
    // from a small driver pool, so a WARM cache built out of them charges the
    // whole machine -- and every DMA path that genuinely needs a pin -- for a
    // cache that must stay free to shrink. The default forwards to the pinned
    // allocator, which on any backend that does not override it is aligned
    // malloc; the HIP backend overrides it with plain pageable memory.
    virtual void *alloc_host_pageable(size_t bytes) {
        return alloc_host(bytes ? bytes : 1);
    }
    virtual void release_host_pageable(void *p) { release_host(p); }
    // Releases host memory an asynchronous copy may still be reading. The
    // default frees it immediately; a backend that stages copies keeps the
    // pointer until the copy has landed, because freeing the source of an
    // in-flight DMA is a use-after-free, and with a greedy decode it shows up
    // as text that is plausible but different from run to run.
    virtual void release_host_deferred(void *p) { release_host_pageable(p); }
    // Fills `dst` (pageable host memory) with `bytes` taken from `mapped_src`,
    // a pointer into the weight mapping. A backend that can read the file
    // directly (see set_weight_pull) does so, because a cold expert should cost
    // one read() into RAM and not a page fault per 4 KiB of mapping; backends
    // without that path copy the mapping. Returns true when `dst` was filled.
    virtual bool read_host(void *dst, const void *mapped_src, size_t bytes) {
        std::memcpy(dst, mapped_src, bytes);
        return true;
    }
    // One file read: `dst` gets `bytes` from the mapping-relative pointer.
    struct ReadReq {
        void *dst;
        const void *mapped_src;
        size_t bytes;
    };
    // A batch of reads issued together, so a backend that owns more than one
    // reader can overlap them. The shape of an expert miss is a *set* of
    // 0.5-2 MiB reads — one expert's three slices, or every expert a layer
    // routed — and reading them one at a time is queue depth 1 on a single
    // file handle: measured on this machine (RX 9070 XT host, NVMe) 593 MiB/s
    // for 1.82 MiB reads, against 1175 MiB/s with two readers, 1446 MiB/s with
    // four, and 3400 MiB/s sequential. Backends that do not override this read
    // one request at a time.
    virtual void read_host_batch(const ReadReq *reqs, int n) {
        for (int i = 0; i < n; i++)
            if (reqs[i].bytes)
                read_host(reqs[i].dst, reqs[i].mapped_src, reqs[i].bytes);
    }
    // Host->device copy whose source is pageable host memory: the WARM -> HOT
    // promotion path. HIP routes it through a small, bounded pinned staging
    // ring so the transfer is a real DMA rather than the driver's internal
    // bounce-buffer copy; backends without a ring forward to upload().
    virtual void upload_paged(void *dst, const void *src, size_t bytes) {
        upload(dst, src, bytes);
    }
    // A batch of pageable-source copies issued together, so a backend with a
    // staging ring can overlap them: one expert is three slices, and three
    // separate calls serialise on the same first buffer. Backends without a
    // ring just forward.
    virtual void upload_paged_batch(void *const dst[], const void *const src[],
                                    const size_t bytes[], int n) {
        for (int i = 0; i < n; i++)
            if (dst[i] && src[i] && bytes[i]) upload_paged(dst[i], src[i], bytes[i]);
    }

    // Small-object device allocations, pooled by the expert cache. A
    // promotion is a handful of ~1-2 MiB slices and the default
    // alloc/release pair is hipMalloc/hipFree, which are
    // device-synchronising: measured on the prefill of a 40-layer MoE, the
    // expert cache issued 13198 uploads and 2594 ms of device time went into
    // them against 1330 ms of device idle. Backends without a pool just
    // forward to alloc/release.
    virtual void *alloc_pooled(size_t bytes) { return alloc(bytes); }
    virtual void release_pooled(void *p, size_t bytes) {
        (void)bytes;
        if (p) release(p);
    }
    virtual void upload(void *dst, const void *src, size_t bytes, size_t off = 0) = 0;
    // Direct copy with no staging side effects. The HIP backend's read()-based
    // weight path calls this per chunk; a backend without that path can simply
    // forward to upload().
    virtual bool upload_now(void *dst, const void *src, size_t bytes, size_t off = 0) {
        upload(dst, src, bytes, off);
        return true;
    }
    // Optional hook for pull-style weight loading; see HipBackend::set_weight_pull.
    //
    // Large weights are better read() out of the file than DMA'd straight from
    // the mapping: a cold mapping faults one 4 KiB page at a time (measured
    // 1208.8 ms to get the 8B into RAM that way, 3.87 GB/s), and the fault path
    // -- not the transfer -- is the cost. The loader owns the mapping, so it
    // hands over the path and the mapping's address range; the backend then owns
    // the whole pipeline (independent handles, pinned buffers, side streams) and
    // overlaps the reads with the copies. `map_base`/`map_size` describe the
    // mapping and `data_off` its tensor data section, which together let the
    // backend turn a source pointer back into a file offset. A backend that
    // ignores this keeps the direct copy.
    virtual void set_weight_pull(const std::string &path, const void *map_base,
                                 size_t map_size, size_t data_off) {
        (void)path; (void)map_base; (void)map_size; (void)data_off;
    }
    virtual void download(void *dst, const void *src, size_t bytes, size_t off = 0) = 0;
    virtual void fill0(void *dst, size_t bytes) = 0;
    virtual void sync() = 0;

    // Completion markers, for a resource that is reused before the work
    // already queued against it has finished.
    //
    // Everything in this engine runs on the default stream, so submitting work
    // is ordered -- but ORDERED SUBMISSION IS NOT COMPLETION. A kernel reading
    // buffer B can still be executing when the host decides B is free and
    // hands it to someone else. Writing B then is a race that corrupts GPU
    // state; on this box it presented as a driver reset, not as wrong numbers.
    //
    // Protocol: make_marker() allocates a token, wait_marker() blocks until
    // the token's previous record has completed, record_marker() marks the
    // point in the stream AFTER everything submitted so far. So the engine
    // waits before touching a reused buffer and records once the work that
    // uses it has been submitted.
    //
    // The default is deliberately conservative and correct: a null token, a
    // no-op record, and a wait that falls back to a full sync(). A backend
    // that does not implement markers is slower, never wrong.
    virtual void *make_marker() { return nullptr; }
    virtual void wait_marker(void *marker) {
        (void)marker;
        sync();
    }
    virtual void record_marker(void *marker) { (void)marker; }
    virtual void release_marker(void *marker) { (void)marker; }
    virtual bool poll_marker(void *marker) { (void)marker; return true; }
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

    // Top-k of one logits row, on the device, so the row never has to be
    // downloaded. `logits` is the output head's buffer, which means it is in
    // the backend's activation type (`act_type()`, f16 here) and not
    // necessarily f32 — there is no dtype argument, so a caller that hands
    // over some other layout gets a plausible answer to the wrong question.
    // Applies `softcap` and the repetition penalty (selected by
    // the bitmask `pen_mask`, one bit per token id) BEFORE ranking — a penalty
    // can raise a token, so ranking first would miss candidates.
    //
    // Writes `k` ids and values, descending, ties broken toward the lower id
    // (the host argmax rule); `mass_out` receives sum(exp((v-max)/temp)) over
    // the whole row so the caller can tell whether the returned set carries
    // enough probability mass for an exact top-p cut. `mass_out` may be null.
    //
    // `cand_cap` bounds the device candidate buffer; *count_out receives how
    // many elements landed at or above the cut, so a value above cand_cap
    // means the set was truncated and the caller should fall back.
    virtual void logits_topk(const void *logits, i64 n, i32 k, const void *pen_mask,
                             f32 rep_pen, f32 softcap, f32 temp, i32 *ids, f32 *vals,
                             f32 *mass_out, void *scratch, i32 cand_cap,
                             i64 *count_out) = 0;

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

    // Rotary embedding applied in place to q and k. `inv_freq` is the
    // rope_dim/2 frequency table and `frac` the rotated fraction of the
    // head, so rot = hd*frac/2 pairs are rotated and the rest are copied
    // through. `neox` picks which channels a pair is made of: false pairs
    // (2i, 2i+1) — llama.cpp's "norm" layout — and true pairs channel i
    // with i + rot, the half-split layout of the NeoX family. It is a
    // property of the file's architecture (ModelConfig::rope_neox) and it
    // has to reach the kernel: only the relative rotation between a query
    // and a key reaches the score, so the wrong pairing stays finite.
    virtual void rope(void *q, void *k, i64 n_head, i64 n_kv, i64 hd, i64 n_tok,
                      i64 pos0, const f32 *inv_freq, f32 scale, f32 frac,
                      bool neox) = 0;

    // Per-head RMSNorm on q/k (Qwen3, Gemma3). No-op when w == nullptr.
    // `wide` selects OLMoE's convention: one RMS over the whole projected row
    // (n_head * hd), the weight spanning the row. False is the Qwen3/Gemma3
    // per-head convention, the weight spanning hd.
    virtual void qk_norm(void *q, void *k, const f32 *wq, const f32 *wk,
                         i64 n_head, i64 n_kv, i64 hd, i64 n_tok, f32 eps,
                         bool wide = false) {
        (void)q; (void)k; (void)wq; (void)wk; (void)n_head; (void)n_kv;
        (void)hd; (void)n_tok; (void)eps; (void)wide;
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
    // cover, a NeoX rope pairing) falls back to the separate rope() +
    // kv_append() + attention() calls the engine then runs instead.
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

    // out[rows, n] = rmsnorm(x[rows, n] + res[rows, n]) * w[n], leaving x
    // holding the sum.
    //
    // A layer ends each sub-block with a residual add and begins the next
    // by normalizing the result, so the add writes a buffer the norm reads
    // straight back. One launch instead of two. The default below is
    // exactly those two calls in order, so a backend that does not
    // override it behaves identically to one that never had the fused
    // path at all.
    virtual void add_rmsnorm(void *out, void *x, const void *res,
                             const f32 *w, i64 rows, i64 n, f32 eps) {
        add_inplace(x, res, rows * n);
        rmsnorm(out, x, w, rows, n, eps);
    }

    // dst <- src, n activation elements
    virtual void copy_act(void *dst, const void *src, i64 n) = 0;

    // dst[i] += alpha * src[i]. The MoE path accumulates routed-expert outputs
    // with their softmax gate weight through this.
    virtual void axpy(void *dst, const void *src, f32 alpha, i64 n) = 0;

    // ---- prism.hadamard activation transform (PrismML's llama.cpp fork) --
    //
    // That fork stores a handful of matmul weights in a rotated basis: each
    // weight's INPUT dimension was multiplied by a per-width +-1 sign vector
    // and then by a normalized Sylvester Walsh-Hadamard over consecutive
    // blocks of `block` elements (matrix entries +-1/sqrt(block), so the
    // transform is its own inverse). The runtime therefore transforms the
    // ACTIVATION the same way immediately before such a matmul -- and, for
    // the one table stored rotated (`token_embd.weight`), once after the
    // lookup.
    //
    // The residual stream stays in the primal basis: only a folded matmul's
    // own input is transformed, and it goes into a scratch buffer, because
    // the same activation also feeds weights that are NOT folded (a Qwen3.5
    // delta-net layer's attention-norm output feeds ssm_alpha/ssm_beta
    // untouched, and a MoE router reads it too).
    //
    // The two orders are not the same operation: the fold is signs-then-FWHT,
    // the inverse after a lookup is FWHT-then-signs. `signs_first` picks one.
    struct HadDesc {
        i64 n = 0;                  // row width in elements; n % block == 0
        i64 block = 1024;           // FWHT block size, a power of two
        const f32 *signs = nullptr; // n values of +-1, backend address space
        bool signs_first = true;    // true: fold; false: inverse-after-lookup
        // ssm_out's grouped-V feature perm, applied first of all:
        //   out[i + j*hd + k*hd*rep] = in[i + k*hd + j*hd*nk]
        // for i in [0,hd), j in [0,rep), k in [0,nk). All zero disables it.
        i64 perm_hd = 0;
        i64 perm_nk = 0;
        i64 perm_rep = 0;
    };
    // out[r, 0..n) <- transform(in[r, 0..n)) for r in [0, rows). `out` may
    // alias `in`: rows never interact, and each stage reads an element before
    // it writes it.
    virtual void hadamard_act(void *out, const void *in, i64 rows,
                              const HadDesc &d) {
        (void)out;
        (void)in;
        (void)rows;
        (void)d;
        KRK_ERROR("hadamard_act: this backend does not implement the "
                  "prism.hadamard activation transform");
        std::abort(); // silently not transforming would decode fluent garbage
    }

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
    // `rows` and `alpha` are in this backend's own address space, the same as
    // dst and src: a device backend is handed buffers its caller allocated
    // through alloc(), so a host copy of either is neither made nor expected.
    // Every expert writes distinct rows of dst, so no atomics are needed.
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

    // Per-group RMS normalization, the Mamba-2 checkpoints' ssm_norm:
    //   o[t, g, i] = x[t, g, i] * rsqrt(mean_j x[t, g, j]^2 + eps) * w[g, i]
    // The weight is NOT shared by the rows. The checkpoint stores one vector
    // per group at `w + g * width` (ssm_norm is [width, n_group]), so the row's
    // group selects its own weight, and the reduction spans `width` channels --
    // the group's block, not one head's. Rows are the (token, group) pairs in
    // the order the scan's output already carries them: row r is group
    // r % n_group of token r / n_group, and `stride` is the element distance
    // between tokens (the group blocks of one token are contiguous). o and x
    // may be the same pointer.
    //
    // Plain rmsnorm() cannot express this: it applies ONE weight vector to all
    // `rows`, and the row's required vector changes every n_group rows.
    virtual void rmsnorm_grouped(void *o, const void *x, const f32 *w, i64 n_tok,
                                 i64 n_group, i64 width, i64 stride, f32 eps) {
        (void)o; (void)x; (void)w; (void)n_tok; (void)n_group; (void)width;
        (void)stride; (void)eps;
    }

    // x <- sigmoid(x), and x <- softplus(x) = log1p(exp(x)). Both in place.
    virtual void sigmoid_act(void *x, i64 n) { (void)x; (void)n; }
    virtual void softplus_act(void *x, i64 n) { (void)x; (void)n; }
    // x <- max(x, 0)^2, in place: squared ReLU, the activation nemotron_h_moe
    // builds every expert and shared-expert MLP with (the reference's
    // LLM_FFN_RELU_SQR). It is not the gated SiLU and not close to it -- silu
    // keeps the sign of the input and saturates, relu^2 zeroes every negative
    // pre-activation and squares the rest -- and this family's files carry no
    // expert gate for a SiLU to ride, so without it the routed bank is a
    // plausible-looking, wrong function of the same weights.
    virtual void relu_sqr_act(void *x, i64 n) { (void)x; (void)n; }

    // x <- alpha * x over n_tok rows of `width` elements spaced `stride` apart,
    // and the elementwise product a[i] *= b[i] (both dense).
    virtual void scale_act(void *x, f32 alpha, i64 n_tok, i64 width, i64 stride) {
        (void)x; (void)alpha; (void)n_tok; (void)width; (void)stride;
    }
    virtual void mul_act(void *a, const void *b, i64 n) { (void)a; (void)b; (void)n; }

    // Per-head broadcast product (laguna's attention output gate):
    //   x[t, h, i] *= g[t, h]     for i in [0, hd)
    // `x` is [n_tok, n_head*hd] and `g` is a dense [n_tok, n_head] row-major
    // buffer, so the two have different lengths: mul_act() is elementwise and
    // cannot express this. Applied to the attention result before the output
    // projection. The Qwen3.5 output gate is per *element* and stays on
    // mul_act(); this is the per-head form.
    virtual void mul_head_broadcast(void *x, const void *g, i64 n_tok, i64 n_head,
                                    i64 hd) {
        (void)x; (void)g; (void)n_tok; (void)n_head; (void)hd;
    }

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
    // `gate` may be null when only the query part is wanted. `q` must *not*
    // alias `packed`: a row's two source halves are twice the width of its
    // destination, so an in-place unpack has one head writing over rows its
    // neighbours have not read yet and the block order decides the result.
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
                             i64 ksize, const void *bias = nullptr) {
        (void)out; (void)in; (void)state; (void)kern; (void)wt;
        (void)n_tok; (void)chan; (void)ksize; (void)bias;
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

    // The Mamba-2 SSD scan (nemotron_h / nemotron_h_moe mamba blocks), over
    // n_tok consecutive tokens of one sequence. The engine supplies the
    // post-conv post-SiLU xBC row and a separate dense dt head, so every
    // per-head gate lives INSIDE this op:
    //   xbc   [n_tok, inner + 2*n_group*d_state] rows (activation-typed):
    //         u   j*head_dim + p                       the inner block
    //         B   inner + g*d_state + s                g = h*n_group/n_head
    //         C   inner + (n_group + g)*d_state + s
    //   dt    [n_tok, n_head] the raw dt head, dense (activation-typed) —
    //         separate because the conv consumes a row exactly inner+b_len
    //         wide and the dt head must not ride through it
    //   a     ssm_a [n_head] f32 device values: the raw per-head A (the file
    //         stores -exp(A_log), so decay = exp(a_h * softplus(...)))
    //   db    ssm_dt.bias [n_head] f32 device values
    //   dp    ssm_d [n_head] f32 device values, or null when the file has none
    //   state f32 [n_head, head_dim, d_state] — persists across calls, layout
    //         [head][p][s] at (h*head_dim + p)*d_state + s (NOT the GDN
    //         [head][s][p] order; the scan never reads it any other way)
    //   out   [n_tok, inner] per pair (y + D*u), activation-typed; the silu(z)
    //         gate and the grouped RMSNorm stay outside, in engine acts — the
    //         same split the GDN path draws between delta_rule and
    //         silu_mul/rmsnorm.
    // Per slot (h, p, s), with dt = softplus(dt[t,h] + db[h]) the step size:
    //   h' = exp(a[h] * dt) * h + dt * u[t,h,p] * B[t,g,s]
    //   y[t,h,p] = sum_s C[t,g,s] * h'[p,s]  (+ dp[p] * u[t,h,p] when has_d)
    // The DECAY takes A and the input term does not -- dt scales the input
    // alone, which is ggml's recurrence (dA = expf(dt_soft_plus*A[h]);
    // x_dt = x[ii]*dt_soft_plus). Scaling the input by a[h]*dt too is the
    // plausible-looking variant: it is a per-head constant, so device and CPU
    // agree perfectly while both are wrong.
    // The scan runs in one pass over the token loop, reading only inputs — the
    // state carried is the slot's own register/stack cell. The scalar CPU
    // backend is the reference oracle.
    virtual void ssd_scan(void *out, void *state, const void *xbc,
                          const void *dt, const void *a, const void *db,
                          const void *dp, i64 n_tok, i64 n_head, i64 head_dim,
                          i64 d_state, i64 n_group, i64 inner, i64 conv_dim,
                          bool has_d) {
        (void)out; (void)state; (void)xbc; (void)dt; (void)a; (void)db;
        (void)dp; (void)n_tok; (void)n_head; (void)head_dim; (void)d_state;
        (void)n_group; (void)inner; (void)conv_dim; (void)has_d;
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
