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

#include <cmath>   // std::tanh in krk_gelu_tanh, called from every FFN path
#include <cstdlib> // std::abort in hadamard_act's default body
#include <cstring>
#include <functional>
#include "krk/common.hpp"
#include "krk/quant.hpp"
#include "krk/vram_guard.hpp"

namespace krk {

enum class BackendKind {
    CPU,
    HIP,
    // A virtual NPU (github.com/riteshnair/npuxl-improved): a software device
    // that runs the GEMM path on its own SIMD kernels. It addresses host
    // memory directly, so like the CPU reference it has no separate device
    // tier -- see has_device_memory().
    NPU,
};

// A per-shard weight pull source. The loader passes a table of these; the
// backend turns a source pointer back into a file offset by checking which
// source's range contains the pointer (integer containment, not pointer
// comparison across unrelated mappings).
struct WeightSource {
    std::string path;
    const u8 *map_base = nullptr;
    size_t map_size = 0;
    size_t data_off = 0; // tensor data section within this shard
};

// GELU, the tanh approximation (ggml's ggml_gelu):
//   0.5x(1 + tanh(sqrt(2/pi)(x + 0.044715 x^3)))
// One definition for both backends, because the CPU arm is the reference the
// device arm is compared against -- two transcriptions of this would differ in
// the last bits and read as a kernel bug.
inline f32 krk_gelu_tanh(f32 v) {
    constexpr f32 kAlpha = 0.7978845608028654f; // sqrt(2/pi)
    constexpr f32 kBeta = 0.044715f;
    return 0.5f * v *
           (1.0f + std::tanh(kAlpha * (v + kBeta * v * v * v)));
}

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
    // hipDeviceAttributeIsLargeBar: 1 yes, 0 no (256 MiB aperture), -1 unknown.
    // A HINT, not a fact -- WDDM has reported 0 on firmware with ReBAR on --
    // so the copy path's own GB/s is what any decision should be based on.
    int large_bar = -1;
    // hipDriverGetVersion()'s integer; 0 when it cannot be read (the CPU
    // backend, an old runtime). Carried into the H2D verdict cache's key: the
    // driver is what stages a pageable source, so a cached copy-path verdict is
    // a statement about THIS driver, and an update has to invalidate it the way
    // a firmware change does. The banner already names the HIP *runtime*
    // version; this is the kernel-mode half underneath it.
    int driver_version = 0;

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

// How the K/V cache stores each element. F16 (the default, and the only arm
// any model ran before this existed) keeps the row as the backend's activation
// type. Int8 stores `kv_dim` signed bytes followed by one f32 scale per 32
// values, so a row is `kv_dim + 4*ceil(kv_dim/32)` bytes whatever the backend's
// activation type is -- the layout is the same on the CPU reference and on the
// device, which is what makes the CPU arm an oracle for the kernel.
//
// Why: at a long context the cache stops being a rounding error and starts
// taking the VRAM the expert tier needs (measured: a 128k run re-clamped its
// expert pool 4697 -> 359 MiB as the KV tier grew). Int8 is 1.78x smaller at
// 32-wide blocks (1 byte + 4/32) against f16's 2, and 3.6x against the CPU
// reference's f32.
enum class KvQuant : i32 { F16 = 0, Int8 = 1 };

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
    // Attention sink: one scalar per query head, `n_head` values, in the
    // backend's own address space (host for the CPU backend, device for HIP) --
    // the same convention qk_norm's weights use. The sink is a DENOMINATOR-only
    // softmax term with no value row behind it:
    //     out = acc / (l + exp(sink[h] - m))
    // so probability mass can settle on a zero vector instead of being spread
    // over the window (StreamingLLM; deepseek4 and gpt-oss both carry
    // `blk.N.attn_sinks.weight` [n_head]). nullptr means no sink, which is the
    // only path that existed before this field and the one that must stay
    // bit-identical: every implementation adds the term strictly inside
    // `if (sinks)`, so a model without the tensor runs the same arithmetic in
    // the same order it always did. On TOP of that, a sink LOGIT of exactly
    // 0 is treated as NO sink (`sinks && sinks[h] != 0`), so a zeroed sink
    // row is BIT-identical to the absent path -- the zeroed-sink arm the
    // test group pins with `==`. llama.cpp adds exp(0-m) unconditionally;
    // a checkpoint carrying exact-zero sink logits therefore differs from it
    // here, deliberately: the zeroed arm must reproduce the pre-feature
    // arithmetic bit for bit, and exact zeros are the only values for which
    // that and the formula can both hold.
    const f32 *sinks = nullptr;
    // Cache element format. False is f16 (or f32 on the CPU reference) and is
    // bit-identical to the pre-feature arithmetic: every int8 path is guarded
    // by this flag, so a run that never sets it cannot change.
    bool kv_int8 = false;
    // Self-only block mask, for a masked-block drafter. When `skip_from >= 0`, a
    // query row at position p >= skip_from ALSO sees every key in
    // [skip_from, p) as masked -- i.e. it keeps the context before skip_from and
    // its own row, and ignores the sibling rows of the same block. Those
    // siblings are parallel candidates for other positions: they carry no answer
    // for this row, and letting them in mixes one candidate's guess into the
    // next one's.
    //
    // The range is [skip_from, p), not [skip_from, p], so a row still reads its
    // own K/V, and row 0 of a block (at p == skip_from) has an empty skip range
    // and sees the whole context normally. -1 (the default) masks nothing and
    // every existing model's arithmetic is unchanged.
    i64 skip_from = -1;
};

// ---------------------------------------------------------------------------
// Multi-head latent attention (DeepSeek-V2/V3/V4), legacy-lite layout.
//
// The cache row is NOT n_kv x head_dim. One row per token holds the compressed
// latent (kv_lora wide, normed at write time) followed by the decoupled RoPE
// key (qk_rope wide, stored UNROTATED so each row rotates at its own position).
// Per token and layer that is 512 + 64 on V2-Lite against a GQA row of
// n_kv*head_dim*2, which is the whole point of MLA.
//
// Attention runs ON the latent, with the two per-head projections folded where
// they cost the least:
//
//   qk_h  = Wk_h^T q_nope_h                      the absorb, per call
//   s_t   = qk_h . c_t + rope(q_pe_h) . rope(k_pe_t)
//   p     = softmax(s * scale)
//   cv    = sum_t p_t c_t                        still latent width
//   out_h = Wv_h^T cv
//
// so neither the cache nor the score ever widens to per-head K and V rows.
// ---------------------------------------------------------------------------

struct MlaDesc {
    i64 n_head = 0;
    i64 qk = 0;        // per-head query width: nope + rope
    i64 nope = 0;
    i64 rope = 0;
    i64 kv_lora = 0;
    i64 v_dim = 0;
    i64 n_tok = 0;      // query rows in this call
    i64 pos0 = 0;       // absolute position of row 0
    i64 pos_stride = 0; // elements between cache positions
    f32 scale = 1.0f;   // 1/sqrt(qk), times any rope magnitude factor
    f32 eps = 1e-6f;
    // Decoupled-rope frequencies, [rope/2], HOST memory (the backend stages
    // them, exactly as AttnDesc's tables are staged). The pairing is the
    // adjacent-channel one (rope_neox=false), which is what this family
    // declares; a file needing the half-split pairing is a different layout
    // and is refused by the loader today.
    const f32 *inv_freq = nullptr;
};

// V4's shared-latent MLA (MlaLayout::Shared), which is a different graph from
// the layout above: ONE normed latent row per token is used as BOTH K (rotated
// on read) and V (as it is), and the output is a grouped low-rank projection.
// There is no decoupled key, so no `nope`/`rope` split of the cache row -- the
// rotary dims are the TAIL of the same latent. Arithmetic: src/mla_shared.cpp.
//
// Weights are the file's own f16 staging: [out][in] rows with ne[0] the
// contracting axis (GGUF's order). The oracle in src/mla_shared.cpp is [in][out]
// for the same matrices, which is the convention the bench transposes between.
struct MlaSharedDesc {
    i64 n_head = 0;
    i64 kv_lora = 0;   // key_length == value_length == head_dim
    i64 qk_rope = 0;   // rotary tail of the latent AND of every query head
    i64 o_groups = 0;
    i64 o_lora = 0;
    i64 n_embd = 0;
    i64 n_tok = 0;      // query rows in this call
    i64 pos0 = 0;       // absolute position of row 0
    i64 pos_stride = 0; // elements between cache positions
    f32 scale = 1.0f;   // 1/sqrt(kv_lora): the score is over the whole latent
    f32 eps = 1e-6f;
    // V4's layers are windowed: keys at or before q_pos - window do not take
    // part. 0 means every cached key. Positions are the cache slots themselves
    // (dense cache), which is what the engine keeps -- the oracle takes an
    // explicit kpos array because the SUITE compacts a cache to test the rule.
    i32 window = 0;
    const f32 *inv_freq = nullptr; // HOST [qk_rope/2]
    // Per-head attention-sink logit, entered into the softmax DENOMINATOR only
    // (never attended). HOST memory, [n_head]; null on a file with no sinks.
    const f32 *sinks = nullptr;
    i64 heads_per_group() const { return o_groups > 0 ? n_head / o_groups : 0; }
    i64 gout() const { return o_groups * o_lora; }
};

// ---------------------------------------------------------------------------

class Backend {
public:
    virtual ~Backend() = default;

    virtual BackendKind kind() const = 0;
    virtual const DeviceCaps &caps() const = 0;

    // True when this backend computes out of a memory space of its own, so
    // alloc() hands back device memory that has to be staged through upload()
    // and a VRAM ledger, expert cache and weight streaming all have something
    // to plan against. False for a backend that computes in place on host
    // memory: the CPU reference and the virtual NPU. The default keeps every
    // kind that is not the CPU reference on the device side, which is what
    // `kind() != BackendKind::CPU` meant at every call site before this
    // existed.
    virtual bool has_device_memory() const {
        return kind() != BackendKind::CPU;
    }

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
    // Bytes this backend has outstanding through alloc()/release(), i.e. the
    // allocator's own ledger rather than the driver's. `device_free_bytes()` is
    // a device-wide probe and moves with every other process on the card, so it
    // cannot say whether a number in this engine's accounting is right; this
    // one can. Zero means the backend does not track it.
    virtual size_t device_live_bytes() const { return 0; }
    // Count and largest single buffer, for the same check: live bytes alone
    // cannot tell one 7 GiB buffer apart from 1,100 cells of 6.3 MiB, and those
    // two mean opposite things.
    virtual size_t device_alloc_count() const { return 0; }
    virtual size_t device_largest_alloc() const { return 0; }
    // Live host-memory availability for safe pageable staging. Zero means the
    // backend cannot obtain a trustworthy reading; callers that allocate RAM
    // tiers fail closed in that case.
    virtual size_t host_available_bytes() const { return 0; }
    virtual size_t host_total_bytes() const { return 0; }

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

    // Shared admission for every pageable staging allocation, regardless of
    // whether it belongs to the expert WARM tier or a KV WARM/COLD move.
    // Individual tier budgets remain upper bounds; this process-wide ledger is
    // the common physical-host-memory ceiling they jointly spend.
    void *alloc_host_pageable_budgeted(size_t bytes) {
        if (bytes == 0) bytes = 1;
        if (!SharedHostRamBudget::reserve(bytes, host_available_bytes(),
                                          host_total_bytes()))
            return nullptr;
        void *p = alloc_host_pageable(bytes);
        if (!p) {
            SharedHostRamBudget::cancel(bytes);
            return nullptr;
        }
        if (!SharedHostRamBudget::commit(p, bytes)) {
            release_host_pageable(p);
            SharedHostRamBudget::cancel(bytes);
            return nullptr;
        }
        return p;
    }
    void release_host_pageable_budgeted(void *p) {
        SharedHostRamBudget::release_and_free(
            p, [this](void *owned) { release_host_pageable(owned); });
    }
    // Releases host memory an asynchronous copy may still be reading. The
    // default frees it immediately; a backend that stages copies keeps the
    // pointer until the copy has landed, because freeing the source of an
    // in-flight DMA is a use-after-free, and with a greedy decode it shows up
    // as text that is plausible but different from run to run.
    virtual void release_host_deferred(void *p) {
        SharedHostRamBudget::release_and_free(
            p, [this](void *owned) { release_host_pageable(owned); });
    }
    // The same deferral for a source that came from alloc_host() (pinned) and
    // is therefore freed with release_host(). A backend that stages copies must
    // keep the pointer until the copy lands, exactly as the pageable arm does;
    // freeing it early is a use-after-free whose symptom is text that varies
    // from run to run rather than a crash.
    virtual void release_host_pinned_deferred(void *p) {
        if (p) release_host(p);
    }


    void release_host_deferred_budgeted(void *p) {
        if (p) release_host_deferred(p);
    }
    // Can a KERNEL read a buffer this backend allocated with alloc_host()?
    //
    // `alloc_host` is the pinned tier (hipHostMalloc on HIP), and whether those
    // bytes are addressable from the device is a property of the runtime and the
    // driver, not a thing to assume: HIP on this box reports
    // `unifiedAddressing: 0` and still maps the pinned pool. The answer was
    // measured (tools/probe_hostread.hip: a kernel read a 4 MiB pinned buffer
    // back exactly, and a 256 MiB one at 24.3 GiB/s against 485 GiB/s for device
    // memory), so it is a fact on this machine and false by default everywhere
    // else until it is measured there too.
    //
    // This is what lets an expert be READ IN PLACE out of the host tier instead
    // of promoted: for a decode row the promotion copies ~1.95 MiB into VRAM to
    // read it once, and the same bytes out of RAM cost one pass at the same
    // order of speed with no VRAM slot and no eviction.
    virtual bool host_load_supported() const { return false; }
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
    // The same batch for a source that is ALREADY pinned. A backend that can
    // DMA straight out of it skips its staging ring entirely, which is the
    // difference between a copy the host thread pays for and one it only
    // enqueues -- see Model::stage_layer, where the next layer's file read
    // happens while this batch is still in flight.
    // `merge` says whether adjacent entries may be issued as ONE copy. It is a
    // property of the SOURCE, not a preference: a merged run must lie inside a
    // single host allocation, because a pinned registration is what the copy
    // is validated against. Passing true for a batch whose sources are one
    // allocation per entry (a WARM tier slot is one hipHostMalloc per expert)
    // is how a run came to span two of them -- the driver answers that with
    // hipErrorInvalidValue, and the caller read it as a dead process.
    virtual void upload_pinned_batch(void *const dst[], const void *const src[],
                                     const size_t bytes[], int n,
                                     bool merge = true) {
        static_cast<void>(merge);
        upload_paged_batch(dst, src, bytes, n);
    }

    // There are two ways to move a PAGEABLE source, and which one is faster is a
    // property of the card's aperture -- a FIRMWARE setting (Above 4G decoding +
    // Resizable BAR), not a constant. The ring stages each chunk through a
    // bounded pinned buffer, which is the only way to reach the bus through a
    // narrow aperture. `direct` hands the pageable source to the driver on one
    // side stream: the driver still stages it internally, but a wide aperture
    // makes that staging free of the windowed bounce copy the ring exists to
    // avoid, so one sync per batch beats the ring's per-chunk bookkeeping and
    // the caller's memcpy disappears entirely.
    //
    // Engine::init times both at the promotion shape and sets this from the
    // result; KRK_WARM_DMA=1/2 forces either path. A backend with a single
    // pageable path says so and ignores the setting -- that is why these are two
    // calls and not one: `selectable` decides whether the measurement means
    // anything, and it is false on the CPU backend, where a promotion is a
    // memcpy either way.
    virtual bool pageable_paths_selectable() const { return false; }
    // Which pageable mechanism to run. A backend that implements this one is
    // asked for a MEASURED choice at init (see `pageable_path_choice`); the
    // default keeps the historical two-way setting working by forwarding.
    //
    // `Blocking` is in the enum because on a wide aperture the driver's own
    // hipMemcpy is the fastest of the three and the engine's ring is the
    // slowest -- the menu was incomplete, not misordered.
    enum class PageablePath { Ring = 0, Direct = 1, Blocking = 2 };
    virtual void set_pageable_path(PageablePath) {}
    virtual void set_pageable_direct(bool on) {
        set_pageable_path(on ? PageablePath::Direct : PageablePath::Ring);
    }

    // Time and bytes the pinned copy stream spent moving these batches, from
    // events on that stream. A caller's wall clock around
    // upload_pinned_batch measures the ENQUEUE and nothing else -- that batch is
    // still in flight when the call returns -- so a promotion rate computed from
    // it reads several times the link's real rate (measured 64.6 GB/s on a link
    // that does 27). Folded lazily: the value lands once the copy completes.
    virtual double pinned_dma_ms() { return 0.0; }
    virtual unsigned long long pinned_dma_bytes() const { return 0; }
    virtual unsigned long long pinned_dma_batches() const { return 0; }
    // The subset the event bracket actually covered. A batch enqueued while every
    // bracket slot is in flight goes untimed, so a rate over `pinned_dma_bytes()`
    // would divide real milliseconds by bytes that were never timed; these two are
    // what the rate is over.
    virtual unsigned long long pinned_dma_timed_bytes() const { return 0; }
    virtual unsigned long long pinned_dma_timed_batches() const { return 0; }
    // Adjacent-source coalescing removed this many per-tensor copies on the
    // pinned path. Reported so an A/B can say whether a difference came from the
    // change or from the machine: 0 means the change could not have mattered.
    virtual unsigned long long pinned_merged() const { return 0; }

    // Opt-in promotion ticket. Source bytes must be copied/staged before return;
    // wait_marker(ticket) makes the copy visible before its first consumer.
    virtual void *upload_expert_staged(void *const dst[], const void *const src[],
                                       const size_t bytes[], int n) {
        upload_paged_batch(dst, src, bytes, n);
        return nullptr;
    }

    // Prefetch ticket: the bytes are DMA'd out of PINNED host memory on a
    // private stream, with no host copy and NO default-stream wait -- the
    // caller's kernels keep running and the consumer pays only a device-side
    // ordering when it asks (wait_marker_stream). That is the difference from
    // upload_expert_staged, whose ring copies through host memory at issue time
    // and whose ticket is host-synchronised at consumption.
    virtual void *upload_expert_prefetch(void *const dst[], const void *const src[],
                                         const size_t bytes[], int n,
                                         bool merge = true) {
        static_cast<void>(merge);
        return upload_expert_staged(dst, src, bytes, n);
    }
    // Orders the default stream behind a prefetch ticket without blocking the
    // host. False when the backend cannot express that (no event to wait on).
    virtual bool wait_marker_stream(void *) { return false; }

    // The same prefetch, with its COPY stream ordered behind `ready` -- an event
    // already recorded on the default stream by record_marker(). A double
    // buffered weight ring needs both directions at once and neither call above
    // can express it: the slot a layer is copied INTO was read by the layer two
    // before it, so that copy must not begin until those kernels are done, while
    // the layer being read must start its copy before the current layer's
    // kernels. One event on each side says exactly that, and the host blocks on
    // neither. A backend that cannot order its copy stream falls back to waiting
    // on the host, which is slower and never wrong.
    virtual void *prefetch_after(void *const dst[], const void *const src[],
                                 const size_t bytes[], int n, void *ready) {
        if (ready) wait_marker(ready);
        return upload_expert_prefetch(dst, src, bytes, n);
    }

    // Small-object device allocations, pooled by the expert cache. A
    // promotion is a handful of ~1-2 MiB slices and the default
    // alloc/release pair is hipMalloc/hipFree, which are
    // device-synchronising: measured on the prefill of a 40-layer MoE, the
    // expert cache issued 13198 uploads and 2594 ms of device time went into
    // them against 1330 ms of device idle. Backends without a pool just
    // forward to alloc/release.
    struct ExpertPoolStats {
        size_t budget = 0, committed = 0, live = 0, reusable = 0;
        u64 allocations = 0, reuses = 0, trims = 0, guard_fallbacks = 0;
        f64 staged_copy_ms = 0, staged_reader_ms = 0, staged_overlap_ms = 0;
        u64 staged_measured = 0, staged_hidden = 0;
        // How a reused cell's guard was honored: on the device (the writer's
        // stream waits on the mark), or by waiting on the host for a path with
        // no single stream to order. The first is the cheap one; the ratio is
        // what says whether the reuse path was actually taken.
        u64 order_device = 0, order_host = 0, order_implicit = 0;
        // Host time actually blocked inside pool_honor_host_ (a
        // hipEventSynchronize on the mark). order_host counts the calls; this
        // is what they cost, and the difference between it and the whole
        // guard's cost (the KRK_POOL_NO_WAIT arm) is time spent on the DEVICE:
        // a stream waiting for a mark delays the copy, and the default stream
        // waits for that copy, so the host sees none of it.
        f64 honor_host_ms = 0;
        // Release marks: how many hipEventRecords the pool issued, and how many
        // slices those records covered. A burst release is ONE mark for its
        // whole burst, so release_slices / release_marks is the sharing factor
        // and a ratio of 1.0 means the burst path was not taken.
        u64 release_marks = 0, release_slices = 0;
        // Reader-done marks recorded at the end of a slot's use, which a reuse
        // then waits on instead of the release-time position. Zero means the
        // slot-mark path never ran (see KRK_POOL_SLOT_MARK).
        u64 reader_marks = 0;
        // Growth refused because it would have taken the card below the VRAM
        // floor. Not an error: the cache serves those experts from WARM, and
        // the count is what says the pool hit the real limit rather than its
        // own plan. It exists because the plan is in requested bytes and the
        // driver also holds memory the engine does not account for (measured
        // 7,000 MiB beside 7,430 MiB promoted on Mellum2.1).
        u64 floor_refusals = 0;
        // Where `alloc` in the expert-path split actually goes. A constrained
        // run grows the pool between trims, and the three candidates -- freeing
        // slabs, asking the driver how much is free, taking a new cell -- have
        // never been separated in the numbers, so a 4.8 s alloc phase could not
        // be attributed to any of them.
        f64 trim_ms = 0, malloc_ms = 0, probe_ms = 0;
        u64 probes = 0;
    };
    // Physical bytes, including alignment. One oversize expert is explicitly
    // allowed by the cache; free cells are trimmed before changing the limit.
    virtual size_t pooled_size(size_t bytes) const { return bytes; }
    virtual void set_expert_pool_budget(size_t bytes) { (void)bytes; }
    virtual void set_expert_pool_owner(const void *owner, size_t bytes) {
        (void)owner; (void)bytes;
    }
    virtual ExpertPoolStats expert_pool_stats() const { return {}; }
    virtual void *alloc_pooled(size_t bytes) { return alloc(bytes); }
    virtual void release_pooled(void *p, size_t bytes) {
        (void)bytes;
        if (p) release(p);
    }
    // Release several cells that are going back at the SAME stream point --
    // an expert's three slices, which is what an eviction returns. The mark a
    // reused cell needs is "the default-stream position its readers are
    // ordered before", and cells released together share that position, so a
    // backend that can express it records one mark for the burst instead of one
    // per cell. It is not an approximation: nothing is submitted between the
    // releases in a burst, so their positions are the same point.
    //
    // Sharing must not go further than a burst. A backend that re-records a
    // mark while a wait on it is still in flight (a rotating pool of marks,
    // say) can put the recorded position after the default stream's
    // `hipStreamWaitEvent(0, paged_ev)` -- the default stream waits on the copy
    // stream there while the copy stream waits on these marks -- and a mark
    // recorded after such a wait, then handed to a copy-stream writer, is a
    // dependency cycle. See HipBackend::PoolMark.
    virtual void release_pooled_batch(void *const ps[], const size_t bytes[], int n) {
        for (int i = 0; i < n; i++) release_pooled(ps[i], bytes[i]);
    }
    // "These cells were last READ here" -- the owner records it once its kernels
    // for them have been enqueued, which is a much tighter point than the
    // release that comes at eviction time. A release mark is a POSITION on the
    // default stream, so the mark recorded at release covers every kernel queued
    // before it, not just these cells' readers: in a decode step the whole token
    // is queued back to back, so a promotion that reused a just-evicted slot
    // waited for the layer's own earlier experts and the layer before it
    // (measured +24% at a saturated budget, docs/PERF-ANALYSIS.md section 12n).
    // Recorded at the end of the slot's own use, the wait is for that slot's
    // readers only. The mark is shared by the cells of one expert -- they are
    // read by the same kernels -- so this is one record per slot use, not per
    // cell. `KRK_POOL_SLOT_MARK=0` restores the release-time mark.
    virtual void pool_note_reader_done(void *const cells[], int n) {
        (void)cells;
        (void)n;
    }
    virtual void upload(void *dst, const void *src, size_t bytes, size_t off = 0) = 0;
    // Direct copy with no staging side effects. The HIP backend's read()-based
    // weight path calls this per chunk; a backend without that path can simply
    // forward to upload().
    virtual bool upload_now(void *dst, const void *src, size_t bytes, size_t off = 0) {
        upload(dst, src, bytes, off);
        return true;
    }
    // Optional hook for pull-style weight loading; see HipBackend::set_weight_sources.
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
    //
    // The shard table lets each tensor's data live in a different shard's mapping.
    // Each entry is { path, map_base, map_size, data_off } — the backend turns a
    // source pointer into a file offset by checking which source contains it (integer
    // range check, not pointer comparison across unrelated mappings).
    virtual void set_weight_sources(const std::vector<WeightSource> &sources) {
        (void)sources;
    }
    // Legacy wrapper: one source = the old single-shard fast path.
    virtual void set_weight_pull(const std::string &path, const void *map_base,
                                 size_t map_size, size_t data_off) {
        std::vector<WeightSource> sources;
        sources.push_back({path, static_cast<const u8 *>(map_base), map_size, data_off});
        set_weight_sources(sources);
    }
    virtual void download(void *dst, const void *src, size_t bytes, size_t off = 0) = 0;

    // Device-to-device byte copy. A hybrid model's recurrent state is f32, not
    // activation-typed, so copy_act cannot move it; speculative decoding needs
    // a snapshot of it before a verification pass and a restore when the pass
    // keeps only a prefix. The default stages through the host (correct
    // everywhere); the device backends override it with one DMA on the default
    // stream, which is what makes a per-round snapshot affordable.
    virtual void copy_device(void *dst, const void *src, size_t bytes) {
        if (!bytes) return;
        std::vector<u8> stage(bytes);
        download(stage.data(), src, bytes, 0);
        upload(dst, stage.data(), bytes);
    }
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
    // Some backends provide a null/no-op marker and implement waits as a full
    // sync (the conservative fallback above); others require real markers to
    // safely reuse a device slot. A failed required marker is an init error.
    virtual bool requires_markers() const { return false; }
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

    // Counters for a run summary, or empty when the backend has none to add.
    // The NPU backend reports what its GEMM hand-over actually did (calls
    // offered, calls served, GFLOPs); the reference and the HIP arms do not
    // keep a parallel ledger and return nothing.
    virtual std::string backend_stats() const { return std::string(); }

    // ---- ops -------------------------------------------------------------

    // out[n_tok, n_embd] <- rows of tok_embd selected by tokens.
    virtual void embed(void *out, const void *tok_embd, DType tt, i64 n_vocab,
                       i64 n_embd, const i32 *tokens, i64 n_tok) = 0;

    // out[rows, n] = rmsnorm(x[rows, n]) * w[n], computed in f32.
    //
    // `row_exp` (nullable) is the per-row power-of-two exponent the residual
    // stream is stored under (see add_residual): row r holds x * 2^-row_exp[r],
    // and this multiplies it back before squaring, which is exact and is a
    // no-op when the array is null or every exponent is zero. It is not a
    // linearity trick but a storage one -- the norm is where a nonlinear block
    // gets true-scale input again.
    virtual void rmsnorm(void *out, const void *x, const f32 *w, i64 rows, i64 n,
                         f32 eps, const i32 *row_exp = nullptr) = 0;

    // out[rows, n_out] = x[rows, n_in] * W[n_out, n_in]^T ; W is quantized.
    virtual void gemm(void *out, const void *x, const void *w, DType wt,
                      i64 n_out, i64 n_in, i64 rows) = 0;

    // Can this backend's GEMM address a CONTRACTION of `k` columns? The CPU
    // reference walks any k; a kernel that vectorizes K in fixed chunks can
    // only take multiples of that chunk, and the answer has to be asked BEFORE
    // the first pass -- a shape the kernel cannot address is a refusal that
    // names the geometry, not a launch that aborts halfway through a prefill.
    // Default true, and every backend that imposes a rule overrides it: a
    // default of false would refuse shapes that run fine.
    virtual bool gemm_k_ok(i64 k) const { return k > 0; }

    // dst[rows, n_out] += x[rows, n_in] * W[n_out, n_in]^T : the residual add
    // that ends a layer's projection, folded into that projection's epilogue
    // so the layer pays one launch instead of two. `dst` is both the addend and
    // the destination, so a caller whose `x` is `dst` must not use this.
    //
    // Returns true when the fused form ran, and the caller must then issue
    // neither the projection nor the add. False means "do it the long way",
    // which is what every backend that does not override this gets, and the
    // caller's fallback is gemm() into a scratch row followed by
    // add_inplace() -- exactly the two calls this exists to remove, so a
    // backend without the fused path behaves as if it never had one.
    //
    // The fused store is bit-identical to that pair, and not by luck: both
    // forms round the projection to the activation type BEFORE summing, and a
    // sum of two activation-typed values is exact in f32, so the f16 result is
    // the same number. A backend that accumulates in a wider type must round
    // the product before the add to keep that promise.
    // `rexp` (nullable) is the residual stream's per-row storage exponent when
    // `dst` is that stream (see add_residual): the contribution this folds in
    // arrives at true scale, so it is divided by 2^rexp[0] -- the decode row's
    // own exponent -- before it is summed. Null, or a zero exponent, leaves the
    // fused store bit-identical to the pair, and a backend that ignores the
    // array keeps the f32 CPU reference's exact behavior.
    virtual bool gemm_accumulate(void *dst, const void *x, const void *w,
                                 DType wt, i64 n_out, i64 n_in, i64 rows,
                                 const i32 *rexp = nullptr) {
        (void)dst; (void)x; (void)w; (void)wt; (void)n_out; (void)n_in;
        (void)rows; (void)rexp;
        return false;
    }

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

    // The MoE router's activation (softmax over n_expert) on the device, from
    // the router GEMM's row in the activation type into f32 probabilities.
    //
    // The engine does this on the host today because the top-k that follows
    // needs the expert ids there anyway (the cache resolves a pointer from
    // them). This op exists so the NUMERICS of moving it can be measured
    // before the dispatch is rewritten: device `expf` plus a tree-reduction
    // sum is not `std::exp` plus a sequential sum, and `KRK_DEVICE_SOFTMAX=1`
    // reports the drift in weights and in selected ids against the host answer.
    virtual bool router_softmax_supported() const { return false; }
    virtual void router_softmax(void *probs, const void *logits, i64 n_rows,
                                i32 n_expert) {
        (void)probs; (void)logits; (void)n_rows; (void)n_expert;
    }

    // Selection and the layer plan, on the device. Same rule as the host: the
    // highest probability wins, a tie keeps the lower expert id, and the
    // weights are the selected probabilities, sum-normalized over the k when
    // `normalize` is set and multiplied by `scale`.
    //
    // `moe_plan_build` turns the ids into the layer's expert-major plan --
    // `off[n_expert+1]`, `rows[n*k]`, `wt[n*k]` -- in token-then-slot order, so
    // it is the same plan the host builds and a differential check compares two
    // orderings that are meant to be identical rather than two that merely
    // happen to agree.
    virtual bool router_topk_supported() const { return false; }
    virtual void router_topk(const void *probs, i64 n_rows, i32 n_expert, i32 k,
                             void *ids, void *wts, i32 normalize, f32 scale) {
        (void)probs; (void)n_rows; (void)n_expert; (void)k; (void)ids;
        (void)wts; (void)normalize; (void)scale;
    }
    // ---- routing statistics, accumulated on the device --------------------
    // A tracking run (`--expert-scan`, `--expert-adapt`) wants one thing per
    // routed layer: which experts this row chose. Collecting that on the host
    // costs the router readback -- a blocking drain per layer, which on a
    // 40-layer stack is 8 ms of a 22 ms token, and it is a tax on the very path
    // the tracking run is supposed to measure. With these two, the device adds
    // each layer's choices into a per-(layer, expert) histogram and the host
    // reads the finished histogram ONCE, after the run.
    //
    // `hits` is u32 and `mass` f32, both [n_layer * n_expert]. `mass` is summed
    // with device atomics, so its last bits depend on the order the rows land
    // in -- it ranks experts, it does not decide text.
    virtual bool scan_accum_supported() const { return false; }
    virtual void scan_accum(const void *probs, const void *ids, i64 n_rows,
                            i32 n_expert, i32 k, i32 layer, void *hits,
                            void *mass) {
        (void)probs; (void)ids; (void)n_rows; (void)n_expert; (void)k;
        (void)layer; (void)hits; (void)mass;
    }

    virtual void moe_plan_build(const void *ids, i64 n_rows, i32 k,
                                i32 n_expert, void *off, void *rows, void *wt,
                                const void *wts) {
        (void)ids; (void)n_rows; (void)k; (void)n_expert; (void)off; (void)rows;
        (void)wt; (void)wts;
    }

    // The whole MoE layer from ONE launch: the expert matrices are chosen by
    // the kernel from `ids` through a device pointer table
    // (`krk/moe_table.hpp`, published by ExpertCache), so no host pointer per
    // expert and no `acquire()` sits between the routing and the GEMMs.
    //
    // This is the stage the round trip is actually paid for. Device top-k and a
    // device plan remove the host scan but still need the ids on the host to
    // learn where the matrices are; with the table they never come back, and
    // the layer stops waiting for its own GPU work to retire (measured: 0.523
    // ms/layer on Mellum2.1, of which 0.077 is the copy and the rest is that
    // wait). What it cannot do is keep the host's bits: the softmax runs on the
    // device, which is a different `expf`, so this is an opt-in arm and the
    // drift it costs is measured rather than assumed.
    //
    // `table` is `[n_expert * n_layer]` entries indexed `layer * n_expert + e`,
    // `xn` is the f16 activation block `[n_tok, n_embd]`, and `out` receives the
    // weighted, f16-rounded sum of the routed experts' down projections
    // (`[n_tok, n_embd]`) -- the buffer the caller would otherwise have
    // scattered into per expert.
    virtual bool moe_dispatch_supported() const { return false; }
    virtual void moe_dispatch(const void *table, i32 layer, i32 n_expert,
                              const void *ids, const void *wts, const void *xn,
                              void *out, i64 n_tok, i64 n_embd, i64 n_ff, i32 k) {
        (void)table; (void)layer; (void)n_expert; (void)ids; (void)wts;
        (void)xn; (void)out; (void)n_tok; (void)n_embd; (void)n_ff; (void)k;
        // No device, no dispatch: moe_dispatch_supported() is false on this
        // backend, so the engine's host path is the one that runs.
    }

    // A group of n_mat projections that all read the same activation
    // x and have no dependency between them (a decode layer's q/k/v,
    // or its gate/up), issued as one launch. Only the M=1 decode row
    // is fused; any backend, dtype mix or prefill shape falls back to
    // one gemm() per matrix. All matrices must share `wt`. Returns
    // true when the fused path ran.
    // `silu` asks for silu_mul() of matrices 0 and 1 to be folded into the
    // group's epilogue: out[0] then receives silu(out[0]) * out[1] and the
    // caller must not issue that activation itself. *silu_fused reports whether
    // that happened -- the group can still have fused without it (a shape or a
    // split-K decision it cannot express), which is why the return value alone
    // is not the answer. The default does the per-matrix gemms and leaves the
    // activation to the caller, so a backend without the folded form behaves
    // exactly as one that never had it.
    // `row_parallel_only` pins every matrix in the group to one row-parallel
    // pass (no split-K top-up). A caller that groups matrices too small to fill
    // the device -- a recurrent layer's dt_rank-wide projections, say -- wants
    // the launch count down, and the top-up would spend a reduce launch per
    // split matrix to buy parallelism it does not need; it also regroups the
    // fp32 partials, so a group with splits at 1 is bit-identical to the
    // separate launches while a split one is not.
    // `wts` gives each matrix its own weight type, for a group whose
    // projections are not all quantised the same way (a real checkpoint will
    // have attn_qkv at one format and ssm_alpha/ssm_beta at another). Null
    // means every matrix uses `wt`.
    // `xs` gives each matrix its own activation source, for a group whose
    // matrices do NOT share one -- the routed experts' down projections read a
    // different silu row each. Null, the usual case, means every matrix reads
    // `x`; `x` is then ignored.
    virtual bool gemm_group(void *const out[], const void *x,
                            const void *const w[], DType wt,
                            const i64 n_out[], i64 n_in, int n_mat,
                            i64 rows, bool silu = false,
                            bool *silu_fused = nullptr,
                            bool row_parallel_only = false,
                            const DType *wts = nullptr,
                            const void *const xs[] = nullptr) {
        (void)silu;
        (void)row_parallel_only;
        (void)wts;
        if (silu_fused) *silu_fused = false;
        for (int m = 0; m < n_mat; m++)
            gemm(out[m], xs ? xs[m] : x, w[m], wts ? wts[m] : wt, n_out[m],
                 n_in, rows);
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

    // Pinned by --prefill-hash: every prefill row must be produced by the same
    // arithmetic a decode step uses, because that hash is the observable a
    // chunk-invariance claim is recorded from. A backend with a faster
    // approximate prefill (a matrix-core flash attention, say) keeps it out of
    // this mode rather than making the hash depend on which kernel ran.
    virtual void set_attn_hash_mode(bool) {}

    // Quantized KV. A backend that cannot read a format says so instead of
    // running the f16 path over int8 bytes, which would be fluent nonsense:
    // Engine::init refuses the run by name when this answers false.
    virtual bool kv_quant_supported(KvQuant q) const {
        return q == KvQuant::F16;
    }

    // Bytes of ONE position's K row (V is the same width) for this backend and
    // format. The engine sizes the cache and the tier slices from this, so a
    // backend that adds a format has one number to answer for instead of
    // every call site learning the layout.
    virtual size_t kv_row_bytes(i64 kv_dim, KvQuant q) const {
        (void)q;
        return static_cast<size_t>(kv_dim) * act_size();
    }

    // ---- multi-head latent attention (see MlaDesc above) ------------------
    // A backend that cannot run MLA says so, and the engine refuses the file
    // rather than running a graph with no attention in it at all. The two ops
    // below are no-ops by default for exactly the reason: a backend that
    // forgets to implement one and still claims support would leave the
    // destination buffer holding whatever the allocator did, which is fluent
    // text that is not the model. The CPU reference implements both, and the
    // suite drives it, so "supported" has something behind it.
    virtual bool mla_supported() const { return false; }
    // cache[pos0 + t] = [ rmsnorm(proj[t][0..kv_lora)) * gamma | proj[t][kv_lora..) ]
    // from ONE [n_tok, kv_lora+rope] projection: the latent and the decoupled
    // key come from the same row, and only the latent half is normed.
    virtual void mla_kv_write(void *cache, const void *proj, const void *gamma,
                              const MlaDesc &d) {
        (void)cache; (void)proj; (void)gamma; (void)d;
    }
    // out[n_tok, n_head*v_dim] from q [n_tok, n_head*qk] and the cache.
    // `wkv_b` is the file's unsplit table, f16 [kv_lora, n_head*(nope+v_dim)],
    // head h at column offset h*(nope+v_dim), K part first.
    virtual void mla_attention(void *out, const void *q, const void *cache,
                               const void *wkv_b, const MlaDesc &d) {
        (void)out; (void)q; (void)cache; (void)wkv_b; (void)d;
    }

    // ---- V4 shared-latent MLA (see MlaSharedDesc) -------------------------
    // A separate flag from mla_supported(): the two layouts are different
    // graphs, and a backend that runs V2-Lite is not thereby able to run V4.
    virtual bool mla_shared_supported() const { return false; }
    // cache[pos0 + t] = rmsnorm(proj[t][0..kv_lora)) * gamma, the WHOLE row
    // (the rotary dims are its tail, so they are normed here unlike the lite
    // layout's decoupled key). `proj` is f16 [n_tok, kv_lora].
    virtual void mla_shared_latent(void *cache, const void *proj,
                                   const void *gamma, const MlaSharedDesc &d) {
        (void)cache; (void)proj; (void)gamma; (void)d;
    }
    // a[n_tok, n_head*kv_lora] f32: per head, softmax over the prefix (window
    // and sink applied) of the rotated latent's dot with that head's query,
    // then that softmax's weighted sum of the UNROTATED latent. `q` is f16
    // [n_tok, n_head*kv_lora]` already projected; its rotary tail is rotated
    // here at the row's own position.
    virtual void mla_shared_attention(void *a, const void *q, const void *cache,
                                      const MlaSharedDesc &d) {
        (void)a; (void)q; (void)cache; (void)d;
    }
    // out[n_tok, n_embd] f16: y[g] = Woa_g over group g's concatenated heads,
    // then Wob over the concatenated y. wo_a is f16 [gout, hpg*kv_lora] with
    // group g's o_lora output rows at [g*o_lora, (g+1)*o_lora); wo_b is f16
    // [n_embd, gout]. Both in the file's order (row = output). `scratch` is the
    // caller's f32 [n_tok, gout] workspace for the intermediate y -- per-call
    // allocation here would be a device malloc on every layer of every token.
    virtual void mla_shared_output(void *out, const void *a, void *scratch,
                                   const void *wo_a, const void *wo_b,
                                   const MlaSharedDesc &d) {
        (void)out; (void)a; (void)scratch; (void)wo_a; (void)wo_b; (void)d;
    }

    // Compact gather out of a KV cache: dst[r] = src[idx[r]], `row_bytes`
    // each, for r in [0, n_rows). Pure byte movement, so no element type is
    // involved and an f16 row and an f32 row gather identically; the index
    // list is always host-visibility-agnostic because it is small and
    // per-query-row (see gather_rows_supported).
    //
    // This exists for QSA. `attention()` takes no mask, and the QSA indexer
    // gives every QUERY ROW its own cell list, which neither `window` (one
    // bound for the whole call) nor `pos0` can express. Compacting the chosen
    // cells and running the ordinary causal kernel over the compact buffer
    // (n_tok = 1, pos0 = cells-1) does express it, without a second attention
    // kernel and without a mask that would have to be carried through every
    // one of them.
    //
    // The default is a NO-OP and gather_rows_supported() is false, which is
    // how a backend says it cannot: a silent no-op gather leaves the compact
    // buffer holding whatever was there before, i.e. fluent text that is not
    // this model (the same shape as the lane_bcast_mul default that shipped
    // HIP-only and made a test flaky).
    virtual bool gather_rows_supported() const { return false; }
    virtual void gather_rows(void *dst, const void *src, const i64 *idx,
                             i64 n_rows, size_t row_bytes) {
        (void)dst; (void)src; (void)idx; (void)n_rows; (void)row_bytes;
    }

    // One sink logit per query head, entered into the softmax denominator
    // at the row's running max with no value term (V4 report Eq. 27).
    // Called between the scores/online-max advance and the accumulation,
    // i.e. after the newest key row has been applied. An implementation
    // must be a no-op for a null `sinks`.
    virtual void attention_sink(void *, const void *, i64, i64, i64, i64) {}

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
    // `wq`/`wk` (nullable, device-side, per-head width) and `eps` carry the
    // QK-norm the chain folds between the projections and the rotation;
    // `wide` names OLMoE's whole-row convention, which the chain refuses,
    // and `neox` selects the half-split rope pairing (refused when the rope
    // fraction makes it partial). A backend that implements none of this
    // returns false and the engine runs the separate chain.
    virtual bool attn_fused_chain(void *out, void *q, void *kcache,
                                    void *vcache, const void *k,
                                    const void *v, const AttnDesc &d,
                                    const f32 *inv_freq,
                                    f32 rope_scale, f32 rope_frac,
                                    const f32 *wq, const f32 *wk, f32 eps,
                                    bool wide, bool neox) {
        (void)out; (void)q; (void)kcache; (void)vcache; (void)k;
        (void)v; (void)d; (void)inv_freq; (void)rope_scale;
        (void)rope_frac; (void)wq; (void)wk; (void)eps; (void)wide;
        (void)neox;
        return false;
    }

    // out = silu(gate) * up
    //
    // `row_len` > 0 with `row_scale` non-null takes the row-scaled form: `n`
    // is `n / row_len` rows of `row_len` elements (one row of the gated
    // intermediate, which is the dimension the down projection reduces over --
    // one token expert activation, or a dense FFN's gated row), the products
    // are stored divided by one power of two per row so each row fits the f16
    // activation buffer, and the multiplier 2^k is written to row_scale[row]
    // for the caller to fold back into whatever consumes that row (the MoE
    // scatter alpha). A consumer that is itself scale-invariant folds nothing
    // back and is exactly right either way: BitNet's `ffn_sub_norm` is an
    // RMSNorm on this row, so it reads the row's direction and never its size.
    // A power-of-two divide is exact in f16, so this buys exponent range and
    // spends no precision, and it is bit-identical to the plain form whenever
    // the row is already in range.
    // A backend whose activations are wider than f16 needs no scale: it stores
    // exactly and fills row_scale with 1.0, so the caller contract holds
    // either way.
    virtual void silu_mul(void *out, const void *gate, const void *up, i64 n,
                          i64 row_len = 0, f32 *row_scale = nullptr) = 0;

    // silu_mul() for `count` independent ONE-ROW buffers, as a single launch:
    // row i is out[i] (operands gate[i], up[i]), `row_len` wide, and scale[i]
    // receives that row's power-of-two scale. This is exactly `count` calls to
    // silu_mul(out[i], gate[i], up[i], row_len, row_len, scale[i]) and has to
    // stay bit-identical to them: the MoE decode path batches its routed
    // experts into one activation launch, and a row that rounded differently
    // here would change the model's text. The default is those calls, so a
    // backend that does not batch is correct by construction.
    virtual void silu_mul_rows(void *const out[], const void *const gate[],
                               const void *const up[], i64 row_len, int count,
                               f32 *const scale[]) {
        for (int i = 0; i < count; i++)
            silu_mul(out[i], gate[i], up[i], row_len, row_len,
                     scale ? scale[i] : nullptr);
    }

    // gpt-oss's expert activation: min(gate, 7) * sigmoid(1.702 * that) *
    // (clamp(up, -7, 7) + 1). Every operand is clamped, so the result is
    // bounded by 7 * 8 and no row scale is needed or filled. A backend that
    // does not implement it leaves the buffer untouched -- which is why this
    // one has no default: a model asking for it must not silently get silu.
    virtual void swiglu_oai(void *out, const void *gate, const void *up,
                            i64 n) = 0;

    // out += bias  for each of `rows` output rows of width n (Qwen2/Phi QKV biases)
    virtual void add_bias_rows(void *out, const f32 *bias, i64 n, i64 rows) = 0;

    // a += b  (activation elementwise)
    virtual void add_inplace(void *a, const void *b, i64 n) = 0;

    // a += b, where `a` is the residual stream and its rows may be stored
    // scaled: row r holds X * 2^-rexp[r] and `b` is the layer's contribution at
    // true scale, so the sum is formed in f32, the smallest exponent that keeps
    // the row inside the activation range is chosen, the row is stored divided
    // by it and rexp[r] is updated for every later reader (the norms un-scale on
    // the way in; see rmsnorm's `row_exp`).
    //
    // This is a STORAGE contract, not a factor that rides the arithmetic. A
    // per-token power of two does not survive the nonlinearities: silu(2^-e g)
    // is not 2^-e silu(g), and q and k both carrying 2^-e put 2^-2e in the
    // attention scores, which softmax reads as a temperature change. So every
    // block keeps computing in true scale and only the buffer's storage moves.
    //
    // `rexp` null or `row_len` <= 0 is the plain add_inplace, and the default
    // below is that plain add -- correct for a backend whose activation type
    // has the range to need no exponent at all (the f32 CPU reference, which
    // leaves every exponent at zero, so its residual stream is bit-identical to
    // what it always was).
    // `bexp` (nullable) is b's own storage exponent when the contribution is
    // already carried scaled: the MoE staging fold adds the shared expert into a
    // buffer the routed experts have been accumulating (and a sum of two in-range
    // rows can leave the range on its own), so that fold has an exponent of its
    // own and this renormalizes it as well. Both sides are un-scaled to true
    // scale first, so the sum is the same number either way.
    virtual void add_residual(void *a, const void *b, i64 rows, i64 row_len,
                              i32 *rexp, const i32 *bexp = nullptr) {
        (void)rows;
        (void)row_len;
        (void)rexp;
        (void)bexp;
        add_inplace(a, b, rows * row_len);
    }

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
                             const f32 *w, i64 rows, i64 n, f32 eps,
                             const i32 *row_exp = nullptr) {
        add_inplace(x, res, rows * n);
        rmsnorm(out, x, w, rows, n, eps, row_exp);
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
    // `row_scale` (nullable) is the row scale a prior silu_mul wrote for these
    // same rows: it multiplies alpha[i] before the add.
    //
    // `row_exp` (nullable) is the destination row's storage exponent, and this
    // is the one place in the MoE chain where a row can leave f16 without any
    // single producer being at fault: the sum of ten experts that are each in
    // range can still be out of it. The row is summed in f32, stored divided by
    // the smallest power of two that keeps its own maximum inside f16, and the
    // exponent is written back for the fold that un-scales it -- the same
    // contract add_residual carries, which is why the destination's exponent
    // slot is passed in and out rather than created here.
    virtual void scatter_axpy_rows(void *dst, const void *src, const i32 *rows,
                                   const f32 *alpha, i64 n_rows, i64 n,
                                   const f32 *row_scale = nullptr,
                                   i32 *row_exp = nullptr) = 0;

    // How many f16 activation stores saturated, in the order [silu product,
    // experts sum, residual row]. A backend whose activations are f32 (the CPU
    // reference) has nothing to saturate and answers zeros, and so does a run
    // that stayed in range -- which is the point of printing it every run:
    // silence was the only other report, and a model that decodes nonsense while
    // a bound fires is exactly the case a stats line should not leave to
    // inference. The residual row is the one store the exponent exists to keep
    // out of: a non-zero count there means a single row's contribution left f16
    // even after its exponent was raised, which no bound can answer.
    virtual void activation_saturation(unsigned long long out[3]) const {
        out[0] = 0;
        out[1] = 0;
        out[2] = 0;
    }

    // The largest residual-row exponent this run needed: 0 for every model
    // whose hidden state fits the activation type, and > 0 exactly when the
    // storage scaling of add_residual did something. Reported so a run carrying
    // a scaled residual stream announces it instead of being inferred.
    virtual int residual_exponent_max() const { return 0; }

    // Zeroes those counters; called once per engine so the numbers describe one
    // run rather than everything the process has done.
    virtual void reset_activation_saturation() {}

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

    // In-place GELU, the tanh approximation ggml's ggml_gelu uses --
    // 0.5x(1 + tanh(sqrt(2/pi)(x + 0.044715 x^3))). gemma's FFN is GeGLU, so
    // its gate matrix runs through this instead of silu; the two activations
    // agree near zero and differ in the tail, which is a plausible-text wrong
    // answer rather than a visible one, so the arch declares which it needs.
    virtual void gelu_act(void *x, i64 n) { (void)x; (void)n; }

    // x <- alpha * x over n_tok rows of `width` elements spaced `stride` apart,
    // and the elementwise product a[i] *= b[i] (both dense).
    virtual void scale_act(void *x, f32 alpha, i64 n_tok, i64 width, i64 stride) {
        (void)x; (void)alpha; (void)n_tok; (void)width; (void)stride;
    }
    virtual void mul_act(void *a, const void *b, i64 n) { (void)a; (void)b; (void)n; }

    // ---- hyper-connections (qwen4exp) ------------------------------------
    //
    // A qwen4exp stack does not carry one residual stream: it carries `hc` of
    // them (4), all n_embd wide, kept token-major as [n_tok, hc, n_embd]. Every
    // sub-block reads a MIX of the lanes and writes back into all of them, so
    // the three operations a plain `x += f(x)` residual cannot express live
    // here. The algebra and the invariant they keep -- all lanes equal in, all
    // lanes equal out while the injection is zero -- are in hc.hpp.
    //
    // A backend that does not implement them says so through hc_supported()
    // rather than running a default that writes nothing: a no-op hc_combine
    // leaves the lane stack at its initial value and the model decodes fluent
    // text that is not this model, which is the exact failure mode the
    // capability flag exists to prevent.
    virtual bool hc_supported() const { return false; }

    // lanes[t, c, e] <- x[t, e]: all hc lanes start as copies of the embedding
    // row (the reference's repeat_4d).
    virtual void hc_init(void *lanes, const void *x, i64 n_tok, i64 hc, i64 ne) {
        (void)lanes; (void)x; (void)n_tok; (void)hc; (void)ne;
    }
    // mixed[t, e] <- (1/hc) * sum_c xn[t, c, e] * sigmoid(gate[t, c, e]).
    // `gate` is pre-sigmoid (the mixer's up-projection output).
    virtual void hc_mix(void *mixed, const void *xn, const void *gate, i64 n_tok,
                        i64 hc, i64 ne) {
        (void)mixed; (void)xn; (void)gate; (void)n_tok; (void)hc; (void)ne;
    }
    // res[t, c, e] += block_out[t, e] * (2 * sigmoid(inject[t, c] / hc)).
    // The 2*sigmoid centring is what makes a zero injection a plain residual
    // add on every lane; see hc_lane_weight().
    virtual void hc_combine(void *res, const void *block_out, const void *inject,
                            i64 n_tok, i64 hc, i64 ne) {
        (void)res; (void)block_out; (void)inject; (void)n_tok; (void)hc; (void)ne;
    }
    // x <- x * sigmoid(x), in place: the mixer's low-rank nonlinearity, and NOT
    // silu_mul() (which needs a second operand) nor sigmoid_act() (which drops
    // the x factor).
    virtual void silu_act(void *x, i64 n) { (void)x; (void)n; }

    // ---- per-layer embedding (PLE), qwen4exp ------------------------------
    //
    // The n-gram block on the layers `ple.layers` names. Its two pieces a plain
    // elementwise op cannot express are the per-LANE dot product and the dilated
    // depthwise convolution along the token axis.

    // Per-lane gate:
    //   s[t,c]       = (sum_e key[t,c,e] * qry[t,c,e]) / sqrt(ne)
    //   out[c*n + t] = sigmoid(sign(s) * sqrt(max(|s|, 1e-6)))
    // `key` and `qry` are token-major [n_tok, hc, ne] and both per-lane normed
    // before they arrive. `out` is LANE-major [hc, n_tok] on purpose: a lane's
    // column is then contiguous, which is what the gated value broadcast below
    // reads one lane at a time.
    //
    // The signed square root is the reference's: it keeps the sign of the dot
    // product while compressing its magnitude, so a near-orthogonal pair lands
    // near 0.5 and an aligned one saturates the sigmoid.
    virtual void ple_gate(void *out, const void *key, const void *qry,
                          i64 n_tok, i64 hc, i64 ne) {
        (void)out; (void)key; (void)qry; (void)n_tok; (void)hc; (void)ne;
    }

    // Depthwise causal convolution along the TOKEN axis:
    //   out[t, c] = sum_k w[k*ch + c] * x[t + (kern-1-k)*dil, c]
    // `x` is [(kern-1)*dil + n_tok, ch] -- the previous tokens' rows in front of
    // this pass's -- and `out` is [n_tok, ch]. `w` is f32 [kern, ch], tap-major
    // (Model::upload_conv1d already produces exactly that layout). `dil` is the
    // n-gram size: the reference dilates by it so tap k looks k n-grams back.
    //
    // The history is the caller's: a chunked prefill and a token-at-a-time
    // decode must see the same window, which is why this op takes the window
    // rather than a ring.
    virtual void conv1d_dw(void *out, const void *x, const void *w,
                           i64 n_tok, i64 ch, i64 kern, i64 dil) {
        (void)out; (void)x; (void)w; (void)n_tok; (void)ch; (void)kern; (void)dil;
    }

    // LFM2.5's short convolution (`lfm2moe`): the mixer on the layers that are
    // not attention. `proj` is the layer's own projection, `[n_tok, 3*ch]` in
    // slices B | C | X, and the op is
    //     bx[t,c] = B[t,c] * X[t,c]
    //     out[t,c] = C[t,c] * sum_k w[k*ch+c] * bx[t-(kern-1)+k, c]
    // with the taps reaching into `state` for the rows before this call. `pos0`
    // is the ABSOLUTE position of row 0: the history is a ring keyed by it, so a
    // chunked prefill and a token-at-a-time decode address the same slots and a
    // call never has to shift the window. `state` is `hist * ch` values wide,
    // where `hist` is the ring's depth -- at least `kern - 1`, and deeper when
    // the model is speculated over (see Engine::sc_hist_).
    //
    // No activation sits between the convolution and the gate -- that is what
    // makes this not `conv1d_silu`, and it is the whole mixer plus the two
    // projections either side of it.
    virtual void shortconv(void *out, const void *proj, void *state,
                           const void *w, i64 n_tok, i64 ch, i64 kern,
                           i64 hist, i64 pos0) {
        (void)out; (void)proj; (void)state; (void)w;
        (void)n_tok; (void)ch; (void)kern; (void)hist; (void)pos0;
    }

    // The gated value broadcast: one value row per token (shared by every lane)
    // times that LANE's gate.
    //   out[t,c,e] = value[t,e] * g[c*n_tok + t]
    // `value` is token-major [n_tok, ne], `g` is lane-major [hc, n_tok] -- the
    // layout ple_gate produces -- and `out` is token-major [n_tok, hc, ne], which
    // is the layout the lane norm and the convolution both read.
    virtual void lane_bcast_mul(void *out, const void *value, const void *g,
                                i64 n_tok, i64 hc, i64 ne) {
        (void)out; (void)value; (void)g; (void)n_tok; (void)hc; (void)ne;
    }

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

// The virtual-NPU backend: the same reference implementation, with the compute
// handed to a VNPU device (github.com/riteshnair/npuxl-improved) — the whole
// GEMM path for `mode` = false, and every op the device has for `mode` = true
// (`--npu` vs `--npu-cpu`). Compiled only when KRK_ENABLE_NPU is defined (see
// CMakeLists.txt); otherwise make_npu_backend() returns nullptr and *err says
// so. `err` is filled on failure in both cases.
Backend *make_npu_backend(int device_index, bool whole_graph, std::string *err);
// True when this build has NPU support compiled in (the device itself may
// still fail to open).
bool npu_available();
// Number of visible HIP devices, or 0 when this build has no HIP support.
int hip_device_count();
// Human-readable runtime/version string for logging, or "unavailable".
std::string hip_runtime_version();

} // namespace krk

#endif // KRK_BACKEND_HPP
