// ============================================================================
//  backend_hip.hip — the HIP Backend implementation.
//
//  This translation unit is compiled once per --offload-arch. The kernel
//  headers pick their compute path from the __gfx*__ macros of that pass, so a
//  fat binary built for {gfx1031, gfx1201} carries a hand-written path for
//  each and device code is selected by the loader, not by a runtime branch.
// ============================================================================
#include <hip/hip_version.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>

#include "krk/backend.hpp"
#include "kernels/attention.hpp"
#include "kernels/fused.hpp"
#include "kernels/gemm.hpp"
#include "kernels/gemm_dp4a.hpp"
#include "kernels/gdn.hpp"

namespace krk {

// ---------------------------------------------------------------------------
// error handling
// ---------------------------------------------------------------------------

const char *hip_err_name(hipError_t e) { return hipGetErrorName(e); }

void hip_check(hipError_t e, const char *what) {
    if (e == hipSuccess) return;
    KRK_ERROR("HIP call failed (%s): %s", hip_err_name(e), what);
    std::abort();
}

// ---------------------------------------------------------------------------
// device capability probing
// ---------------------------------------------------------------------------

namespace {

std::string prop_gcn_arch(const hipDeviceProp_t &p) {
    std::string s = p.gcnArchName[0] ? p.gcnArchName : "";
    const size_t colon = s.find(':');
    if (colon != std::string::npos) s.resize(colon);
    if (s.empty()) s = "unknown";
    return s;
}

GfxFamily family_from_gfx(const std::string &gfx) {
    // gfx1030-1032 are RDNA2, gfx1100-1153 RDNA3/3.5, gfx1200/1201 RDNA4.
    auto starts = [&gfx](const char *p) { return gfx.rfind(p, 0) == 0; };
    if (starts("gfx120")) return GfxFamily::Rdna4;
    if (starts("gfx11")) return GfxFamily::Rdna3;
    if (starts("gfx103") || starts("gfx101")) return GfxFamily::Rdna2;
    return GfxFamily::Unknown;
}

} // namespace

int hip_device_count() {
    int n = 0;
    if (hipGetDeviceCount(&n) != hipSuccess) return 0;
    return n;
}

// HIP packs its version as major*10^7 + minor*10^5 + patch.
static std::string hip_version_string(int v) {
    return format("%d.%d.%d", v / 10000000, (v / 100000) % 100, v % 100000);
}

// Two HIP versions can disagree at run time: the SDK the binary was compiled
// against (HIP_VERSION) and the runtime DLL the loader actually resolved.
// Windows searches the executable's directory before System32, so staging the
// SDK's amdhip64_<major>.dll next to the binaries pins the runtime; without
// that, a driver-installed runtime of a different version silently wins.
// Report both when they differ instead of printing a plausible number.
std::string hip_runtime_version() {
    int v = 0;
    if (hipRuntimeGetVersion(&v) != hipSuccess) return "unavailable";
#if defined(HIP_VERSION)
    if (HIP_VERSION != v)
        return format("%s (SDK %s)", hip_version_string(v).c_str(),
                      hip_version_string(HIP_VERSION).c_str());
#endif
    return hip_version_string(v);
}

bool hip_runtime_matches_sdk() {
#if defined(HIP_VERSION)
    int v = 0;
    return hipRuntimeGetVersion(&v) == hipSuccess && v == HIP_VERSION;
#else
    return true;
#endif
}

// Real CU count from the KFD topology, or 0 when it cannot be read (Windows,
// a container without /sys, an unexpected layout). Every GPU node reports
// `simd_count`, which is SIMDs — two per CU — alongside a zero
// `cpu_cores_count` that distinguishes it from the CPU node. The largest
// such node wins, so a multi-GPU host still reports one card's width.
i64 kfd_compute_units() {
#if defined(__linux__)
    i64 best = 0;
    for (int node = 0; node < 64; node++) {
        char path[160];
        std::snprintf(path, sizeof(path),
                      "/sys/class/kfd/kfd/topology/nodes/%d/properties", node);
        std::FILE *f = std::fopen(path, "r");
        if (!f) continue;
        i64 simds = -1, cores = -1;
        char key[64];
        long long val = 0;
        while (std::fscanf(f, "%63s %lld", key, &val) == 2) {
            if (std::strcmp(key, "simd_count") == 0) simds = val;
            else if (std::strcmp(key, "cpu_cores_count") == 0) cores = val;
        }
        std::fclose(f);
        if (simds > 0 && cores == 0) best = std::max(best, simds / 2);
    }
    return best;
#else
    return 0;
#endif
}

// ---------------------------------------------------------------------------
// HipBackend
// ---------------------------------------------------------------------------

class HipBackend final : public Backend {
public:
    HipBackend(int index, const hipDeviceProp_t &p, const DeviceCaps &c)
        : caps_(c), index_(index) {
        (void)p;
        // Decode-layer GEMM fusion (kernels/fused.hpp): the
        // q/k/v and gate/up groups of a decode layer ride one
        // launch each. KRK_FUSED_LAYER=0 restores the separate
        // gemm() launches (A/B and fallback switch).
        const char *fe = std::getenv("KRK_FUSED_LAYER");
        fused_layer_ = !fe || std::strcmp(fe, "0") != 0;
        // Fused non-GEMM decode chain (kernels/attention.hpp):
        // RoPE + KV append + attention from one launch on the
        // decode row. KRK_FUSED_ATTN=0 restores the three
        // separate launches (A/B and fallback switch).
        const char *ae = std::getenv("KRK_FUSED_ATTN");
        attn_fused_ = !ae || std::strcmp(ae, "0") != 0;
        // int8 DP4A prefill GEMM for Q8_0 weights (kernels/gemm_dp4a.hpp).
        // Opt-in, unlike the two switches above: that path quantizes the
        // activations to int8, so its logits are close to the fp16 path's but
        // not bit-identical, and no default in this tree is allowed to be
        // lossy. KRK_GEMM_INT8=1 enables it for measurement; anything else
        // (including its absence) leaves the fp16 tile in place.
        const char *ie = std::getenv("KRK_GEMM_INT8");
        int8_gemm_ = ie && std::strcmp(ie, "0") != 0;
    }

    BackendKind kind() const override { return BackendKind::HIP; }
    const DeviceCaps &caps() const override { return caps_; }
    DType act_type() const override { return DType::F16; }
    size_t act_size() const override { return 2; }

    // ---- memory ----------------------------------------------------------
    void *alloc(size_t bytes) override {
        if (bytes == 0) bytes = 1;
        const size_t rounded = (bytes + 255u) & ~static_cast<size_t>(255u);
        void *p = nullptr;
        const hipError_t e = hipMalloc(&p, rounded);
        if (e != hipSuccess) {
            oom(rounded, e);
        }
        allocated_ += rounded;
        return p;
    }
    void release(void *p) override {
        if (p) KRK_HIP(hipFree(p));
    }
    // The expert cache's L2 tier. Page-locked so demote (device->host) and
    // promote (host->device) are straight DMAs; unpinned staging would spend
    // more time in the driver's bounce buffer than on the copy. Pinned memory
    // is a finite pool, so a failure degrades the tier instead of aborting the
    // run: the caller falls back to dropping the evicted expert to the mapping.
    void *alloc_host(size_t bytes) override {
        if (bytes == 0) bytes = 1;
        void *p = nullptr;
        if (hipHostMalloc(&p, bytes) != hipSuccess) {
            static_cast<void>(hipGetLastError()); // clear the sticky error, keep going
            KRK_WARN("expert L2 tier: pinned host allocation of %zu bytes failed",
                     bytes);
            return nullptr;
        }
        return p;
    }
    void release_host(void *p) override {
        if (p) KRK_HIP(hipHostFree(p));
    }
    void upload(void *dst, const void *src, size_t bytes, size_t off) override {
        KRK_HIP(hipMemcpy(static_cast<char *>(dst) + off, src, bytes,
                          hipMemcpyHostToDevice));
    }
    void download(void *dst, const void *src, size_t bytes, size_t off) override {
        KRK_HIP(hipMemcpy(dst, static_cast<const char *>(src) + off, bytes,
                          hipMemcpyDeviceToHost));
    }
    void fill0(void *dst, size_t bytes) override {
        KRK_HIP(hipMemset(dst, 0, bytes));
    }
    void sync() override { KRK_HIP(hipDeviceSynchronize()); }

    // Out of device memory. The bare hipErrorOutOfMemory tells the user
    // nothing they can act on, and the amount that failed to allocate is
    // usually only one slice of a much larger budget: report what the
    // card has, what this process already holds, and the flags that
    // actually move the numbers.
    [[noreturn]] void oom(size_t want, hipError_t e) const {
        size_t free_b = 0, total_b = 0;
        hipMemGetInfo(&free_b, &total_b);
        KRK_ERROR("out of VRAM: could not allocate %.1f MiB on %s", 
                  static_cast<double>(want) / (1024.0 * 1024.0), 
                  caps_.name.c_str());
        KRK_ERROR("  device total %.2f GiB, free %.2f GiB, already held by "
                  "this process %.2f GiB", 
                  static_cast<double>(total_b) / (1024.0 * 1024.0 * 1024.0),
                  static_cast<double>(free_b) / (1024.0 * 1024.0 * 1024.0),
                  static_cast<double>(allocated_) / (1024.0 * 1024.0 * 1024.0));
        KRK_ERROR("  (HIP reported %s)", hip_err_name(e));
        KRK_ERROR("  another process may be holding VRAM; check with rocm-smi "
                  "--showmeminfo vram");
        KRK_ERROR("  to fit in this much VRAM, try a smaller model, or:");
        KRK_ERROR("    --ctx N          smaller context (KV cache is the big term)");
        KRK_ERROR("    --chunk N        smaller prefill batch (cuts workspace)");
        KRK_ERROR("    --expert-cache-mb 0   for MoE, bound expert residency");
        KRK_ERROR("    --expert-cache-slots 1 --expert-l2-mb 0");
        std::abort();
    }

    void download_f32(f32 *dst, const void *src, i64 n) override {
        std::vector<u16> tmp(static_cast<size_t>(n));
        KRK_HIP(hipMemcpy(tmp.data(), src, static_cast<size_t>(n) * 2,
                          hipMemcpyDeviceToHost));
        for (i64 i = 0; i < n; i++) dst[i] = fp16_to_fp32(tmp[static_cast<size_t>(i)]);
    }

    std::string describe() const override {
        return format("%s (%s, %s, %d CU, wave32, %s)", caps_.name.c_str(),
                      caps_.gfx.c_str(), caps_.family_name(), caps_.cu_count,
                      caps_.has_wmma ? (caps_.has_wmma_gfx12 ? "WMMA gfx12" : "WMMA")
                                     : "packed-math");
    }

    // ---- ops -------------------------------------------------------------
    void embed(void *out, const void *tok_embd, DType tt, i64 n_vocab, i64 n_embd,
               const i32 *tokens, i64 n_tok) override {
        if (n_tok <= 0) return;
        const size_t row_bytes = dtype_row_bytes(tt, n_embd);
        // `tokens` is host memory (the Backend contract — the CPU backend
        // reads it on the host), but the gather kernel indexes it per
        // thread on the device. Stage the ids into a persistent scratch
        // so the prefill chunk and every decode step reuse one buffer.
        const size_t need = static_cast<size_t>(n_tok) * sizeof(i32);
        if (need > tok_cap_) {
            if (tok_dev_) KRK_HIP(hipFree(tok_dev_));
            tok_cap_ = (need + 255u) & ~static_cast<size_t>(255u);
            tok_dev_ = nullptr;
            KRK_HIP(hipMalloc(&tok_dev_, tok_cap_));
        }
        KRK_HIP(hipMemcpy(tok_dev_, tokens, need, hipMemcpyHostToDevice));
        embed_kernel<128><<<static_cast<unsigned>(n_tok), 128>>>(
            static_cast<_Float16 *>(out), static_cast<const u8 *>(tok_embd),
            static_cast<int>(tt), n_vocab, n_embd, row_bytes,
            static_cast<const i32 *>(tok_dev_), n_tok);
    }

    void rmsnorm(void *out, const void *x, const f32 *w, i64 rows,
                 i64 n, f32 eps) override {
        rmsnorm_kernel<256><<<static_cast<unsigned>(rows), 256>>>(
            static_cast<_Float16 *>(out),
            static_cast<const _Float16 *>(x), w, n, eps);
    }

    void gemm(void *out, const void *x, const void *w, DType wt, i64 n_out,
              i64 n_in, i64 rows) override {
        if (rows <= 0 || n_out <= 0 || n_in <= 0) return;
        const size_t row_bytes = dtype_row_bytes(wt, n_in);
        const _Float16 *xh = static_cast<const _Float16 *>(x);
        _Float16 *oh = static_cast<_Float16 *>(out);
        const u8 *wh = static_cast<const u8 *>(w);

        // The kernels address K in 32-value chunks, which every supported
        // format and every real model satisfies. Fail loudly rather than
        // silently mis-reading a stubby row.
        if ((n_in & 31) != 0) {
            KRK_ERROR("gemm: n_in=%lld is not a multiple of 32",
                      static_cast<long long>(n_in));
            std::abort();
        }

        if (rows == 1) {
            // One 256-thread block covers THREADS/32 rows (one warp
            // per row), so the grid is n_out warps, not n_out blocks —
            // launching n_out blocks leaves 7/8 of them empty.
            gemv_kernel<256><<<static_cast<unsigned>((n_out + 7) / 8),
                                256>>>(
                wh, static_cast<int>(wt), n_out, n_in, row_bytes, xh, oh);
        } else if (caps_.has_wmma) {
#if defined(KRK_GFX11) || defined(KRK_GFX12)
            const unsigned bx = static_cast<unsigned>(
                (n_out + wmma_cfg::BN - 1) / wmma_cfg::BN);
            const unsigned by = static_cast<unsigned>(
                (rows + wmma_cfg::BM - 1) / wmma_cfg::BM);
            // Split-K: a short prefill GEMM runs only a
            // handful of blocks, and §2b showed its cost is
            // the per-block staging chain, not CU occupancy.
            // Slicing K shortens the chain per block; the
            // fp32 partials are summed by a second,
            // memory-bound kernel. The cost model in
            // pick_split leaves long-chain shapes (gate/up,
            // down) and large grids unsplit, so only the
            // shapes that pay take the extra reduce.
            i64 slice = n_in;
            const int split = pick_split(static_cast<int>(bx * by),
                                         n_in, wmma_cfg::BK,
                                         caps_.cu_count, &slice);
            const dim3 grid(bx, by,
                            static_cast<unsigned>(split));
            f32 *parts = nullptr;
            i64 pstride = 0;
            if (split > 1) {
                const size_t need =
                    static_cast<size_t>(split) *
                    static_cast<size_t>(rows) *
                    static_cast<size_t>(n_out) * sizeof(f32);
                if (need > split_cap_) {
                    if (split_dev_) KRK_HIP(hipFree(split_dev_));
                    split_cap_ =
                        (need + 255u) & ~static_cast<size_t>(255u);
                    split_dev_ = nullptr;
                    KRK_HIP(hipMalloc(&split_dev_, split_cap_));
                }
                parts = split_dev_;
                pstride = rows * n_out;
            }
            // Q4_K takes the compile-time-specialized staging
            // pipeline; every other dtype keeps the runtime path.
            if (wt == DType::Q4_K)
                gemm_wmma_kernel<wmma_cfg, 0,
                                 static_cast<int>(DType::Q4_K)><<<grid, wmma_cfg::THREADS>>>(
                    wh, static_cast<int>(wt), n_out, n_in,
                    row_bytes, xh, rows, oh, split, slice,
                    parts, pstride);
            else
                gemm_wmma_kernel<wmma_cfg, 0, -1><<<grid, wmma_cfg::THREADS>>>(
                    wh, static_cast<int>(wt), n_out, n_in,
                    row_bytes, xh, rows, oh, split, slice,
                    parts, pstride);
            if (split > 1) {
                const i64 total = rows * n_out;
                gemm_reduce_kernel<256>
                    <<<static_cast<unsigned>((total + 255) / 256),
                        256>>>(parts, split, pstride, total, oh);
            }
#endif
        } else if (int8_gemm_ && dp4a_ok(static_cast<int>(wt), rows, n_out, n_in)) {
            // The opt-in int8 path (see the constructor and the header of
            // kernels/gemm_dp4a.hpp). Q8_0 only, because Q8_0's payload is
            // already int8; rows >= 16, because a tile below that spends more
            // on its own staging than on the rows it has. Everything else
            // keeps the fp16 tile, which is also what happens when the flag
            // is absent.
            //
            // Quantized-activation staging: [rows, n_in] codes plus one fp32
            // scale per 32-value block. Grown on demand and kept across calls,
            // like the split-K partials, so the hot path never allocates.
            const size_t need =
                static_cast<size_t>(rows) * static_cast<size_t>(n_in);
            if (need > int8q_cap_) {
                if (int8q_) KRK_HIP(hipFree(int8q_));
                int8q_cap_ = (need + 255u) & ~static_cast<size_t>(255u);
                int8q_ = nullptr;
                KRK_HIP(hipMalloc(&int8q_, int8q_cap_));
            }
            const size_t sneed = static_cast<size_t>(rows) *
                                 static_cast<size_t>(n_in / 32) * sizeof(f32);
            if (sneed > int8s_cap_) {
                if (int8s_) KRK_HIP(hipFree(int8s_));
                int8s_cap_ = (sneed + 255u) & ~static_cast<size_t>(255u);
                int8s_ = nullptr;
                KRK_HIP(hipMalloc(&int8s_, int8s_cap_));
            }
            gemm_dp4a_launch(wh, n_out, n_in, row_bytes, xh, rows, oh,
                             reinterpret_cast<i8 *>(int8q_), int8s_);
        } else {
            const dim3 grid(
                static_cast<unsigned>((n_out + simt_cfg::BN - 1) / simt_cfg::BN),
                static_cast<unsigned>((rows + simt_cfg::BM - 1) / simt_cfg::BM));
            gemm_simt_kernel<<<grid, simt_cfg::THREADS>>>(
                wh, static_cast<int>(wt), n_out, n_in, row_bytes, xh, rows, oh);
        }
    }

    bool gemm_group(void *const out[], const void *x,
                    const void *const w[], DType wt,
                    const i64 n_out[], i64 n_in, int n_mat,
                    i64 rows) override {
        // The fused kernels express the M=1 decode row only
        // (row-parallel over n_out, no M dimension); prefill
        // rows and anything the group cannot express keep the
        // per-matrix gemm() path, which is what the engine ran
        // before the fusion existed.
        if (rows != 1 || n_mat < 1 || n_mat > 8 || !fused_layer_) {
            for (int m = 0; m < n_mat; m++)
                gemm(out[m], x, w[m], wt, n_out[m], n_in, rows);
            return false;
        }
        // Same 32-channel addressing rule as gemm().
        if ((n_in & 31) != 0) {
            KRK_ERROR("gemm_group: n_in=%lld is not a multiple of 32",
                      static_cast<long long>(n_in));
            std::abort();
        }
        const u8 *wp[8];
        const _Float16 *in[8];
        _Float16 *op[8];
        i64 nin[8];
        size_t rb[8];
        i64 need = 0; // worst-case fp32 partials, elements
        for (int m = 0; m < n_mat; m++) {
            wp[m] = static_cast<const u8 *>(w[m]);
            in[m] = static_cast<const _Float16 *>(x);
            op[m] = static_cast<_Float16 *>(out[m]);
            nin[m] = n_in;
            rb[m] = dtype_row_bytes(wt, n_in);
            need += 8 * n_out[m]; // the split policy tops out at 8 slices
        }
        // The fused group shares the split-K partials scratch;
        // its part_cap counts fp32 elements, split_cap_ bytes.
        const size_t need_bytes =
            static_cast<size_t>(need) * sizeof(f32);
        if (need_bytes > split_cap_) {
            if (split_dev_) KRK_HIP(hipFree(split_dev_));
            split_cap_ =
                (need_bytes + 255u) & ~static_cast<size_t>(255u);
            split_dev_ = nullptr;
            KRK_HIP(hipMalloc(&split_dev_, split_cap_));
        }
        fused_layer_launch(wp, in, op, n_out, nin, rb, n_mat,
                           split_dev_, split_cap_ / sizeof(f32),
                           caps_.cu_count, static_cast<int>(wt));
        return true;
    }

    void rope(void *q, void *k, i64 n_head, i64 n_kv, i64 hd, i64 n_tok, i64 pos0,
              const f32 *inv_freq, f32 scale, f32 frac) override {
        // `inv_freq` is host memory (Model keeps the table in a
        // std::vector), but the kernel reads it per thread on the
        // device; stage_freq() stages it, caching the upload.
        const f32 *fd = stage_freq(inv_freq, hd);
        stage_pos(pos0);
        const dim3 gq(static_cast<unsigned>(n_head), static_cast<unsigned>(n_tok));
        rope_kernel<<<gq, 128>>>(static_cast<_Float16 *>(q), n_head, hd,
                                 pos_dev_, fd, scale, frac);
        const dim3 gk(static_cast<unsigned>(n_kv), static_cast<unsigned>(n_tok));
        rope_kernel<<<gk, 128>>>(static_cast<_Float16 *>(k), n_kv, hd,
                                 pos_dev_, fd, scale, frac);
    }

    void qk_norm(void *q, void *k, const f32 *wq, const f32 *wk, i64 n_head,
                 i64 n_kv, i64 hd, i64 n_tok, f32 eps) override {
        if (wq) {
            head_norm_kernel<128>
                <<<dim3(static_cast<unsigned>(n_head), static_cast<unsigned>(n_tok)),
                   128>>>(static_cast<_Float16 *>(q), wq, n_head, hd, eps);
        }
        if (wk) {
            head_norm_kernel<128>
                <<<dim3(static_cast<unsigned>(n_kv), static_cast<unsigned>(n_tok)),
                   128>>>(static_cast<_Float16 *>(k), wk, n_kv, hd, eps);
        }
    }

    void kv_append(void *kcache, void *vcache, const void *k, const void *v,
                   const AttnDesc &d) override {
        _Float16 *kc = static_cast<_Float16 *>(kcache) + d.layer * d.layer_stride;
        _Float16 *vc = static_cast<_Float16 *>(vcache) + d.layer * d.layer_stride;
        const i64 total = d.n_tok * d.pos_stride;
        const unsigned blocks = static_cast<unsigned>((total + 255) / 256);
        stage_pos(d.pos0);
        kv_append_kernel<<<blocks, 256>>>(kc, vc,
                                          static_cast<const _Float16 *>(k),
                                          static_cast<const _Float16 *>(v),
                                          pos_dev_, d.n_tok, d.pos_stride);
    }

    void attention(void *out, const void *q, const void *kcache, const void *vcache,
                   const AttnDesc &d) override {
        const void *kc =
            static_cast<const _Float16 *>(kcache) + d.layer * d.layer_stride;
        const void *vc =
            static_cast<const _Float16 *>(vcache) + d.layer * d.layer_stride;

        // Decode (M = 1): one block per head with warp-strided keys instead of
        // the tiled kernel's single-lane channel-per-thread split, which only
        // fills 9 blocks on a 32-CU device. Falls back for head widths the
        // packed lane slice cannot cover.
        if (d.n_tok == 1 && attention_decode_supported(d.hd)) {
            stage_pos(d.pos0);
            attention_decode(out, q, kc, vc, d.n_head, d.n_kv, d.hd,
                             pos_dev_, d.pos_stride, d.causal ? 1 : 0,
                             d.scale);
            return;
        }

        // Size the key tile so the two staged tiles stay near 32 KiB of LDS.
        const int tile_k = static_cast<int>(
            std::min<i64>(64, std::max<i64>(8, 8192 / std::max<i64>(1, d.hd))));
        const size_t smem = attn_smem_bytes(tile_k, d.hd);
        if (smem > caps_.lds_bytes) {
            KRK_ERROR("attention: %zu bytes of LDS requested, device has %d", smem,
                      caps_.lds_bytes);
            std::abort();
        }
        const dim3 grid(static_cast<unsigned>(d.n_head),
                        static_cast<unsigned>(d.n_tok));
        KRK_HIP(hipFuncSetAttribute(
            reinterpret_cast<const void *>(&attention_kernel),
            hipFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem + 512)));
        attention_kernel<<<grid, kAttnBlock, smem>>>(
            static_cast<_Float16 *>(out), static_cast<const _Float16 *>(q),
            static_cast<const _Float16 *>(kc), static_cast<const _Float16 *>(vc),
            d.n_head, d.n_kv, d.hd, d.n_tok, d.pos0, d.pos_stride, d.causal ? 1 : 0,
            d.scale, tile_k);
    }

    bool attn_fused_chain(void *out, void *q, void *kcache, void *vcache,
                            const void *k, const void *v, const AttnDesc &d,
                            const f32 *inv_freq, f32 rope_scale,
                            f32 rope_frac) override {
        // The fused kernel expresses the single-query-token decode row
        // only, and only the head widths its packed lane slice covers
        // (the same rule as the decode attention path above). Anything
        // else keeps the separate rope() + kv_append() + attention()
        // chain the engine runs as the fallback.
        if (d.n_tok != 1 || !attn_fused_decode_supported(d.hd) ||
            !attn_fused_)
            return false;
        const f32 *fd = stage_freq(inv_freq, d.hd);
        stage_pos(d.pos0);
        attn_fused_decode(out, q,
                          static_cast<_Float16 *>(kcache) +
                              d.layer * d.layer_stride,
                          static_cast<_Float16 *>(vcache) +
                              d.layer * d.layer_stride,
                          k, v, d.n_head, d.n_kv, d.hd, pos_dev_,
                          d.pos_stride, fd, rope_scale, d.scale,
                          rope_frac);
        return true;
    }

    void silu_mul(void *out, const void *gate, const void *up, i64 n) override {
        const unsigned blocks = static_cast<unsigned>((n + 255) / 256);
        silu_mul_kernel<<<blocks, 256>>>(static_cast<_Float16 *>(out),
                                         static_cast<const _Float16 *>(gate),
                                         static_cast<const _Float16 *>(up), n);
    }

    void add_bias_rows(void *out, const f32 *bias, i64 n, i64 rows) override {
        const i64 total = n * rows;
        const unsigned blocks = static_cast<unsigned>((total + 255) / 256);
        add_bias_rows_kernel<<<blocks, 256>>>(static_cast<_Float16 *>(out), bias, n,
                                              total);
    }

    void add_inplace(void *a, const void *b, i64 n) override {
        const unsigned blocks = static_cast<unsigned>((n + 255) / 256);
        add_inplace_kernel<<<blocks, 256>>>(static_cast<_Float16 *>(a),
                                            static_cast<const _Float16 *>(b), n);
    }

    void copy_act(void *dst, const void *src, i64 n) override {
        const unsigned blocks = static_cast<unsigned>((n + 255) / 256);
        copy_h2h_kernel<<<blocks, 256>>>(static_cast<_Float16 *>(dst),
                                         static_cast<const _Float16 *>(src), n);
    }

    void axpy(void *dst, const void *src, f32 alpha, i64 n) override {
        const unsigned blocks = static_cast<unsigned>((n + 255) / 256);
        axpy_kernel<<<blocks, 256>>>(static_cast<_Float16 *>(dst),
                                     static_cast<const _Float16 *>(src), alpha, n);
    }

    void upload_i32(i32 *dst, const i32 *src, i64 n) override {
        KRK_HIP(hipMemcpy(dst, src, static_cast<size_t>(n) * sizeof(i32),
                          hipMemcpyHostToDevice));
    }

    // ---- gated delta net (Qwen3.5 / Qwen3-Next) -------------------------
    // kernels/gdn.hpp, one launch per op, f16 activations over the f32
    // recurrent state. The model loader consults gdn_supported() and refuses a
    // recurrent checkpoint rather than running it on a backend that would
    // silently skip these.
    bool gdn_supported() const override { return true; }

    void l2norm(void *o, const void *x, i64 n_tok, i64 n_head, i64 hd,
                i64 stride, f32 eps) override {
        if (n_tok <= 0 || n_head <= 0 || hd <= 0) return;
        const dim3 grid(static_cast<unsigned>(n_head),
                        static_cast<unsigned>(std::min<i64>(n_tok, 65535)));
        _Float16 *od = static_cast<_Float16 *>(o);
        const _Float16 *xd = static_cast<const _Float16 *>(x);
        // hd is the reduction width, so the block matches it: one thread per
        // channel at 128, which is what Qwen3.5's head width is.
        if (hd <= 64)
            gdn_l2norm_kernel<64><<<grid, 64>>>(od, xd, hd, stride, eps);
        else
            gdn_l2norm_kernel<128><<<grid, 128>>>(od, xd, hd, stride, eps);
    }

    void sigmoid_act(void *x, i64 n) override {
        if (n <= 0) return;
        const unsigned blocks = static_cast<unsigned>((n + 255) / 256);
        gdn_sigmoid_kernel<<<blocks, 256>>>(static_cast<_Float16 *>(x), n);
    }

    void softplus_act(void *x, i64 n) override {
        if (n <= 0) return;
        const unsigned blocks = static_cast<unsigned>((n + 255) / 256);
        gdn_softplus_kernel<<<blocks, 256>>>(static_cast<_Float16 *>(x), n);
    }

    void scale_act(void *x, f32 alpha, i64 n_tok, i64 width, i64 stride) override {
        if (n_tok <= 0 || width <= 0) return;
        const dim3 grid(static_cast<unsigned>((width + 255) / 256),
                        static_cast<unsigned>(std::min<i64>(n_tok, 65535)));
        gdn_scale_act_kernel<<<grid, 256>>>(static_cast<_Float16 *>(x), alpha,
                                            width, stride, n_tok);
    }

    void mul_act(void *a, const void *b, i64 n) override {
        if (n <= 0) return;
        const unsigned blocks = static_cast<unsigned>((n + 255) / 256);
        gdn_mul_act_kernel<<<blocks, 256>>>(static_cast<_Float16 *>(a),
                                            static_cast<const _Float16 *>(b), n);
    }

    void add_bias_cols(void *x, const f32 *bias, i64 n_row, i64 n_col) override {
        if (n_row <= 0 || n_col <= 0) return;
        const dim3 grid(1, static_cast<unsigned>(std::min<i64>(n_row, 65535)));
        gdn_add_bias_cols_kernel<<<grid, 64>>>(static_cast<_Float16 *>(x), bias,
                                               n_col, n_row);
    }

    void scale_cols(void *x, const f32 *col, i64 n_row, i64 n_col) override {
        if (n_row <= 0 || n_col <= 0) return;
        const dim3 grid(1, static_cast<unsigned>(std::min<i64>(n_row, 65535)));
        gdn_scale_cols_kernel<<<grid, 64>>>(static_cast<_Float16 *>(x), col, n_col,
                                            n_row);
    }

    void scale_rows(void *x, const void *alpha, i64 n_row, i64 n) override {
        if (n_row <= 0 || n <= 0) return;
        const dim3 grid(1, static_cast<unsigned>(std::min<i64>(n_row, 65535)));
        gdn_scale_rows_kernel<<<grid, 256>>>(static_cast<_Float16 *>(x),
                                             static_cast<const _Float16 *>(alpha),
                                             n, n_row);
    }

    void qwen3_next_split(void *q, void *gate, const void *packed, i64 n_tok,
                          i64 n_head, i64 hd) override {
        if (n_tok <= 0 || n_head <= 0 || hd <= 0) return;
        const dim3 grid(static_cast<unsigned>(n_head),
                        static_cast<unsigned>(std::min<i64>(n_tok, 65535)));
        gdn_qwen3_next_split_kernel<<<grid, 128>>>(static_cast<_Float16 *>(q),
                                                   static_cast<_Float16 *>(gate),
                                                   static_cast<const _Float16 *>(packed),
                                                   hd, n_tok);
    }

    void conv1d_silu(void *out, const void *in, void *state, const void *kern,
                     DType wt, i64 n_tok, i64 chan, i64 ksize) override {
        if (n_tok <= 0 || chan <= 0 || ksize <= 0) return;
        // The taps are a constant weight: decode them once per layer, not once
        // per token. This is the only place a quantized buffer is read on the
        // delta net's hot path.
        const _Float16 *taps = stage_gdn_taps(kern, wt, ksize * chan);
        gdn_conv1d_silu_launch(static_cast<_Float16 *>(out),
                               static_cast<const _Float16 *>(in),
                               static_cast<f32 *>(state), taps, n_tok, chan,
                               ksize);
    }

    void delta_rule(void *out, void *state, const void *q, const void *k,
                    const void *v, const void *g, const void *beta, i64 n_tok,
                    i64 n_k_head, i64 n_v_head, i64 d_state, i64 hd,
                    i64 row_stride) override {
        gdn_delta_rule_launch(
            static_cast<_Float16 *>(out), static_cast<f32 *>(state),
            static_cast<const _Float16 *>(q), static_cast<const _Float16 *>(k),
            static_cast<const _Float16 *>(v), static_cast<const _Float16 *>(g),
            static_cast<const _Float16 *>(beta), n_tok, n_k_head, n_v_head,
            d_state, hd, row_stride);
    }

    void gather_rows(void *dst, const void *src, const i32 *rows, i64 n_rows,
                     i64 n) override {
        if (n_rows <= 0 || n <= 0) return;
        gather_rows_kernel<<<static_cast<unsigned>(n_rows), 256>>>(
            static_cast<_Float16 *>(dst),            static_cast<const _Float16 *>(src), rows,
            n);
    }

    void scatter_axpy_rows(void *dst, const void *src, const i32 *rows,
                           const f32 *alpha, i64 n_rows, i64 n) override {
        if (n_rows <= 0 || n <= 0) return;
        // alpha is host memory; stage it next to the row plan (pinned-style
        // staging through a small scratch keeps this one memcpy).
        std::vector<f32> a(static_cast<size_t>(n_rows));
        std::memcpy(a.data(), alpha, static_cast<size_t>(n_rows) * sizeof(f32));
        f32 *alpha_dev = static_cast<f32 *>(alloc(a.size() * sizeof(f32)));
        KRK_HIP(hipMemcpy(alpha_dev, a.data(), a.size() * sizeof(f32),
                          hipMemcpyHostToDevice));
        scatter_axpy_rows_kernel<<<static_cast<unsigned>(n_rows), 256>>>(
            static_cast<_Float16 *>(dst), static_cast<const _Float16 *>(src),
            rows, alpha_dev, n);
        release(alpha_dev);
    }

    ~HipBackend() override {
        if (tok_dev_) {
            (void)hipFree(tok_dev_);
            tok_dev_ = nullptr;
        }
        if (freq_dev_) {
            (void)hipFree(freq_dev_);
            freq_dev_ = nullptr;
        }
        if (pos_dev_) {
            (void)hipFree(pos_dev_);
            pos_dev_ = nullptr;
        }
        if (split_dev_) {
            (void)hipFree(split_dev_);
            split_dev_ = nullptr;
        }
        if (int8q_) {
            (void)hipFree(int8q_);
            int8q_ = nullptr;
        }
        if (int8s_) {
            (void)hipFree(int8s_);
            int8s_ = nullptr;
        }
        for (const GdnTaps &e : gdn_taps_) (void)hipFree(e.dev);
        gdn_taps_.clear();
    }

private:
    DeviceCaps caps_;
    int index_ = 0;
    size_t allocated_ = 0;
    // Decode-layer GEMM fusion on/off (see the constructor).
    bool fused_layer_ = true;
    // Fused non-GEMM decode chain on/off (see the constructor).
    bool attn_fused_ = true;
    // Opt-in int8 DP4A prefill GEMM on Q8_0 weights (see the constructor).
    bool int8_gemm_ = false;
    // Quantized activations for that path: [rows, n_in] int8 codes and one
    // fp32 scale per 32-value block, grown on demand.
    void *int8q_ = nullptr;
    size_t int8q_cap_ = 0;
    f32 *int8s_ = nullptr;
    size_t int8s_cap_ = 0;
    // Device staging for the host token-id array embed() reads (grown
    // on demand; see embed()).
    void *tok_dev_ = nullptr;
    size_t tok_cap_ = 0;
    // Device staging for the host inv_freq table rope() reads; the
    // (pointer, length, signature) triple caches the upload.
    void *freq_dev_ = nullptr;
    size_t freq_cap_ = 0;
    const f32 *freq_src_ = nullptr;
    size_t freq_n_ = 0;
    f32 freq_sig_ = 0.0f;
    // Device-side position scalar for the decode path (rope,
    // kv_append, decode attention). Those kernels read pos0 from
    // device memory so a captured HIP graph can replay a whole
    // decode step without re-binding kernel arguments. One 8-byte
    // upload per distinct position covers the three calls of a
    // step, which all share the same value.
    i64 *pos_dev_ = nullptr;
    i64 pos_staged_ = 0;
    bool pos_valid_ = false;
    // fp32 partials for split-K prefill GEMMs (gemm():
    // one buffer per split x row x channel, grown on
    // demand to the largest split GEMM the model runs).
    f32 *split_dev_ = nullptr;
    size_t split_cap_ = 0;
    // Decoded short-convolution taps, one entry per recurrent layer (see
    // stage_gdn_taps). The engine walks the layers in order, so the scan is one
    // entry per layer compared against one pointer.
    struct GdnTaps {
        const void *src;
        int type;
        i64 n;
        _Float16 *dev;
    };
    std::vector<GdnTaps> gdn_taps_;

    const _Float16 *stage_gdn_taps(const void *kern, DType wt, i64 n) {
        const int t = static_cast<int>(wt);
        for (const GdnTaps &e : gdn_taps_)
            if (e.src == kern && e.type == t && e.n == n) return e.dev;
        // Whole 32-value chunks only, which every real conv width satisfies
        // (and is the same limit the host dequant has).
        if (n <= 0 || (n & 31) != 0) {
            KRK_ERROR("gdn conv taps: %lld values is not a whole number of "
                      "32-value chunks",
                      static_cast<long long>(n));
            std::abort();
        }
        _Float16 *dev =
            static_cast<_Float16 *>(alloc(static_cast<size_t>(n) * sizeof(_Float16)));
        const unsigned blocks = static_cast<unsigned>(((n / 32) + 255) / 256);
        gdn_dequant_taps_kernel<<<blocks, 256>>>(dev,
                                                 static_cast<const u8 *>(kern), t, n);
        gdn_taps_.push_back({kern, t, n, dev});
        return dev;
    }

    void stage_pos(i64 pos0) {
        if (pos_valid_ && pos_staged_ == pos0) return;
        if (!pos_dev_) KRK_HIP(hipMalloc(&pos_dev_, sizeof(i64)));
        KRK_HIP(hipMemcpy(pos_dev_, &pos0, sizeof(i64),
                          hipMemcpyHostToDevice));
        pos_staged_ = pos0;
        pos_valid_ = true;
    }

    // Stages the host inv_freq table for the device-side RoPE
    // kernels (rope(), attn_fused_chain()); the (pointer, length,
    // signature) triple caches the upload, so the layers of a
    // step share one memcpy.
    const f32 *stage_freq(const f32 *inv_freq, i64 hd) {
        const size_t n = static_cast<size_t>(hd / 2);
        const f32 sig = n >= 2 ? inv_freq[1] : 0.0f;
        if (inv_freq != freq_src_ || n != freq_n_ || sig != freq_sig_) {
            const size_t need = n * sizeof(f32);
            if (need > freq_cap_) {
                if (freq_dev_) KRK_HIP(hipFree(freq_dev_));
                freq_cap_ = (need + 255u) & ~static_cast<size_t>(255u);
                freq_dev_ = nullptr;
                KRK_HIP(hipMalloc(&freq_dev_, freq_cap_));
            }
            KRK_HIP(hipMemcpy(freq_dev_, inv_freq, need,
                              hipMemcpyHostToDevice));
            freq_src_ = inv_freq;
            freq_n_ = n;
            freq_sig_ = sig;
        }
        return static_cast<const f32 *>(freq_dev_);
    }
};

// ---------------------------------------------------------------------------
// factory
// ---------------------------------------------------------------------------

Backend *make_hip_backend(int device_index, std::string *err) {
    int count = 0;
    if (hipGetDeviceCount(&count) != hipSuccess || count <= 0) {
        if (err) *err = "no HIP devices are visible (check the HIP SDK install and PATH)";
        return nullptr;
    }
    if (device_index < 0 || device_index >= count) device_index = 0;
    if (hipSetDevice(device_index) != hipSuccess) {
        if (err) *err = "hipSetDevice failed";
        return nullptr;
    }

    hipDeviceProp_t props{};
    if (hipGetDeviceProperties(&props, device_index) != hipSuccess) {
        if (err) *err = "hipGetDeviceProperties failed";
        return nullptr;
    }

    DeviceCaps c;
    c.name = props.name;
    c.gfx = prop_gcn_arch(props);
    c.family = family_from_gfx(c.gfx);
    c.wave_size = 32; // every RDNA target here is wave32
    c.cu_count = props.multiProcessorCount;
    // HIP under-reports multiProcessorCount on some gfx10 parts: a RX 6700 XT
    // (gfx1031, 40 CUs) comes back as 20. Every occupancy heuristic here is
    // sized against cu_count, so believing it halves the launch geometry and
    // leaves half the card idle. KFD's per-node topology is authoritative —
    // simd_count is SIMDs, two per CU — so prefer it and keep the HIP value
    // only as a fallback.
    {
        const i64 real_cus = kfd_compute_units();
        if (real_cus > 0) c.cu_count = static_cast<int>(real_cus);
    }
    c.lds_bytes = props.sharedMemPerBlock > 0 ? props.sharedMemPerBlock : 65536;
    c.vram_total = props.totalGlobalMem;
    c.vram_free = props.totalGlobalMem * 9 / 10;
    c.has_wmma = (c.family == GfxFamily::Rdna3 || c.family == GfxFamily::Rdna4);
    c.has_wmma_gfx12 = (c.family == GfxFamily::Rdna4);
    c.has_dot2 = (c.family == GfxFamily::Rdna2 || c.family == GfxFamily::Rdna3);

    KRK_INFO("HIP %s | %s | %s | %d CU | %.1f GiB VRAM", hip_runtime_version().c_str(),
             c.name.c_str(), c.gfx.c_str(), c.cu_count,
             static_cast<f64>(c.vram_total) / (1024.0 * 1024.0 * 1024.0));
    if (!hip_runtime_matches_sdk())
        KRK_WARN("the loaded HIP runtime is not the SDK this binary was built "
                 "against; stage the SDK's amdhip64_*.dll next to the binary "
                 "to pin the runtime");
    if (c.family == GfxFamily::Unknown)
        KRK_WARN("unknown gfx target '%s': kernels fall back to the SIMT path",
                 c.gfx.c_str());

    return new HipBackend(device_index, props, c);
}

} // namespace krk
