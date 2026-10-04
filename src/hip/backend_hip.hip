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
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <thread>

#include "krk/backend.hpp"
#include "kernels/attention.hpp"
#include "kernels/topk.hpp"
#include "kernels/fused.hpp"
#include "kernels/gemm.hpp"
#include "kernels/gemm_dp4a.hpp"
#include "kernels/gdn.hpp"
#include "op_time.hpp"

// Declared here rather than at their definitions (below the class) so the
// backend's upload path can call them; see the definitions for the measurements.
static constexpr size_t kTouchMinBytes = 64u * 1024u;
void prefault(const void *src, size_t bytes);
static double phase_now() {
    return static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::steady_clock::now().time_since_epoch()).count()) / 1000.0;
}
// Function pointer: inside the class, unqualified `prefault` would find a
// member, so the upload path takes the address through this instead.
using PrefaultFn = void (*)(const void *, size_t);
static const PrefaultFn kPrefaultFn = &prefault;

// Pulls a range into residency so the following copy is a straight memcpy.
// Reading one byte per page is enough -- the fault is the cost, not the byte.
// Measured cold on the 8B mapping: 1 byte/page 459 ms, sequential 64B stride
// 668 ms, full memcpy 1374 ms, so the sparse touch is both the cheapest and the
// one that leaves memory bandwidth for the DMA.
void prefault(const void *src, size_t bytes) {
    const volatile char *p = static_cast<const volatile char *>(src);
    const size_t pages = bytes >> 12;
    size_t acc = 0;
    for (size_t i = 0; i < pages; i++) acc += static_cast<size_t>(p[i << 12]);
    // Defeat dead-code elimination without branching on the accumulated value.
    if (acc == static_cast<size_t>(-1)) std::fprintf(stderr, " ");
}

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

// The four top-k launches for one logits row, typed on how the row is stored.
// Nothing between them touches the host: stage 2 writes the cut and stage 3
// reads it back from device memory, so the whole ranking is one async chain.
template <typename T, int kThreads>
void topk_launch(const T *logits, const u32 *pm, f32 rep_pen, f32 softcap,
                 f32 inv_temp, i32 n, i32 k, u32 *hist,
                 unsigned long long *gmax, u32 *cut, unsigned long long *count,
                 float *mass, unsigned long long *cand, i32 cand_cap, i32 *ids,
                 f32 *vals, unsigned long long *count_out, unsigned blocks,
                 i32 span) {
    topk_hist_kernel<T, kThreads><<<blocks, kThreads>>>(
        logits, pm, rep_pen, softcap, hist, gmax, n, span);
    topk_scan_kernel<<<1, 1>>>(hist, k, cut);
    topk_collect_kernel<T, kThreads><<<blocks, kThreads>>>(
        logits, pm, rep_pen, softcap, cut, gmax, cand, cand_cap, count, mass,
        inv_temp, n, span);
    topk_final_kernel<kThreads><<<1, kThreads>>>(cand, count, cand_cap, k, ids,
                                                vals, count_out);
}

// The GEMM ladder this backend walks, and the switch that pins a rung for
// measurement. See HipBackend::gemm_rung for how the device picks.
enum class GemmRung { kWmma, kDp4a, kSimt };
enum class GemmPath { kAuto, kWmma, kDp4a, kSimt };

// Small hot-path H2D staging (see HipBackend::stage_upload): one pinned
// allocation, kStageSlots slots, rotated per use, each guarded by its own
// event. kStageMaxBytes is the largest upload that rides the ring; a larger
// source is a weight that goes through the pull path or a pageable copy.
static constexpr size_t kStageMaxBytes = 64u * 1024u;
static constexpr int kStageSlots = 64;

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
        // Which rung of the GEMM ladder to use, for measurement and for
        // pinning a path when comparing two builds. Absent or "auto" means the
        // device decides (see gemm_rung).
        const char *pe = std::getenv("KRK_GEMM_PATH");
        if (pe == nullptr || std::strcmp(pe, "auto") == 0) path_ = GemmPath::kAuto;
        else if (std::strcmp(pe, "wmma") == 0) path_ = GemmPath::kWmma;
        else if (std::strcmp(pe, "dp4a") == 0) path_ = GemmPath::kDp4a;
        else if (std::strcmp(pe, "simt") == 0) path_ = GemmPath::kSimt;
        else {
            KRK_WARN("KRK_GEMM_PATH=%s is not one of auto/wmma/dp4a/simt; "
                     "using the device's own choice", pe);
            path_ = GemmPath::kAuto;
        }
        // A rung that was asked for and is not there is a warning, not a
        // failure: the ladder falls through to the next one.
        if (path_ == GemmPath::kWmma && !caps_.has_wmma)
            KRK_WARN("KRK_GEMM_PATH=wmma, but %s has no WMMA; the ladder will "
                     "fall through", caps_.gfx.c_str());
        if (path_ == GemmPath::kDp4a && !int8_gemm_)
            KRK_WARN("KRK_GEMM_PATH=dp4a also turns on the int8 activations, "
                     "which are lossy by construction");
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
    size_t device_free_bytes() const override {
        size_t free_b = 0, total_b = 0;
        if (hipMemGetInfo(&free_b, &total_b) != hipSuccess) return caps_.vram_free;
        return free_b;
    }
    // Free lists keyed by exact rounded size. The expert cache allocates and
    // releases the same few slice sizes thousands of times per chunk, and
    // hipMalloc/hipFree synchronise the device, so the pool is the difference
    // between overlapping the promotion copies and waiting on the allocator.
    // Bounded twice over, because a pool that only grows is a leak with extra
    // steps: blocks over 4 MiB are never kept (a workspace-sized block is a
    // resident workspace), and the whole pool holds at most 256 MiB of free
    // blocks. The first version kept up to 512 blocks per size with no total
    // cap and pushed the card into its own allocations on a 1.2 GiB model.
    // A pooled block remembers the moment it became reusable. An expert
    // evicted on the host side may still be read by kernels already queued on
    // the default stream, and the pool is exactly what would hand that block to
    // the next expert before they are done: measured on a 3 GiB VRAM budget the
    // engine emitted different text on every run until this event existed.
    struct PoolBlock {
        void *p = nullptr;
        hipEvent_t ev = nullptr; // recorded on the default stream at release
    };
    void *alloc_pooled(size_t bytes) override {
        const size_t rounded = (bytes + 255u) & ~static_cast<size_t>(255u);
        auto it = pool_.find(rounded);
        if (it != pool_.end() && !it->second.empty()) {
            std::vector<PoolBlock> &v = it->second;
            // Prefer a block the device is already done with; the events are
            // in release order, so the first ready one is also the oldest.
            size_t pick = 0;
            for (size_t i = 0; i < v.size(); i++) {
                if (!v[i].ev || hipEventQuery(v[i].ev) == hipSuccess) {
                    pick = i;
                    break;
                }
                static_cast<void>(hipGetLastError()); // hipErrorNotReady is not
                                                      // a sticky failure
            }
            PoolBlock b = v[pick];
            v.erase(v.begin() + pick);
            if (b.ev) {
                // Nothing was ready: wait for that block's reuse point rather
                // than allocating new VRAM. The wait is bounded by the queue
                // the device had already been given.
                KRK_HIP(hipEventSynchronize(b.ev));
                (void)hipEventDestroy(b.ev);
            }
            pool_bytes_ -= rounded;
            return b.p;
        }
        return alloc(rounded);
    }
    void release_pooled(void *p, size_t bytes) override {
        if (!p) return;
        const size_t rounded = (bytes + 255u) & ~static_cast<size_t>(255u);
        if (rounded <= (4u << 20) && pool_bytes_ + rounded <= kPoolCapBytes) {
            hipEvent_t ev = nullptr;
            if (hipEventCreateWithFlags(&ev, hipEventDisableTiming) != hipSuccess) {
                static_cast<void>(hipGetLastError());
                hipFree(p); // no guard to be had: take the synchronising free
                return;
            }
            (void)hipEventRecord(ev, 0);
            pool_[rounded].push_back(PoolBlock{p, ev});
            pool_bytes_ += rounded;
            return;
        }
        hipFree(p);
    }
    static constexpr size_t kPoolCapBytes = 256u << 20;
    std::unordered_map<size_t, std::vector<PoolBlock>> pool_;
    size_t pool_bytes_ = 0;
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

    // The WARM tier, in pageable RAM. This is where the expert set lives when
    // it does not fit in VRAM: tens of GiB on a big box, swappable, and free to
    // shrink, so it must not be drawn from the pinned pool (that pool is small,
    // non-swappable, and shared with the pull ring and the hot-path staging
    // ring, all of which stop working when it is exhausted). The WARM -> HOT
    // promotion stages through a bounded pinned ring instead (upload_paged), so
    // every DMA keeps a real pinned source of about 16 MiB whatever WARM costs.
    void *alloc_host_pageable(size_t bytes) override {
        return host_alloc(bytes ? bytes : 1, 64);
    }
    void release_host_pageable(void *p) override {
        if (p) host_free(p);
    }
    // WARM memory given back while a promotion copy may still be reading it.
    // The copies are ordered back into the default stream, so an event recorded
    // there marks the point after which the buffer is really free. Freeing it
    // eagerly is a use-after-free that a greedy decode reports as "plausible
    // text, different every run" -- which is exactly what the 3 GiB-VRAM arm
    // did before this list existed.
    struct DeferredHostFree {
        void *p = nullptr;
        hipEvent_t ev = nullptr;
    };
    static constexpr size_t kDeferredMax = 256;
    std::vector<DeferredHostFree> host_pending_;
    void release_host_deferred(void *p) override {
        if (!p) return;
        if (host_pending_.size() >= kDeferredMax) host_deferred_flush_(false);
        hipEvent_t ev = nullptr;
        if (hipEventCreateWithFlags(&ev, hipEventDisableTiming) != hipSuccess) {
            static_cast<void>(hipGetLastError());
            release_host_pageable(p); // no guard available: free it the safe way
            return;
        }
        (void)hipEventRecord(ev, 0);
        host_pending_.push_back(DeferredHostFree{p, ev});
    }
    // Frees every pending buffer whose copy has landed. `drain` waits for the
    // device first and is for teardown, when no copy may outlive the backend.
    void host_deferred_flush_(bool drain) {
        if (drain) (void)hipDeviceSynchronize();
        size_t keep = 0;
        for (size_t i = 0; i < host_pending_.size(); i++) {
            DeferredHostFree d = host_pending_[i];
            bool done = drain || !d.ev;
            if (!done) {
                done = hipEventQuery(d.ev) == hipSuccess;
                if (!done) static_cast<void>(hipGetLastError());
            }
            if (!done) {
                host_pending_[keep++] = d;
                continue;
            }
            if (d.ev) (void)hipEventDestroy(d.ev);
            release_host_pageable(d.p);
        }
        host_pending_.resize(keep);
    }
    // A cold expert is read() straight into its WARM slot. Through the mapping
    // it would fault one 4 KiB page at a time (the same 3.87 GB/s fault path
    // upload()'s comment describes); read() out of the page cache is several
    // times that and leaves no mapping page resident, so the tier does not
    // double-count the model in RAM. Falls back to the mapping copy whenever
    // the file path is unavailable, so callers never see an unfilled buffer.
    bool read_host(void *dst, const void *mapped_src, size_t bytes) override {
        if (bytes == 0) return true;
        KRK_TIME_OP("warm_read");
        const u8 *s = static_cast<const u8 *>(mapped_src);
        if (pull_path_.empty() || !pull_map_ || pull_disabled_ || s < pull_map_ ||
            s + bytes > pull_map_ + pull_map_size_) {
            std::memcpy(dst, mapped_src, bytes);
            return true;
        }
        if (!warm_reader_.ok() && !warm_reader_.open(pull_path_)) {
            std::memcpy(dst, mapped_src, bytes);
            return true;
        }
        // As in load_pull_: the pointer's distance from the map base IS the
        // file offset, because the tensor directory already folded the data
        // section in when it set t.data.
        const size_t off = static_cast<size_t>(s - pull_map_);
        if (warm_reader_.at(off, dst, bytes)) return true;
        std::memcpy(dst, mapped_src, bytes); // read failed: never hand back garbage
        return true;
    }
    // WARM -> HOT. A pageable hipMemcpy works but performs the whole transfer
    // inside the driver's bounce buffer; this stages the copy through the small
    // bounded pinned ring instead, double-buffered, so the DMA overlaps the
    // next memcpy. A batch smaller than one chunk rides the hot-path staging
    // ring (stage_upload) and never allocates anything.
    //
    // Each chunk lands in a pinned ring slot and is copied on that slot's own
    // non-blocking stream, so the copy does not queue behind the kernels already
    // on the default stream. A single, device-wide hipDeviceSynchronize per
    // slice (the first version) drained the whole pipeline: 4321 experts
    // promoted at 1.9 GB/s. Copies on the default stream instead are correct by
    // construction but wait for the kernels ahead of them: 2.0 GB/s. A host wait
    // on the side stream is what shipped, because an event the default stream
    // waits on did not make the copy visible here (see paged_publish_).
    void upload_paged(void *dst, const void *src, size_t bytes) override {
        if (bytes == 0) return;
        KRK_TIME_OP("upload_paged");
        if (!host_pending_.empty()) host_deferred_flush_(false);
        if (paged_blocking_()) {
            // A/B switch: the blocking pageable copy, ordered behind everything
            // already queued on the default stream. Slow, and it drains the
            // queue, but it is the reference the side-stream path is measured
            // against -- and the fallback when the pinned ring cannot be built.
            KRK_HIP(hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice));
            return;
        }
        if (bytes <= kStageMaxBytes && stage_upload(dst, src, bytes)) return;
        if (!paged_ready_()) {
            // Pinned memory is finite and losing the ring must degrade the
            // path, not the run (same rule as the pull fallback).
            KRK_HIP(hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice));
            return;
        }
        int used[kPagedBufs];
        int nused = 0;
        const char *s = static_cast<const char *>(src);
        char *d = static_cast<char *>(dst);
        for (size_t off = 0; off < bytes; off += paged_chunk_) {
            const size_t len = std::min(paged_chunk_, bytes - off);
            const int slot = paged_take_();
            if (slot < 0) { // ring unusable: fall back to the blocking copy
                KRK_HIP(hipMemcpy(d + off, s + off, len, hipMemcpyHostToDevice));
                continue;
            }
            std::memcpy(paged_buf_[slot], s + off, len);
            KRK_HIP(hipMemcpyAsync(d + off, paged_buf_[slot], len,
                                   hipMemcpyHostToDevice, paged_stream_[slot]));
            KRK_HIP(hipEventRecord(paged_ev_[slot], paged_stream_[slot]));
            paged_used_[slot] = true;
            if (nused < kPagedBufs) used[nused++] = slot;
        }
        paged_publish_(used, nused);
    }
    // One expert is three slices, and three separate calls serialise: each one
    // waits for the buffer it is about to refill. Batching them shares a single
    // ring cursor, so slice N+1 stages while slice N is still in flight.
    void upload_paged_batch(void *const dst[], const void *const src[],
                            const size_t bytes[], int n) override {
        if (n <= 0) return;
        KRK_TIME_OP("upload_paged");
        i64 batch_bytes = 0;
        if (!host_pending_.empty()) host_deferred_flush_(false);
        if (paged_blocking_()) {
            for (int i = 0; i < n; i++)
                if (dst[i] && src[i] && bytes[i])
                    KRK_HIP(hipMemcpy(dst[i], src[i], bytes[i], hipMemcpyHostToDevice));
            return;
        }
        if (!paged_ready_()) {
            for (int i = 0; i < n; i++)
                if (dst[i] && src[i] && bytes[i])
                    upload_paged(dst[i], src[i], bytes[i]);
            return;
        }
        if (paged_direct_() && paged_ready_()) {
            // KRK_WARM_DMA=2. The driver stages a pageable source into its own
            // pinned buffer; issuing all three slices on one side stream then
            // leaves one host wait for the batch instead of one per slice.
            for (int i = 0; i < n; i++) {
                if (!dst[i] || !src[i] || !bytes[i]) continue;
                KRK_HIP(hipMemcpyAsync(dst[i], src[i], bytes[i],
                                       hipMemcpyHostToDevice, paged_stream_[0]));
                batch_bytes += static_cast<i64>(bytes[i]);
            }
            KRK_HIP(hipStreamSynchronize(paged_stream_[0]));
            krk_op_scope.bytes(batch_bytes);
            return;
        }
        int used[kPagedBufs];
        int nused = 0;
        for (int i = 0; i < n; i++) {
            if (!dst[i] || !src[i] || !bytes[i]) continue;
            char *d = static_cast<char *>(dst[i]);
            const char *s = static_cast<const char *>(src[i]);
            for (size_t off = 0; off < bytes[i]; off += paged_chunk_) {
                const size_t len = std::min(paged_chunk_, bytes[i] - off);
                const int slot = paged_take_();
                if (slot < 0) {
                    KRK_HIP(hipMemcpy(d + off, s + off, len, hipMemcpyHostToDevice));
                    continue;
                }
                std::memcpy(paged_buf_[slot], s + off, len);
                KRK_HIP(hipMemcpyAsync(d + off, paged_buf_[slot], len,
                                       hipMemcpyHostToDevice, paged_stream_[slot]));
                KRK_HIP(hipEventRecord(paged_ev_[slot], paged_stream_[slot]));
                batch_bytes += static_cast<i64>(len);
                paged_used_[slot] = true;
                if (nused < kPagedBufs) used[nused++] = slot;
            }
        }
        krk_op_scope.bytes(batch_bytes);
        paged_publish_(used, nused);
    }
    // Closes a batch of side-stream copies. The host waits for the copies --
    // not for the kernel queue: each copy runs on its own stream, beside the
    // kernels, and the wait is only for that stream. When this returns, the
    // weights are in VRAM, the staging slot is idle and so is the WARM source.
    //
    // The host wait is the correct part, and it was measured three ways. An
    // event the default stream waits on (recorded after the copy: per slot, or
    // per batch from a ring) did not make the copy visible to the next kernel on
    // this runtime. The prefetch arm emitted garbage on roughly one run in four
    // with each of them, and 4/4 clean once the host waited here. Copies issued
    // on the default stream instead are correct by construction but queue behind
    // the kernels already on it, which measured 2.0 GB/s against 3.2-4.3 GB/s
    // for this shape.
    void paged_publish_(const int used[], int nused) {
        for (int i = 0; i < nused; i++)
            if (paged_stream_[used[i]])
                KRK_HIP(hipStreamSynchronize(paged_stream_[used[i]]));
    }
    // Claims a ring slot to write into, waiting only on the copy that last used
    // it -- an event, not the device. Returns -1 when the ring cannot be used.
    int paged_take_() {
        for (int tries = 0; tries < kPagedBufs; tries++) {
            const int i = paged_next_;
            paged_next_ = (paged_next_ + 1) % kPagedBufs;
            if (!paged_used_[i]) return i; // nothing has filled this slot yet
            if (hipEventSynchronize(paged_ev_[i]) == hipSuccess) return i;
            static_cast<void>(hipGetLastError());
        }
        return -1;
    }
    // Lazily builds the promotion ring. Chunk size, and with it the pinned
    // footprint (kPagedBufs * chunk), is sweepable with KRK_WARM_STAGE_KIB.
    // KRK_WARM_DMA=0 restores the blocking copy for the WARM -> HOT path, so a
    // change here can be measured against what it replaced. KRK_WARM_DMA=2
    // skips the ring and has the runtime stage a pageable source itself, on one
    // side stream -- the same double copy without the ring bookkeeping, and one
    // sync per batch instead of one per slice.
    static bool paged_blocking_() {
        static const bool v = [] {
            const char *e = std::getenv("KRK_WARM_DMA");
            return e && e[0] == '0';
        }();
        return v;
    }
    static bool paged_direct_() {
        static const bool v = [] {
            const char *e = std::getenv("KRK_WARM_DMA");
            return e && e[0] == '2';
        }();
        return v;
    }
    bool paged_ready_() {
        if (paged_on_) return true;
        if (paged_failed_) return false;
        if (const char *e = std::getenv("KRK_WARM_STAGE_KIB")) {
            const long k = std::atol(e);
            if (k >= 64 && k <= 8192) paged_chunk_ = static_cast<size_t>(k) << 10;
        }
        for (int i = 0; i < kPagedBufs; i++) {
            if (hipHostMalloc(&paged_buf_[i], paged_chunk_) != hipSuccess) {
                static_cast<void>(hipGetLastError());
                KRK_WARN("warm promotion ring: pinned staging of %zu bytes failed",
                         paged_chunk_);
                paged_failed_ = true;
                return false;
            }
            // Non-blocking: a plain stream has implicit synchronization with
            // the legacy default stream every kernel runs on, which is the
            // queueing this whole exercise exists to avoid.
            if (hipStreamCreateWithFlags(&paged_stream_[i], hipStreamNonBlocking) !=
                    hipSuccess ||
                hipEventCreateWithFlags(&paged_ev_[i], hipEventDisableTiming) !=
                    hipSuccess) {
                static_cast<void>(hipGetLastError());
                paged_failed_ = true;
                return false;
            }
        }
        paged_on_ = true;
        return true;
    }
    void paged_release_() {
        for (int i = 0; i < kPagedBufs; i++) {
            if (paged_buf_[i]) {
                (void)hipHostFree(paged_buf_[i]);
                paged_buf_[i] = nullptr;
            }
            if (paged_ev_[i]) {
                (void)hipEventDestroy(paged_ev_[i]);
                paged_ev_[i] = nullptr;
            }
            if (paged_stream_[i]) {
                (void)hipStreamDestroy(paged_stream_[i]);
                paged_stream_[i] = nullptr;
            }
            paged_used_[i] = false;
        }
        paged_on_ = false;
    }
    // Installs the pull-style weight path (see Backend::set_weight_pull). The
    // loader hands over the path and the mapping's address range; this class
    // opens its own handles and owns the pinned buffers and the side streams.
    // Without it every upload takes the direct hipMemcpy, which is also the
    // CPU/other-backend behaviour.
    void set_weight_pull(const std::string &path, const void *map_base,
                         size_t map_size, size_t data_off) override {
        pull_path_ = path;
        pull_map_ = static_cast<const u8 *>(map_base);
        pull_map_size_ = map_size;
        pull_data_off_ = data_off;
        // The ring is built on first use (pull_ready_), not here: a run that
        // never pulls a large tensor never pins a buffer or opens a handle.
        pull_on_ = false;
        // A/B switch. KRK_WEIGHT_PULL=0 restores the direct hipMemcpy from the
        // mapping, which is what every measurement of the pull path has to be
        // compared against -- the probes that produced its numbers ran in
        // sequence, so the later arms had a warmer page cache and the ordering
        // alone was worth several GB/s.
        if (const char *v = std::getenv("KRK_WEIGHT_PULL"))
            if (v[0] == '0') pull_disabled_ = true;
    }

    void upload(void *dst, const void *src, size_t bytes, size_t off) override {
        KRK_TIME_OP("upload");
        if (stage_upload(static_cast<char *>(dst) + off, src, bytes)) return;
        // A hipMemcpy whose source is a cold file mapping faults one 4 KiB page
        // at a time: measured 1208.8 ms to get the 8B into RAM that way
        // (3.87 GB/s), and no amount of DMA tuning helps -- a pinned bounce
        // buffer or an async copy on top of the mapping still lands at
        // 3.9-4.4 GB/s, so it is the fault path itself, not the transfer.
        //
        // read() into a warm page cache is 6.2 GB/s single-threaded and 16.3
        // with four threads on separate handles, so large weights go through
        // pull_ (see load_pull_ below), which overlaps the reads with the
        // copies. Everything else -- small tensors, converted F32 staging,
        // buffers that are not in the mapping -- takes the direct copy below.
        if (pull_map_ && !pull_disabled_ && bytes >= kTouchMinBytes &&
            load_pull_(dst, off, src, bytes))
            return;
        KRK_HIP(hipMemcpy(static_cast<char *>(dst) + off, src, bytes,
                          hipMemcpyHostToDevice));
    }
    // The direct copy, with no staging side effects.
    bool upload_now(void *dst, const void *src, size_t bytes, size_t off) override {
        if (stage_upload(static_cast<char *>(dst) + off, src, bytes)) return true;
        KRK_HIP(hipMemcpy(static_cast<char *>(dst) + off, src, bytes,
                          hipMemcpyHostToDevice));
        return true;
    }

    // ---------------------------------------------------------------------
    // Weight-load pipeline
    //
    // Reads a large tensor out of the model file and copies it to VRAM with the
    // two stages overlapping, which is where the whole win is. Measured on the
    // 8B (4795 MiB, gfx1201, warm page cache), end to end including the copy:
    //
    //   A serial read 32 MiB + copy, 1 thread            945.9 ms   4.95 GB/s
    //   B 4 reader threads, 1 MiB chunks, serial copy    705.1 ms   6.64 GB/s
    //   C pipelined, 8 chunks in flight, async copies    389.1 ms  12.03 GB/s
    //     (repeat: 962.5 / 704.6 / 374.4 -- 4.86 / 6.65 / 12.51)
    //
    // Two facts drive the shape of it:
    //
    //  * A file HANDLE on Windows serialises concurrent ReadFile calls. Four
    //    threads on one handle: 7.4 GB/s. Four threads on four handles: 15.9.
    //    So each reader opens its own FileReader, on every platform.
    //  * Small chunks win once the reads are parallel -- 4 threads x 1 MiB read
    //    at 16.3 GB/s against 9.3 for 32 MiB chunks -- and 1 MiB buffers are
    //    also what keeps pinning a handful of them cheap.
    //
    // The threads are created per tensor and joined before it returns, because
    // the loader is synchronous: one tensor is the most that is ever in flight.
    // Creating eight threads per large tensor costs ~50 us against the 100+ ms
    // of reading it, and it keeps the whole thing free of lifetime problems.
    //
    // Fallback-safe: if the file side cannot be used, or the range is not in the
    // mapping, this returns false and upload() takes the direct hipMemcpy.
    bool load_pull_(void *dst, size_t dst_off, const void *src, size_t bytes) {
        const u8 *s = static_cast<const u8 *>(src);
        if (s < pull_map_ || s + bytes > pull_map_ + pull_map_size_) return false;
        // `src` is a pointer into the mapping, so its distance from the map base
        // IS the file offset: the tensor directory already folded the data
        // section in when it set `t.data = data_base_ + t.offset`. Adding
        // pull_data_off_ here again reads every tensor from twice the data
        // section offset -- fast, plausible-looking, and the wrong bytes.
        const size_t file_off = static_cast<size_t>(s - pull_map_);

        if (!pull_ready_()) return false;
        const size_t chunk = pull_chunk_();
        const size_t nchunk = (bytes + chunk - 1) / chunk;
        const int nthread = static_cast<int>(std::min<size_t>(pull_threads_(), nchunk));

        // Each worker owns a CONTIGUOUS slice of the tensor, not every nth
        // chunk. Round-robin looked equivalent and is not: a worker taking
        // chunks 0, 8, 16... reads 1 MiB every 8 MiB, and the strided pattern
        // costs the OS read-ahead and the handle's own sequential hint. The
        // probe that measured 12.0 GB/s pipelined gave each thread one
        // contiguous run, and matching that shape is what closes the gap.
        const size_t per = (bytes + static_cast<size_t>(nthread) - 1) /
                           static_cast<size_t>(nthread);

        std::atomic<int> fail{0};
        // KRK_PULL_VERIFY=1 compares every pulled chunk against the same
        // offset in the mapping. The mapping is the ground truth the direct
        // hipMemcpy reads, so a mismatch is exactly the corruption a wrong
        // offset or a racing buffer would produce. It costs the page faults
        // the pull path exists to avoid, so it is a diagnostic, never a default.

        auto worker = [&](int slot) {
            const size_t begin = static_cast<size_t>(slot) * per;
            if (begin >= bytes) return;
            const size_t stop = std::min(begin + per, bytes);
            int which = 0;
            int local = 0; // chunks this worker has taken, for the reuse guard
            for (size_t c_off = begin; c_off < stop; c_off += chunk) {
                if (fail.load()) break;
                const size_t c_len = std::min(chunk, stop - c_off);

                hipStream_t st = pull_stream_[slot][which];
                char *buf = pull_buf_[slot][which];
                // `buf` was filled and queued two iterations ago. Drain that
                // copy before overwriting it -- but only that far back: syncing
                // in front of every copy (the obvious version) serialises the
                // two stages and measured 1144 ms against 374 for this.
                if (local >= 2 && hipStreamSynchronize(st) != hipSuccess) {
                    fail.store(1);
                    break;
                }
                if (!pull_readers_[slot].at(file_off + c_off, buf, c_len)) {
                    fail.store(1);
                    break;
                }
                if (hipMemcpyAsync(static_cast<char *>(dst) + dst_off + c_off, buf, c_len,
                                   hipMemcpyHostToDevice, st) != hipSuccess) {
                    fail.store(1);
                    break;
                }
                which ^= 1;
                local++;
            }
        };

        std::vector<std::thread> ts;
        ts.reserve(nthread);
        for (int i = 0; i < nthread; i++) ts.emplace_back(worker, i);
        for (auto &t : ts) t.join();
        // Drain EVERY buffer in the ring. This loop covered streams 0 and 1
        // only -- a leftover from a two-buffer ring. With kPullBufs = 4 the
        // last two chunks of every worker were never waited on, so the engine
        // read weights that had not landed yet. It is silent (the bytes are in
        // flight, not missing) and model-dependent, which is why it produced
        // garbage on SmolLM2 while the 8B looked fine.
        for (int i = 0; i < nthread; i++)
            for (int j = 0; j < kPullBufs; j++)
                KRK_HIP(hipStreamSynchronize(pull_stream_[i][j]));
        // The copies went out on hipStreamNonBlocking streams, which have NO
        // implicit ordering against the legacy default stream every kernel in
        // this engine runs on. Draining the pull streams is therefore not enough
        // for the caller to observe the weights: without this barrier the first
        // kernel that reads the tensor can start before the DMA lands. It is
        // silent (the bytes are in flight, not absent) and model-dependent --
        // SmolLM2's single pulled tensor produced garbage text while the 8B,
        // whose tensors are copied in a different order, looked fine.
        KRK_HIP(hipDeviceSynchronize());
        return fail.load() == 0;
    }

    // Tunables, overridable for a sweep (KRK_PULL_THREADS / KRK_PULL_CHUNK_KIB).
    static int pull_threads_() {
        static const int v = [] {
            const char *e = std::getenv("KRK_PULL_THREADS");
            const int n = e ? std::atoi(e) : 0;
            return (n >= 1 && n <= kPullThreads) ? n : kPullThreads;
        }();
        return v;
    }
    static size_t pull_chunk_() {
        static const size_t v = [] {
            const char *e = std::getenv("KRK_PULL_CHUNK_KIB");
            const long k = e ? std::atol(e) : 0;
            return (k >= 64 && k <= 8192) ? static_cast<size_t>(k) << 10 : kPullChunk;
        }();
        return v;
    }

    // Lazily builds the ring: per worker, one independent handle (a Windows
    // HANDLE serialises concurrent ReadFile, so sharing one measured 7.4 GB/s
    // against 15.9 for separate handles), two pinned buffers and two streams.
    // Kept for the life of the backend -- a second load (a draft model, say)
    // reuses it.
    bool pull_ready_() {
        if (pull_on_) return true;
        if (pull_path_.empty() || !pull_map_) return false;
        for (int i = 0; i < kPullThreads; i++) {
            if (!pull_readers_[i].open(pull_path_)) return false;
            for (int j = 0; j < kPullBufs; j++) {
                void *p = alloc_host(kPullChunk);
                if (!p) return false;
                pull_buf_[i][j] = static_cast<char *>(p);
                // Non-blocking: a plain hipStreamCreate stream has implicit
                // synchronization with the legacy default stream, which the
                // engine's kernels run on, so every H2D copy would queue behind
                // it instead of running concurrently.
                if (hipStreamCreateWithFlags(&pull_stream_[i][j], hipStreamNonBlocking) !=
                    hipSuccess)
                    return false;
            }
        }
        pull_on_ = true;
        return true;
    }

    void pull_release_() {
        for (int i = 0; i < kPullThreads; i++) {
            for (int j = 0; j < kPullBufs; j++) {
                if (pull_stream_[i][j]) {
                    hipStreamDestroy(pull_stream_[i][j]);
                    pull_stream_[i][j] = nullptr;
                }
                if (pull_buf_[i][j]) {
                    release_host(pull_buf_[i][j]);
                    pull_buf_[i][j] = nullptr;
                }
            }
            pull_readers_[i].close();
        }
        pull_on_ = false;
    }

    void download(void *dst, const void *src, size_t bytes, size_t off) override {
        KRK_TIME_OP("download");
        KRK_HIP(hipMemcpy(dst, static_cast<const char *>(src) + off, bytes,
                          hipMemcpyDeviceToHost));
    }
    void fill0(void *dst, size_t bytes) override {
        KRK_TIME_OP("fill0");
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
        KRK_TIME_OP("download_f32");
        std::vector<u16> tmp(static_cast<size_t>(n));
        KRK_HIP(hipMemcpy(tmp.data(), src, static_cast<size_t>(n) * 2,
                          hipMemcpyDeviceToHost));
        for (i64 i = 0; i < n; i++) dst[i] = fp16_to_fp32(tmp[static_cast<size_t>(i)]);
    }

    // Which GEMM family to run, decided from the DEVICE rather than from how
    // this binary was built: WMMA where the part has it, DP4A for the formats
    // and shapes it can stage (when that path is enabled), and the packed-math
    // SIMT tile everywhere else. Two reasons it is a runtime ladder. A build may
    // hold several targets, so a build-time answer would be the wrong device's;
    // and a rung that is not present must fall THROUGH to the next one instead
    // of launching nothing, which is what the previous shape of this code did
    // whenever the device had WMMA but the kernel had not been compiled in.
    GemmRung gemm_rung(DType wt, i64 rows, i64 n_out, i64 n_in) const {
        const bool wmma_present = caps_.has_wmma
#if defined(KRK_WMMA_COMPILED)
                                  ;
#else
                                  && false;
#endif
        // Forcing a rung means exactly that: a forced path is not silently
        // upgraded, and a forced rung the device lacks falls through with the
        // warning that was printed once at startup.
        const bool force_wmma = path_ == GemmPath::kWmma;
        const bool force_dp4a = path_ == GemmPath::kDp4a;
        const bool force_simt = path_ == GemmPath::kSimt;
        if (!force_dp4a && !force_simt && wmma_present) return GemmRung::kWmma;
        if (!force_simt && !force_wmma && (int8_gemm_ || force_dp4a) &&
            dp4a_ok(static_cast<int>(wt), rows, n_out, n_in))
            return GemmRung::kDp4a;
        return GemmRung::kSimt;
    }

    // The highest rung this device can reach, for the banner: it names the
    // family the engine would prefer, not what every individual call takes
    // (a shape or format outside DP4A's rules still lands on the SIMT tile).
    const char *gemm_rung_name() const {
        switch (gemm_rung(DType::Q4_K, 64, 256, 256)) {
            case GemmRung::kWmma: return caps_.has_wmma_gfx12 ? "WMMA gfx12" : "WMMA";
            case GemmRung::kDp4a: return "DP4A int8";
            default: return "packed-math";
        }
    }

    std::string describe() const override {
        return format("%s (%s, %s, %d CU, wave32, %s)", caps_.name.c_str(),
                      caps_.gfx.c_str(), caps_.family_name(), caps_.cu_count,
                      gemm_rung_name());
    }

    // ---- ops -------------------------------------------------------------
    void embed(void *out, const void *tok_embd, DType tt, i64 n_vocab, i64 n_embd,                const i32 *tokens, i64 n_tok) override {
        KRK_TIME_OP("embed");
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
        // Same rule as stage_pos: the ring declines anything over
        // kStageMaxBytes, and an upload that silently did not happen would
        // embed whatever the buffer happened to hold.
        if (!stage_upload(tok_dev_, tokens, need))
            KRK_HIP(hipMemcpy(tok_dev_, tokens, need, hipMemcpyHostToDevice));
        embed_kernel<128><<<static_cast<unsigned>(n_tok), 128>>>(
            static_cast<_Float16 *>(out), static_cast<const u8 *>(tok_embd),
            static_cast<int>(tt), n_vocab, n_embd, row_bytes,
            static_cast<const i32 *>(tok_dev_), n_tok);
    }

    void rmsnorm(void *out, const void *x, const f32 *w, i64 rows,
                 i64 n, f32 eps) override {
        KRK_TIME_OP("rmsnorm");
        rmsnorm_kernel<256><<<static_cast<unsigned>(rows), 256>>>(
            static_cast<_Float16 *>(out),
            static_cast<const _Float16 *>(x), w, n, eps);
    }
    // Top-k of the logits row, entirely on the device (kernels/topk.hpp).
    // Four launches and no readback between them: stage 2 writes the cut and
    // stage 3 reads it back from device memory, so the only transfer at the
    // end is k ids, k values, the mass and the count -- a few hundred bytes
    // where the download this replaces was 993 kB, and none of the host
    // partial_sort over 248320 elements that used to follow it.
    //
    // `logits` is the head's output buffer, so it is in the *activation* type
    // (f16 on the fast path), not f32. The dispatch below is the only thing
    // that knows it; the kernels are templated on the load.
    //
    // Scratch layout (caller-owned; 4096 B header then cand_cap u64 keys):
    //   0 hist[512] u32 | 2048 gmax u64 | 2056 cut u32 | 2064 count u64
    //   2072 mass f32   | 4096 cand[cand_cap] u64
    void logits_topk(const void *logits, i64 n, i32 k, const void *pen_mask,
                     f32 rep_pen, f32 softcap, f32 temp, i32 *ids, f32 *vals,
                     f32 *mass_out, void *scratch, i32 cand_cap,
                     i64 *count_out) override {
        KRK_TIME_OP("logits_topk");
        constexpr int kThreads = 256;
        if (n <= 0 || k <= 0) return;
        u8 *sc = static_cast<u8 *>(scratch);
        u32 *hist = reinterpret_cast<u32 *>(sc);
        auto *gmax = reinterpret_cast<unsigned long long *>(sc + 2048);
        u32 *cut = reinterpret_cast<u32 *>(sc + 2056);
        auto *count = reinterpret_cast<unsigned long long *>(sc + 2064);
        float *mass = reinterpret_cast<float *>(sc + 2072);
        auto *cand = reinterpret_cast<unsigned long long *>(sc + 4096);

        // One block per ~16 chunks per thread, capped at the device width so
        // a 248k row does not launch a thousand blocks of mostly-idle work.
        const i64 want = (n + kThreads * 16 - 1) / (kThreads * 16);
        const unsigned blocks =
            static_cast<unsigned>(want < 1 ? 1 : (want > 1024 ? 1024 : want));
        const i32 span = static_cast<i32>((n + blocks - 1) / blocks);

        KRK_HIP(hipMemsetAsync(hist, 0, 2048));
        KRK_HIP(hipMemsetAsync(gmax, 0, 8));
        KRK_HIP(hipMemsetAsync(count, 0, 8));
        KRK_HIP(hipMemsetAsync(mass, 0, 4));

        const f32 inv_temp = temp > 0.0f ? 1.0f / temp : 1.0f;
        const u32 *pm = static_cast<const u32 *>(pen_mask);
        auto *co = reinterpret_cast<unsigned long long *>(count_out);

        // k == 1 and no mass wanted: one pass, one max, two launches. This is
        // the greedy decode path and the only one wired up today.
        if (k == 1 && mass_out == nullptr) {
            KRK_HIP(hipMemsetAsync(gmax, 0, 8));
            if (act_type() == DType::F16) {
                topk_argmax_kernel<_Float16, kThreads>
                    <<<blocks, kThreads>>>(
                        static_cast<const _Float16 *>(logits), pm, rep_pen,
                        softcap, static_cast<i32>(n), span, gmax);
            } else {
                topk_argmax_kernel<f32, kThreads><<<blocks, kThreads>>>(
                    static_cast<const f32 *>(logits), pm, rep_pen, softcap,
                    static_cast<i32>(n), span, gmax);
            }
            topk_unpack_kernel<<<1, 1>>>(
                gmax, reinterpret_cast<unsigned long long *>(count_out), ids, vals);
            KRK_HIP(hipGetLastError());
            (void)inv_temp;
            (void)cut;
            (void)hist;
            (void)mass;
            (void)cand;
            (void)cand_cap;
            return;
        }

        if (act_type() == DType::F16) {
            topk_launch<_Float16, kThreads>(
                static_cast<const _Float16 *>(logits), pm, rep_pen, softcap,
                inv_temp, static_cast<i32>(n), k, hist, gmax, cut, count, mass,
                cand, cand_cap, ids, vals, co, blocks, span);
        } else {
            topk_launch<f32, kThreads>(
                static_cast<const f32 *>(logits), pm, rep_pen, softcap, inv_temp,
                static_cast<i32>(n), k, hist, gmax, cut, count, mass, cand,
                cand_cap, ids, vals, co, blocks, span);
        }
        KRK_HIP(hipGetLastError());
        // count_out is written by the final kernel; mass is computed into the
        // scratch for anyone who asks for it through the four-stage path.
        (void)mass_out;
    }


    void gemm(void *out, const void *x, const void *w, DType wt, i64 n_out,               i64 n_in, i64 rows) override {
        KRK_TIME_OP("gemm");
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

        const GemmRung rung = gemm_rung(wt, rows, n_out, n_in);
        // The rung the shape actually took is the first thing to know when a
        // GEMM looks slow: WMMA is only reachable for rows > 1, and the decode
        // row runs a GEMV with no tile at all.
        krk_op_scope.retag(rows == 1                 ? "gemm(gemv)"
                           : rung == GemmRung::kWmma ? "gemm(wmma)"
                           : rung == GemmRung::kDp4a ? "gemm(dp4a)"
                                                     : "gemm(simt)");
        krk_op_scope.bytes(static_cast<i64>(dtype_row_bytes(wt, n_in)) * n_out);
        krk_op_scope.flops(2 * rows * n_out * n_in);
        if (rows == 1) {
            // One 256-thread block covers THREADS/32 rows (one warp
            // per row), so the grid is n_out warps, not n_out blocks —
            // launching n_out blocks leaves 7/8 of them empty.
            gemv_kernel<256><<<static_cast<unsigned>((n_out + 7) / 8),
                                256>>>(
                wh, static_cast<int>(wt), n_out, n_in, row_bytes, xh, oh);
        } else if (rung == GemmRung::kWmma) {
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
        } else if (rung == GemmRung::kDp4a) {
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
            // The dispatcher picks the kernel for this weight format; every
            // format it stages was checked against the host decoder by
            // tools/gemm_i8k_verify.cpp (built as kraken-i8k).
            gemm_dp4a_dispatch(static_cast<int>(wt), wh, n_out, n_in, row_bytes, xh,
                               rows, oh, reinterpret_cast<i8 *>(int8q_), int8s_);
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
        KRK_TIME_OP("gemm_group");
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
              const f32 *inv_freq, f32 scale, f32 frac, bool neox) override {
        KRK_TIME_OP("rope");
        // `inv_freq` is host memory (Model keeps the table in a
        // std::vector), but the kernel reads it per thread on the
        // device; stage_freq() stages it, caching the upload.
        const f32 *fd = stage_freq(inv_freq, hd);
        stage_pos(pos0);
        const dim3 gq(static_cast<unsigned>(n_head), static_cast<unsigned>(n_tok));
        rope_kernel<<<gq, 128>>>(static_cast<_Float16 *>(q), n_head, hd,
                                 pos_dev_, fd, scale, frac, neox);
        const dim3 gk(static_cast<unsigned>(n_kv), static_cast<unsigned>(n_tok));
        rope_kernel<<<gk, 128>>>(static_cast<_Float16 *>(k), n_kv, hd,
                                 pos_dev_, fd, scale, frac, neox);
    }

    void qk_norm(void *q, void *k, const f32 *wq, const f32 *wk, i64 n_head,
                 i64 n_kv, i64 hd, i64 n_tok, f32 eps) override {
        KRK_TIME_OP("qk_norm");
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
        KRK_TIME_OP("kv_append");
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
        KRK_TIME_OP("attention");
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
            // Split the key range so the grid fills the card. n_head alone is
            // 8 blocks on 64 CUs for this model; the split is sized from the
            // key count so a short context still collapses to one block per
            // head and takes the original path unchanged.
            const i64 n_keys = d.pos0 + 1;
            // A windowed layer only visits the visible range, so the split
            // count is sized from that rather than from the whole history.
            const i64 n_vis = d.window > 0 && d.window < n_keys ? d.window : n_keys;
            int n_split = attention_decode_splits(
                n_vis, d.n_head, caps_.cu_count);
            // A/B knob: KRK_ATTN_SPLIT=0 forces one block per head so the
            // split path can be diffed against it on identical input. The two
            // are algebraically the same computation, so the text must match.
            if (const char *kv = std::getenv("KRK_ATTN_SPLIT"))
                if (kv[0] == '0') n_split = 1;
            f32 *part = nullptr;
            if (n_split > 1) {
                const size_t per = static_cast<size_t>(d.n_head) *
                                   static_cast<size_t>(n_split) *
                                   static_cast<size_t>(2 + kWaveSize * (d.hd / 32)) *
                                   sizeof(f32);
                if (per > attn_part_cap_) {
                    if (attn_part_) KRK_HIP(hipFree(attn_part_));
                    attn_part_ = nullptr;
                    attn_part_cap_ = (per + 255u) & ~static_cast<size_t>(255u);
                    KRK_HIP(hipMalloc(&attn_part_, attn_part_cap_));
                }
                part = attn_part_;
            }
            attention_decode(out, q, kc, vc, d.n_head, d.n_kv, d.hd,
                             pos_dev_, d.pos_stride, d.causal ? 1 : 0,
                             d.window, d.scale, nullptr, part, n_split,
                             caps_.cu_count);
            return;
        }

        // Size the key tile so the two staged tiles stay near 32 KiB of LDS.
        const int tile_k = static_cast<int>(
            std::min<i64>(64, std::max<i64>(8, 8192 / std::max<i64>(1, d.hd))));
        const size_t smem = attn_smem_bytes(tile_k, d.hd);

        // Query-tiled prefill: QT tokens share one staged K/V tile instead of
        // every token staging the whole history alone. QT=8 is what the
        // register budget allows (kMaxOv * QT accumulators per thread); the
        // tile is cut back if LDS cannot hold the wider q tile.
        // KNOWN WRONG -- opt in with KRK_ATTN_QTILE=1, see attn_qtile_on().
        constexpr int kQT = 8;
        // The q tile adds QT*hd halves on top of the two key tiles, so the key
        // tile has to shrink to fit LDS. Shrinking is the right knob because
        // the sharing factor (QT) is what buys the win, while tile_k only
        // controls barrier count -- 16 keys already keeps every lane of the
        // staging loops busy at kAttnBlock = 128.
        int qtile_k = tile_k;
        while (qtile_k > 8 &&
               attn_qtile_smem_bytes<kQT>(qtile_k, d.hd) > caps_.lds_bytes)
            qtile_k /= 2;
        if (const char *fk = std::getenv("KRK_ATTN_QTILE_K")) {
            const int want = std::atoi(fk);
            if (want > 0 &&
                attn_qtile_smem_bytes<kQT>(want, d.hd) <= caps_.lds_bytes)
                qtile_k = want;
        }
        if (attn_qtile_on() && d.window == 0 &&
            attn_qtile_smem_bytes<kQT>(qtile_k, d.hd) <= caps_.lds_bytes) {
            const size_t qsmem = attn_qtile_smem_bytes<kQT>(qtile_k, d.hd);
            const dim3 grid(static_cast<unsigned>(d.n_head),
                            static_cast<unsigned>((d.n_tok + kQT - 1) / kQT));
            KRK_HIP(hipFuncSetAttribute(
                reinterpret_cast<const void *>(&attention_qtile_kernel<kQT>),
                hipFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(qsmem + 512)));
            attention_qtile_kernel<kQT><<<grid, kAttnBlock, qsmem>>>(
                static_cast<_Float16 *>(out), static_cast<const _Float16 *>(q),
                static_cast<const _Float16 *>(kc), static_cast<const _Float16 *>(vc),
                d.n_head, d.n_kv, d.hd, d.n_tok, d.pos0, d.pos_stride,
                d.causal ? 1 : 0, d.scale, qtile_k);
            return;
        }

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
            d.window, d.scale, tile_k);
    }

    bool attn_fused_chain(void *out, void *q, void *kcache, void *vcache,
                            const void *k, const void *v, const AttnDesc &d,
                            const f32 *inv_freq, f32 rope_scale,
                            f32 rope_frac) override {
        KRK_TIME_OP("attn_fused");
        // The fused kernel expresses the single-query-token decode row
        // only, and only the head widths its packed lane slice covers
        // (the same rule as the decode attention path above). Anything
        // else keeps the separate rope() + kv_append() + attention()
        // chain the engine runs as the fallback. The kernel rotates
        // adjacent channel pairs, which is the llama-family layout, so a
        // NeoX-pairing model (ModelConfig::rope_neox) stays on that
        // fallback until the packed lane slice can express the pair.
        if (d.n_tok != 1 || !attn_fused_decode_supported(d.hd) ||
            !attn_fused_ || d.window > 0)
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
        KRK_TIME_OP("silu_mul");
        const unsigned blocks = static_cast<unsigned>((n + 255) / 256);
        silu_mul_kernel<<<blocks, 256>>>(static_cast<_Float16 *>(out),
                                         static_cast<const _Float16 *>(gate),
                                         static_cast<const _Float16 *>(up), n);
    }

    void add_bias_rows(void *out, const f32 *bias, i64 n, i64 rows) override {
        KRK_TIME_OP("add_bias_rows");
        const i64 total = n * rows;
        const unsigned blocks = static_cast<unsigned>((total + 255) / 256);
        add_bias_rows_kernel<<<blocks, 256>>>(static_cast<_Float16 *>(out), bias, n,
                                              total);
    }

    void add_inplace(void *a, const void *b, i64 n) override {
        KRK_TIME_OP("add_inplace");
        const unsigned blocks = static_cast<unsigned>((n + 255) / 256);
        add_inplace_kernel<<<blocks, 256>>>(static_cast<_Float16 *>(a),
                                            static_cast<const _Float16 *>(b), n);
    }

    void copy_act(void *dst, const void *src, i64 n) override {
        KRK_TIME_OP("copy_act");
        const unsigned blocks = static_cast<unsigned>((n + 255) / 256);
        copy_h2h_kernel<<<blocks, 256>>>(static_cast<_Float16 *>(dst),
                                         static_cast<const _Float16 *>(src), n);
    }

    void axpy(void *dst, const void *src, f32 alpha, i64 n) override {
        KRK_TIME_OP("axpy");
        const unsigned blocks = static_cast<unsigned>((n + 255) / 256);
        axpy_kernel<<<blocks, 256>>>(static_cast<_Float16 *>(dst),
                                     static_cast<const _Float16 *>(src), alpha, n);
    }

    void upload_i32(i32 *dst, const i32 *src, i64 n) override {
        KRK_TIME_OP("upload_i32");
        const size_t bytes = static_cast<size_t>(n) * sizeof(i32);
        if (stage_upload(dst, src, bytes)) return;
        KRK_HIP(hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice));
    }

    // ---- gated delta net (Qwen3.5 / Qwen3-Next) -------------------------
    // kernels/gdn.hpp, one launch per op, f16 activations over the f32
    // recurrent state. The model loader consults gdn_supported() and refuses a
    // recurrent checkpoint rather than running it on a backend that would
    // silently skip these.
    bool gdn_supported() const override { return true; }

    void l2norm(void *o, const void *x, i64 n_tok, i64 n_head, i64 hd,
                i64 stride, f32 eps) override {
        KRK_TIME_OP("l2norm");
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
        KRK_TIME_OP("sigmoid_act");
        if (n <= 0) return;
        const unsigned blocks = static_cast<unsigned>((n + 255) / 256);
        gdn_sigmoid_kernel<<<blocks, 256>>>(static_cast<_Float16 *>(x), n);
    }

    void softplus_act(void *x, i64 n) override {
        KRK_TIME_OP("softplus_act");
        if (n <= 0) return;
        const unsigned blocks = static_cast<unsigned>((n + 255) / 256);
        gdn_softplus_kernel<<<blocks, 256>>>(static_cast<_Float16 *>(x), n);
    }

    void scale_act(void *x, f32 alpha, i64 n_tok, i64 width, i64 stride) override {
        KRK_TIME_OP("scale_act");
        if (n_tok <= 0 || width <= 0) return;
        const dim3 grid(static_cast<unsigned>((width + 255) / 256),
                        static_cast<unsigned>(std::min<i64>(n_tok, 65535)));
        gdn_scale_act_kernel<<<grid, 256>>>(static_cast<_Float16 *>(x), alpha,
                                            width, stride, n_tok);
    }

    void mul_act(void *a, const void *b, i64 n) override {
        KRK_TIME_OP("mul_act");
        if (n <= 0) return;
        const unsigned blocks = static_cast<unsigned>((n + 255) / 256);
        gdn_mul_act_kernel<<<blocks, 256>>>(static_cast<_Float16 *>(a),
                                            static_cast<const _Float16 *>(b), n);
    }

    // Laguna's per-head output gate: x[t, h, :] *= g[t, h].
    void mul_head_broadcast(void *x, const void *g, i64 n_tok, i64 n_head,
                            i64 hd) override {
        KRK_TIME_OP("mul_head_bcast");
        if (n_tok <= 0 || n_head <= 0 || hd <= 0) return;
        const i64 total = n_tok * n_head * hd;
        const unsigned blocks = static_cast<unsigned>((total + 255) / 256);
        gdn_mul_head_broadcast_kernel<<<blocks, 256>>>(
            static_cast<_Float16 *>(x), static_cast<const _Float16 *>(g),
            n_head, hd, total);
    }

    void add_bias_cols(void *x, const f32 *bias, i64 n_row, i64 n_col) override {
        KRK_TIME_OP("add_bias_cols");
        if (n_row <= 0 || n_col <= 0) return;
        const dim3 grid(1, static_cast<unsigned>(std::min<i64>(n_row, 65535)));
        gdn_add_bias_cols_kernel<<<grid, 64>>>(static_cast<_Float16 *>(x), bias,
                                               n_col, n_row);
    }

    void scale_cols(void *x, const f32 *col, i64 n_row, i64 n_col) override {
        KRK_TIME_OP("scale_cols");
        if (n_row <= 0 || n_col <= 0) return;
        const dim3 grid(1, static_cast<unsigned>(std::min<i64>(n_row, 65535)));
        gdn_scale_cols_kernel<<<grid, 64>>>(static_cast<_Float16 *>(x), col, n_col,
                                            n_row);
    }

    void scale_rows(void *x, const void *alpha, i64 n_row, i64 n) override {
        KRK_TIME_OP("scale_rows");
        if (n_row <= 0 || n <= 0) return;
        const dim3 grid(1, static_cast<unsigned>(std::min<i64>(n_row, 65535)));
        gdn_scale_rows_kernel<<<grid, 256>>>(static_cast<_Float16 *>(x),
                                             static_cast<const _Float16 *>(alpha),
                                             n, n_row);
    }

    void qwen3_next_split(void *q, void *gate, const void *packed, i64 n_tok,
                          i64 n_head, i64 hd) override {
        KRK_TIME_OP("q3_split");
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
        KRK_TIME_OP("conv1d_silu");
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
        KRK_TIME_OP("delta_rule");
        gdn_delta_rule_launch(
            static_cast<_Float16 *>(out), static_cast<f32 *>(state),
            static_cast<const _Float16 *>(q), static_cast<const _Float16 *>(k),
            static_cast<const _Float16 *>(v), static_cast<const _Float16 *>(g),
            static_cast<const _Float16 *>(beta), n_tok, n_k_head, n_v_head,
            d_state, hd, row_stride);
    }

    void gather_rows(void *dst, const void *src, const i32 *rows, i64 n_rows,
                     i64 n) override {
        KRK_TIME_OP("gather_rows");
        krk_op_scope.bytes(n_rows * n * 2);
        if (n_rows <= 0 || n <= 0) return;
        gather_rows_kernel<<<static_cast<unsigned>(n_rows), 256>>>(
            static_cast<_Float16 *>(dst),            static_cast<const _Float16 *>(src), rows,
            n);
    }

    void scatter_axpy_rows(void *dst, const void *src, const i32 *rows,
                           const f32 *alpha, i64 n_rows, i64 n) override {
        KRK_TIME_OP("scatter_axpy");
        krk_op_scope.bytes(n_rows * n * 2);
        if (n_rows <= 0 || n <= 0) return;
        // alpha lives in this backend's address space, exactly like `rows`: the
        // caller is a device backend's user and hands over buffers it allocated
        // here. This used to memcpy it as if it were host memory, which is a
        // read of device memory from the host: harmless-looking on a Linux
        // driver that maps device pages into the process, an access violation
        // (0xC0000005) on Windows, and undefined everywhere in between. The MoE
        // path's first use on Windows hit it, in Engine::moe_ffn. Staging the
        // copy also cost a per-expert H2D transfer that the caller had already
        // paid for when it uploaded the same weights.
        scatter_axpy_rows_kernel<<<static_cast<unsigned>(n_rows), 256>>>(
            static_cast<_Float16 *>(dst), static_cast<const _Float16 *>(src),
            rows, alpha, n);
    }

    ~HipBackend() override {
        pull_release_();
        paged_release_();
        for (int i = 0; i < kStageSlots; i++) {
            if (stage_ev_[i]) {
                (void)hipEventDestroy(stage_ev_[i]);
                stage_ev_[i] = nullptr;
            }
        }
        host_deferred_flush_(true); // no copy may outlive the backend
        OpTime::get().finish();
        if (tok_dev_) {
            (void)hipFree(tok_dev_);
            tok_dev_ = nullptr;
        }
        for (FreqSlot &s : freq_) {
            if (s.dev) {
                (void)hipFree(s.dev);
                s.dev = nullptr;
            }
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
        if (attn_part_) {
            (void)hipFree(attn_part_);
            attn_part_ = nullptr;
        }
        for (const GdnTaps &e : gdn_taps_) (void)hipFree(e.dev);
        gdn_taps_.clear();
        if (stage_pin_) {
            (void)hipHostFree(stage_pin_);
            stage_pin_ = nullptr;
        }
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
    GemmPath path_ = GemmPath::kAuto;
    // Quantized activations for that path: [rows, n_in] int8 codes and one
    // fp32 scale per 32-value block, grown on demand.
    void *int8q_ = nullptr;
    size_t int8q_cap_ = 0;
    f32 *int8s_ = nullptr;
    size_t int8s_cap_ = 0;
    // Decode-attention key-split partials: [n_head][n_split][2 + 32*VPT].
    // One allocation, grown on demand, reused every layer and every token.
    f32 *attn_part_ = nullptr;
    size_t attn_part_cap_ = 0;
    // Device staging for the host token-id array embed() reads (grown
    // on demand; see embed()).
    void *tok_dev_ = nullptr;
    size_t tok_cap_ = 0;
    // Device staging for the host inv_freq table rope() reads.
    //
    // This has to be more than one slot. A hybrid-attention model interleaves
    // two rope tables (inv_freq_ on the full-attention layers, inv_freq_swa_ on
    // the windowed ones), so a single-entry cache misses at every layer
    // boundary -- and the miss it takes is a BLOCKING hipMemcpy, which drains
    // everything already queued on the stream behind it. On laguna-xs2 that
    // measured 1296 us of host time per rope() call against 3.2 us on a
    // non-hybrid MoE: 51.8 ms of a ~99 ms decode step spent copying a table
    // that is 256 bytes and already resident four layers earlier.
    //
    // A 4-entry LRU keyed on (pointer, length, signature) keeps both tables
    // resident for a whole step, so the upload happens once per distinct table
    // per run instead of once per layer boundary.
    static constexpr int kFreqSlots = 4;
    struct FreqSlot {
        void *dev = nullptr;
        size_t cap = 0;
        const f32 *src = nullptr;
        size_t n = 0;
        f32 sig = 0.0f;
        i64 stamp = 0;   // LRU clock; ties break on the lowest slot index
    };
    FreqSlot freq_[kFreqSlots] = {};
    i64 freq_clock_ = 0;
    // Device-side position scalar for the decode path (rope,
    // kv_append, decode attention). Those kernels read pos0 from
    // device memory so a captured HIP graph can replay a whole
    // decode step without re-binding kernel arguments. One 8-byte
    // upload per distinct position covers the three calls of a
    // step, which all share the same value.
    i64 *pos_dev_ = nullptr;
    i64 pos_staged_ = 0;
    bool pos_valid_ = false;
    // Pinned staging ring for the small hot-path H2D transfers (see
    // stage_upload): one allocation, kStageSlots slots, each guarded by an
    // event that marks the copy the slot still owes the device.
    void *stage_pin_ = nullptr;
    size_t stage_stride_ = 0;
    // One event per slot: it marks the copy the slot still owes the device.
    hipEvent_t stage_ev_[kStageSlots] = {};
    // Set by the loader (see set_read_upload): pulls a tensor in with
    // read() into a pinned buffer and copies from there. Returns false
    // to fall back to the direct hipMemcpy.
    // Pull-style weight path (see load_pull_). The ring is built on first use so
    // a run that never loads a large tensor never pins anything.
    static constexpr int kPullThreads = 4;
    static constexpr size_t kPullChunk = 1u << 20;
    // Buffers (and streams) per worker. Two is enough: the worker syncs the
    // buffer it is about to refill, which was queued two iterations ago, so the
    // copy has a full read's worth of time to land.
    static constexpr int kPullBufs = 2;
    std::string pull_path_;
    const u8 *pull_map_ = nullptr;
    size_t pull_map_size_ = 0;
    size_t pull_data_off_ = 0;
    bool pull_on_ = false;
    bool pull_disabled_ = false;
    FileReader pull_readers_[kPullThreads];
    // WARM fill path. Its own handle, like every pull reader: a Windows HANDLE
    // serialises concurrent ReadFile, so sharing one with the pull ring would
    // put cold-expert reads and weight pulls in each other's way.
    FileReader warm_reader_;
    // Bounded pinned staging ring for WARM -> HOT promotions (upload_paged).
    // Small on purpose: it is the DMA's in-flight pipeline, not a cache, and
    // nothing in it outlives the copy. Total pinned bytes is kPagedBufs *
    // paged_chunk_, 16 MiB by default -- sitting beside a WARM tier that may be
    // tens of GiB, which is the point of keeping the two allocations apart.
    static constexpr int kPagedBufs = 4;
    static constexpr size_t kPagedChunk = 4u << 20;
    void *paged_buf_[kPagedBufs] = {};
    hipStream_t paged_stream_[kPagedBufs] = {};
    // One event per slot, recorded on that slot's stream. It guards the slot
    // against being refilled before its own copy landed, and it is what the
    // default stream waits on to observe the weights.
    hipEvent_t paged_ev_[kPagedBufs] = {};
    bool paged_used_[kPagedBufs] = {};
    int paged_next_ = 0;
    bool paged_on_ = false;
    bool paged_failed_ = false;
    size_t paged_chunk_ = kPagedChunk;
    char *pull_buf_[kPullThreads][kPullBufs] = {};
    hipStream_t pull_stream_[kPullThreads][kPullBufs] = {};
    int stage_slot_ = 0;
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

    // ---- host->device staging ------------------------------------------------
//
// A pageable hipMemcpy is synchronous with respect to the *host* and is
// ordered after everything already queued on the null stream. Called between
// decode steps it therefore blocks until the previous token's entire kernel
// stream has retired, which is a full pipeline drain per token for the sake of
// four bytes: measured on the 0.8B, embed() spent 274 us of host time per call
// (65 times its kernel launch) and the decode step's host column put ~17% of
// the token inside that one call. It is the single clearest reason the step is
// launch-bound — the GPU is drained at the top of every token, so it cannot
// overlap with the host enqueueing the next one.
//
// Staging through pinned memory and issuing hipMemcpyAsync on the same stream
// keeps the ordering the kernels rely on (the DMA is still ahead of every
// kernel that reads the buffer) while returning to the host at once.
//
// The pinned source has to be rotated: the DMA reads it asynchronously, so the
// host must not overwrite a slot an earlier copy is still reading. The first
// version simply trusted the rotation depth, on the theory that a 4-byte DMA
// retires long before the ring wraps. That theory is wrong: the retire time is
// not the bound, the queue position is. A prefill layer queues thousands of
// kernels, every upload sits behind all of them, and a slot rewritten while its
// DMA was still queued made that DMA read the *new* bytes -- a plan or alpha
// belonging to a different expert, silently. Measured on a 682-token prompt: 5
// runs, 4 distinct outputs with the unguarded ring, against 3 runs and one
// output with KRK_STAGE_SYNC=1 (blocking copies). The guard is one event per
// slot: a slot is reused only once its own copy has completed, and when all 64
// slots are still in flight the ring takes the blocking copy instead.
//
// Above kStageMaxBytes the pageable copy stays: those are the one-shot weight
// loads, where a pinned slot the size of a tensor is neither available nor
// worth having, and nothing is queued behind them to overlap with.

// KRK_STAGE_SYNC=1 takes the direct blocking copy for every small upload,
// which is what the callers do when this returns false. It exists to separate
// the staging ring from the copy itself, so a change to the ring can be
// measured against the copy it replaced.
static bool stage_sync_() {
    static const bool v = [] {
        const char *e = std::getenv("KRK_STAGE_SYNC");
        return e && e[0] == '1';
    }();
    return v;
}

// KRK_STAGE_WAIT=1 keeps the ring and its pinned source but host-waits each
// copy, which separates two different claims: "the slot was reused too early"
// and "an async H2D copy on the legacy default stream is not ordered before
// the kernels queued behind it". The first is fixed by the per-slot event
// guard; the second is a runtime property, not a policy.
static bool stage_wait_() {
    static const bool v = [] {
        const char *e = std::getenv("KRK_STAGE_WAIT");
        return e && e[0] == '1';
    }();
    return v;
}

// Returns false when the caller should fall back to the blocking pageable copy.
bool stage_upload(void *dst, const void *src, size_t bytes) {
    if (bytes == 0) return true;
    if (bytes > kStageMaxBytes) return false;
    if (stage_sync_()) {
        // The A/B shape, not a reason to skip the copy: stage_pos and embed()
        // call this for their side effect, so returning false here left the
        // position and the token ids un-uploaded and the run answered "S".
        KRK_HIP(hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice));
        return true;
    }
    // One allocation, at full stride, for the life of the backend. The ring
    // used to grow to fit the largest upload seen so far and hipHostFree the
    // old block to do it; a DMA still queued against that block then read
    // memory the allocator had handed back, which is the same use-after-free
    // the pooled-VRAM guard exists for, in the one place with no guard at all.
    // Four MiB pinned once is cheaper than a second allocation path.
    if (!stage_pin_) {
        if (hipHostMalloc(&stage_pin_, kStageSlots * kStageMaxBytes) !=
            hipSuccess) {
            // Pinned memory is a finite pool. Losing it must degrade the
            // staging path, not the run: fall back to the blocking copy.
            (void)hipGetLastError();
            stage_pin_ = nullptr;
            return false;
        }
        stage_stride_ = kStageMaxBytes;
        stage_slot_ = 0;
    }
    // A slot may only be overwritten once the async copy that last read it has
    // completed. Without the guard the ring was a silent correctness hole: the
    // slot rotation is 32 deep, a prefill layer queues far more than 32 small
    // uploads behind its own GEMMs, and a slot handed to a new upload while its
    // DMA still sat in the queue made that DMA read the new bytes -- the plan
    // and alpha of a *different* expert, silently, at a rate that varied with
    // queue depth. It showed up as "plausible text, different every run" on a
    // long prompt (5 runs, 4 distinct outputs; 3 runs with KRK_STAGE_SYNC=1, one
    // output) while a 31-token prompt never tripped it.
    int pick = -1;
    for (int i = 0; i < kStageSlots; i++) {
        const int s = (stage_slot_ + 1 + i) % kStageSlots;
        hipEvent_t ev = stage_ev_[s];
        if (!ev) {
            pick = s;
            break;
        }
        if (hipEventQuery(ev) == hipSuccess) {
            pick = s;
            break;
        }
        static_cast<void>(hipGetLastError()); // hipErrorNotReady is not sticky
    }
    if (pick < 0) {
        // Every slot is still in flight: take the blocking copy rather than
        // overwrite one. Correct, and rare -- 64 slots of 64 KiB.
        KRK_HIP(hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice));
        return true;
    }
    if (!stage_ev_[pick])
        KRK_HIP(hipEventCreateWithFlags(&stage_ev_[pick], hipEventDisableTiming));
    char *p = static_cast<char *>(stage_pin_) +
              static_cast<size_t>(pick) * stage_stride_;
    std::memcpy(p, src, bytes);
    KRK_HIP(hipMemcpyAsync(dst, p, bytes, hipMemcpyHostToDevice, 0));
    KRK_HIP(hipEventRecord(stage_ev_[pick], 0));
    if (stage_wait_()) KRK_HIP(hipStreamSynchronize(0));
    stage_slot_ = pick;
    return true;
}

// Opt-IN. The query-tiled prefill kernel is FAST but not yet correct: with
    // identical inputs it disagrees with attention_kernel, and the error grows
    // with the number of key tiles (rel 6e-2 on a single tile, 5.3e-1 at
    // tile_k = 1). The per-key score matrix is verified correct against a
    // double-precision CPU reference, so the defect is downstream of it, in
    // the per-query softmax/normalizer/accumulator stage. It is off by default
    // because a wrong-but-fluent prefill is strictly worse than a slow one;
    // probe_attn_qtile.hip reproduces the comparison.
//   KRK_ATTN_QTILE=1  use it (needs attention_qtile_kernel to be fixed)
//   KRK_ATTN_QTILE_K=N  force the key tile size when running the probe
    static bool attn_qtile_on() {
        const char *v = std::getenv("KRK_ATTN_QTILE");
        return v && v[0] == '1';
    }

    void stage_pos(i64 pos0) {
        if (pos_valid_ && pos_staged_ == pos0) return;
        if (!pos_dev_) KRK_HIP(hipMalloc(&pos_dev_, sizeof(i64)));
        if (!stage_upload(pos_dev_, &pos0, sizeof(i64)))
            KRK_HIP(hipMemcpy(pos_dev_, &pos0, sizeof(i64),
                                  hipMemcpyHostToDevice));
        pos_staged_ = pos0;
        pos_valid_ = true;
    }

    // Stages the host inv_freq table for the device-side RoPE
    // kernels (rope(), attn_fused_chain()). Keyed on the (pointer,
    // length, signature) triple so the layers of a step share one
    // memcpy -- and held in a small LRU rather than a single slot, so a
    // model that interleaves two tables does not re-upload one of them at
    // every layer boundary (see the freq_ members).
    const f32 *stage_freq(const f32 *inv_freq, i64 hd) {
        const size_t n = static_cast<size_t>(hd / 2);
        const f32 sig = n >= 2 ? inv_freq[1] : 0.0f;
        FreqSlot *victim = &freq_[0];
        for (int i = 0; i < kFreqSlots; i++) {
            FreqSlot &s = freq_[i];
            if (s.dev && s.src == inv_freq && s.n == n && s.sig == sig) {
                s.stamp = ++freq_clock_;
                return static_cast<const f32 *>(s.dev);
            }
            if (s.stamp <= victim->stamp) victim = &s;
        }
        const size_t need = n * sizeof(f32);
        if (need > victim->cap) {
            if (victim->dev) KRK_HIP(hipFree(victim->dev));
            victim->cap = (need + 255u) & ~static_cast<size_t>(255u);
            victim->dev = nullptr;
            KRK_HIP(hipMalloc(&victim->dev, victim->cap));
        }
        KRK_HIP(hipMemcpy(victim->dev, inv_freq, need,
                          hipMemcpyHostToDevice));
        victim->src = inv_freq;
        victim->n = n;
        victim->sig = sig;
        victim->stamp = ++freq_clock_;
        return static_cast<const f32 *>(victim->dev);
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
        if (real_cus > 0) {
            c.cu_count = static_cast<int>(real_cus);
        } else if (c.family != GfxFamily::Unknown) {
            // No KFD to ask (Windows, or a kernel without the topology). What
            // HIP reports on RDNA is the WGP count, and a WGP is two CUs: the
            // 6700 XT's 40 CUs arrive as 20 (which KFD confirmed), and the
            // 9070 XT's 64 arrive as the 32 this backend used to believe. Every
            // occupancy heuristic, every grid and pick_split's cost model are
            // sized against cu_count, so believing half of it halves the
            // launch geometry on exactly the machines that cannot be corrected
            // any other way.
            c.cu_count *= 2;
        }
        // A/B switch for the above, and the escape hatch if a part reports
        // something that is neither CUs nor WGPs.
        if (const char *cu = std::getenv("KRK_CU_OVERRIDE")) {
            const int v = std::atoi(cu);
            if (v > 0) {
                KRK_WARN("KRK_CU_OVERRIDE=%d replaces the detected %d CU", v,
                         c.cu_count);
                c.cu_count = v;
            }
        }
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
