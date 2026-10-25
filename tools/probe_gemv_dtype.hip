// probe_gemv_dtype.hip — is the 27B's projection GEMV slow because of its QUANT
// FORMAT, or because of its SHAPE?
//
// The trace cannot answer that. Every `gemm*` row in it is an average over every
// dtype the model happens to use, and the two models in hand move two variables
// at once: the 27B is mostly Q8_0/Q6_K with n_out=5120, the 8B is all Q4_K with
// n_out=4096..14336. `gemm_group(fused)` measures 167.8 GB/s on one and 563.4
// GB/s on the other, and that difference has been blamed on both.
//
// This runs the same `gemv_kernel<256>` the decode path launches — same grid
// formula, same 256 threads, B=1 — back to back with nothing else in the queue,
// at each model's real shapes for each of its dtypes. Kernel quality, shape and
// format are separated by construction.
//
// Read GB/s as the bus figure and Gval/s as the comparable one: a format with
// more bytes per value reaches a higher GB/s at the same arithmetic rate, so only
// GB/s between DIFFERENT dtypes is apples to oranges, and %peak only means
// anything for the byte count it was computed from.
//
// Build (the KRK_BUILD_PROBES block in CMakeLists.txt does not configure at the
// moment -- it lists probe sources that are not in the tree -- so this target is
// not in it; build it directly with the same flags the other probes use):
//
//   G:/ROCM10RT-gfx1201/lib/llvm/bin/clang++.exe -DKRK_ENABLE_HIP=1 \
//     -DKRK_OFFLOAD_GFX1201=1 -D__HIP_ROCclr__=1 -D_CRT_SECURE_NO_WARNINGS \
//     -Iinclude -Isrc -Isrc/hip -isystem G:/ROCM10RT-gfx1201/include \
//     --rocm-device-lib-path=G:/ROCM10RT-gfx1201/lib/llvm/amdgcn/bitcode \
//     -O3 -std=c++17 --offload-arch=gfx1201 -Xclang --dependent-lib=msvcrt \
//     -ffast-math -fno-finite-math-only -LG:/ROCM10RT-gfx1201/lib -lamdhip64 \
//     -o build-hip/probe_gemv_dtype.exe -x hip tools/probe_gemv_dtype.hip
//   ./build-hip/probe_gemv_dtype.exe
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "krk_hip.hpp"
#include "kernels/dequant.hpp"
#include "kernels/gemm.hpp"

using namespace krk;

namespace {

struct Fmt {
    DType t;
    const char *name;
    int block_vals;   // values covered by one block
    int block_bytes;  // bytes one block occupies
    int scale_off;    // where the block's fp16 scale sits
};

const Fmt kFmts[] = {
    {DType::Q4_K, "Q4_K", 256, 144, 0},
    {DType::Q5_K, "Q5_K", 256, 176, 0},
    {DType::Q6_K, "Q6_K", 256, 210, 208},
    {DType::Q8_0, "Q8_0", 32, 34, 0},
    {DType::F16, "F16", 1, 2, -1},
};

struct Shape {
    const char *name;
    int n_out;
    int n_in;
};

const Shape kShapes[] = {
    {"27B o_proj    (gemm_acc)", 5120, 6144},
    {"27B down_proj (gemm_acc)", 5120, 17408},
    {"27B gate/up   (gemm)", 17408, 5120},
    {"27B lm_head   (gemm)", 248320, 5120},
    {"8B  q/o       (both)", 4096, 4096},
    {"8B  up/gate   (both)", 14336, 4096},
};

// The same arithmetic dtype_row_bytes() does, kept local so the probe links
// without the engine library: a row is n/block_vals whole blocks.
static size_t row_bytes(const Fmt &f, int n_in) {
    return (size_t)(n_in / f.block_vals) * (size_t)f.block_bytes;
}

// INCOMPRESSIBLE weights, with each block's scale set to fp16 1.0.
//
// The first version wrote zeros everywhere, and zeros are the one filler this
// measurement must not use: a VRAM buffer of zeros is what every write-zero and
// compression path in the memory system is built to move cheaply, which is a
// property of the FILL and not of the kernel. A pseudo-random stream is the
// honest filler, and a real checkpoint's bytes are equally arbitrary. The scale
// stays 1.0 so nothing underflows into denormal arithmetic, and a wrong layout
// assumption degrades to "random bytes", which is still the right test.
void fill(std::vector<u8> &b, const Fmt &f) {
    u32 s = 0x9E3779B9u;
    for (size_t i = 0; i < b.size(); i++) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;   // xorshift32
        b[i] = static_cast<u8>(s >> 24);
    }
    if (f.scale_off < 0) {
        // F16 rows: arbitrary bytes include NaNs, whose arithmetic cost is not
        // the question. A 1.0 pattern is incompressible enough for a 2-byte
        // format and keeps every product finite.
        for (size_t i = 0; i + 1 < b.size(); i += 2) {
            b[i] = 0x00; b[i + 1] = 0x3c;
        }
        return;
    }
    for (size_t off = 0; off + (size_t)f.block_bytes <= b.size();
         off += (size_t)f.block_bytes) {
        b[off + (size_t)f.scale_off] = 0x3c;      // fp16 1.0, little endian
        b[off + (size_t)f.scale_off + 1] = 0x00;
    }
}

// The probe has to flush L2 between iterations or it measures the cache: a
// 31.9 MiB matrix "reached" 1383 GB/s on the first run of this probe, on a card
// whose HBM does ~600, and the model never sees that matrix twice in a row.
__global__ void l2_flush_kernel(const u32 *__restrict__ p, size_t n32,
                                u32 *__restrict__ sink) {
    u32 acc = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n32;
         i += (size_t)gridDim.x * blockDim.x)
        acc ^= p[i];
    if (acc == 0xFFFFFFFFu) sink[0] = acc; // never taken on a zero fill
}

// One weight matrix as the walk sees it.
struct Mat {
    u8 *w;
    size_t bytes;
    int n_out, n_in;
    DType t;
    size_t rb;
};

} // namespace

int main() {
    hipDeviceProp_t prop{};
    hipGetDeviceProperties(&prop, 0);
    const int cu = prop.multiProcessorCount;
    // No %peak column: the property these runtimes report for memory clock is
    // not the GDDR data rate, and a wrong denominator silently redefines every
    // number in the table. The honest reference is the big-shape rows below,
    // which are HBM-bound whatever the header says.
    std::printf("device: %s  CUs=%d  bus=%d bit (measured HBM band: see the "
                ">=600 MiB rows)\n\n",
                prop.gcnArchName, cu, prop.memoryBusWidth);

    _Float16 *x = nullptr, *o = nullptr;
    const int xcap = 65536;
    if (hipMalloc(&x, xcap * sizeof(_Float16)) != hipSuccess) return 1;
    hipMemset(x, 0, (size_t)xcap * sizeof(_Float16));

    // 256 MiB is four times the largest L2 this family ships; reading it
    // between iterations is what makes each timed call start cold, which is the
    // only condition the model ever runs in.
    const size_t flush_bytes = 256u << 20;
    u32 *flush = nullptr, *sink = nullptr;
    if (hipMalloc(&flush, flush_bytes) != hipSuccess) return 1;
    if (hipMalloc(&sink, 4) != hipSuccess) return 1;
    hipMemset(flush, 0, flush_bytes);
    // Enough blocks to touch every line of the flush buffer in one pass, capped
    // at what the grid allows; the flush is the condition, not the measurement.
    const unsigned fblocks = (unsigned)std::min<size_t>(
        (flush_bytes / 4 + 255) / 256, 4096u);

    std::printf("%-26s %-5s %7s %6s %8s %9s %9s %9s\n", "shape", "dtype",
                "n_out", "n_in", "MiB", "us/call", "GB/s", "Gval/s");

    for (const Shape &s : kShapes) {
        for (const Fmt &f : kFmts) {
            const size_t rb = row_bytes(f, s.n_in);
            const size_t bytes = rb * (size_t)s.n_out;
            if (bytes == 0 || bytes > (size_t)12 << 30) continue;
            u8 *w = nullptr;
            if (hipMalloc(&w, bytes) != hipSuccess) {
                std::printf("%-26s %-5s  alloc %.0f MiB failed\n", s.name, f.name,
                            (double)bytes / 1048576.0);
                continue;
            }
            std::vector<u8> host(bytes);
            fill(host, f);
            hipMemcpy(w, host.data(), bytes, hipMemcpyHostToDevice);
            std::vector<u8>().swap(host);

            if (o) { hipFree(o); o = nullptr; }
            if (hipMalloc(&o, (size_t)s.n_out * sizeof(_Float16)) != hipSuccess)
                return 1;

            const unsigned grid = (unsigned)((s.n_out + 7) / 8);
            for (int i = 0; i < 3; i++)
                gemv_kernel<256><<<grid, 256>>>(w, (int)f.t, s.n_out, s.n_in, rb,
                                                x, o);
            hipDeviceSynchronize();

            // Each iteration is timed on its own, after a flush that is not
            // timed: the flush is the condition, not the measurement.
            const int iters = bytes > (256u << 20) ? 6 : 20;
            hipEvent_t a, b;
            hipEventCreate(&a); hipEventCreate(&b);
            double total_us = 0.0;
            for (int i = 0; i < iters; i++) {
                l2_flush_kernel<<<fblocks, 256>>>(flush,
                                                  flush_bytes / 4, sink);
                hipEventRecord(a, 0);
                gemv_kernel<256><<<grid, 256>>>(w, (int)f.t, s.n_out, s.n_in, rb,
                                                x, o);
                hipEventRecord(b, 0);
                hipEventSynchronize(b);
                float ms = 0.f;
                hipEventElapsedTime(&ms, a, b);
                total_us += (double)ms * 1000.0;
            }
            const double us = total_us / iters;
            const double gbs = (double)bytes / (us * 1e3);
            const double gvals =
                (double)s.n_out * (double)s.n_in / (us * 1e3);
            std::printf("%-26s %-5s %7d %6d %8.1f %9.2f %9.1f %9.1f\n", s.name,
                        f.name, s.n_out, s.n_in, (double)bytes / 1048576.0, us,
                        gbs, gvals);
            hipEventDestroy(a); hipEventDestroy(b);
            hipFree(w);
        }
        std::printf("\n");
    }

    // ---- phase 2: a WORKING SET, not one matrix ----------------------------
    //
    // Phase 1 reads the same matrix over and over, so its pages, page-table
    // entries and DRAM rows are the ones the last iteration touched. The model
    // never does that: a 27B layer's o_proj and down_proj are each a different
    // buffer, and one token walks 13.9 GiB of them. If the engine's 168-180 GB/s
    // is that difference, it will show up here as a many-matrix rate well below
    // the single-matrix rate; if it does not, the engine is adding the
    // difference itself and the next probe is an engine trace, not a kernel.
    {
        const int kLayers = 64;
        std::vector<Mat> mats;
        // The 27B's own mix: o_proj at 6144 K, down_proj at 17408 K, alternating
        // Q8_0 and Q6_K the way its layers are.
        for (int l = 0; l < kLayers; l++) {
            const bool down = (l & 1) != 0;
            const Fmt &f = (l & 2) ? kFmts[2] : kFmts[3]; // Q6_K / Q8_0
            Mat m{};
            m.n_out = 5120;
            m.n_in = down ? 17408 : 6144;
            m.t = f.t;
            m.rb = row_bytes(f, m.n_in);
            m.bytes = m.rb * (size_t)m.n_out;
            if (hipMalloc(&m.w, m.bytes) != hipSuccess) { m.w = nullptr; break; }
            std::vector<u8> host(m.bytes);
            fill(host, f);
            hipMemcpy(m.w, host.data(), m.bytes, hipMemcpyHostToDevice);
            mats.push_back(m);
        }
        size_t total = 0;
        for (const Mat &m : mats) total += m.bytes;
        if (!mats.empty()) {
            // Warm every matrix once, the way a first token does.
            for (const Mat &m : mats) {
                const unsigned g = (unsigned)((m.n_out + 7) / 8);
                gemv_kernel<256><<<g, 256>>>(m.w, (int)m.t, m.n_out, m.n_in, m.rb,
                                             x, o);
            }
            hipDeviceSynchronize();
            // Each matrix is timed on its own, flush OUTSIDE the window.
            // Timing the whole walk instead put the 256 MiB flush of every step
            // inside the measurement and read 101 GB/s -- a number that was
            // ~75% flush, i.e. a property of the probe and not of the walk.
            hipEvent_t a, b;
            hipEventCreate(&a); hipEventCreate(&b);
            double sum_ms = 0.0;
            for (const Mat &m : mats) {
                l2_flush_kernel<<<fblocks, 256>>>(flush, flush_bytes / 4, sink);
                const unsigned g = (unsigned)((m.n_out + 7) / 8);
                hipEventRecord(a, 0);
                gemv_kernel<256><<<g, 256>>>(m.w, (int)m.t, m.n_out, m.n_in, m.rb,
                                             x, o);
                hipEventRecord(b, 0);
                hipEventSynchronize(b);
                float ms = 0.f;
                hipEventElapsedTime(&ms, a, b);
                sum_ms += (double)ms;
            }
            const double ms = sum_ms;
            std::printf("working set: %d matrices, %.2f GiB, %.1f ms total, "
                        "%.1f GB/s\n",
                        (int)mats.size(), (double)total / 1073741824.0, ms,
                        (double)total / (ms * 1e6));
            // Sustained, not burst. The engine holds this work for seconds;
            // a 7 ms burst can ride a clock the sustained case cannot. If the
            // per-pass rate decays across passes, the in-situ gap is the power
            // budget and no kernel or scheduler change moves it.
            std::printf("sustained:");
            for (int pass = 0; pass < 8; pass++) {
                hipEventRecord(a, 0);
                for (const Mat &m : mats) {
                    const unsigned g = (unsigned)((m.n_out + 7) / 8);
                    gemv_kernel<256><<<g, 256>>>(m.w, (int)m.t, m.n_out, m.n_in,
                                                 m.rb, x, o);
                }
                hipEventRecord(b, 0);
                hipEventSynchronize(b);
                float pms = 0.f;
                hipEventElapsedTime(&pms, a, b);
                std::printf(" %.0f", (double)total / ((double)pms * 1e6));
            }
            std::printf(" GB/s (8 back-to-back walks, warm L2)\n");
            hipEventDestroy(a); hipEventDestroy(b);
        }
        for (Mat &m : mats) if (m.w) hipFree(m.w);
    }

    // ---- phase 3: the same walk, the engine's ALLOCATION LAYOUT -----------
    //
    // The engine holds 817 separate device buffers for the 27B (`vram ledger:
    // allocator live 14860 in 817 buffers, largest 995`) -- every projection,
    // norm, bias, workspace and activation is its own hipMalloc, allocated in
    // load order. Phase 2's 64 buffers are not that shape: they are allocated
    // back to back with nothing between them, so their pages and their VA order
    // are as friendly as an allocator can make them.
    //
    // This rebuilds the SAME walk with the engine's furniture interleaved: per
    // layer, two matrices and twelve small buffers of the sizes the 27B's load
    // histogram reports (64-256K x48, 256K-1M x96 over 64 layers, plus a 40 KB
    // norm/bias each). Only the placement changes -- same bytes, same order,
    // same kernels -- so a difference here is the layout and nothing else.
    {
        struct Small { void *p; size_t b; };
        std::vector<void *> keep;
        std::vector<Mat> mats;
        const size_t kSmall[12] = {160u << 10, 160u << 10, 480u << 10, 40u << 10,
                                   40u << 10, 480u << 10, 160u << 10, 40u << 10,
                                   160u << 10, 480u << 10, 40u << 10, 40u << 10};
        for (int l = 0; l < 64; l++) {
            const bool down = (l & 1) != 0;
            const Fmt &f = (l & 2) ? kFmts[2] : kFmts[3];
            for (int k = 0; k < 2; k++) {
                Mat m{};
                m.n_out = 5120;
                m.n_in = (k == 0) ? (down ? 17408 : 6144) : 6144;
                m.t = f.t;
                m.rb = row_bytes(f, m.n_in);
                m.bytes = m.rb * (size_t)m.n_out;
                if (hipMalloc(&m.w, m.bytes) != hipSuccess) { m.w = nullptr; break; }
                std::vector<u8> host(m.bytes);
                fill(host, f);
                hipMemcpy(m.w, host.data(), m.bytes, hipMemcpyHostToDevice);
                mats.push_back(m);
                // ... and the furniture the engine allocates between them.
                for (size_t sb : kSmall) {
                    void *sp2 = nullptr;
                    if (hipMalloc(&sp2, sb) == hipSuccess) keep.push_back(sp2);
                }
            }
        }
        size_t total = 0;
        for (const Mat &m : mats) total += m.bytes;
        if (!mats.empty()) {
            for (const Mat &m : mats) {
                const unsigned g = (unsigned)((m.n_out + 7) / 8);
                gemv_kernel<256><<<g, 256>>>(m.w, (int)m.t, m.n_out, m.n_in, m.rb,
                                             x, o);
            }
            hipDeviceSynchronize();
            hipEvent_t a, b;
            hipEventCreate(&a); hipEventCreate(&b);
            double sum_ms = 0.0;
            for (const Mat &m : mats) {
                l2_flush_kernel<<<fblocks, 256>>>(flush, flush_bytes / 4, sink);
                const unsigned g = (unsigned)((m.n_out + 7) / 8);
                hipEventRecord(a, 0);
                gemv_kernel<256><<<g, 256>>>(m.w, (int)m.t, m.n_out, m.n_in, m.rb,
                                             x, o);
                hipEventRecord(b, 0);
                hipEventSynchronize(b);
                float ms = 0.f;
                hipEventElapsedTime(&ms, a, b);
                sum_ms += (double)ms;
            }
            std::printf("layout (interleaved, %d small buffers): %d matrices, "
                        "%.2f GiB, %.1f ms, %.1f GB/s\n",
                        (int)keep.size(), (int)mats.size(),
                        (double)total / 1073741824.0, sum_ms,
                        (double)total / (sum_ms * 1e6));
            hipEventDestroy(a); hipEventDestroy(b);
        }
        for (Mat &m : mats) if (m.w) hipFree(m.w);
        for (void *p : keep) if (p) hipFree(p);
    }

    std::printf(
        "\nReading it: at one shape, a dtype stuck far below the others is a slow\n"
        "quant path and is fixable in dot_chunks. If every dtype lands in the\n"
        "same band at one shape and a different band at another, the format is\n"
        "innocent and the shape (grid width per byte) is the ceiling. If the\n"
        "working-set line lands where phase 1 does, the engine's lower rate is\n"
        "the engine's; if it lands near the engine's, the working set is the\n"
        "cost and no kernel change recovers it.\n");
    return 0;
}
