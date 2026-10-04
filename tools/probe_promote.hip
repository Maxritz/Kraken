// probe_promote.hip -- how fast can cold expert weights reach VRAM?
//
// A decode token on laguna-xs2 costs ~97 ms, and ~89 ms of that is the expert
// path. Two experiments localise it:
//
//   --expert-stub (routers only, no expert weights at all)  ->  8.5 ms/token
//   --expert-warm-prefetch 1 (COLD misses 0, 0 MiB read from the file)
//                                                             -> 130 ms/token
//
// The second one is the interesting one: with every weight already in host RAM
// and no file read anywhere in decode, the token got SLOWER. The prefetch run
// reports "4613 promotions, 8371 MiB in 4605.8 ms = 1.8 GB/s (998.4 us per
// expert)". So the cost is not the file. It is the pageable -> VRAM copy of
// each 2.06 MiB expert, and at 1.8 GB/s it is an order of magnitude off what
// this link can do.
//
// The shape of the current path explains the number: `upload_paged_batch`
// stages the source through a pinned ring and issues one async copy per 4 MiB
// chunk on a side stream, then HOST-WAITS for it (paged_publish_). An expert is
// 2.06 MiB, so that is exactly one copy in flight for the whole run, and the
// host blocks on each one. One in-flight copy cannot saturate a link.
//
// This probe prices the alternatives on the real workload shape (2.06 MiB
// experts, pageable source, 4 GiB moved):
//
//   0  sync per copy on a side stream      -- what ships
//   1  blocking hipMemcpy per copy         -- the pre-ring fallback
//   2  K copies queued on the DEFAULT stream, one drain at the end
//   3  K copies on K side streams, host wait at the end
//   4  as 2, but staged through pinned memory first (the ring, without the wait)
//
// 2 and 4 need no host wait at all and stay correct by construction: the
// consumer GEMM is on the same stream, so it cannot start before the copy
// lands. That is the property the current path is paying ~1 ms per expert to
// get another way.
//
// Build:
//   hipcc -O3 --offload-arch=gfx1201 -I include -I src -I src/hip \
//       tools/probe_promote.hip -o probe_promote
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "krk_hip.hpp"

using namespace krk;

namespace krk {
const char *hip_err_name(hipError_t e) { return hipGetErrorName(e); }
void hip_check(hipError_t e, const char *what) {
    if (e != hipSuccess) {
        std::fprintf(stderr, "FATAL %s: %s\n", what, hipGetErrorName(e));
        std::exit(1);
    }
}
}  // namespace krk

static double pnow_us() {
    return static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch()).count()) / 1e3;
}

// One expert's three slices: 2.06 MiB total, which is what laguna-xs2 stores.
static const size_t kExpertBytes = 2u << 20;
static const size_t kTotalBytes = 4ull << 30; // enough to time steadily
static const size_t kStageChunk = 4u << 20;   // upload_paged's chunk size

static const char *mode_name(int mode, int k) {
    switch (mode) {
    case 0: return "sync per copy on a side stream";
    case 1: return "blocking hipMemcpy per copy";
    case 2: return "K async on the DEFAULT stream, one drain";
    case 3: return "K async on K side streams, drain at end";
    default: return "K async on DEFAULT, staged through pinned";
    }
}

int main(int argc, char **argv) {
    size_t total = kTotalBytes;
    if (argc > 1) total = (static_cast<size_t>(std::atoll(argv[1])) << 20);

    hip_check(hipSetDevice(0), "setDevice");
    hipDeviceProp_t prop{};
    hip_check(hipGetDeviceProperties(&prop, 0), "props");
    std::printf("device: %s  CUs=%d\n", prop.gcnArchName,
                prop.multiProcessorCount);
    std::printf("shape : %.2f MiB per expert, %.0f MiB total, %zu copies\n\n",
                kExpertBytes / 1048576.0, total / 1048576.0,
                total / kExpertBytes);

    const size_t nexpert = total / kExpertBytes;
    // Destination: one contiguous VRAM slab, written round-robin. Real VRAM
    // pressure does not change a DMA's cost, and a slab keeps the probe free of
    // the allocator.
    void *vram = nullptr;
    hip_check(hipMalloc(&vram, kExpertBytes * 64), "malloc vram");
    // Source: PAGEABLE, because that is what the WARM tier is on purpose
    // (alloc_host_pageable), and a pageable source is exactly what makes the
    // shipped path slow.
    void *host = std::malloc(kExpertBytes * 64);
    if (!host) { std::fprintf(stderr, "host malloc failed\n"); return 1; }
    std::memset(host, 0x5a, kExpertBytes * 64);
    void *pinned = nullptr;
    hip_check(hipHostMalloc(&pinned, kStageChunk * 8, hipHostMallocDefault),
              "hostMalloc pinned");

    const int kStreams = 8;
    hipStream_t side[kStreams];
    for (int i = 0; i < kStreams; i++)
        hip_check(hipStreamCreateWithFlags(&side[i], hipStreamNonBlocking),
                  "stream create");

    std::printf("%-46s %6s %10s %10s %9s\n", "mode", "depth", "GB/s", "us/copy",
                "s total");

    for (int mode = 0; mode <= 4; mode++) {
        for (int k : {1, 2, 4, 8}) {
            if (mode == 0 && k != 1) continue; // this mode is k = 1 by definition
            hipDeviceSynchronize();
            const double t0 = pnow_us();
            int inflight = 0;
            for (size_t i = 0; i < nexpert; i++) {
                char *d = static_cast<char *>(vram) +
                          (i % 64) * kExpertBytes;
                const char *s = static_cast<const char *>(host) +
                                (i % 64) * kExpertBytes;
                switch (mode) {
                case 0: {
                    hipMemcpyAsync(d, s, kExpertBytes, hipMemcpyHostToDevice,
                                   side[0]);
                    hipStreamSynchronize(side[0]);
                    inflight = 0;
                    break;
                }
                case 1:
                    hipMemcpy(d, s, kExpertBytes, hipMemcpyHostToDevice);
                    break;
                case 2:
                    hipMemcpyAsync(d, s, kExpertBytes, hipMemcpyHostToDevice, 0);
                    if (++inflight == k) {
                        hipDeviceSynchronize();
                        inflight = 0;
                    }
                    break;
                case 3: {
                    const int st = static_cast<int>(i) % k;
                    hipMemcpyAsync(d, s, kExpertBytes, hipMemcpyHostToDevice,
                                   side[st]);
                    if (++inflight == k) {
                        for (int j = 0; j < k; j++) hipStreamSynchronize(side[j]);
                        inflight = 0;
                    }
                    break;
                }
                default: {
                    // The staging ring, minus the host wait: memcpy into pinned,
                    // then an async copy on the default stream. The pinned
                    // buffer per stream is what bounds the depth.
                    const int st = static_cast<int>(i) % k;
                    char *p = static_cast<char *>(pinned) +
                               static_cast<size_t>(st) * kStageChunk;
                    std::memcpy(p, s, kExpertBytes);
                    hipMemcpyAsync(d, p, kExpertBytes, hipMemcpyHostToDevice, 0);
                    if (++inflight == k) {
                        hipDeviceSynchronize();
                        inflight = 0;
                    }
                    break;
                }
                }
            }
            hipDeviceSynchronize();
            const double us = pnow_us() - t0;
            const double gb = static_cast<double>(nexpert * kExpertBytes) / 1e9;
            std::printf("%-46s %6d %10.2f %10.1f %9.2f\n", mode_name(mode, k), k,
                        gb / (us / 1e6), us / static_cast<double>(nexpert),
                        us / 1e6);
            std::fflush(stdout);
        }
        std::printf("\n");
    }

    // What the workload actually needs: 347 MiB per token moves at the rate
    // above, so ms/token = 347 MiB / (GB/s * 1e6/1024) -- printed so the table
    // can be read as a decode time rather than as a bandwidth.
    std::printf("347 MiB/token, i.e. ms/token at each rate:\n");
    for (double gbs : {1.8, 4.0, 8.0, 16.0, 25.0})
        std::printf("  %5.1f GB/s -> %6.1f ms/token\n", gbs,
                    347.0 / (gbs * 1e6 / 1024.0 / 1000.0));

    for (int i = 0; i < kStreams; i++) hipStreamDestroy(side[i]);
    hipHostFree(pinned);
    std::free(host);
    hipFree(vram);
    return 0;
}