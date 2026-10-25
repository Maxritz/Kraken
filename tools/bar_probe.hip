// bar_probe -- what the aperture actually is, and which copy path it pays for.
//
// Why this exists: `hipDeviceAttributeIsLargeBar` is a HINT (it reads 0 on this
// machine, yet a 1.5 GiB pageable copy runs at 22.9 GB/s, which a 256 MiB
// aperture cannot do). The engine's own H2D probe only ever copies 16/32/64
// MiB -- every one of which fits inside a 256 MiB bounce window -- so it cannot
// see the aperture, and it never measures the one path a wide aperture makes
// fastest: the driver's own blocking pageable copy.
//
// Modes here mirror the engine's, one for one:
//   blocking   hipMemcpy pageable -> device                (engine: paged_blocking)
//   async      hipMemcpyAsync pageable on a side stream    (engine: direct arm)
//   ring       memcpy into pinned slot, async DMA from it  (engine: staging ring)
//   pinned     hipMemcpyAsync from pinned                  (engine: pinned arm)
//   mapped     device reads host-pinned memory in place    (engine: not used)
//
// Build:
//   clang++ --offload-arch=gfx1201 -O3 -o build-hip/bar_probe.exe tools/bar_probe.hip
//
// Usage: bar_probe [max_mib]
#include <hip/hip_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch())
        .count();
}

// A read of every byte, on the device, so a mapped case measures the bus and not
// a memcpy the host performed.
__global__ void sum_bytes(const unsigned char *p, size_t n, unsigned *out) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned acc = 0;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n; i += stride) acc += p[i];
    if (acc == 0xFFFFFFFFu) atomicAdd(out, 1u);
}

static double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

struct Rig {
    void *dev = nullptr;
    void *pag = nullptr;   // pageable host
    void *pin = nullptr;   // pinned host
    void *slot = nullptr;  // one pinned ring slot
    size_t bytes = 0;
    size_t slot_bytes = 4u << 20;
    hipStream_t s = nullptr;
    hipEvent_t ev = nullptr;
};

static double run_mode(Rig &r, const char *mode) {
    std::vector<double> gbs;
    const size_t n = r.slot_bytes < r.bytes ? r.slot_bytes : r.bytes;
    for (int it = 0; it < 7; it++) {
        const double t0 = now_ms();
        if (std::strcmp(mode, "blocking") == 0) {
            hipMemcpy(r.dev, r.pag, r.bytes, hipMemcpyHostToDevice);
        } else if (std::strcmp(mode, "async") == 0) {
            hipMemcpyAsync(r.dev, r.pag, r.bytes, hipMemcpyHostToDevice, r.s);
            hipStreamSynchronize(r.s);
        } else if (std::strcmp(mode, "ring") == 0) {
            // Chunked exactly as the engine does it: memcpy into a pinned slot,
            // async DMA out of it, one chunk in flight at a time.
            for (size_t off = 0; off < r.bytes; off += n) {
                const size_t len = r.bytes - off < n ? r.bytes - off : n;
                std::memcpy(r.slot, (const char *)r.pag + off, len);
                hipMemcpyAsync((char *)r.dev + off, r.slot, len,
                               hipMemcpyHostToDevice, r.s);
                hipStreamSynchronize(r.s);
            }
        } else if (std::strcmp(mode, "pinned") == 0) {
            hipMemcpyAsync(r.dev, r.pin, r.bytes, hipMemcpyHostToDevice, r.s);
            hipStreamSynchronize(r.s);
        } else {
            return 0.0;
        }
        const double ms = now_ms() - t0;
        if (it > 0 && ms > 0.0) gbs.push_back(r.bytes / (ms * 1e6));
    }
    return median(gbs);
}

static void clear_error() {
    hipError_t e = hipGetLastError();
    (void)e;
}

int main(int argc, char **argv) {
    const size_t max_mib = argc > 1 ? (size_t)std::atoi(argv[1]) : 1024;

    hipDeviceProp_t p{};
    if (hipGetDeviceProperties(&p, 0) != hipSuccess) {
        std::printf("no device\n");
        return 1;
    }
    int large_bar = -1;
    if (hipDeviceGetAttribute(&large_bar, hipDeviceAttributeIsLargeBar, 0) !=
        hipSuccess) {
        clear_error();
        large_bar = -1;
    }
    std::printf("%s (%s)\n", p.name, p.gcnArchName);
    std::printf("runtime says large BAR: %s\n\n",
                large_bar == 1 ? "yes"
                               : large_bar == 0 ? "NO (256 MiB) <-- claim under test"
                                                : "unknown");

    Rig r;
    hipStreamCreate(&r.s);
    hipEventCreate(&r.ev);
    if (hipMalloc(&r.dev, max_mib << 20) != hipSuccess) {
        std::printf("device allocation of %zu MiB failed\n", max_mib);
        return 1;
    }
    hipHostMalloc(&r.pag, max_mib << 20, hipHostMallocDefault);  // pinned? no: see below
    // hipHostMalloc is pinned by definition; the pageable arm needs malloc.
    hipHostFree(r.pag);
    r.pag = std::malloc(max_mib << 20);
    hipHostMalloc(&r.pin, max_mib << 20, hipHostMallocDefault);
    hipHostMalloc(&r.slot, 4u << 20, hipHostMallocDefault);
    if (!r.pag || !r.pin || !r.slot) {
        std::printf("host allocation of %zu MiB failed\n", max_mib);
        return 1;
    }
    std::memset(r.pag, 1, max_mib << 20);  // fault the pages in: a measurement of
    std::memset(r.pin, 2, max_mib << 20);  // first touch is not a copy rate

    std::printf("%8s  %8s  %8s  %8s  %8s   %s\n", "size", "blocking", "async",
                "ring", "pinned", "fastest");
    for (size_t mib = 2; mib <= max_mib; mib *= (mib < 256 ? 2 : 4)) {
        r.bytes = mib << 20;
        r.slot_bytes = 4u << 20;
        // Warm each arm once: the first pageable copy is where the driver picks
        // its staging strategy, and charging that to the rate measures startup.
        const char *modes[] = {"blocking", "async", "ring", "pinned"};
        for (const char *m : modes) run_mode(r, m);
        double b = run_mode(r, "blocking");
        double a = run_mode(r, "async");
        double g = run_mode(r, "ring");
        double pi = run_mode(r, "pinned");
        const double best = std::max(std::max(b, a), std::max(g, pi));
        const char *who = best == b ? "blocking" : best == a ? "async"
                          : best == g ? "ring" : "pinned";
        std::printf("%6zu M  %8.2f  %8.2f  %8.2f  %8.2f   %s\n", mib, b, a, g,
                    pi, who);
        if (mib == max_mib) break;
    }

    // The point of a wide aperture: the device can read host memory in place, so
    // a weight never has to be copied to VRAM to be used. Measured as a kernel
    // reading every byte.
    std::printf("\ndevice reads host-pinned memory in place (mapped)\n");
    for (size_t mib = 16; mib <= max_mib; mib *= 4) {
        void *host = nullptr;
        if (hipHostMalloc(&host, mib << 20, hipHostMallocMapped) != hipSuccess) {
            clear_error();
            std::printf("  %5zu MiB  host-mapped allocation failed\n", mib);
            continue;
        }
        std::memset(host, 3, mib << 20);
        void *dev = nullptr;
        if (hipHostGetDevicePointer(&dev, host, 0) != hipSuccess) {
            clear_error();
            std::printf("  %5zu MiB  no device pointer for host memory\n", mib);
            hipHostFree(host);
            continue;
        }
        unsigned *out = nullptr;
        hipMalloc(&out, sizeof(unsigned));
        std::vector<double> gbs;
        for (int it = 0; it < 5; it++) {
            const double t0 = now_ms();
            hipLaunchKernelGGL(sum_bytes, dim3(512), dim3(256), 0, r.s,
                               (const unsigned char *)dev, mib << 20, out);
            hipStreamSynchronize(r.s);
            const double ms = now_ms() - t0;
            if (it > 0 && ms > 0.0) gbs.push_back((mib << 20) / (ms * 1e6));
        }
        hipFree(out);
        std::printf("  %5zu MiB  %6.2f GB/s\n", mib, median(gbs));
        hipHostFree(host);
        if (mib * 4 > max_mib) break;
    }

    // How much of a small copy's cost is the COPY and how much is the number of
    // calls. The engine's streamed-layer path issues one DMA per tensor, so a
    // 1.67 MB layer can be 32 copies of 52 KB -- and the op timeline reports
    // that as 482 us, which is 3.5 GB/s. Fragmenting the same bytes here says
    // which of the two it is: if 32 x 52 KiB costs what the timeline says, the
    // cost is per-call and coalescing adjacent tensors is the fix; if the single
    // 1.67 MB copy costs the same, the cost is the transfer and coalescing
    // would buy nothing.
    std::printf("\nfragmentation: same bytes, one copy against many\n");
    {
        const size_t total = 1677721;     // 1.67 MB, the streamed-layer shape
        const size_t frag = total / 32;   // 32 tensors, one DMA each
        void *d = nullptr, *p = nullptr;
        hipMalloc(&d, total + 4096);
        hipHostMalloc(&p, total + 4096, hipHostMallocDefault);
        std::memset(p, 7, total + 4096);
        auto one = [&](bool fragmented) {
            std::vector<double> gbs;
            for (int it = 0; it < 9; it++) {
                const double t0 = now_ms();
                if (fragmented) {
                    for (size_t off = 0; off < total; off += frag)
                        hipMemcpyAsync((char *)d + off, (const char *)p + off,
                                       frag, hipMemcpyHostToDevice, r.s);
                } else {
                    hipMemcpyAsync(d, p, total, hipMemcpyHostToDevice, r.s);
                }
                hipStreamSynchronize(r.s);
                const double ms = now_ms() - t0;
                if (it > 0 && ms > 0.0) gbs.push_back(total / (ms * 1e6));
            }
            return std::vector<double>{median(gbs), gbs.empty() ? 0.0 : gbs.front()};
        };
        one(false);
        one(true);
        const std::vector<double> whole = one(false), pieces = one(true);
        std::printf("  %zu bytes as 1 copy      %6.2f GB/s\n", total, whole[0]);
        std::printf("  %zu bytes as 32 x %zu KiB %6.2f GB/s\n", total, frag >> 10,
                    pieces[0]);
        std::printf("  per-call cost of the fragmented form: %.0f us over 32 calls\n",
                    whole[0] > 0.0 ? (total / (whole[0] * 1e6) * 32 * 1000 *
                                      (1.0 - whole[0] / (pieces[0] > 0.0 ? pieces[0] : 1.0)))
                                     : 0.0);
        hipFree(d);
        hipHostFree(p);
    }

    hipFree(r.dev);
    hipHostFree(r.pin);
    hipHostFree(r.slot);
    std::free(r.pag);
    return 0;
}
