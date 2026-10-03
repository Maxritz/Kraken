// H2D transfer probe: why do the weight uploads run at ~4 GB/s?
//
// Measured on the 8B load path: 4.79 GB of weights land in ~1200 ms, i.e.
// 3.3-4.7 GB/s, and every individual tensor shows the same rate (ffn_down
// 8.4 ms / 36 MB = 4.5 GB/s). PCIe 4.0 x16 should be ~25 GB/s. The upload path
// is `hipMemcpy` straight out of the mmap, which is PAGEABLE memory, and the
// driver has to bounce pageable copies through an internal pinned buffer.
//
// This probe separates the two explanations:
//   (a) the link is the limit        -> pinned is also ~4 GB/s
//   (b) pageable bounce is the limit -> pinned is several times faster
//
// It also measures the pinned-ring variant (many small pinned copies), because
// that is what a fix would actually do for a tensor larger than one ring slot.
//
// Build:
//   hipcc -O3 --offload-arch=gfx1201 -I include -I src -I src/hip \
//       tools/probe_h2d.hip -o probe_h2d
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "krk/common.hpp"
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
} // namespace krk

static double ph_now_ms() {
    return static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::steady_clock::now().time_since_epoch()).count()) / 1000.0;
}

static void report(const char *what, size_t bytes, double ms) {
    std::printf("  %-34s %8.1f MiB  %9.2f ms  %7.2f GB/s\n", what,
                static_cast<double>(bytes) / 1048576.0, ms,
                static_cast<double>(bytes) / 1073741824.0 / (ms / 1000.0));
}

int main_file(const char *path);

int main(int argc, char **argv) {
    // Sizes chosen to span the ring threshold (64 KiB) up to a real weight
    // tensor (350 MB).
    const size_t sizes[] = {1u << 20, 8u << 20, 64u << 20, 350u << 20};
    const int reps = 5;

    for (size_t sz : sizes) {
        std::vector<char> pageable(sz);
        std::memset(pageable.data(), 0x5a, sz);

        void *dev = nullptr;
        KRK_HIP(hipMalloc(&dev, sz));
        KRK_HIP(hipMemset(dev, 0, sz));

        // --- pageable (what the loader does today for >64 KiB) -------------
        double best = 1e30;
        for (int i = 0; i < reps; i++) {
            const double t0 = ph_now_ms();
            KRK_HIP(hipMemcpy(dev, pageable.data(), sz, hipMemcpyHostToDevice));
            KRK_HIP(hipDeviceSynchronize());
            const double t1 = ph_now_ms();
            if (t1 - t0 < best) best = t1 - t0;
        }
        report("pageable hipMemcpy", sz, best);

        // --- pinned, one copy ---------------------------------------------
        void *pin = nullptr;
        if (hipHostMalloc(&pin, sz) == hipSuccess) {
            std::memcpy(pin, pageable.data(), sz);
            best = 1e30;
            for (int i = 0; i < reps; i++) {
                const double t0 = ph_now_ms();
                KRK_HIP(hipMemcpy(dev, pin, sz, hipMemcpyHostToDevice));
                KRK_HIP(hipDeviceSynchronize());
                const double t1 = ph_now_ms();
                if (t1 - t0 < best) best = t1 - t0;
            }
            report("pinned hipMemcpy", sz, best);

            // --- pinned ring: 1 MiB slots, the shape a fix would use -------
            const size_t slot = 1u << 20;
            const size_t n_slot = sz / slot;
            if (n_slot >= 1) {
                best = 1e30;
                for (int i = 0; i < reps; i++) {
                    const double t0 = ph_now_ms();
                    for (size_t c = 0; c < n_slot; c++) {
                        std::memcpy(pin, pageable.data() + c * slot, slot);
                        KRK_HIP(hipMemcpy(static_cast<char *>(dev) + c * slot, pin,
                                          slot, hipMemcpyHostToDevice));
                    }
                    KRK_HIP(hipDeviceSynchronize());
                    const double t1 = ph_now_ms();
                    if (t1 - t0 < best) best = t1 - t0;
                }
                report("pinned ring 1 MiB + memcpy in", sz, best);
            }
            (void)hipHostFree(pin);
        } else {
            (void)hipGetLastError();
            std::printf("  %-34s (pinned alloc failed)\n", "pinned hipMemcpy");
        }

        // --- host-side memcpy alone, to bound what staging costs -----------
        {
            std::vector<char> dst(sz);
            best = 1e30;
            for (int i = 0; i < reps; i++) {
                const double t0 = ph_now_ms();
                std::memcpy(dst.data(), pageable.data(), sz);
                const double t1 = ph_now_ms();
                if (t1 - t0 < best) best = t1 - t0;
            }
            report("host memcpy (staging cost)", sz, best);
        }

        (void)hipFree(dev);
        std::printf("\n");
    }
    if (argc > 1) return main_file(argv[1]);
    return 0;
}
// Second half: is the loader's 4.5 GB/s a property of the SOURCE being a
// file-backed mmap rather than anonymous RAM? Same hipMemcpy, same size, two
// sources. If the file source is slower, the loader is disk/page-fault bound
// and pinning the destination would not help.
int main_file(const char *path) {
    // Use a real model file, mmap it, and copy from inside it.
    std::printf("=== file-backed source vs anonymous, hipMemcpy pageable ===\n");
    MappedFile mf;
    std::string err;
    if (!mf.open(path, &err)) {
        std::printf("  cannot map %s: %s\n", path, err.c_str());
        return 1;
    }
    const size_t sz = 350u << 20;
    if (mf.size() < static_cast<i64>(sz)) {
        std::printf("  file too small (%lld < %zu)\n",
                    static_cast<long long>(mf.size()), sz);
        return 1;
    }
    void *dev = nullptr;
    KRK_HIP(hipMalloc(&dev, sz));
    std::vector<char> anon(sz, 0x5a);

    const int reps = 4;
    for (int pass = 0; pass < 2; pass++) {
        // pass 0 = first touch (page cache may be cold), pass 1 = repeat
        double b_anon = 1e30, b_file = 1e30;
        for (int i = 0; i < reps; i++) {
            double t0 = ph_now_ms();
            KRK_HIP(hipMemcpy(dev, anon.data(), sz, hipMemcpyHostToDevice));
            KRK_HIP(hipDeviceSynchronize());
            double t1 = ph_now_ms();
            if (t1 - t0 < b_anon) b_anon = t1 - t0;

            t0 = ph_now_ms();
            KRK_HIP(hipMemcpy(dev, reinterpret_cast<const char *>(mf.data()) + (pass * sz),
                              sz, hipMemcpyHostToDevice));
            KRK_HIP(hipDeviceSynchronize());
            t1 = ph_now_ms();
            if (t1 - t0 < b_file) b_file = t1 - t0;
        }
        std::printf("  pass %d (offset %zu MiB):\n", pass, (pass * sz) >> 20);
        report("  anonymous RAM source", sz, b_anon);
        report("  file mmap source", sz, b_file);
    }
    // CPU-only read of the same file range, to bound what the disk gives us.
    {
        std::vector<char> sink(sz);
        const double t0 = ph_now_ms();
        std::memcpy(sink.data(), reinterpret_cast<const char *>(mf.data()), sz);
        const double t1 = ph_now_ms();
        report("  host memcpy from file mmap", sz, t1 - t0);
    }
    (void)hipFree(dev);
    return 0;
}
