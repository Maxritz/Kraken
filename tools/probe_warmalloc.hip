// Is the warmup's allocation cost PER CALL or PER PAGE?
//
// The engine's WARM tier is fully pinned on this box (`hipHostMalloc` per
// expert slice): a 17.2 GiB tier of 10,240 experts is 30,720 calls of ~0.56 MiB,
// and the load phase charges 4.5-5.5 s to them. Two explanations fit that
// number and they call for opposite fixes:
//
//   per call  -> one arena per layer (40 calls instead of 30,720) removes it.
//                This is the hypothesis the arena test was asked to settle.
//   per page  -> an arena removes NOTHING. The driver still probes and locks
//                4.22 M pages; only the loop around them gets shorter, and that
//                loop is a syscall the arena keeps paying once per arena.
//
// The sweep below holds BYTES constant (one 17.2 GiB tier) and moves how many
// experts share an allocation, from one slice per call to one whole-tier block.
// It separates the two costs the arena question actually turns on:
//
//   per call  -> the rate improves as calls fall, even at a fixed page count
//   per page  -> the rate tracks the BLOCK size: the driver probes and locks a
//                long run of pages far cheaper than 137-page runs, and no
//                amount of call batching at small block size recovers that.
//
// The touch column is separate on purpose: pages the driver lays down at
// allocation and pages faulted on first write are different costs, and the
// engine's read faults whatever the allocator left untouched.
//
// Build and run (no project headers: this measures the RUNTIME):
//   hipcc -O2 -o /tmp/probe_warmalloc.exe tools/probe_warmalloc.hip \
//         --offload-arch=gfx1201
//   /tmp/probe_warmalloc.exe
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;

static double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

struct Arm {
    const char *name;
    size_t count;
    size_t bytes;
};

// One tier, `experts_per_block` experts per allocation. Returns alloc ms.

static void run(const Arm &a, bool touch) {
    std::vector<void *> p;
    p.reserve(a.count);
    // Time the allocation alone. A failed call is not retried: a refusal is an
    // answer about the pinned pool, not a reason to loop.
    double best_alloc = 1e30;
    for (int rep = 0; rep < 2; rep++) { // two reps, keep the better: the first
                                        // pays page-table faults in the driver
        const Clock::time_point t0 = Clock::now();
        for (size_t i = 0; i < a.count; i++) {
            void *q = nullptr;
            if (hipHostMalloc(&q, a.bytes) != hipSuccess) {
                (void)hipGetLastError();
                break;
            }
            p.push_back(q);
        }
        const double alloc = ms_since(t0);
        if (alloc < best_alloc) best_alloc = alloc;

        // First write over every byte: the cost the driver deferred, which the
        // engine's read then pays in whichever thread does the pread.
        double touch_ms = 0.0;
        if (touch) {
            const Clock::time_point w0 = Clock::now();
            for (void *q : p) std::memset(q, 0x5a, a.bytes);
            touch_ms = ms_since(w0);
        }
        const Clock::time_point f0 = Clock::now();
        for (void *q : p) (void)hipHostFree(q);
        const double free_ms = ms_since(f0);
        p.clear();
        if (rep == 1) {
            const double got = double(a.count) * double(a.bytes);
            const double pages = got / 4096.0;
            std::printf(
                "  %-22s %6zu calls x %8.3f MiB  %9.1f ms  %8.1f us/call "
                "%7.2f ns/page  %6.2f GB/s | free %7.1f ms | touch %7.1f ms\n",
                a.name, a.count, double(a.bytes) / 1048576.0, best_alloc,
                best_alloc * 1000.0 / double(a.count ? a.count : 1),
                best_alloc * 1e6 / pages, got / (best_alloc * 1e6),
                free_ms, touch_ms);
        }
    }
}

int main() {
    hipDeviceProp_t prop{};
    int rt = 0;
    (void)hipRuntimeGetVersion(&rt);
    if (hipGetDeviceProperties(&prop, 0) == hipSuccess)
        std::printf("device: %s  host-pinned pool test, runtime %d\n", prop.name,
                    rt);

    const size_t kExpertSlice = 601216; // ~0.573 MiB, the 35B's shape/3
    const size_t kExperts = 10240;
    const size_t kSlices = 3;
    const size_t kPerExpert = kExpertSlice * kSlices;
    const size_t kTotal = kPerExpert * kExperts;

    std::printf("\none %5.1f GiB tier, pinned, block = experts per allocation\n",
                double(kTotal) / 1073741824.0);
    std::printf("  %7s %8s %12s %10s %11s %10s %9s\n", "blk/exp", "calls",
                "blk MiB", "alloc ms", "us/call", "ns/page", "GB/s");

    // Slice-per-call is the engine's shape today, and it is the row the arena
    // has to beat: three calls per expert, so blocks of 1/3 expert.
    const size_t blocks[] = {0, 1, 2, 4, 8, 16, 32, 64, 256, 10240};
    for (size_t b : blocks) {
        const size_t bytes = b == 0 ? kExpertSlice : kPerExpert * b;
        const size_t count = kTotal / bytes;
        std::vector<void *> p;
        p.reserve(count);
        const Clock::time_point t0 = Clock::now();
        for (size_t i = 0; i < count; i++) {
            void *q = nullptr;
            if (hipHostMalloc(&q, bytes) != hipSuccess) {
                (void)hipGetLastError();
                break;
            }
            p.push_back(q);
        }
        const double alloc = ms_since(t0);
        const double got = double(p.size()) * double(bytes);
        const Clock::time_point f0 = Clock::now();
        for (void *q : p) (void)hipHostFree(q);
        const double free_ms = ms_since(f0);
        std::printf("  %7s %8zu %12.2f %10.1f %11.1f %10.2f %9.2f | free %7.1f ms\n",
                    b == 0 ? "1/3" : std::to_string(b).c_str(), p.size(),
                    double(bytes) / 1048576.0, alloc,
                    p.empty() ? 0.0 : alloc * 1000.0 / double(p.size()),
                    alloc * 1e6 / (got / 4096.0), got / (alloc * 1e6), free_ms);
        p.clear();
    }
    return 0;
}
