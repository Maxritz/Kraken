// probe_sync.hip -- what does a decode token pay for its HOST-side operations?
//
// A laguna-xs2 decode token costs ~96 ms while the weights it streams are
// ~1.3 G parameters of dense projection plus the lm_head, which at the measured
// 570 GB/s is under 10 ms. So ~85 ms is not arithmetic; it is the host
// round-tripping the device 40 times per token.
//
// One round trip per layer: moe_ffn needs the router row on the host to build
// the expert plan, and the plan is a host array. Getting it there costs
//     gemm -> hipDeviceSynchronize -> blocking hipMemcpy(D2H, 512 B)
// forty times per token. Removing the explicit hipDeviceSynchronize is worth
// only ~3.5 ms/token (measured, KRK_ROUTER_SYNC A/B), which says the barrier
// itself is not the cost -- so either the blocking pageable copy is, or
// something else in the loop is.
//
// This probe prices each of those operations on its own, at decode-realistic
// queue depths, so the answer is a number rather than a guess:
//
//   * an empty kernel launch, host time only
//   * hipDeviceSynchronize with an empty queue (pure barrier overhead)
//   * hipDeviceSynchronize behind N queued kernels (does it scale with depth?)
//   * blocking pageable D2H of the router row, idle and behind a queue
//   * the same copy into PINNED memory, which skips the driver's bounce buffer
//   * async D2H on a side stream + hipStreamSynchronize
//
// A row is only interesting against its depth: the engine's queue at layer 20
// holds the projection kernels of layers 0..19, so "cost behind a queue" is the
// number that matters, not "cost on an idle device".
//
// Build:
//   hipcc -O3 --offload-arch=gfx1201 -I include -I src -I src/hip \
//       tools/probe_sync.hip -o probe_sync
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

// Busy kernel of a known length, so "the queue had work in it" is a statement
// about the device rather than a hope. clock64() is per-wavefront on AMD, so
// one block of 64 threads is two wavefronts on the same counter.
__global__ void spin_kernel(long long cycles, float *sink) {
    const long long t0 = clock64();
    float acc = 0.f;
    while (clock64() - t0 < cycles) acc += 1.0000001f;
    if (acc < 0.f) *sink = acc; // never true: keeps the loop alive
}

static double pu_now_us() {
    return static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch()).count()) / 1e3;
}

// Median of `reps` samples: the minimum is the least polluted by whatever else
// the machine was doing, and the median is what a run actually pays.
static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

static float *g_sink = nullptr;
static long long g_cycles = 0;

// Queues `n` spins of g_cycles each without waiting. The caller is expected to
// consume them with the operation under test.
static void queue_spins(int n) {
    for (int i = 0; i < n; i++) spin_kernel<<<1, 64>>>(g_cycles, g_sink);
}

static void row(const char *what, int depth, double us) {
    std::printf("%-44s depth=%-5d %10.2f us\n", what, depth, us);
}

int main(int argc, char **argv) {
    int reps = 200;
    if (argc > 1) reps = std::atoi(argv[1]);

    hip_check(hipSetDevice(0), "setDevice");
    hipDeviceProp_t prop{};
    hip_check(hipGetDeviceProperties(&prop, 0), "props");
    std::printf("device: %s  CUs=%d  memClk=%d kHz\n\n", prop.gcnArchName,
                prop.multiProcessorCount, prop.memoryClockRate);

    hip_check(hipMalloc(&g_sink, 4), "malloc sink");
    hip_check(hipMemset(g_sink, 0, 4), "memset sink");

    // Calibrate: how many cycles is 50 us of spin on this card? One block, so
    // the queue depth is exact and the calibration is not a grid-shape guess.
    for (long long c : {1000LL, 20000LL, 100000LL}) {
        spin_kernel<<<1, 64>>>(c, g_sink);
        hipDeviceSynchronize();
        const double t0 = pu_now_us();
        spin_kernel<<<1, 64>>>(c, g_sink);
        hipDeviceSynchronize();
        std::printf("spin calibration: %8lld cycles -> %8.2f us\n", c,
                    pu_now_us() - t0);
    }
    g_cycles = 20000; // ~ the 50 us above on this clock

    // Device-side reference for one spin, so a host number can be checked
    // against the work it claims to be waiting for.
    {
        std::vector<double> s;
        for (int r = 0; r < 50; r++) {
            const double t0 = pu_now_us();
            queue_spins(1);
            hipDeviceSynchronize();
            s.push_back(pu_now_us() - t0);
        }
        std::printf("\none queued spin + hipDeviceSynchronize: %.2f us\n\n",
                    median(s));
    }

    struct Row {
        const char *what;
        int depth;
        std::vector<double> us;
    };
    std::vector<Row> rows;
    auto add = [&](const char *what, int depth, double us) {
        rows.push_back(Row{what, depth, {us}});
    };

    // ---- launch: host time only, nothing waited on -----------------------
    for (int depth : {0, 40}) {
        std::vector<double> s;
        for (int r = 0; r < reps; r++) {
            queue_spins(depth);
            const double t0 = pu_now_us();
            spin_kernel<<<1, 64>>>(g_cycles, g_sink);
            s.push_back(pu_now_us() - t0);
            hipDeviceSynchronize();
        }
        add("empty-ish kernel launch (host time)", depth, median(s));
    }

    // ---- hipDeviceSynchronize, empty queue -------------------------------
    {
        std::vector<double> s;
        for (int r = 0; r < reps; r++) {
            hipDeviceSynchronize();
            const double t0 = pu_now_us();
            hipDeviceSynchronize();
            s.push_back(pu_now_us() - t0);
        }
        add("hipDeviceSynchronize (empty queue)", 0, median(s));
    }

    // ---- hipDeviceSynchronize behind a queue -----------------------------
    for (int depth : {1, 10, 40, 200}) {
        std::vector<double> s;
        for (int r = 0; r < reps; r++) {
            hipDeviceSynchronize();
            queue_spins(depth);
            const double t0 = pu_now_us();
            hipDeviceSynchronize();
            s.push_back(pu_now_us() - t0);
        }
        add("hipDeviceSynchronize behind spins", depth, median(s));
    }

    // ---- D2H copies -------------------------------------------------------
    // 512 B is the laguna router row: 256 experts stored as fp16, which is what
    // download_f32 moves. 1 KiB is the same shape for a slightly wider model.
    void *dev = nullptr;
    hip_check(hipMalloc(&dev, 4096), "malloc dev");
    hip_check(hipMemset(dev, 0x3c, 4096), "memset dev");
    void *pageable = std::malloc(4096);
    void *pinned = nullptr;
    hip_check(hipHostMalloc(&pinned, 4096, hipHostMallocDefault), "hostMalloc");
    hipStream_t side = nullptr;
    hip_check(hipStreamCreateWithFlags(&side, hipStreamNonBlocking), "stream");

    for (int depth : {0, 10, 40}) {
        std::vector<double> s;
        for (int r = 0; r < reps; r++) {
            hipDeviceSynchronize();
            queue_spins(depth);
            const double t0 = pu_now_us();
            hipMemcpy(pageable, dev, 512, hipMemcpyDeviceToHost);
            s.push_back(pu_now_us() - t0);
        }
        add("blocking D2H 512 B, PAGEABLE", depth, median(s));
    }
    for (int depth : {0, 10, 40}) {
        std::vector<double> s;
        for (int r = 0; r < reps; r++) {
            hipDeviceSynchronize();
            queue_spins(depth);
            const double t0 = pu_now_us();
            hipMemcpy(pinned, dev, 512, hipMemcpyDeviceToHost);
            s.push_back(pu_now_us() - t0);
        }
        add("blocking D2H 512 B, PINNED", depth, median(s));
    }
    for (int depth : {0, 10, 40}) {
        std::vector<double> s;
        for (int r = 0; r < reps; r++) {
            hipDeviceSynchronize();
            queue_spins(depth);
            const double t0 = pu_now_us();
            hipMemcpyAsync(pinned, dev, 512, hipMemcpyDeviceToHost, side);
            hipStreamSynchronize(side);
            s.push_back(pu_now_us() - t0);
        }
        add("async D2H 512 B on side stream + sync", depth, median(s));
    }
    // What a host wait costs when the copy is on the DEFAULT stream and already
    // ordered behind the queue -- the shape a fused router would need if the
    // plan stayed on the device.
    for (int depth : {0, 40}) {
        std::vector<double> s;
        for (int r = 0; r < reps; r++) {
            hipDeviceSynchronize();
            queue_spins(depth);
            const double t0 = pu_now_us();
            hipMemcpyAsync(pinned, dev, 512, hipMemcpyDeviceToHost, 0);
            hipStreamSynchronize(0);
            s.push_back(pu_now_us() - t0);
        }
        add("async D2H 512 B on DEFAULT stream + sync", depth, median(s));
    }
    // The host-side half of download_f32: a fresh std::vector<u16> per call.
    {
        std::vector<double> s;
        for (int r = 0; r < reps; r++) {
            const double t0 = pu_now_us();
            std::vector<u16> tmp(256);
            s.push_back(pu_now_us() - t0);
            if (tmp[0] == 0xffffu) std::fprintf(stderr, "unreachable\n");
        }
        add("std::vector<u16>(256) allocation", 0, median(s));
    }

    std::printf("\n%-44s %-11s %10s\n", "operation", "queue", "host us");
    for (const Row &r : rows)
        std::printf("%-44s %-11d %10.2f\n", r.what, r.depth,
                    r.us[0]);
    std::printf(
        "\nThe engine does 40 of the depth=40 rows per decode token. Multiply\n"
        "the last column by 40 and compare with the ~96 ms/token it takes:\n"
        "that product is the part of the token spent round-tripping the\n"
        "device, and it is the budget any fused router has to recover.\n");

    hipStreamDestroy(side);
    hipHostFree(pinned);
    std::free(pageable);
    hipFree(dev);
    hipFree(g_sink);
    return 0;
}