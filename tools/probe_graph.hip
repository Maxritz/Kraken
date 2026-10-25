// probe_graph.hip -- what does a decode step pay to SUBMIT its kernels, and what
// does a captured graph replay cost instead?
//
// The open question after the router A/B: a 35B-A3B decode step is 48.6 ms and a
// per-op trace records ~340 ops per token, of which 41% of the device timeline is
// idle. Two very different causes produce that shape:
//
//   (a) the host cannot enqueue faster than the device retires, so the step is
//       submission-bound: the fix is to collapse 340 launches into one graph
//       replay;
//   (b) the device work itself is the 59%, and the "idle" is the recorder's own
//       22-24 us floor landing between real kernels: the fix is kernels, not
//       graphs, and a graph would buy nothing.
//
// This probe separates them at the engine's own granularity: N dependent
// launches of a kernel whose device time is known, submitted op-by-op, against
// the same N nodes captured once and replayed.
//
//   arm            what it prices
//   submit/tiny    337 launches of a ~0 us kernel: pure submission cost
//   submit/work    337 launches of a ~15 us kernel: submission vs device
//   graph/tiny     the same 337 nodes replayed from a graph
//   graph/work     the same, with real work behind each node
//
// If submit/work is much slower than graph/work the step is host-bound and the
// graph is the fix; if they are the same, submission is already hidden and no
// amount of graph work will move the 48.6 ms.
//
// Build (no CMake target: this is a measurement, not part of the engine):
//   hipcc -O3 --offload-arch=gfx1201 tools/probe_graph.hip -o probe_graph
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define HK(e, w)                                                              \
    do {                                                                       \
        hipError_t _r = (e);                                                    \
        if (_r != hipSuccess) {                                                 \
            std::fprintf(stderr, "FATAL %s: %s\n", (w), hipGetErrorName(_r));    \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

static double now_us() {
    return static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                   std::chrono::steady_clock::now().time_since_epoch())
                                   .count()) /
           1e3;
}

// A dependent op: reads the previous op's output, writes the next one's input.
// `reps` sets the device time, and the four arrays are touched so the compiler
// cannot fold the chain away and the dependency is real.
__global__ void work_kernel(float *__restrict__ a, float *__restrict__ b, int n,
                            int reps, int seed) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = a[i] + static_cast<float>(seed);
    float acc = x;
    #pragma unroll 1
    for (int r = 0; r < reps; r++) acc = fmaf(acc, 1.0000001f, x);
    b[i] = acc;
}

static float median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 0.0f : v[v.size() / 2];
}

struct Arm {
    double host_us;
    double wall_us;
};

// Submits `k` dependent launches on `s` and reports the host time spent
// enqueuing and the wall time to completion. Host time is sampled immediately
// after the last launch (before the sync), which is exactly the window in which
// the engine is "submitting" rather than waiting.
static Arm submit_arm(hipStream_t s, int k, float *a, float *b, int n, int reps,
                      int blocks, int threads) {
    HK(hipStreamSynchronize(s), "pre-sync");
    const double t0 = now_us();
    for (int i = 0; i < k; i++)
        work_kernel<<<blocks, threads, 0, s>>>(a, b, n, reps, i);
    const double t1 = now_us();
    HK(hipStreamSynchronize(s), "post-sync");
    const double t2 = now_us();
    return {t1 - t0, t2 - t0};
}

int main(int argc, char **argv) {
    int k = 337; // ops per decode token in the 30B-A3B trace
    if (argc > 1) k = std::atoi(argv[1]);

    hipDeviceProp_t p{};
    HK(hipGetDeviceProperties(&p, 0), "props");
    std::printf("device %s (%s)\n", p.name, p.gcnArchName);
    std::printf("ops per batch: %d\n\n", k);

    const int n = 4096, threads = 256;
    const int blocks = (n + threads - 1) / threads;
    float *a = nullptr, *b = nullptr;
    HK(hipMalloc(&a, sizeof(float) * n), "malloc a");
    HK(hipMalloc(&b, sizeof(float) * n), "malloc b");
    HK(hipMemset(a, 0, sizeof(float) * n), "memset");
    HK(hipMemset(b, 0, sizeof(float) * n), "memset");

    hipStream_t s = nullptr;
    HK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking), "stream");

    // ---- how long is one op on the device? ---------------------------------
    // Calibrated from BATCH wall time, not from a single evented launch. A lone
    // kernel behind hipEventRecord on an idle GPU measures 14-15 us even when it
    // is empty: the event pair plus the clock ramp is the whole number, so a
    // single-launch calibration overstates every op by ~50x and makes the graph
    // arm look free. The batch is what a decode step actually is.
    auto batch_us = [&](int reps, int count) {
        std::vector<double> v;
        for (int i = 0; i < 5; i++) {
            Arm r = submit_arm(s, count, a, b, n, reps, blocks, threads);
            v.push_back(r.wall_us / count);
        }
        return median(v);
    };
    const double tiny_us = batch_us(0, 200);
    int reps = 0;
    double work_us = tiny_us;
    for (int r = 100; r <= 20000; r += 100) {
        reps = r;
        work_us = batch_us(r, 200);
        if (work_us - tiny_us >= 15.0) break;
    }
    std::printf("one op on the device, from batch wall time:\n");
    std::printf("  %.2f us with reps=0, %.2f us with reps=%d (%.2f us of work)\n\n",
                tiny_us, work_us, reps, work_us - tiny_us);

    // ---- warm the clocks before any timed arm -----------------------------
    // The first arm of this probe measured 26% slower than the last for the same
    // work: a 9070 XT idling at its low clock p-state takes seconds to ramp, and
    // whichever arm runs first pays for it. That bias is larger than the effect
    // being measured, so the clocks are warmed and the arms alternate order on
    // every sample instead of running one batch after the other.
    for (int i = 0; i < 400; i++)
        work_kernel<<<blocks, threads, 0, s>>>(a, b, n, reps, i);
    HK(hipStreamSynchronize(s), "warm");

    // ---- capture both graphs once -----------------------------------------
    // Two graphs, because they answer different questions: an empty one prices
    // the driver's per-node replay cost, and the work one prices it against real
    // device time. Both are reported so neither can be quoted as "graph
    // overhead" without the other.
    auto capture = [&](int r, size_t *nodes_out) {
        hipGraph_t g = nullptr;
        hipGraphExec_t ex = nullptr;
        HK(hipStreamBeginCapture(s, hipStreamCaptureModeRelaxed), "begin capture");
        for (int i = 0; i < k; i++)
            work_kernel<<<blocks, threads, 0, s>>>(a, b, n, r, i);
        HK(hipStreamEndCapture(s, &g), "end capture");
        size_t nodes = 0;
        HK(hipGraphGetNodes(g, nullptr, &nodes), "node count");
        HK(hipGraphInstantiate(&ex, g, nullptr, nullptr, 0), "instantiate");
        HK(hipGraphDestroy(g), "destroy graph");
        *nodes_out = nodes;
        return ex;
    };
    size_t nodes_t = 0, nodes_w = 0;
    hipGraphExec_t ex_t = capture(0, &nodes_t);
    hipGraphExec_t ex_w = capture(reps, &nodes_w);

    // One replay, host-visible submit window and wall-to-completion, same shape
    // as submit_arm so the two are comparable line for line.
    auto replay_arm = [&](hipGraphExec_t ex) {
        HK(hipStreamSynchronize(s), "pre-replay");
        const double t0 = now_us();
        HK(hipGraphLaunch(ex, s), "graph launch");
        const double t1 = now_us();
        HK(hipStreamSynchronize(s), "post-replay");
        const double t2 = now_us();
        return Arm{t1 - t0, t2 - t0};
    };

    std::vector<double> s_h_t, s_w_t, s_h_w, s_w_w, g_h_t, g_w_t, g_h_w, g_w_w;
    for (int i = 0; i < 9; i++) {
        const bool graph_first = (i & 1) != 0;
        for (int pass = 0; pass < 2; pass++) {
            const bool do_graph = (pass == 0) == graph_first;
            if (do_graph) {
                Arm g = replay_arm(ex_w);
                g_h_w.push_back(g.host_us);
                g_w_w.push_back(g.wall_us);
                Arm gt = replay_arm(ex_t);
                g_h_t.push_back(gt.host_us);
                g_w_t.push_back(gt.wall_us);
            } else {
                Arm w = submit_arm(s, k, a, b, n, reps, blocks, threads);
                s_h_w.push_back(w.host_us);
                s_w_w.push_back(w.wall_us);
                Arm t = submit_arm(s, k, a, b, n, 0, blocks, threads);
                s_h_t.push_back(t.host_us);
                s_w_t.push_back(t.wall_us);
            }
        }
    }
    const double h_t = median(s_h_t), w_t = median(s_w_t);
    const double h_w = median(s_h_w), w_w = median(s_w_w);
    const double h_gt = median(g_h_t), w_gt = median(g_w_t);
    const double h_gw = median(g_h_w), w_g = median(g_w_w);

    std::printf("                        host us   wall us   wall/op   host/op\n");
    std::printf("submit, empty op      %9.1f %9.1f %9.2f %9.2f\n", h_t, w_t, w_t / k, h_t / k);
    std::printf("submit, %2d us op      %9.1f %9.1f %9.2f %9.2f\n", static_cast<int>(work_us),
                h_w, w_w, w_w / k, h_w / k);
    std::printf("graph, empty op (%zu)  %9.1f %9.1f %9.2f %9.2f\n", nodes_t, h_gt, w_gt,
                w_gt / k, h_gt / k);
    std::printf("graph, %2d us op (%zu)  %9.1f %9.1f %9.2f %9.2f\n",
                static_cast<int>(work_us), nodes_w, h_gw, w_g, w_g / k, h_gw / k);
    std::printf("\n");
    std::printf("submission per launch:      %.2f us\n", h_t / k);
    std::printf("graph replay per node:      %.2f us\n", h_gw / k);
    std::printf("graph speedup, %.1f us work/op: %.2fx\n", work_us, w_w / w_g);
    std::printf("graph speedup, empty op:        %.2fx\n", w_t / w_gt);
    std::printf("per-node gap recovered:     %.2f us\n", (w_w - w_g) / k);
    std::printf("\n");

    // ---- what the split means for a decode token --------------------------
    std::printf("A decode token at %d ops costs (us):\n", k);
    std::printf("  op-by-op   wall                = %.1f\n", w_w);
    std::printf("  graph      replay wall         = %.1f\n", w_g);
    std::printf("  saved per token                = %.1f\n", w_w - w_g);

    HK(hipGraphExecDestroy(ex_t), "destroy exec");
    HK(hipGraphExecDestroy(ex_w), "destroy exec");
    HK(hipStreamDestroy(s), "destroy stream");
    HK(hipFree(a), "free");
    HK(hipFree(b), "free");
    return 0;
}
