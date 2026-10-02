// Two questions that have to be measured before any int8 GEMM is worth
// writing, and before any claim about how fast the fp16 path "could" be.
//
//   1. Is v_dot4c_i32_i8 lane-local or cross-lane on gfx1031?
//   2. What is the real v_dot2c_f32_f16 issue ceiling?
//
// Both earlier probes for these were wrong in instructive ways:
//
//   * A lane-semantics probe that gave every lane a DIFFERENT operand and then
//     asked whether the results matched cannot distinguish anything: different
//     inputs must produce different outputs. The correct test gives every lane
//     the SAME operand and checks the results agree; a second test gives each
//     lane a distinct operand and checks which hypothesis each lane satisfies.
//
//   * A peak probe that reports a number while the achieved rate is still
//     climbing with ILP is measuring LATENCY, not throughput. The rate has to
//     be flat across the deepest chains before any of it is a ceiling.
//
// Build:
//   hipcc -O3 --offload-arch=gfx1031 -I include -I src -I src/hip \
//       tools/probe_gemm_tile.hip -o probe
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "krk_hip.hpp"

using namespace krk;

// This file has to carry its own HIP_CHECK: it was written when the shared
// header still defined one, and the header has since stopped. Without it the
// probe does not compile at all (17 `use of undeclared identifier` errors),
// which is how the stale 0.81 TFLOP/s number in docs/TODO.md outlived its
// probe.
#define HIP_CHECK(x)                                                              \
    do {                                                                          \
        const hipError_t e_ = (x);                                                \
        if (e_ != hipSuccess) {                                                   \
            std::printf("HIP error %s at %s:%d\n", hipGetErrorString(e_),         \
                        __FILE__, __LINE__);                                      \
            std::exit(1);                                                         \
        }                                                                         \
    } while (0)

// ===========================================================================
//  1. DP4A lane semantics
// ===========================================================================

// Test A: identical operands in all 32 lanes.
//   lane-local  -> every lane returns sum(a bytes) * sum(1s) = 1+2+3+4 = 10
//   cross-lane  -> lanes pair up and see two lanes' worth of data; the results
//                  differ between even and odd lanes.
__global__ void probe_same_input(int *out) {
    const unsigned a = 0x04030201u; // bytes 1,2,3,4
    const unsigned b = 0x01010101u; // all ones
    const int r = __builtin_amdgcn_sdot4(static_cast<int>(a),
                                         static_cast<int>(b), 0, 0);
    if (threadIdx.x < 32) out[threadIdx.x] = r;
}

// Test B: operand varies per lane, the other operand is all ones.
//   lane-local  -> lane L computes (L+1)+(L+2)+(L+3)+(L+4) = 4L + 10
//   cross-lane  -> lane L also folds in lane L+/-1's bytes, so the value will
//                  not follow 4L+10 and will not be constant across lanes.
// The harness below fits BOTH models to the measured values and reports which
// one they match.
__global__ void probe_lane_indexed(int *out) {
    const unsigned b = 0x01010101u;
    const unsigned a = 0x04030201u + (threadIdx.x & 31) * 0x01010101u;
    const int r = __builtin_amdgcn_sdot4(static_cast<int>(a),
                                         static_cast<int>(b), 0, 0);
    if (threadIdx.x < 32) out[threadIdx.x] = r;
}

// Test C: the accumulator is added, not multiplied into.
__global__ void probe_accumulator(int *out) {
    const unsigned a = 0x04030201u, b = 0x01010101u;
    if (threadIdx.x < 32)
        out[threadIdx.x] = __builtin_amdgcn_sdot4(static_cast<int>(a),
                                                  static_cast<int>(b), 100, 0);
}

void run_lane_probes() {
    int *d = nullptr;
    HIP_CHECK(hipMalloc(&d, 32 * sizeof(int)));
    std::vector<int> h(32);

    std::printf("=== 1. DP4A (v_dot4c_i32_i8) lane semantics ===\n");

    HIP_CHECK(hipMemset(d, 0, 32 * sizeof(int)));
    probe_same_input<<<1, 32>>>(d);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h.data(), d, 32 * sizeof(int), hipMemcpyDeviceToHost));
    bool same_all = true;
    for (int l = 0; l < 32; l++)
        if (h[l] != h[0]) same_all = false;
    std::printf("  A. identical operands in all lanes (a=1,2,3,4 b=1,1,1,1)\n");
    std::printf("     lane0=%d lane1=%d lane16=%d lane31=%d   want all 10\n",
                h[0], h[1], h[16], h[31]);
    std::printf("     all 32 lanes agree? %s\n", same_all ? "YES" : "NO");

    HIP_CHECK(hipMemset(d, 0, 32 * sizeof(int)));
    probe_lane_indexed<<<1, 32>>>(d);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h.data(), d, 32 * sizeof(int), hipMemcpyDeviceToHost));
    int own_match = 0, prev_match = 0, next_match = 0;
    for (int l = 0; l < 32; l++) {
        const int own = 4 * l + 10;
        const int prev = 4 * ((l + 31) % 32) + 10;
        const int next = 4 * ((l + 1) % 32) + 10;
        if (h[l] == own) own_match++;
        if (h[l] == prev) prev_match++;
        if (h[l] == next) next_match++;
    }
    std::printf("  B. operand varies per lane, other operand all ones\n");
    std::printf("     lane0=%d lane1=%d lane2=%d  (own-byte model 10,14,18)\n",
                h[0], h[1], h[2]);
    std::printf("     lanes matching OWN bytes      : %2d/32\n", own_match);
    std::printf("     lanes matching PREV lane bytes: %2d/32\n", prev_match);
    std::printf("     lanes matching NEXT lane bytes: %2d/32\n", next_match);

    HIP_CHECK(hipMemset(d, 0, 32 * sizeof(int)));
    probe_accumulator<<<1, 32>>>(d);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h.data(), d, 32 * sizeof(int), hipMemcpyDeviceToHost));
    std::printf("  C. accumulator passthrough (c=100) -> %d (want 110)\n\n", h[0]);

    if (same_all && own_match == 32)
        std::printf("  VERDICT: LANE-LOCAL. Each lane consumes only its own four\n"
                    "  bytes, so an ordinary row-major int8 layout feeds sdot4\n"
                    "  directly and no cross-lane reduction or shuffle is needed.\n"
                    "  The 'cross-lane' caveat does not hold on gfx1031.\n\n");
    else
        std::printf("  VERDICT: NOT LANE-LOCAL. An int8 GEMM must permute its\n"
                    "  layout to match the pairing above; do not assume row-major.\n\n");
    HIP_CHECK(hipFree(d));
}

// ===========================================================================
//  2. v_dot2c_f32_f16 issue ceiling
// ===========================================================================

// The rate climbs with ILP until the pipes saturate, then flattens. The
// ceiling is the FLAT part, so the sweep has to be deep enough to see it.
//
// THIS SECTION MEASURED 0.81 TFLOP/s AND WAS WRONG THREE WAYS. The third is
// the one that produced the number, and all three are fixed here:
//
//   1. d_dot2() takes a POINTER and reinterprets it as a packed pair. Pointing
//      it at an indexed _Float16 array forces that array to be addressable —
//      local memory — which is exactly the trap krk_hip.hpp warns about for
//      register-resident operands. The packed register form (d_dot2_pk, u32
//      operands) is the one this probe needs.
//   2. It ran with cus = multiProcessorCount = 20 on a 40-CU part, so the
//      grid it launched was half the size its own table claimed.
//   3. 512 dot2 per thread over 160 blocks x 256 threads is 0.084 GFLOP per
//      launch: tens of microseconds of work whose timing is dominated by
//      launch ramp and drain. The old probe reported that overhead as a pipe
//      ceiling. Work is now ~21 GFLOP per launch at the narrowest row, the
//      throughput and per-launch time are both printed, and the last block
//      re-runs the same kernel at 1/16 the grid so a rate that is still
//      climbing with work per launch is visibly not a ceiling.
//
// Reference: the corrected register-only ceiling on this card is ~25.3
// TFLOP/s, reproduced by tools/dot2_ceiling.cpp and independent of this file.
constexpr int kTotalDot2 = 8192;  // per thread, held constant across NACC

template <int NACC>
__global__ void peak_dot2(f32 *out, int reps) {
    // Two shared operand pairs: register cost NACC + 4 instead of 3*NACC.
    u32 a[2], b[2];
#pragma unroll
    for (int i = 0; i < 2; i++) {
        a[i] = 0x3c003c00u + static_cast<u32>(i * 3 + 1) * 0x00010001u;
        b[i] = 0x34003400u + static_cast<u32>(i * 5 + 2) * 0x00010001u;
    }
    f32 acc[NACC];
#pragma unroll
    for (int i = 0; i < NACC; i++) acc[i] = 0.0f;

#pragma unroll 1
    for (int r = 0; r < reps; r++) {
#pragma unroll
        for (int i = 0; i < NACC; i++)
            acc[i] = d_dot2_pk(a[i & 1], b[i & 1], acc[i]);
    }

    f32 s = 0.0f;
#pragma unroll
    for (int i = 0; i < NACC; i++) s += acc[i];
    if (s == 1234.5f) out[0] = s; // never true; keeps the work live
}

struct PeakRow {
    double ms;          // per launch
    double tflops;
    double waves_per_cu;
    bool ok;
};

template <int NACC>
PeakRow measure_dot2(int cus, int blocks, int threads, double clk) {
    const int reps = kTotalDot2 / NACC;
    PeakRow r{};
    f32 *out = nullptr;
    HIP_CHECK(hipMalloc(&out, 4096));
    HIP_CHECK(hipMemset(out, 0, 4096));

    peak_dot2<NACC><<<blocks, threads>>>(out, reps);
    HIP_CHECK(hipDeviceSynchronize());
    const hipError_t le = hipGetLastError();
    if (le != hipSuccess) {
        std::printf("  NACC=%-3d SKIPPED (%s)\n", NACC, hipGetErrorString(le));
        hipFree(out);
        return r;
    }

    hipEvent_t e0, e1;
    hipEventCreate(&e0);
    hipEventCreate(&e1);
    hipEventRecord(e0);
    for (int i = 0; i < 20; i++) peak_dot2<NACC><<<blocks, threads>>>(out, reps);
    hipEventRecord(e1);
    HIP_CHECK(hipEventSynchronize(e1));
    float ms = 0;
    hipEventElapsedTime(&ms, e0, e1);
    hipEventDestroy(e0);
    hipEventDestroy(e1);
    hipFree(out);

    const double flops = static_cast<double>(blocks) * threads * kTotalDot2 * 4.0;
    r.ms = ms / 20.0;
    r.tflops = flops / (r.ms * 1e-3) / 1e12;
    r.waves_per_cu = static_cast<double>(blocks) * (threads / 32.0) / 2.0 / cus;
    r.ok = true;
    (void)clk;
    return r;
}

template <int NACC>
void peak_row(int cus, int blocks, int threads, double clk, const char *tag) {
    const PeakRow r = measure_dot2<NACC>(cus, blocks, threads, clk);
    if (!r.ok) return;
    const double macs_per_clk_cu = r.tflops * 0.5e12 / (clk * 1e9) / cus;
    std::printf("  %-8s %6d x %3d  %9.3f ms  %7.2f TFLOP/s  %6.1f MAC/clk/CU\n",
                tag, blocks, threads, r.ms, r.tflops, macs_per_clk_cu);
}

void run_peak_probe(int cus, double clk) {
    std::printf("=== 2. v_dot2_f32_f16 issue ceiling (register-only, packed) ===\n");
    std::printf("  %d dot2 per thread, constant across NACC\n", kTotalDot2);
    std::printf("  %-8s %16s %10s %12s %12s\n", "NACC", "grid x threads", "ms",
                "TFLOP/s", "MAC/clk/CU");
    peak_row<1>(cus, cus * 64, 256, clk, "1");
    peak_row<2>(cus, cus * 64, 256, clk, "2");
    peak_row<4>(cus, cus * 64, 256, clk, "4");
    peak_row<8>(cus, cus * 64, 256, clk, "8");
    peak_row<16>(cus, cus * 64, 256, clk, "16");
    peak_row<32>(cus, cus * 64, 256, clk, "32");

    std::printf("\n  flatness check at NACC=16 (same kernel, smaller grid):\n");
    peak_row<16>(cus, cus * 4, 256, clk, "16/16x");
    peak_row<16>(cus, cus * 64, 256, clk, "16/1x");
    peak_row<16>(cus, cus * 256, 256, clk, "16/4x");
    std::printf("  (the pipe ceiling is the FLAT part of this table; a row that\n"
                "   scales with grid size is measuring launch ramp, not the pipe.)\n\n");
}

int main(int argc, char **argv) {
    hipDeviceProp_t p{};
    HIP_CHECK(hipGetDeviceProperties(&p, 0));
    // HIP under-reports this part as 20 CU; the real count is 40. Never size a
    // grid off multiProcessorCount here.
    const int cus = argc > 1 ? std::atoi(argv[1]) : 40;
    const double clk = p.clockRate / 1e6;
    std::printf("device: %s  %s  CU(HIP)=%d CU(used)=%d  clock %.2f GHz\n\n", p.name,
                p.gcnArchName, p.multiProcessorCount, cus, clk);
    run_lane_probes();
    run_peak_probe(cus, clk);
    return 0;
}