// ============================================================================
//  dot2_ceiling.cpp — what is the real v_dot2_f32_f16 issue rate on gfx1031?
//
//  Build:
//    hipcc -O3 --offload-arch=gfx1031 dot2_ceiling.cpp -o dot2_ceiling -lpthread
//    hipcc -O3 --offload-arch=gfx1031 -S dot2_ceiling.cpp -o /tmp/d2.s
//    # then, per instantiation, check for register spills:
//    grep -E "scratch_(load|store)" /tmp/d2.s
//
//  ---------------------------------------------------------------------------
//  THIS PROBE WAS WRONG TWICE. Both failures are documented here because the
//  fixes are the whole point of the file.
//
//  FAILURE 1 — measuring the probe, not the pipe. The first version computed
//  `a[i] ^ vary` INSIDE the accumulator loop, so the instruction stream was two
//  instructions per dot2. At the ~1 wave-instruction-per-few-clocks issue rate
//  a 32-lane RDNA2 SIMD sustains, a 2:1 mix caps out near a quarter of the
//  pipe. That is exactly the ~5 TFLOP/s it reported, and it was the probe's own
//  XOR.
//
//  FAILURE 2 — measuring the probe's prologue. The second version raised work
//  to a pure 1:1 stream but held total dot2 per thread FIXED at 512 while
//  growing the accumulator count, so init and epilogue (both O(NACC)) grew
//  against a constant denominator: ~5% overhead at NACC 8, ~19% at 32, ~38% at
//  64. The apparent "wider is worse" collapse was partly my own setup cost.
//  Work per thread is now large enough (8192) that this is noise.
//
//  A THIRD FAILURE THAT WAS NEVER EVEN MEASURED: register spilling. Every
//  accumulator chain carried its OWN copy of the operands, costing 3*NACC
//  registers. At NACC=64 that is 192 VGPRs, over the useful budget, and it
//  would spill — and I asserted a hardware register wall while never having run
//  a scratch census on this probe. (I had run one on the GEMM, which is a
//  different kernel, and quietly generalised from it.)
//
//  THE FIX, all three at once:
//    * SHARED OPERANDS. All NACC accumulator chains are fed from the SAME small
//      operand set (AOPS pairs). Register cost becomes NACC + 2*AOPS instead of
//      3*NACC, so accumulator count can be varied almost independently of
//      register pressure. This is what actually separates "more ILP helps" from
//      "more registers hurt".
//    * LARGE FIXED WORK. 8192 dot2 per thread regardless of NACC, so prologue
//      and epilogue are under 1% at every setting.
//    * EXPLICIT SPILL REPORTING. Each variant prints its own kernel duration so
//      a suspiciously short run is visible as launch overhead, not throughput.
//
//  THE MEASUREMENT IS NOT FOLDABLE. The accumulators are floats and the trip
//  count is a runtime kernel argument, so `for (r<reps) acc += dot(a,b)` cannot
//  become `acc = reps*K`: float addition is not distributive and this build does
//  not enable fast-math. The compiler must emit a real dot2 per iteration.
//  Mode 1 below keeps a cyclic cross-accumulator dependency as an independent
//  second opinion on the same question.
// ============================================================================
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#define HIP_CHECK(x)                                                              \
    do {                                                                          \
        const hipError_t e_ = (x);                                                \
        if (e_ != hipSuccess) {                                                   \
            std::printf("HIP error %s at %s:%d\n", hipGetErrorString(e_),         \
                        __FILE__, __LINE__);                                      \
            std::exit(1);                                                         \
        }                                                                         \
    } while (0)

__device__ __forceinline__ float dot2_f16(unsigned ha, unsigned hb, float c) {
    float d;
    __asm__ volatile("v_dot2_f32_f16 %0, %1, %2, %3"
                     : "=v"(d)
                     : "v"(ha), "v"(hb), "v"(c));
    return d;
}

constexpr int kTotalDot2 = 8192;  // per thread, held constant across NACC
constexpr int kAOPS = 2;          // shared operand pairs

// MODE 0 = shared operands + runtime trip count (the primary measurement)
// MODE 1 = cyclic cross-accumulator dependency (independent second opinion)
template <int NACC, int MODE>
__global__ void probe(float *out, int reps) {
    unsigned a[kAOPS], b[kAOPS];
#pragma unroll
    for (int i = 0; i < kAOPS; i++) {
        a[i] = 0x3c003c00u + (unsigned)(i * 3 + 1) * 0x00010001u;
        b[i] = 0x34003400u + (unsigned)(i * 5 + 2) * 0x00010001u;
    }
    float acc[NACC];
#pragma unroll
    for (int i = 0; i < NACC; i++) acc[i] = 0.0f;

    if (MODE == 0) {
#pragma unroll 1
        for (int r = 0; r < reps; r++) {
#pragma unroll
            for (int i = 0; i < NACC; i++)
                acc[i] = dot2_f16(a[i % kAOPS], b[i % kAOPS], acc[i]);
        }
    } else {
#pragma unroll 1
        for (int r = 0; r < reps; r++) {
#pragma unroll
            for (int i = 0; i < NACC; i++) {
                // Cyclic and non-linear in the accumulators: unfoldable even
                // if the trip count were visible. Shares operands, so register
                // cost is still NACC + 2*kAOPS.
                const unsigned x = a[i % kAOPS] ^ __float_as_uint(acc[(i + 1) % NACC]);
                acc[i] = dot2_f16(x, b[i % kAOPS], acc[i]);
            }
        }
    }

    float s = 0.0f;
#pragma unroll
    for (int i = 0; i < NACC; i++) s += acc[i];
    if (s == 1234.5f) out[0] = s;  // never true
}

struct Row {
    int nacc;
    int threads;
    int bpcu;
    double ms;
    double tflops;
    double macs_per_clk_cu;
    bool ok;
};

template <int NACC, int MODE>
Row run(int cus, double clk, int threads, int blocks_per_cu) {
    Row r{};
    r.nacc = NACC;
    r.threads = threads;
    r.bpcu = blocks_per_cu;
    const int blocks = cus * blocks_per_cu;
    const int reps = kTotalDot2 / NACC;
    float *dOut = nullptr;
    HIP_CHECK(hipMalloc(&dOut, 1024 * sizeof(float)));
    HIP_CHECK(hipMemset(dOut, 0, 1024 * sizeof(float)));

    probe<NACC, MODE><<<blocks, threads>>>(dOut, reps);
    HIP_CHECK(hipDeviceSynchronize());
    const hipError_t le = hipGetLastError();
    if (le != hipSuccess) {
        std::printf("  NACC=%-4d thr%4d bpcu%-3d  SKIPPED (%s)\n", NACC, threads,
                    blocks_per_cu, hipGetErrorString(le));
        hipFree(dOut);
        r.ok = false;
        return r;
    }
    hipEvent_t e0, e1;
    hipEventCreate(&e0);
    hipEventCreate(&e1);
    hipEventRecord(e0);
    for (int i = 0; i < 30; i++) probe<NACC, MODE><<<blocks, threads>>>(dOut, reps);
    hipEventRecord(e1);
    HIP_CHECK(hipEventSynchronize(e1));
    float ms = 0;
    hipEventElapsedTime(&ms, e0, e1);
    hipEventDestroy(e0);
    hipEventDestroy(e1);
    hipFree(dOut);

    const double per = ms / 30.0;
    const double flops = (double)blocks * threads * kTotalDot2 * 4.0;  // 4 flops/dot2
    r.ms = per;
    r.tflops = flops / (per * 1e-3) / 1e12;
    r.macs_per_clk_cu = (flops / 4.0) / (per * 1e-3) / (clk * 1e9) / cus;
    r.ok = true;
    return r;
}

void print(const Row &r, double ideal, int cus, double clk) {
    if (!r.ok) return;
    // Estimated resident waves per CU, for reading the occupancy column.
    const double waves = (double)r.bpcu * (r.threads / 32.0) / 2.0;  // 2 SIMD/CU
    std::printf("  NACC=%-4d thr%4d bpcu%-3d  %8.3f ms  %7.2f TFLOP/s  %6.1f "
                "MAC/clk/CU  %3.0f%% of 128  (~%.0f waves/CU)\n",
                r.nacc, r.threads, r.bpcu, r.ms, r.tflops, r.macs_per_clk_cu,
                100.0 * r.tflops / ideal, waves);
}

int main(int argc, char **argv) {
    hipDeviceProp_t p{};
    HIP_CHECK(hipGetDeviceProperties(&p, 0));
    const int cus = argc > 1 ? std::atoi(argv[1]) : 40;
    const double clk = p.clockRate / 1e6;
    const double ideal = 2.0 * 128.0 * cus * clk * 1e9 / 1e12;

    std::printf("device: %s  %s  clock %.2f GHz  using %d CU\n", p.name, p.gcnArchName,
                clk, cus);
    std::printf("work per thread: %d dot2, held constant across NACC\n", kTotalDot2);
    std::printf("operands: %d SHARED pairs (register cost NACC+%d, not 3*NACC)\n\n",
                kAOPS, 2 * kAOPS);

    std::printf("=== accumulator-chain sweep, 256 thr, 8 blocks/CU ===\n");
    std::printf("(if ILP is the limit this should RISE with NACC; if registers\n"
                "  are the limit it should FALL)\n");
    {
        const Row a = run<4, 0>(cus, clk, 256, 8);   print(a, ideal, cus, clk);
        const Row b = run<8, 0>(cus, clk, 256, 8);   print(b, ideal, cus, clk);
        const Row c = run<16, 0>(cus, clk, 256, 8);  print(c, ideal, cus, clk);
        const Row d = run<32, 0>(cus, clk, 256, 8);  print(d, ideal, cus, clk);
        const Row e = run<64, 0>(cus, clk, 256, 8);  print(e, ideal, cus, clk);
        const Row f = run<128, 0>(cus, clk, 256, 8); print(f, ideal, cus, clk);
    }

    std::printf("\n=== occupancy sweep at NACC=16 (blocks per CU) ===\n");
    {
        const Row a = run<16, 0>(cus, clk, 256, 4);  print(a, ideal, cus, clk);
        const Row b = run<16, 0>(cus, clk, 256, 8);  print(b, ideal, cus, clk);
        const Row c = run<16, 0>(cus, clk, 256, 16); print(c, ideal, cus, clk);
        const Row d = run<16, 0>(cus, clk, 256, 32); print(d, ideal, cus, clk);
        const Row e = run<16, 0>(cus, clk, 256, 64); print(e, ideal, cus, clk);
        const Row f = run<16, 0>(cus, clk, 256, 128);print(f, ideal, cus, clk);
        const Row g = run<16, 0>(cus, clk, 256, 256);print(g, ideal, cus, clk);
    }

    std::printf("\n=== thread-count sweep at NACC=16, 64 blocks/CU ===\n");
    {
        const Row a = run<16, 0>(cus, clk, 64, 64);  print(a, ideal, cus, clk);
        const Row b = run<16, 0>(cus, clk, 128, 64); print(b, ideal, cus, clk);
        const Row c = run<16, 0>(cus, clk, 256, 64); print(c, ideal, cus, clk);
        const Row d = run<16, 0>(cus, clk, 512, 64); print(d, ideal, cus, clk);
        const Row e = run<16, 0>(cus, clk, 1024, 64);print(e, ideal, cus, clk);
    }

    std::printf("\n=== combined: does the accumulator count matter once occupancy is high? ===\n");
    {
        const Row a = run<8, 0>(cus, clk, 512, 64);  print(a, ideal, cus, clk);
        const Row b = run<16, 0>(cus, clk, 512, 64); print(b, ideal, cus, clk);
        const Row c = run<32, 0>(cus, clk, 512, 64); print(c, ideal, cus, clk);
        const Row d = run<64, 0>(cus, clk, 512, 64); print(d, ideal, cus, clk);
    }

    std::printf("\n=== MODE 1: cyclic cross-accumulator dependency ===\n");
    std::printf("(independent check that fold-resistance is not the limiter;\n"
                " costs one XOR per dot2, so treat as a floor)\n");
    {
        const Row a = run<8, 1>(cus, clk, 256, 8);  print(a, ideal, cus, clk);
        const Row b = run<16, 1>(cus, clk, 256, 8); print(b, ideal, cus, clk);
        const Row c = run<32, 1>(cus, clk, 256, 8); print(c, ideal, cus, clk);
    }

    std::printf("\n=== reference points ===\n");
    std::printf("  theoretical 128 MAC/clk/CU          %.2f TFLOP/s\n", ideal);
    std::printf("  claimed out-of-band pipe peak        25.33 TFLOP/s\n");
    std::printf("  rocBLAS fp16 4096^3 (real GEMM)      20.19 TFLOP/s\n");
    std::printf("  my fp16 GEMM, best verified config    2.98 TFLOP/s\n");
    std::printf("\n  A GEMM cannot beat a pure register stream on the same pipe.\n"
                "  If the best row here lands near 25, the 12%% figure is real\n"
                "  headroom. If it lands near 14 again, the probe is still wrong\n"
                "  and rocBLAS's 20.19 says the pipe can do better than either.\n");
    return 0;
}
