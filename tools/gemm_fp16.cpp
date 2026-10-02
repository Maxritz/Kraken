// ============================================================================
//  gemm_fp16.cpp — fp16 GEMM for gfx1031 (RX 6700 XT, Wave32, no matrix cores)
//
//  Build:
//    hipcc -O3 --offload-arch=gfx1031 gemm_fp16.cpp -o gemm_fp16 -lpthread
//
//  Instruction census (the kernel is only worth anything if v_dot2 survived):
//    hipcc -O3 --offload-arch=gfx1031 -S gemm_fp16.cpp -o /tmp/g16.s
//    grep -c 'v_dot2_f32_f16' /tmp/g16.s
//    # expected: one per k-pair per (i,j) pair in the unrolled inner loop, and
//    # ZERO v_fmac_f32 in the gemm kernel.
//
//  What this file is careful about, and why:
//
//   1. NO WMMA, NO MATRIX CORES. gfx1031 has neither. The only fp16 dot the
//      hardware has is v_dot2_f32_f16: two fp16 MACs into an fp32 accumulator,
//      lane-local, exactly like the int8 DP4A. Everything below is built on
//      that one instruction.
//
//   2. THE INTRINSIC. There is no __builtin_amdgcn_dot2_f32_f16, and the
//      CLAMPING form v_dot2c_f32_f16 cannot be written as inline asm at all:
//      its read-modify-write destination rejects the AMDGPU "+r"/"+f"
//      constraints. The NON-clamping v_dot2_f32_f16 takes four operands and an
//      "=v" output, and assembles. That exact form is what src/hip/krk_hip.hpp
//      already ships and what the production gemm_simt_kernel runs on, so it
//      is known-good on this toolchain rather than newly discovered here.
//
//   3. NO FULL CPU REFERENCE. A 4096^3 reference is 69 GMAC on one core and
//      takes minutes. Verification samples output elements on a fixed stride
//      and computes a full O(K) dot for each, so a wrong inner loop is still
//      caught, deterministically, in under a second.
//
//   4. NO WRONG NUMBERS. Every TFLOP/s printed here was measured by this
//      binary on the device it ran on. Where a comparison point is external
//      (rocBLAS, Triton, the pipe ceiling) it is labelled as external.
//
//  Accumulation is fp32 throughout. gfx1031 has no tf32 and no fp32 accumulate
//  at reduced precision, so fp16 in, fp32 accumulate is the only correct
//  option for K=4096.
// ============================================================================
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <cmath>
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

// ---------------------------------------------------------------------------
//  The one arithmetic instruction this file is built on.
//
//  v_dot2_f32_f16 vD, vA, vB, vC  ->  vD = vC + a.lo*b.lo + a.hi*b.hi, with the
//  two fp16 products summed in fp32. Non-clamping. Four operands, "=v" output.
//  (The clamping v_dot2c form takes only three and does not accept these
//  constraints on gfx1031 — see the note at the top.)
// ---------------------------------------------------------------------------
__device__ __forceinline__ float dot2_f16(unsigned ha, unsigned hb, float c) {
    float d;
    __asm__ volatile("v_dot2_f32_f16 %0, %1, %2, %3"
                     : "=v"(d)
                     : "v"(ha), "v"(hb), "v"(c));
    return d;
}

__device__ __forceinline__ float to_f32(__half h) { return static_cast<float>(h); }

// ============================================================================
//  C[M,N] = A[M,K] * B[K,N],  A and B fp16 row-major, C fp32, fp32 accumulate
//
//  LDS BUDGET, 64 KiB per workgroup on gfx1031 (half a CDNA part, and the
//  binding constraint for fp16 tiles):
//
//      As  = STAGES * BM * BK * 2 B = 2 * 128 * 32 * 2 = 16384 B  (16 KiB)
//      Bs  = STAGES * BN * BK * 2 B = 2 * 128 * 32 * 2 = 16384 B  (16 KiB)
//      total                                = 32768 B  (32 KiB of 64 KiB)
//
//  So a double-buffered 128x128x32 fp16 tile fits in HALF the LDS with room to
//  spare. The next step up, 256x128, would need 48 KiB and still fit; 256x256
//  would need 64 KiB and fail to launch with "out of resource: shared memory".
//  The limit here is registers, not LDS: TM=TN=8 costs 64 fp32 accumulators,
//  and that is what caps the block at 128x128 before spills appear.
//
//  DP4A-equivalent layout reasoning. v_dot2_f32_f16 is lane-local and consumes
//  two CONSECUTIVE k of one row per operand register, so the LDS tiles are
//  [row][k] with k contiguous, and the inner loop walks k in PAIRS. Thread
//  (ty,tx) owns output rows [ty*TM,+TM) and columns [tx*TN,+TN); those are
//  disjoint across threads, so accumulators stay in private registers and no
//  wave reduction is needed anywhere.
//
//  The transpose trap. B is [K][N] but the tile is [N][K]. A 2x2 half transpose
//  per (k-pair, column-pair) is mandatory; loading one 32-bit word per
//  (k-pair, column-pair) fills only the k=2q slots and leaves k=2q+1 garbage,
//  which silently corrupts half of every k-pair.
//
//  SPLIT-K writes int32/fp32 partials reduced by a fixed-order kernel, never
//  float atomicAdd: atomic ordering is non-deterministic and reproducibility
//  matters more than the last few percent.
// ============================================================================
template <int BM, int BN, int BK, int TM, int TN, int STAGES, int SPLIT>
__global__ void __launch_bounds__((BM / TM) * (BN / TN))
    gemm_f16_kernel(const __half *__restrict__ A, const __half *__restrict__ B,
                    float *__restrict__ C, float *__restrict__ partial, int M,
                    int N, int K, int lda, int ldb, int ldc) {
    constexpr int TX = BN / TN;
    constexpr int TY = BM / TM;
    constexpr int THREADS = TX * TY;
    constexpr int KP = BK / 2;    // k-PAIRS per tile row
    constexpr int CP = BN / 2;    // column-PAIRS per tile row

    __shared__ __align__(16) __half As[STAGES][BM * BK];
    __shared__ __align__(16) __half Bs[STAGES][BN * BK];

    const int tid = threadIdx.x;
    const int tx = tid % TX;
    const int ty = tid / TX;
    const int row_base = blockIdx.y * BM;
    const int col_base = blockIdx.x * BN;

    const int kt_total = K / BK;
    const int kt_per = (kt_total + SPLIT - 1) / SPLIT;
    const int kt_begin = blockIdx.z * kt_per;
    int kt_end = kt_begin + kt_per;
    if (kt_end > kt_total) kt_end = kt_total;
    const int k_tiles = kt_end > kt_begin ? kt_end - kt_begin : 0;

    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; i++)
#pragma unroll
        for (int j = 0; j < TN; j++) acc[i][j] = 0.0f;

    auto load_stage = [&](int s, int kt) {
        const int k0 = kt * BK;
        // A is [M][K] and the tile is [M][K]: straight copy, k contiguous.
        __half *Ad = As[s];
        for (int idx = tid; idx < BM * KP; idx += THREADS) {
            const int r = idx / KP;
            const int q = idx - r * KP;
            const int gr = row_base + r;
            *reinterpret_cast<int *>(Ad + r * BK + q * 2) =
                (gr < M) ? *reinterpret_cast<const int *>(A + static_cast<size_t>(gr) * lda +
                                                          k0 + q * 2)
                         : 0;
        }
        // B is [K][N] and the tile is [N][K]: a 2x2 half transpose. Two reads
        // of two columns each, scattered into four tile slots.
        __half *Bd = Bs[s];
        for (int idx = tid; idx < KP * CP; idx += THREADS) {
            const int q = idx / CP;
            const int c2 = idx - q * CP;
            const int gc = col_base + c2 * 2;
            const int kk = k0 + q * 2;
            __half2 r0, r1;
            r0.x = (__half)0.0f;
            r0.y = (__half)0.0f;
            r1.x = r0.x;
            r1.y = r0.y;
            if (gc + 1 < N) {
                r0 = *reinterpret_cast<const __half2 *>(B + static_cast<size_t>(kk) * ldb + gc);
                r1 = *reinterpret_cast<const __half2 *>(B + static_cast<size_t>(kk + 1) * ldb + gc);
            } else if (gc < N) {
                r0.x = B[static_cast<size_t>(kk) * ldb + gc];
                r1.x = B[static_cast<size_t>(kk + 1) * ldb + gc];
            }
            Bd[(c2 * 2 + 0) * BK + q * 2 + 0] = r0.x;
            Bd[(c2 * 2 + 0) * BK + q * 2 + 1] = r1.x;
            Bd[(c2 * 2 + 1) * BK + q * 2 + 0] = r0.y;
            Bd[(c2 * 2 + 1) * BK + q * 2 + 1] = r1.y;
        }
    };

    if (k_tiles > 0) load_stage(0, kt_begin);
    __syncthreads();

    int stage = 0;
    for (int kt = 0; kt < k_tiles; kt++) {
        const int nxt = (stage + 1) % STAGES;
        if (kt + 1 < k_tiles) load_stage(nxt, kt_begin + kt + 1);

        const __half *Ac = As[stage] + ty * TM * BK;
        const __half *Bc = Bs[stage] + tx * TN * BK;

#pragma unroll
        for (int q = 0; q < KP; q++) {
            unsigned a[TM], b[TN];
#pragma unroll
            for (int i = 0; i < TM; i++)
                a[i] = *reinterpret_cast<const unsigned *>(Ac + i * BK + q * 2);
#pragma unroll
            for (int j = 0; j < TN; j++)
                b[j] = *reinterpret_cast<const unsigned *>(Bc + j * BK + q * 2);
#pragma unroll
            for (int i = 0; i < TM; i++)
#pragma unroll
                for (int j = 0; j < TN; j++)
                    acc[i][j] = dot2_f16(a[i], b[j], acc[i][j]);
        }

        __syncthreads();
        stage = nxt;
    }

    if (SPLIT == 1) {
#pragma unroll
        for (int i = 0; i < TM; i++) {
            const int gr = row_base + ty * TM + i;
            if (gr >= M) continue;
#pragma unroll
            for (int j = 0; j < TN; j++) {
                const int gc = col_base + tx * TN + j;
                if (gc >= N) continue;
                C[static_cast<size_t>(gr) * ldc + gc] = acc[i][j];
            }
        }
    } else {
#pragma unroll
        for (int i = 0; i < TM; i++) {
            const int gr = row_base + ty * TM + i;
            if (gr >= M) continue;
#pragma unroll
            for (int j = 0; j < TN; j++) {
                const int gc = col_base + tx * TN + j;
                if (gc >= N) continue;
                partial[(static_cast<size_t>(blockIdx.z) * M + gr) * N + gc] = acc[i][j];
            }
        }
    }
}

// fp32 -> fp16 on the device. __float2half is legal HERE (device code) and is
// the only conversion on this toolchain that is trustworthy; both host routes
// produce non-finite values.
__global__ void to_half_kernel(__half *__restrict__ dst,
                               const float *__restrict__ src, size_t n) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += static_cast<size_t>(blockDim.x) * gridDim.x)
        dst[i] = __float2half(src[i]);
}

// Fixed-order sum of the per-split fp32 partials. Deterministic by construction.
__global__ void reduce_kernel(const float *__restrict__ part, float *__restrict__ C,
                              int M, int N, int split, int ldc) {
    const int total = M * N;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += blockDim.x * gridDim.x) {
        float s = 0.0f;
        for (int k = 0; k < split; k++)
            s += part[static_cast<size_t>(k) * total + i];
        C[static_cast<size_t>(i / N) * ldc + (i % N)] = s;
    }
}

// ---------------------------------------------------------------------------
//  SAMPLED CPU REFERENCE
// ---------------------------------------------------------------------------
//  Exhaustive below kFullBudgetMacs, fixed-stride sample above it. Each
//  sampled element is a full O(K) dot in double precision, so the printed max
//  error is meaningful and not a sampling artefact.
constexpr double kFullBudgetMacs = 5e8;
constexpr size_t kSampleCount = 4096;

struct Check {
    double maxabs;
    double maxrel;
    size_t bad;
    size_t nonfinite;
    size_t checked;
    bool exhaustive;
};

Check verify(const std::vector<float> &got, const std::vector<float> &AF,
             const std::vector<float> &BF, int M, int N, int K, int lda, int ldb) {
    const size_t total = static_cast<size_t>(M) * N;
    Check r{0.0, 0.0, 0, 0, 0, true};

    auto one = [&](size_t i) {
        const int m = static_cast<int>(i / static_cast<size_t>(N));
        const int n = static_cast<int>(i % static_cast<size_t>(N));
        double s = 0.0;
        for (int k = 0; k < K; k++)
            s += static_cast<double>(AF[static_cast<size_t>(m) * lda + k]) *
                 static_cast<double>(BF[static_cast<size_t>(k) * ldb + n]);
        const double g = got[i];
        // A non-finite result is a FAILURE, always, and must be counted before
        // any tolerance comparison. Without this guard a NaN slips through:
        // fabs(NaN - x) is NaN, and NaN > tol is false, so a broken kernel
        // reports "ok". A TM=TN=4 variant of this kernel did exactly that.
        const bool g_finite = std::isfinite(g);
        if (!g_finite) {
            r.nonfinite++;
            if (r.nonfinite <= 4)
                std::printf("      NONFINITE at [%d,%d] got %g ref %g\n",
                            static_cast<int>(i / static_cast<size_t>(N)),
                            static_cast<int>(i % static_cast<size_t>(N)),
                            static_cast<int>(i / static_cast<size_t>(N)) % 128,
                            static_cast<int>(i % static_cast<size_t>(N)) % 128, g, s);
        }
        const double e = std::fabs(g - s);
        const double rel = e / (std::fabs(s) > 1e-6 ? std::fabs(s) : 1.0);
        if (e > r.maxabs) r.maxabs = e;
        if (rel > r.maxrel) r.maxrel = rel;
        // Tolerance: fp32 accumulation of K products. Relative 1e-3 is loose
        // enough for summation-order differences and tight enough that a real
        // indexing bug (which is O(1) relative, not O(1e-6)) cannot hide.
        // NaN fails too, via the finite check above.
        if (!g_finite || !(rel <= 1e-3)) r.bad++;
        r.checked++;
    };

    if (static_cast<double>(total) * K <= kFullBudgetMacs) {
        for (size_t i = 0; i < total; i++) one(i);
    } else {
        r.exhaustive = false;
        const size_t stride = total / kSampleCount ? total / kSampleCount : 1;
        for (size_t i = 0; i < total; i += stride) one(i);
    }
    return r;
}

int main(int argc, char **argv) {
    hipDeviceProp_t p{};
    HIP_CHECK(hipGetDeviceProperties(&p, 0));
    const int hip_cus = p.multiProcessorCount;
    const int cus = argc > 1 ? std::atoi(argv[1]) : 40;  // HIP under-reports 2x
    const double clk = p.clockRate / 1e6;
    std::printf("device: %s  %s  clock %.2f GHz\n", p.name, p.gcnArchName, clk);
    std::printf("CU count: HIP says %d, using %d\n\n", hip_cus, cus);

    std::printf("=== LDS budget (64 KiB per workgroup on gfx1031) ===\n");
    {
        constexpr int BM = 128, BN = 128, BK = 32, ST = 2;
        const int as = ST * BM * BK * (int)sizeof(__half);
        const int bs = ST * BN * BK * (int)sizeof(__half);
        std::printf("  As = %d x %d x %d x 2 B = %6d B (%.0f KiB)\n", ST, BM, BK, as,
                    as / 1024.0);
        std::printf("  Bs = %d x %d x %d x 2 B = %6d B (%.0f KiB)\n", ST, BN, BK, bs,
                    bs / 1024.0);
        std::printf("  total = %d B (%.0f KiB) of 65536 B -> %.0f%% of LDS\n", as + bs,
                    (as + bs) / 1024.0, 100.0 * (as + bs) / 65536.0);
        std::printf("  next step 256x128x32 double buffered = %d B -> still fits\n",
                    2 * 256 * 32 * 2 + 2 * 128 * 32 * 2);
        std::printf("  256x256x32 double buffered = %d B -> EXCEEDS LDS, will not launch\n\n",
                    2 * 256 * 32 * 2 * 2);
    }

    struct Case { int M, N, K; const char *name; };
    const Case cases[] = {
        {256, 256, 256, "small 256^3 (exhaustive)"},
        {1024, 1024, 1024, "mid   1024^3 (sampled)"},
        {4096, 4096, 4096, "big   4096^3 (sampled)"},
    };

    std::printf("=== fp16 GEMM, 128x128x32, TM=TN=8, 2-stage pipeline ===\n");
    std::printf("%-26s %6s %9s %10s %11s %9s\n", "case", "split", "ms", "TFLOP/s",
                "max abs err", "checked");

    bool all_ok = true;
    for (const Case &c : cases) {
        std::vector<__half> A(static_cast<size_t>(c.M) * c.K);
        std::vector<__half> B(static_cast<size_t>(c.K) * c.N);
        // Deterministic small-magnitude values, all exact multiples of 0.25 in
        // [-2, 2] and therefore EXACTLY representable in fp16, so the reference
        // can read the host floats directly and the only error in play is
        // accumulation order.
        //
        // The fp16 encoding is done IN A KERNEL, on the device, because every
        // host-side route to it is broken on this ROCm: __float2half is
        // device-only and (__half)x compiles but returns non-finite garbage for
        // some values. Either way the INPUTS came out inf/NaN, the CPU
        // reference inherited them, and the comparison agreed on garbage —
        // a harness that reports "ok" while verifying nothing.
        std::vector<float> AF(A.size()), BF(B.size());
        // The cast to int is load-bearing. With size_t arithmetic,
        // "(x % 17) - 8" UNDERSHOWS to ~1.8e19 whenever the remainder is below
        // 8, so "small" inputs become 4.6e18, every product overflows fp32,
        // and both the kernel and the reference return inf/NaN — which a naive
        // comparison reports as agreement. Every run of this harness before
        // that cast was checking garbage against garbage.
        for (size_t i = 0; i < AF.size(); i++)
            AF[i] = static_cast<float>(static_cast<int>((i * 37 + 11) % 17) - 8) * 0.25f;
        for (size_t i = 0; i < BF.size(); i++)
            BF[i] = static_cast<float>(static_cast<int>((i * 53 + 7) % 17) - 8) * 0.25f;

        const size_t out_elems = static_cast<size_t>(c.M) * c.N;
        __half *dA, *dB;
        float *dAF, *dBF, *dC, *dP;
        HIP_CHECK(hipMalloc(&dA, A.size() * sizeof(__half)));
        HIP_CHECK(hipMalloc(&dB, B.size() * sizeof(__half)));
        HIP_CHECK(hipMalloc(&dAF, AF.size() * sizeof(float)));
        HIP_CHECK(hipMalloc(&dBF, BF.size() * sizeof(float)));
        HIP_CHECK(hipMalloc(&dC, out_elems * sizeof(float)));
        HIP_CHECK(hipMalloc(&dP, 8 * out_elems * sizeof(float)));
        HIP_CHECK(hipMemcpy(dAF, AF.data(), AF.size() * sizeof(float),
                            hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(dBF, BF.data(), BF.size() * sizeof(float),
                            hipMemcpyHostToDevice));
        to_half_kernel<<<(A.size() + 255) / 256, 256>>>(dA, dAF, A.size());
        to_half_kernel<<<(B.size() + 255) / 256, 256>>>(dB, dBF, B.size());
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipGetLastError());

        for (int split : {1, 2, 4}) {
            const dim3 grid(static_cast<unsigned>((c.N + 127) / 128),
                            static_cast<unsigned>((c.M + 127) / 128),
                            static_cast<unsigned>(split));
            auto once = [&] {
                if (split == 1)
                    gemm_f16_kernel<128, 128, 32, 8, 8, 2, 1>
                        <<<grid, 256>>>(dA, dB, dC, dP, c.M, c.N, c.K, c.K, c.N, c.N);
                else if (split == 2)
                    gemm_f16_kernel<128, 128, 32, 8, 8, 2, 2>
                        <<<grid, 256>>>(dA, dB, dC, dP, c.M, c.N, c.K, c.K, c.N, c.N);
                else
                    gemm_f16_kernel<128, 128, 32, 8, 8, 2, 4>
                        <<<grid, 256>>>(dA, dB, dC, dP, c.M, c.N, c.K, c.K, c.N, c.N);
                if (split > 1) {
                    const int total = c.M * c.N;
                    reduce_kernel<<<(total + 255) / 256, 256>>>(dP, dC, c.M, c.N, split,
                                                               c.N);
                }
            };

            once();
            HIP_CHECK(hipDeviceSynchronize());
            HIP_CHECK(hipGetLastError());

            hipEvent_t e0, e1;
            hipEventCreate(&e0);
            hipEventCreate(&e1);
            hipEventRecord(e0);
            for (int i = 0; i < 10; i++) once();
            hipEventRecord(e1);
            HIP_CHECK(hipEventSynchronize(e1));
            float ms = 0;
            hipEventElapsedTime(&ms, e0, e1);
            hipEventDestroy(e0);
            hipEventDestroy(e1);
            const double per = ms / 10.0;

            std::vector<float> got(out_elems);
            HIP_CHECK(hipMemcpy(got.data(), dC, out_elems * sizeof(float),
                                hipMemcpyDeviceToHost));
            const Check ck = verify(got, AF, BF, c.M, c.N, c.K, c.K, c.N);
            if (ck.bad) all_ok = false;

            const double tflops = 2.0 * c.M * c.N * c.K / (per * 1e-3) / 1e12;
            std::printf("%-26s %6d %9.3f %10.2f %11.3e %9s%s%s\n", c.name, split, per,
                        tflops, ck.maxabs, ck.exhaustive ? "exhaustive" : "sampled",
                        ck.bad ? "  MISMATCH" : " ok",
                        ck.nonfinite ? "  NON-FINITE" : "");
        }
        std::printf("\n");
        HIP_CHECK(hipFree(dA));
        HIP_CHECK(hipFree(dB));
        HIP_CHECK(hipFree(dAF));
        HIP_CHECK(hipFree(dBF));
        HIP_CHECK(hipFree(dC));
        HIP_CHECK(hipFree(dP));
    }

    std::printf("=== external comparison points, measured elsewhere on this card ===\n");
    std::printf("  rocBLAS fp16 4096^3      20.19 TFLOP/s\n");
    std::printf("  Triton tl.dot 4096^3     15.04 TFLOP/s\n");
    std::printf("  v_dot2 pipe ceiling       25.33 TFLOP/s (register-only, out of band)\n");
    std::printf("%s\n", all_ok ? "ALL CASES MATCH THE SAMPLED CPU REFERENCE"
                               : "*** A CASE IS WRONG — the TFLOP/s above is meaningless ***");
    return all_ok ? 0 : 1;
}