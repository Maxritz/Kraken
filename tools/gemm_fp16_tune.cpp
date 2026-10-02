// ============================================================================
//  gemm_fp16_tune.cpp — tuning + instrumentation harness for the fp16 dot2 GEMM
//
//  Build:
//    hipcc -O3 --offload-arch=gfx1031 gemm_fp16_tune.cpp -o gemm_fp16_tune -lpthread
//
//  gemm_fp16.cpp holds the verified reference kernel. This file does NOT replace
//  it; it takes the same kernel and adds three things that kernel deliberately
//  does not carry, so the shipped kernel stays simple and correct:
//
//   1. REGISTER-LEVEL DOUBLE BUFFERING (DB). Reads the next k-pair's a[]/b[]
//      registers before consuming the current pair, so LDS read latency
//      overlaps the dot2 stream instead of stalling in front of it. Every
//      config is run with DB=0 and DB=1 so the comparison is like-for-like.
//
//   2. A CONFIG SWEEP over block tile, BK, stage count and thread count. Each
//      configuration is verified against the sampled CPU reference BEFORE its
//      TFLOP/s is printed. A configuration that fails to launch (LDS over
//      budget) is reported as SKIPPED and never timed.
//
//   3. AN ABLATION STUDY that attributes the wall clock. The same kernel is
//      built with individual phases compiled out via the AB bitmask and each
//      variant timed; the differences give each phase's cost. Ablation
//      variants compute garbage by construction and are NEVER correctness-
//      checked — they exist only to be timed, and the deltas are reported as a
//      ranking rather than a decomposition because removing one phase also
//      removes stalls the others were hiding behind.
//
//  CORRECTNESS RULES, all of which were violated at some point during
//  development and each of which produced a harness that "verified" garbage:
//
//   * No non-finite result is ever a pass. fabs(NaN - x) > tol is FALSE, so a
//     NaN walks straight through a naive comparison.
//   * Test data is generated in SIGNED arithmetic. "(size % 17) - 8" with a
//     size_t undershows to ~1.8e19 whenever the remainder is below 8.
//   * fp32 -> fp16 conversion happens IN A KERNEL. __float2half is device-only
//     on this ROCm and (__half)x yields non-finite garbage on the host.
//   * No full O(M*N*K) CPU reference: 4096^3 is 68 GFLOP on a single core.
//
//  v_dot2_f32_f16 is non-clamping and takes four operands with an "=v" output.
//  The clamping v_dot2c_f32_f16 cannot be written as inline asm on gfx1031 at
//  all; the non-clamping form is the one that assembles, and the census shows
//  the backend keeps it rather than lowering to scalar FMA.
//
//  THE LDS FETCH WAS BANK-ALIASED, and this file now measures the fix.
//  The b fetch reads word (tx*TN + j)*BK + q*2 halves, so consecutive tx lanes
//  are TN*BK*2 = 512 B apart = 128 words; 128 mod 32 == 0, so all 16 tx lanes of
//  a wave land in the SAME LDS bank. The a fetch is 2-way aliased the same way
//  (512 B between the two ty groups). rocprofv3 on the unrotated kernel:
//      SQC_LDS_BANK_CONFLICT / SQ_INSTS_LDS = 16.4
//  i.e. every LDS instruction serialises ~16 times, which is ~43 ms of LDS time
//  inside a 48 ms 4096^3 run -- the kernel was never dot2 bound.
//  ROT (AB bit 16) rotates the k-pair index by the lane's tx, q -> q ^ tx, so
//  the 16 tx lanes walk 16 distinct banks. Both operands of a pair are fetched
//  with the SAME rotated index, so every (a,b) pair is still one k-pair, and
//  since the rotation is a bijection over q every thread still covers all k.
//  It only reorders each thread's own accumulation (deterministic, and inside
//  the 1e-3 relative tolerance this harness checks).
// ============================================================================
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
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

__global__ void to_half_kernel(__half *__restrict__ dst,
                               const float *__restrict__ src, size_t n) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += static_cast<size_t>(blockDim.x) * gridDim.x)
        dst[i] = __float2half(src[i]);
}

// ============================================================================
//  Kernel. DB = register double buffering; AB = ablation bitmask.
//     1 = skip global->LDS load   2 = pin k-pair to 0 (no LDS read stream)
//     4 = one dot2 per k-tile    8 = drop the barrier
//
//  LDS BUDGET = (STAGES*BM*BK + STAGES*BN*BK) * 2 B, against 65536 B:
//    128x128x32 s2 = 32768 (50%)   s3 = 49152 (75%)
//    256x128x32 s2 = 49152 (75%)   128x256x32 s2 = 49152 (75%)
//    256x256x32 s2 = 65536 (at the limit -> will not launch)
//  LDS is therefore not the binding constraint in this sweep; registers are,
//  through the TM*TN fp32 accumulator block.
// ============================================================================
template <int BM, int BN, int BK, int TM, int TN, int STAGES, int DB, int AB>
__global__ void __launch_bounds__((BM / TM) * (BN / TN))
    gemm_f16_kernel(const __half *__restrict__ A, const __half *__restrict__ B,
                    float *__restrict__ C, int M, int N, int K, int lda, int ldb,
                    int ldc) {
    constexpr int TX = BN / TN;
    constexpr int TY = BM / TM;
    constexpr int THREADS = TX * TY;
    constexpr int KP = BK / 2;
    constexpr int CP = BN / 2;
    constexpr int NO_GLOBAL = AB & 1;
    constexpr int PIN_Q = AB & 2;
    constexpr int NO_MATH = AB & 4;
    constexpr int NO_BARRIER = AB & 8;
    constexpr int ROT = AB & 16;      // NOT an ablation: the conflict-free fetch

    __shared__ __align__(16) __half As[STAGES][BM * BK];
    __shared__ __align__(16) __half Bs[STAGES][BN * BK];

    const int tid = threadIdx.x;
    const int tx = tid % TX;
    const int ty = tid / TX;
    const int row_base = blockIdx.y * BM;
    const int col_base = blockIdx.x * BN;
    const int k_tiles = K / BK;

    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; i++)
#pragma unroll
        for (int j = 0; j < TN; j++) acc[i][j] = 0.0f;

    auto load_stage = [&](int s, int kt) {
        const int k0 = kt * BK;
        __half *Ad = As[s];
        for (int idx = tid; idx < BM * KP; idx += THREADS) {
            const int r = idx / KP, q = idx - r * KP, gr = row_base + r;
            *reinterpret_cast<int *>(Ad + r * BK + q * 2) =
                (gr < M) ? *reinterpret_cast<const int *>(
                               A + static_cast<size_t>(gr) * lda + k0 + q * 2)
                         : 0;
        }
        // B is [K][N], the tile is [N][K]: a 2x2 half transpose. Loading one
        // 32-bit word here fills only k=2q and leaves k=2q+1 garbage, which
        // silently corrupts half of every k-pair.
        __half *Bd = Bs[s];
        for (int idx = tid; idx < KP * CP; idx += THREADS) {
            const int q = idx / CP, c2 = idx - q * CP;
            const int gc = col_base + c2 * 2, kk = k0 + q * 2;
            __half2 r0, r1;
            r0.x = (__half)0.0f; r0.y = (__half)0.0f;
            r1.x = (__half)0.0f; r1.y = (__half)0.0f;
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

    auto fetch = [&](int st, int q, unsigned a[TM], unsigned b[TN]) {
        // q ^ tx keeps lanes out of each other's banks; mask keeps it in range
        // for configs with TX > KP (128x256 has TX = 32).
        const int qq = ROT ? ((q ^ tx) & (KP - 1)) : q;
#pragma unroll
        for (int i = 0; i < TM; i++)
            a[i] = *reinterpret_cast<const unsigned *>(
                As[st] + (ty * TM + i) * BK + qq * 2);
#pragma unroll
        for (int j = 0; j < TN; j++)
            b[j] = *reinterpret_cast<const unsigned *>(
                Bs[st] + (tx * TN + j) * BK + qq * 2);
    };
    auto consume = [&](unsigned a[TM], unsigned b[TN]) {
#pragma unroll
        for (int i = 0; i < TM; i++)
#pragma unroll
            for (int j = 0; j < TN; j++) acc[i][j] = dot2_f16(a[i], b[j], acc[i][j]);
    };

    if (!NO_GLOBAL) load_stage(0, 0);
    if (!NO_BARRIER) __syncthreads();

    int stage = 0;
    for (int kt = 0; kt < k_tiles; kt++) {
        const int nxt = (stage + 1) % STAGES;
        if (!NO_GLOBAL && kt + 1 < k_tiles) load_stage(nxt, kt + 1);

        if (NO_MATH) {
            unsigned a[TM], b[TN];
            fetch(stage, 0, a, b);
            acc[0][0] = dot2_f16(a[0], b[0], acc[0][0]);
        } else if (DB) {
            unsigned ac[TM], bc[TN], an[TM], bn[TN];
            fetch(stage, 0, ac, bc);
#pragma unroll
            for (int q = 0; q < KP; q++) {
                // Issue the NEXT k-pair's LDS reads before consuming this one.
                if (q + 1 < KP) fetch(stage, q + 1, an, bn);
                consume(ac, bc);
#pragma unroll
                for (int i = 0; i < TM; i++) ac[i] = an[i];
#pragma unroll
                for (int j = 0; j < TN; j++) bc[j] = bn[j];
            }
        } else {
            for (int q = 0; q < KP; q++) {
                const int qq = PIN_Q ? 0 : q;
                unsigned a[TM], b[TN];
                fetch(stage, qq, a, b);
                consume(a, b);
            }
        }

        if (!NO_BARRIER) __syncthreads();
        stage = nxt;
    }

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
}

// ---------------------------------------------------------------------------
//  Sampled CPU reference
// ---------------------------------------------------------------------------
constexpr double kFullBudgetMacs = 5e8;
constexpr size_t kSampleCount = 4096;

struct Check {
    double maxabs;
    size_t bad;
    size_t checked;
    bool exhaustive;
};

Check verify(const std::vector<float> &got, const std::vector<float> &AF,
             const std::vector<float> &BF, int M, int N, int K, int lda, int ldb) {
    const size_t total = static_cast<size_t>(M) * N;
    Check r{0.0, 0, 0, true};
    auto one = [&](size_t i) {
        const int m = static_cast<int>(i / static_cast<size_t>(N));
        const int n = static_cast<int>(i % static_cast<size_t>(N));
        double s = 0.0;
        for (int k = 0; k < K; k++)
            s += static_cast<double>(AF[static_cast<size_t>(m) * lda + k]) *
                 static_cast<double>(BF[static_cast<size_t>(k) * ldb + n]);
        const double g = got[i];
        if (!std::isfinite(g)) {  // must precede any tolerance comparison
            r.bad++;
            r.checked++;
            return;
        }
        const double e = std::fabs(g - s);
        const double rel = e / (std::fabs(s) > 1e-6 ? std::fabs(s) : 1.0);
        if (e > r.maxabs) r.maxabs = e;
        if (!(rel <= 1e-3)) r.bad++;
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

struct Problem {
    int M = 0, N = 0, K = 0;
    std::vector<float> AF, BF, got;
    __half *dA = nullptr;
    __half *dB = nullptr;
    float *dC = nullptr;

    void init(int m, int n, int k) {
        M = m; N = n; K = k;
        AF.resize(static_cast<size_t>(M) * K);
        BF.resize(static_cast<size_t>(K) * N);
        got.resize(static_cast<size_t>(M) * N);
        for (size_t i = 0; i < AF.size(); i++)
            AF[i] = static_cast<float>(static_cast<int>((i * 37 + 11) % 17) - 8) * 0.25f;
        for (size_t i = 0; i < BF.size(); i++)
            BF[i] = static_cast<float>(static_cast<int>((i * 53 + 7) % 17) - 8) * 0.25f;

        float *dAF = nullptr, *dBF = nullptr;
        HIP_CHECK(hipMalloc(&dA, AF.size() * sizeof(__half)));
        HIP_CHECK(hipMalloc(&dB, BF.size() * sizeof(__half)));
        HIP_CHECK(hipMalloc(&dAF, AF.size() * sizeof(float)));
        HIP_CHECK(hipMalloc(&dBF, BF.size() * sizeof(float)));
        HIP_CHECK(hipMalloc(&dC, static_cast<size_t>(M) * N * sizeof(float)));
        HIP_CHECK(hipMemcpy(dAF, AF.data(), AF.size() * sizeof(float), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(dBF, BF.data(), BF.size() * sizeof(float), hipMemcpyHostToDevice));
        to_half_kernel<<<(AF.size() + 255) / 256, 256>>>(dA, dAF, AF.size());
        to_half_kernel<<<(BF.size() + 255) / 256, 256>>>(dB, dBF, BF.size());
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipGetLastError());
        hipFree(dAF);
        hipFree(dBF);
    }
    void free_all() { hipFree(dA); hipFree(dB); hipFree(dC); }
};

template <int BM, int BN, int BK, int TM, int TN, int STAGES, int DB, int AB>
double time_kernel(Problem &pr, int iters, bool *ok) {
    constexpr int THREADS = (BM / TM) * (BN / TN);
    const dim3 grid((pr.N + BN - 1) / BN, (pr.M + BM - 1) / BM);
    auto launch = [&] {
        gemm_f16_kernel<BM, BN, BK, TM, TN, STAGES, DB, AB>
            <<<grid, THREADS>>>(pr.dA, pr.dB, pr.dC, pr.M, pr.N, pr.K, pr.K, pr.N, pr.N);
    };
    launch();
    HIP_CHECK(hipDeviceSynchronize());
    const hipError_t le = hipGetLastError();
    if (le != hipSuccess) {  // e.g. LDS over budget
        *ok = false;
        return 0.0;
    }
    hipEvent_t e0, e1;
    hipEventCreate(&e0);
    hipEventCreate(&e1);
    hipEventRecord(e0);
    for (int i = 0; i < iters; i++) launch();
    hipEventRecord(e1);
    HIP_CHECK(hipEventSynchronize(e1));
    float ms = 0;
    hipEventElapsedTime(&ms, e0, e1);
    hipEventDestroy(e0);
    hipEventDestroy(e1);
    *ok = true;
    return ms / iters;
}

template <int BM, int BN, int BK, int TM, int TN, int STAGES, int DB, int AB>
void sweep_one(Problem &pr, double &best, std::string &bestname, double &best_db0) {
    bool ok = false;
    const double ms = time_kernel<BM, BN, BK, TM, TN, STAGES, DB, AB>(pr, 10, &ok);
    char name[128];
    std::snprintf(name, sizeof(name), "%3dx%3dx%2d s%d TM%d TN%d thr%4d db%d%s", BM, BN,
                  BK, STAGES, TM, TN, (BM / TM) * (BN / TN), DB,
                  (AB & 16) ? " ROT" : "    ");
    if (!ok) {
        std::printf("  %s  SKIPPED (launch failed: %s)\n", name,
                    hipGetErrorString(hipGetLastError()));
        return;
    }
    HIP_CHECK(hipMemcpy(pr.got.data(), pr.dC, pr.got.size() * sizeof(float),
                        hipMemcpyDeviceToHost));
    const Check ck = verify(pr.got, pr.AF, pr.BF, pr.M, pr.N, pr.K, pr.K, pr.N);
    const double tflops = 2.0 * pr.M * pr.N * pr.K / (ms * 1e-3) / 1e12;
    std::printf("  %s %8.3f %9.2f  %s maxerr %.1e%s\n", name, ms, tflops,
                ck.exhaustive ? "exhaustive" : "sampled  ", ck.maxabs,
                ck.bad ? "  MISMATCH" : "");
    if (!ck.bad && tflops > best) {
        best = tflops;
        bestname = name;
    }
    if (!ck.bad && DB == 0 && (AB & 16) == 0 && tflops > best_db0) best_db0 = tflops;
}

int main(int argc, char **argv) {
    hipDeviceProp_t p{};
    HIP_CHECK(hipGetDeviceProperties(&p, 0));
    const double clk = p.clockRate / 1e6;
    const int M = argc > 1 ? std::atoi(argv[1]) : 4096;

    std::printf("device: %s  %s  clock %.2f GHz\n", p.name, p.gcnArchName, clk);
    std::printf("(HIP reports %d CU; the true count is 40, so per-CU figures are\n"
                " off by 2x if you use it)\n",
                p.multiProcessorCount);

    Problem pr;
    pr.init(M, M, M);

    std::printf("\n=== config sweep at %d^3 (verified before timing) ===\n", M);
    std::printf("  %-36s %8s %9s  %s\n", "BMxBNxBK st TM TN thr db", "ms",
                "TFLOP/s", "correctness");
    double best = 0, best_db0 = 0;
    std::string bestname = "(none)";

#define CONFIG(BM, BN, BK, TM, TN, ST, DB) \
    sweep_one<BM, BN, BK, TM, TN, ST, DB, 0>(pr, best, bestname, best_db0)
#define CONFIG_ROT(BM, BN, BK, TM, TN, ST, DB) \
    sweep_one<BM, BN, BK, TM, TN, ST, DB, 16>(pr, best, bestname, best_db0)

    CONFIG(128, 128, 32, 8, 8, 2, 0);
    CONFIG(128, 128, 32, 8, 8, 2, 1);
    CONFIG(128, 128, 32, 8, 8, 3, 0);
    CONFIG(128, 128, 32, 8, 8, 3, 1);
    CONFIG(128, 128, 16, 8, 8, 3, 0);
    CONFIG(128, 128, 16, 8, 8, 3, 1);
    CONFIG(128, 128, 64, 8, 8, 2, 0);
    CONFIG(128, 128, 32, 8, 4, 2, 0);
    CONFIG(128, 128, 32, 4, 8, 2, 0);
    CONFIG(128, 128, 32, 4, 4, 2, 1);
    CONFIG(128, 64, 32, 8, 4, 2, 0);
    CONFIG(64, 128, 32, 4, 8, 2, 0);
    CONFIG(256, 128, 32, 8, 8, 2, 0);
    CONFIG(256, 128, 32, 8, 8, 2, 1);
    CONFIG(128, 256, 32, 8, 8, 2, 0);
    CONFIG(128, 256, 32, 8, 8, 2, 1);
    CONFIG(64, 64, 32, 4, 4, 2, 0);
    CONFIG(256, 256, 32, 8, 8, 2, 0);  // expected: LDS over budget

    // The same configs with the lane-rotated fetch (AB bit 16). These are real
    // configurations, verified exactly like the rows above.
    std::printf("\n=== same configs with the bank-conflict-free fetch (q ^ tx) ===\n");
    CONFIG_ROT(128, 128, 32, 8, 8, 2, 0);
    CONFIG_ROT(128, 128, 32, 8, 8, 3, 0);
    CONFIG_ROT(128, 128, 16, 8, 8, 3, 0);
    CONFIG_ROT(128, 128, 32, 8, 8, 2, 1);
    CONFIG_ROT(128, 128, 32, 4, 4, 2, 0);
    CONFIG_ROT(256, 128, 32, 8, 8, 2, 0);
    CONFIG_ROT(64, 128, 32, 4, 8, 2, 0);
    CONFIG_ROT(128, 64, 32, 8, 4, 2, 0);

    std::printf("\n  best verified config : %s at %.2f TFLOP/s\n", bestname.c_str(), best);
    std::printf("  best non-rotated, no db : %.2f TFLOP/s\n", best_db0);
    std::printf("  (compare db0/db1 and rotated/unrotated PAIRWISE in the tables\n"
                "   above: best-vs-best_db0 is not a db measurement once the best\n"
                "   row uses the rotated fetch, which is worth ~2.5x by itself.)\n");
    std::printf("  reference: rocBLAS 20.19, Triton 15.04, pipe ceiling 25.33\n");

    std::printf("\n=== ablation at %d^3, 128x128x32 s2 TM=TN=8 db1 (ROTATED fetch) ===\n", M);
    std::printf("  (these variants compute garbage by construction; they are timed\n"
                "   only and never correctness-checked)\n");
    {
        struct V { const char *name; double ms; bool ok; };
        V vs[5];
        bool ok = false;
        vs[0].name = "full (rotated)";
        vs[0].ms = time_kernel<128, 128, 32, 8, 8, 2, 1, 16>(pr, 10, &ok);
        vs[0].ok = ok;
        vs[1].name = "no global->LDS load";
        vs[1].ms = time_kernel<128, 128, 32, 8, 8, 2, 1, 16 | 1>(pr, 10, &ok);
        vs[1].ok = ok;
        vs[2].name = "no LDS register read";
        vs[2].ms = time_kernel<128, 128, 32, 8, 8, 2, 1, 16 | 2>(pr, 10, &ok);
        vs[2].ok = ok;
        vs[3].name = "no dot2 math";
        vs[3].ms = time_kernel<128, 128, 32, 8, 8, 2, 1, 16 | 4>(pr, 10, &ok);
        vs[3].ok = ok;
        vs[4].name = "no barriers";
        vs[4].ms = time_kernel<128, 128, 32, 8, 8, 2, 1, 16 | 8>(pr, 10, &ok);
        vs[4].ok = ok;

        const double full_ms = vs[0].ms;
        const double warps = ((double)((M + 127) / 128) * ((M + 127) / 128)) * 8.0;
        const double ktiles = (double)(M / 32);
        std::printf("  %-24s %9s %15s %10s\n", "variant", "ms", "cyc/k-tile/warp",
                    "vs full");
        for (const V &v : vs) {
            if (!v.ok) { std::printf("  %-24s  SKIPPED\n", v.name); continue; }
            const double cyc = v.ms * 1e-3 * clk * 1e9;
            std::printf("  %-24s %9.3f %15.1f %9.1f%%\n", v.name, v.ms,
                        cyc / (warps * ktiles), 100.0 * v.ms / full_ms);
        }
        std::printf("\n  attributed cost = full minus variant:\n");
        struct A { const char *n; const V *v; };
        const A as[] = {{"global->LDS load", &vs[1]}, {"LDS register read", &vs[2]},
                        {"dot2 math stream ", &vs[3]}, {"barriers         ", &vs[4]}};
        for (const A &a : as) {
            if (!a.v->ok) continue;
            const double d = full_ms - a.v->ms;
            std::printf("    %s : %7.3f ms  (%5.1f%% of full)\n", a.n, d,
                        100.0 * d / full_ms);
        }
        std::printf("\n  NOTE: the deltas overlap and need not sum to 100%% -- removing\n"
                    "  one phase also removes stalls the others were hiding behind.\n"
                    "  Read them as a ranking, not a decomposition.\n");
    }

    pr.free_all();
    std::printf("\n=== external reference points, measured elsewhere on this card ===\n");
    std::printf("  rocBLAS fp16 4096^3   20.19 TFLOP/s\n");
    std::printf("  Triton tl.dot 4096^3  15.04 TFLOP/s\n");
    std::printf("  v_dot2 pipe ceiling    25.33 TFLOP/s (register-only)\n");
    return 0;
}
