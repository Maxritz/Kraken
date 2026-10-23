// probe_hostread.hip -- can a kernel READ pinned host memory, and at what rate?
//
// The question this settles: on this box a MoE decode token promotes ~318 MiB
// WARM->VRAM to deliver ~19.5 MiB of expert weights it actually uses, because
// each routed expert is read once and then evicted. If a kernel can read the
// pinned WARM slab in place, that becomes one PCIe pass over the bytes that are
// needed instead of a copy of bytes that are not. Before that is worth building,
// three things have to be facts and not hopes:
//
//   1. a `hipHostMalloc` pointer is a legal kernel argument here at all (ROCm on
//      WDDM maps the pinned pool; a device fault or a silent zero would say no),
//   2. the values read back are the values written (a bandwidth number from a
//      kernel that read zeros would be a very fast lie),
//   3. the rate is in the same class as the promotion copy it replaces, so the
//      trade is one pass instead of two and not "much slower per byte".
//
// Build (same toolchain as the engine, ad hoc -- this is a probe, not a target):
//   clang++.exe --offload-arch=gfx1201 -O3 -std=c++17 \
//     -IG:/ROCM10RT-gfx1201/include probe_hostread.hip -o probe_hostread.exe \
//     -LG:/ROCM10RT-gfx1201/lib -lamdhip64
#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define HIPCHK(x)                                                              \
    do {                                                                       \
        hipError_t e_ = (x);                                                   \
        if (e_ != hipSuccess) {                                                \
            std::printf("HIP error: %s  (%s:%d)\n", hipGetErrorString(e_),     \
                        __FILE__, __LINE__);                                   \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

namespace {

using u64 = unsigned long long;

// Contiguous 16-byte reads, one accumulator per thread. That is the access
// pattern a quantized weight walk has (contiguous blocks, no reuse), so the rate
// this prints is the rate the expert GEMM would see -- not a peak measured with
// a pattern the real kernel does not use.
__global__ void read_sum(const uint4 *__restrict__ src, u64 n4,
                         unsigned long long *__restrict__ out) {
    const u64 tid = static_cast<u64>(blockIdx.x) * blockDim.x + threadIdx.x;
    const u64 stride = static_cast<u64>(gridDim.x) * blockDim.x;
    u64 acc = 0;
    for (u64 i = tid; i < n4; i += stride) {
        const uint4 v = src[i];
        acc += static_cast<u64>(v.x) + v.y + v.z + v.w;
    }
    // ONE slot per thread. The first version of this probe summed into
    // `out[tid & 1023]`, so 128 threads raced on the same word and the total
    // came out at exactly 1/128 of the truth -- which read as "the pinned buffer
    // is not mapped" when the reads had in fact been correct all along. A
    // correctness check that cannot distinguish a broken read from a broken
    // check is worth less than no check.
    out[tid] = acc;
}

double wall_ms(std::chrono::steady_clock::time_point a) {
    const auto b = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(b - a).count();
}

double timed_read(const uint4 *src, u64 n4, unsigned long long *dst, int reps) {
    // Warm, then measure: the first launch pays module load, and this probe is
    // about the steady-state read rate.
    read_sum<<<512, 256>>>(src, n4, dst);
    HIPCHK(hipDeviceSynchronize());
    double best = 1e30;
    for (int r = 0; r < reps; r++) {
        const auto t0 = std::chrono::steady_clock::now();
        read_sum<<<512, 256>>>(src, n4, dst);
        HIPCHK(hipDeviceSynchronize());
        const double ms = wall_ms(t0);
        if (ms < best) best = ms;
    }
    return best;
}

} // namespace

int main() {
    hipDeviceProp_t prop{};
    HIPCHK(hipGetDeviceProperties(&prop, 0));
    std::printf("device      %s (%s)\n", prop.name, prop.gcnArchName);
    std::printf("unified addressing: %d\n", prop.unifiedAddressing);

    // ---- 1. is a pinned host pointer readable at all? -----------------------
    const u64 small = 4ull << 20; // 4 MiB
    const u64 small_n4 = small / 16;
    void *pin_small = nullptr;
    HIPCHK(hipHostMalloc(&pin_small, small, hipHostMallocDefault));
    auto *h = static_cast<uint4 *>(pin_small);
    for (u64 i = 0; i < small_n4; i++)
        h[i] = make_uint4(0x01010101u, 0x01010101u, 0x01010101u, 0x01010101u);
    const int kSlots = 512 * 256;
    unsigned long long *d_out = nullptr;
    HIPCHK(hipMalloc(&d_out, static_cast<size_t>(kSlots) * sizeof(unsigned long long)));
    HIPCHK(hipMemset(d_out, 0, static_cast<size_t>(kSlots) * sizeof(unsigned long long)));

    read_sum<<<512, 256>>>(reinterpret_cast<const uint4 *>(pin_small), small_n4,
                           d_out);
    HIPCHK(hipDeviceSynchronize());
    std::vector<unsigned long long> back(static_cast<size_t>(kSlots), 0);
    HIPCHK(hipMemcpy(back.data(), d_out, back.size() * sizeof(unsigned long long),
                     hipMemcpyDeviceToHost));
    u64 total = 0;
    for (u64 v : back) total += v;
    // Every uint4 is 16 bytes of 0x01, so each contributes 4 * 0x01010101.
    const u64 expect = small_n4 * 4ull * 0x01010101ull;
    std::printf("pinned read: %llu of %llu (%.6f%%)\n", total, expect,
                expect ? 100.0 * static_cast<double>(total) /
                             static_cast<double>(expect)
                       : 0.0);
    if (total != expect) {
        std::printf("VERDICT: a kernel cannot read this pinned buffer correctly "
                    "(a zero sum is the usual failure: the runtime did not map "
                    "it), so an in-place expert read is not available here\n");
        return 2;
    }

    // ---- 2. rate: same kernel over device memory and over pinned memory -----
    const u64 big = 256ull << 20;
    const u64 big_n4 = big / 16;
    void *dev = nullptr;
    HIPCHK(hipMalloc(&dev, big));
    HIPCHK(hipMemset(dev, 0, big));
    void *pin_big = nullptr;
    HIPCHK(hipHostMalloc(&pin_big, big, hipHostMallocDefault));
    std::memset(pin_big, 0, big);

    // Interleaved, best-of-N each: a sequential pair would let the page tables
    // and the driver's mapping cache warm between arms and credit the second one.
    double dev_best = 1e30, pin_best = 1e30;
    for (int round = 0; round < 3; round++) {
        const double d = timed_read(static_cast<const uint4 *>(dev), big_n4, d_out, 3);
        const double p =
            timed_read(static_cast<const uint4 *>(pin_big), big_n4, d_out, 3);
        if (d < dev_best) dev_best = d;
        if (p < pin_best) pin_best = p;
    }
    const double gib = static_cast<double>(big) / (1024.0 * 1024.0 * 1024.0);
    const double dev_gbs = gib / (dev_best / 1000.0);
    const double pin_gbs = gib / (pin_best / 1000.0);
    std::printf("device read  %7.1f ms  %6.1f GiB/s\n", dev_best, dev_gbs);
    std::printf("pinned read  %7.1f ms  %6.1f GiB/s  (%.0f%% of device)\n",
                pin_best, pin_gbs, 100.0 * pin_gbs / dev_gbs);

    // ---- 3. the trade the engine would actually make ------------------------
    // One decode token on the REAP-288 file promotes ~318 MiB WARM->VRAM to use
    // ~19.5 MiB of it. In place, that is 19.5 MiB over PCIe and no device copy.
    const double routed_mib = 318.0, used_mib = 19.5;
    const double promote_ms = routed_mib / 1024.0 / 11.0 * 1000.0; // measured 11 GB/s
    const double inplace_ms = used_mib / 1024.0 / pin_gbs * 1000.0;
    std::printf("per token: promote %.1f MiB = %.1f ms at 11 GiB/s; "
                "in-place %.1f MiB = %.1f ms at %.1f GiB/s\n",
                routed_mib, promote_ms, used_mib, inplace_ms, pin_gbs);
    std::printf("VERDICT: %s\n",
                pin_gbs > 4.0
                    ? "pinned memory is readable in place and fast enough that "
                      "one PCIe pass beats a copy of 16x the bytes"
                    : "pinned memory is readable but too slow in place; the "
                      "promotion copy stays");

    HIPCHK(hipFree(dev));
    HIPCHK(hipHostFree(pin_big));
    HIPCHK(hipHostFree(pin_small));
    HIPCHK(hipFree(d_out));
    return 0;
}
