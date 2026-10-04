// probe_gemv.hip — is gemv_kernel slow, or is it being measured wrong?
//
// The decode trace attributes 42.2 ms to 1234 gemm(gemv) calls, which works out
// to 35.1 GB/s on a card whose GDDR6 peak is ~896 GB/s. That number comes from
// a device span, which includes launch-to-arrival idle, so it does NOT
// distinguish "the kernel is slow" from "the op is starved". This probe runs
// the same kernel back to back with nothing else in the queue, which does.
//
// It also separates kernel quality from SHAPE: the decode shapes are small
// (1024x2048 Q4_K = 1.15 MB), so a large matrix is timed too. If the large
// shape reaches a few hundred GB/s and the small one does not, the kernel is
// fine and the geometry is the problem. If neither does, the kernel is.
//
// Output is a table: shape, rows, cols, grid, blocks/CU, us/call, GB/s.
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "krk_hip.hpp"
#include "kernels/dequant.hpp"
#include "kernels/gemm.hpp"

using namespace krk;

// A Q4_K superblock is 144 bytes covering 256 values, so a row of n values
// costs (n/256)*144 bytes. Verified against dtype_row_bytes below.
static size_t q4k_row_bytes(int n) { return (size_t)(n / 256) * 144; }

static void fill_q4k(std::vector<u8> &b) {
    // Scales live at the head of each 144-byte superblock. Writing 0x3c 0x00
    // there makes them fp16 1.0, so the dequant path stays finite and no lane
    // can hit a denormal that would change the timing.
    std::memset(b.data(), 0x5a, b.size());
    for (size_t i = 0; i + 144 <= b.size(); i += 144) {
        b[i + 0] = 0x3c; b[i + 1] = 0x00;  // d = 1.0
        b[i + 2] = 0x3c; b[i + 3] = 0x00;  // m = 1.0
    }
}

struct Shape {
    const char *name;
    int n_out;
    int n_in;
};

int main() {
    hipDeviceProp_t prop{};
    hipGetDeviceProperties(&prop, 0);
    const int cu = prop.multiProcessorCount;
    std::printf("device: %s  SMs(CUs)=%d  memClk=%d kHz  busWidth=%d bit\n",
                prop.gcnArchName, cu, prop.memoryClockRate,
                prop.memoryBusWidth);
    const double peak_gbs =
        2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8.0) / 1.0e6;
    std::printf("theoretical peak: %.0f GB/s\n\n", peak_gbs);

    const Shape shapes[] = {
        {"decode gate/up [1024,2048]", 1024, 2048},   // laguna-xs2 expert
        {"decode down   [2048,1024]", 2048, 1024},   // laguna-xs2 expert
        {"large         [8192,2048]", 8192, 2048},   // 8x the work, same kernel
        {"large         [16384,2048]", 16384, 2048},  // 16x
    };

    u8 *w = nullptr;
    _Float16 *x = nullptr, *o = nullptr;
    const int xcap = 65536;

    std::printf("%-28s %6s %6s %8s %9s %9s %9s %7s\n", "shape", "n_out",
                "n_in", "grid", "blocks/CU", "us/call", "GB/s", "%peak");

    for (const Shape &s : shapes) {
        const size_t rb = q4k_row_bytes(s.n_in);
        const size_t bytes = rb * (size_t)s.n_out;
        if (w) hipFree(w);
        std::vector<u8> host(bytes);
        fill_q4k(host);
        if (hipMalloc(&w, bytes) != hipSuccess) { std::printf("alloc fail\n"); return 1; }
        hipMemcpy(w, host.data(), bytes, hipMemcpyHostToDevice);
        if (!x) {
            if (hipMalloc(&x, xcap * sizeof(_Float16)) != hipSuccess) return 1;
            hipMemset(x, 0, xcap * sizeof(_Float16));
        }
        if (o) hipFree(o);
        if (hipMalloc(&o, (size_t)s.n_out * sizeof(_Float16)) != hipSuccess) return 1;

        const unsigned grid = (unsigned)((s.n_out + 7) / 8);

        // Warm up: first launch pays module load and any lazy allocation.
        for (int i = 0; i < 5; i++)
            gemv_kernel<256><<<grid, 256>>>(w, (int)DType::Q4_K, s.n_out,
                                            s.n_in, rb, x, o);
        hipDeviceSynchronize();

        const int iters = 200;
        hipEvent_t a, b;
        hipEventCreate(&a); hipEventCreate(&b);
        hipEventRecord(a, 0);
        for (int i = 0; i < iters; i++)
            gemv_kernel<256><<<grid, 256>>>(w, (int)DType::Q4_K, s.n_out,
                                            s.n_in, rb, x, o);
        hipEventRecord(b, 0);
        hipEventSynchronize(b);
        float ms = 0.f;
        hipEventElapsedTime(&ms, a, b);
        const double us = (double)ms * 1000.0 / iters;
        const double gbs = (double)bytes / (us * 1e3);
        std::printf("%-28s %6d %6d %8u %9.2f %9.2f %9.1f %6.1f%%\n", s.name,
                    s.n_out, s.n_in, grid, (double)grid / cu, us, gbs,
                    100.0 * gbs / peak_gbs);
        hipEventDestroy(a); hipEventDestroy(b);
    }

    std::printf(
        "\nReading the table: a row near %%peak would mean the kernel is fine\n"
        "and the decode geometry is what starves it. Every row stuck at a\n"
        "similar low %%peak regardless of size would mean the kernel itself is\n"
        "the ceiling, and no grid change can recover it.\n");
    return 0;
}