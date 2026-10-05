// cpu_expert_ceiling.cpp — is the CPU expert fallback actually worth building?
//
//   kraken-cpu-expert-ceiling model.gguf [--layer N] [--routed N]
//
// WHY THIS EXISTS
// ---------------
// A Laguna decode token is 75.1-75.7 ms on this machine and 60% of the
// generation phase is spent moving expert weights: 19.43 GiB of routed
// experts against a 12,425 MiB VRAM budget, so ~82 of the 312 routed
// experts are cold every token and cost ~160 MiB of NVMe read plus ~160 MiB
// of H2D. Measured; see docs/traces/decode-dissection.txt.
//
// The proposed fix (spec section 22, Fiddler) is: do not move a cold expert,
// COMPUTE it. For one decode token the activation a cold expert needs is
// m x n_embd floats -- 8 KB at m=1 -- against a ~1.95 MiB weight, so moving
// the weight costs ~250x more than moving the operand.
//
// That argument only decides WHERE to compute. It does not say whether the
// CPU can keep up. This answers that directly, on the real file, with the
// real dequantiser from src/quant.cpp:
//
//   cold experts per token:  C      (routed x (1 - VRAM residency))
//   decode-token CPU cost =  C * t_expert / threads
//
// printed against the measured 48.8 ms/token of read + xfer, which is what
// it has to beat.
//
// SHAPE NOTES, because getting these wrong is silent
// --------------------------------------------------
// The expert tensors are 3D: ne = [n_in, n_ff, n_expert]. So for the gate and
// up matrices ne[0] is n_in and ne[1] is n_ff, while for the down matrix
// ne[0] is n_ff and ne[1] is n_in. Reading n_expert off ne[1] gets n_ff and
// the whole thing reads garbage.
//
// The work is row-wise, exactly as gemv_kernel does it: dequantize ONE row,
// dot it, move on. Dequantizing a whole [n_ff, n_in] matrix needs a
// n_ff*n_in buffer, which is 4.8 MB of scratch for a 2048-wide Laguna expert
// and would have made this tool measure the allocator instead of the maths.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "krk/gguf.hpp"
#include "krk/quant.hpp"

using namespace krk;
using Clock = std::chrono::steady_clock;

namespace {

double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// One expert's three slices, copied out of the mapping into ordinary pageable
// host memory. That IS the WARM tier's representation -- ExpertCache allocates
// exactly these bytes with alloc_host_pageable -- so this measures the real
// steady state, not a best case the runtime cannot reach.
struct Expert {
    std::vector<u8> gate, up, down;
    DType dt = DType::Unknown;
    i64 n_in = 0; // inner dim of gate/up, and rows of down
    i64 n_ff = 0; // rows of gate/up, and inner dim of down
};

bool load_slice(const GgufTensor *t, i64 expert, i64 n_expert, std::vector<u8> &out) {
    if (!t || !t->data || t->n_bytes == 0 || n_expert <= 0) return false;
    const size_t per = t->n_bytes / static_cast<size_t>(n_expert);
    const u8 *src = t->data + static_cast<size_t>(expert) * per;
    out.assign(src, src + per);
    return true;
}

// One expert for ONE activation row, matching what the engine computes on the
// GPU: gate/up over [1, n_in], silu(gate) * up, down over [1, n_ff].
// Row-wise, O(n_in + n_ff) scratch.
void run_expert(const Expert &e, const f32 *x, f32 *h, f32 *row, f32 *y) {
    const i64 nin = e.n_in, nff = e.n_ff;
    const size_t rs_in = dtype_row_bytes(e.dt, nin); // gate/up row stride
    const size_t rs_ff = dtype_row_bytes(e.dt, nff); // down row stride
    for (i64 j = 0; j < nff; j++) {
        dequant_row(e.dt, e.gate.data() + static_cast<size_t>(j) * rs_in, row, nin);
        f32 g = 0.0f;
        for (i64 i = 0; i < nin; i++) g += row[i] * x[i];
        dequant_row(e.dt, e.up.data() + static_cast<size_t>(j) * rs_in, row, nin);
        f32 u = 0.0f;
        for (i64 i = 0; i < nin; i++) u += row[i] * x[i];
        h[j] = (g / (1.0f + std::exp(-g))) * u; // silu(g) * u
    }
    for (i64 i = 0; i < nin; i++) {
        dequant_row(e.dt, e.down.data() + static_cast<size_t>(i) * rs_ff, row, nff);
        f32 s = 0.0f;
        for (i64 j = 0; j < nff; j++) s += row[j] * h[j];
        y[i] = s;
    }
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s model.gguf [--layer N] [--routed N]\n", argv[0]);
        return 2;
    }
    const std::string path = argv[1];
    i32 layer = 3;
    i32 routed = 312; // Laguna XS.2: 39 sparse layers x top-8
    for (int i = 2; i < argc; i++) {
        if (!std::strcmp(argv[i], "--layer") && i + 1 < argc) layer = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--routed") && i + 1 < argc) routed = std::atoi(argv[++i]);
    }

    Gguf g;
    std::string err;
    if (!g.load(path, &err)) {
        std::fprintf(stderr, "load failed: %s\n", err.c_str());
        return 1;
    }

    char nm[128];
    const GgufTensor *t[3] = {nullptr, nullptr, nullptr};
    static const char *which[3] = {"gate", "up", "down"};
    for (int i = 0; i < 3; i++) {
        std::snprintf(nm, sizeof nm, "blk.%d.ffn_%s_exps.weight", layer, which[i]);
        t[i] = g.tensor(nm);
        if (!t[i]) {
            std::fprintf(stderr, "missing %s\n", nm);
            return 1;
        }
        if (t[i]->n_dims != 3) {
            std::fprintf(stderr, "%s has %d dims, expected 3 [in, mid, n_expert]\n", nm,
                         t[i]->n_dims);
            return 1;
        }
    }
    // gate/up: [n_in, n_ff, n_expert].  down: [n_ff, n_in, n_expert].
    const i64 n_in = static_cast<i64>(t[0]->ne[0]);
    const i64 n_ff = static_cast<i64>(t[0]->ne[1]);
    const i64 n_expert = static_cast<i64>(t[0]->ne[2]);
    if (static_cast<i64>(t[2]->ne[0]) != n_ff || static_cast<i64>(t[2]->ne[1]) != n_in ||
        static_cast<i64>(t[2]->ne[2]) != n_expert) {
        std::fprintf(stderr, "down [%lld, %lld, %lld] does not match gate [%lld, %lld, %lld]\n",
                     static_cast<long long>(t[2]->ne[0]), static_cast<long long>(t[2]->ne[1]),
                     static_cast<long long>(t[2]->ne[2]), static_cast<long long>(t[0]->ne[0]),
                     static_cast<long long>(t[0]->ne[1]), static_cast<long long>(t[0]->ne[2]));
        return 1;
    }
    const size_t bytes = t[0]->n_bytes / static_cast<size_t>(n_expert) +
                         t[1]->n_bytes / static_cast<size_t>(n_expert) +
                         t[2]->n_bytes / static_cast<size_t>(n_expert);

    std::printf("model    %s\n", path.c_str());
    std::printf("layer %d  %lld experts, n_in %lld, n_ff %lld\n", layer,
                static_cast<long long>(n_expert), static_cast<long long>(n_in),
                static_cast<long long>(n_ff));
    std::printf("expert   %s, %zu bytes (%.3f MiB), 3 slices\n", dtype_name(t[0]->type), bytes,
                bytes / (1024.0 * 1024.0));
    std::printf("params   %.2f M\n", 2.0 * static_cast<double>(n_in) * static_cast<double>(n_ff) / 1e6);
    std::printf("\n");

    const i32 kPool = 32; // distinct experts, so threads are not reading one
    std::vector<Expert> pool(static_cast<size_t>(kPool));
    for (i32 e = 0; e < kPool; e++) {
        Expert &x = pool[static_cast<size_t>(e)];
        x.dt = t[0]->type;
        x.n_in = n_in;
        x.n_ff = n_ff;
        if (!load_slice(t[0], e, n_expert, x.gate) || !load_slice(t[1], e, n_expert, x.up) ||
            !load_slice(t[2], e, n_expert, x.down)) {
            std::fprintf(stderr, "slice read failed\n");
            return 1;
        }
    }

    std::vector<f32> x(static_cast<size_t>(n_in));
    for (i64 i = 0; i < n_in; i++)
        x[static_cast<size_t>(i)] = 0.001f * static_cast<float>((i % 17) - 8);

    // Warm-up pass: touch every pool expert once so the timed loop is not
    // measuring first-touch faults on the memcpy'd slices.
    {
        std::vector<f32> h(static_cast<size_t>(n_ff)), row(static_cast<size_t>(std::max(n_in, n_ff))),
            y(static_cast<size_t>(n_in));
        for (i32 e = 0; e < kPool; e++)
            run_expert(pool[static_cast<size_t>(e)], x.data(), h.data(), row.data(), y.data());
    }

    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    std::printf("hardware_concurrency %u\n\n", hw);
    std::printf("threads  ms/expert  experts/s   GB/s ram   cold/tok ms   tok/s overlap"
                "   tok/s serial\n");
    std::printf("-------  ---------  ----------  ---------  ------------  ---------------"
                "  -------------\n");

    // Measured on this machine, docs/traces/decode-dissection.txt.
    const double kResidentMs = 26.2;   // everything that is not read/xfer
    const double kReadXferMs = 48.8;   // read + xfer, per token, 312 routed
    const double kColdFrac = 0.36;     // 1 - 64% VRAM residency for this file

    for (unsigned nt : {1u, 2u, 4u, 8u, 12u, 16u, 24u, 32u, hw}) {
        if (nt > hw) continue;
        const int reps = 8;
        const Clock::time_point t0 = Clock::now();
        std::vector<std::thread> th;
        th.reserve(nt);
        for (unsigned u = 0; u < nt; u++) {
            th.emplace_back([&, u]() {
                std::vector<f32> h(static_cast<size_t>(n_ff));
                std::vector<f32> row(static_cast<size_t>(std::max(n_in, n_ff)));
                std::vector<f32> y(static_cast<size_t>(n_in));
                for (int r = 0; r < reps; r++)
                    for (i32 e = static_cast<i32>(u); e < kPool; e += static_cast<i32>(nt))
                        run_expert(pool[static_cast<size_t>(e)], x.data(), h.data(),
                                   row.data(), y.data());
            });
        }
        for (auto &t2 : th) t2.join();
        const double el = ms_since(t0);
        const int total = reps * ((kPool + static_cast<i32>(nt) - 1) / static_cast<i32>(nt));
        const double per_ms = el / static_cast<double>(total);
        // per_ms is the wall cost of one expert on ONE thread. The threads run
        // concurrently, so the machine's aggregate throughput is nt / per_ms and
        // not 1 / per_ms. An earlier version of this tool printed the latter and
        // made 8 threads look slower than 1.
        const double per_s = static_cast<double>(nt) * 1000.0 / per_ms; // per second
        const double gbs = per_s * static_cast<double>(bytes) / 1e9;

        const double cold = static_cast<double>(routed) * kColdFrac;
        const double cpu_ms = cold * per_ms / static_cast<double>(nt);
        const double overlap_ms = kResidentMs + cpu_ms; // CPU work hides under the GPU's
        const double serial_ms = kReadXferMs + cpu_ms; // CPU work does NOT hide
        std::printf("%7u  %9.3f  %10.1f  %9.2f  %12.2f  %14.1f  %13.1f%s\n", nt, per_ms,
                    per_s, gbs, cpu_ms, 1000.0 / overlap_ms, 1000.0 / serial_ms,
                    cpu_ms < kReadXferMs ? "   beats read+xfer" : "");
    }

    std::printf("\nMeasured on this machine, for comparison:\n");
    std::printf("  current decode                       75.1-75.7 ms/tok (13.2-13.3 tok/s)\n");
    std::printf("  of which read+xfer                    %.1f ms/tok\n", kReadXferMs);
    std::printf("  of which everything else              %.1f ms/tok\n", kResidentMs);
    std::printf("\n'tok/s overlap' REPLACES the 48.8 ms of read+xfer with the CPU cost, assuming\n");
    std::printf("the CPU threads run cold experts while the GPU runs the resident ones --\n");
    std::printf("the point of the split. 'tok/s serial' KEEPS read+xfer and ADDS the CPU\n");
    std::printf("cost, which is what a version with no overlap between the two would pay.\n");
    std::printf("The truth is between them: the GPU cannot proceed past a layer until\n");
    std::printf("that layer's cold experts are done, so the overlap is bounded.\n");
    return 0;
}