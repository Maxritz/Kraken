// probe_tq_file.cpp — validate the ternary decoders against real GGUF bytes,
// and measure the zero-block density that the MoE paging policy would exploit.
//
// Two jobs:
//   1. Decode every TQ1_0/TQ2_0 tensor in a file and report the statistics of
//      the reconstructed weights. A correct decoder gives trit values in
//      {-d, 0, +d} with d in a plausible range and ~1/3 exact zeros. Garbage
//      output from a model run is then attributable: either the weights decode
//      sanely (the file/quant is the problem) or they do not (the decoder is).
//   2. Count all-zero quant blocks, which is what the zero-skip page-in policy
//      would avoid reading.
//
// Build:
//   c++ -O2 -std=c++17 -Iinclude tools/probe_tq_file.cpp src/gguf.cpp \
//       src/quant.cpp src/common.cpp -o probe_tq_file
#include "krk/gguf.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>

using namespace krk;

int main(int argc, char **argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: probe_tq_file <model.gguf>\n");
        return 2;
    }
    Gguf g;
    std::string err;
    if (!g.load(argv[1], &err)) {
        std::fprintf(stderr, "load failed: %s\n", err.c_str());
        return 1;
    }

    // ---- type census ------------------------------------------------------
    std::map<std::string, std::string> dummy; // unused; kept out of the way
    std::map<DType, std::pair<int, size_t>> census;
    for (const GgufTensor &t : g.tensors()) {
        auto &e = census[t.type];
        e.first++;
        e.second += t.n_bytes;
    }
    std::printf("=== tensor type census ===\n");
    for (const auto &kv : census)
        std::printf("  %-8s %4d tensors  %8.1f MiB\n", dtype_name(kv.first),
                    kv.second.first, kv.second.second / 1048576.0);

    // ---- per-tensor decode stats -----------------------------------------
    std::printf("\n=== ternary tensors: decode statistics ===\n");
    int n_tq = 0;
    size_t tot_blocks = 0, tot_zero_blocks = 0, tot_bytes = 0, zero_bytes = 0;
    for (const GgufTensor &t : g.tensors()) {
        if (t.type != DType::TQ1_0 && t.type != DType::TQ2_0) continue;
        n_tq++;
        const i64 n = t.n_elements;
        const int bs = dtype_block_size(t.type);
        const int bb = dtype_block_bytes(t.type);
        const i64 nblk = n / bs;
        if (nblk <= 0) continue;

        std::vector<f32> out(static_cast<size_t>(std::min<i64>(n, 4096)));
        dequant_row(t.type, t.data, out.data(), static_cast<i64>(out.size()));

        int zeros = 0, pos = 0, neg = 0;
        f32 amax = 0;
        for (f32 v : out) {
            if (v == 0.0f) zeros++;
            else if (v > 0) pos++;
            else neg++;
            amax = std::fabs(v) > amax ? std::fabs(v) : amax;
        }
        const f64 zf = 100.0 * zeros / static_cast<f64>(out.size());

        // Zero-block census over the whole tensor.
        size_t zb = 0;
        const u8 *p = t.data;
        for (i64 b = 0; b < nblk; b++, p += bb) {
            bool all_zero = true;
            // Cheap test first: TQ2_0 zero is the 0x55 pattern. For TQ1_0 the
            // canonical zero block is 0x80 in qs and 0x7F in qh.
            if (t.type == DType::TQ2_0) {
                for (int i = 0; i < 64 && all_zero; i++)
                    if (p[i] != 0x55) all_zero = false;
            } else {
                for (int i = 0; i < 48 && all_zero; i++)
                    if (p[i] != 0x80) all_zero = false;
                for (int i = 0; i < 4 && all_zero; i++)
                    if (p[48 + i] != 0x7F) all_zero = false;
            }
            if (all_zero) zb++;
        }
        tot_blocks += static_cast<size_t>(nblk);
        tot_zero_blocks += zb;
        tot_bytes += t.n_bytes;
        zero_bytes += zb * static_cast<size_t>(bb);

        std::printf("  %-44s %-6s %6.2fM el  dmax=%.5g  zero=%5.1f%%  "
                    "zeroblk=%5.1f%% (%zu/%lld)\n",
                    t.name.c_str(), dtype_name(t.type), n / 1e6, amax, zf,
                    100.0 * static_cast<f64>(zb) / static_cast<f64>(nblk), zb,
                    static_cast<long long>(nblk));
        (void)pos;
        (void)neg;
    }

    if (n_tq == 0) {
        std::printf("  (no ternary tensors in this file)\n");
        return 0;
    }
    std::printf("\n  %d ternary tensors, %.1f MiB payload\n", n_tq,
                tot_bytes / 1048576.0);
    std::printf("  all-zero blocks: %zu/%zu (%.2f%%) = %.2f MiB of %.1f MiB "
                "that a zero-skipping page-in never has to read\n",
                tot_zero_blocks, tot_blocks,
                100.0 * static_cast<f64>(tot_zero_blocks) / static_cast<f64>(tot_blocks),
                zero_bytes / 1048576.0, tot_bytes / 1048576.0);
    return 0;
}