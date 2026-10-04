// probe_zero_blocks.cpp — measure the yield of the "skip all-zero quant blocks"
// idea, per dtype, on a real file.
//
// The premise behind zero-block skipping is that a quantized block can be
// entirely zero, so a page-in never has to read it. This measures how often
// that actually happens, which decides whether the optimisation is worth
// building. Decoding is done with the same reference decoder the engine uses,
// so the answer is in terms of values the model would really see.
//
// Build:
//   c++ -O2 -std=c++17 -Iinclude tools/probe_zero_blocks.cpp src/gguf.cpp \
//       src/quant.cpp src/common.cpp -o probe_zero_blocks
#include "krk/gguf.hpp"

#include <cmath>
#include <cstdio>
#include <map>
#include <string>

using namespace krk;

int main(int argc, char **argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: probe_zero_blocks <model.gguf> [max_mib_per_tensor]\n");
        return 2;
    }
    const f64 max_mib = argc > 2 ? std::atof(argv[2]) : 64.0;

    Gguf g;
    std::string err;
    if (!g.load(argv[1], &err)) {
        std::fprintf(stderr, "load failed: %s\n", err.c_str());
        return 1;
    }

    std::printf("file: %s\n", argv[1]);
    std::printf("zero-block yield per dtype (a block is 256 or 32 values; a\n"
                "\"zero block\" is one whose decoded values are ALL exactly 0)\n\n");
    std::printf("  %-8s %8s %10s %10s %9s %9s\n", "dtype", "tensors",
                "blocks", "zeroblks", "blk%", "elem%");

    struct Acc {
        int tensors = 0;
        size_t blocks = 0, zero_blocks = 0;
        size_t elems = 0, zero_elems = 0;
    };
    std::map<DType, Acc> acc;

    for (const GgufTensor &t : g.tensors()) {
        if (!t.dtype_known || t.data == nullptr) continue;
        const int bs = dtype_block_size(t.type);
        const int bb = dtype_block_bytes(t.type);
        if (bs <= 0 || bb <= 0) continue;
        const i64 nblk = t.n_elements / bs;
        if (nblk <= 0) continue;

        Acc &a = acc[t.type];
        a.tensors++;

        // Cap the work: whole-tensor decoding of a 964 MB file is minutes.
        const i64 max_blocks =
            static_cast<i64>(max_mib * 1048576.0 / bb);
        const i64 scan = nblk < max_blocks ? nblk : max_blocks;

        std::vector<f32> out(static_cast<size_t>(bs));
        const u8 *p = t.data;
        for (i64 b = 0; b < scan; b++, p += bb) {
            dequant_row(t.type, p, out.data(), bs);
            bool all_zero = true;
            size_t z = 0;
            for (int i = 0; i < bs; i++) {
                if (out[i] != 0.0f) all_zero = false;
                else z++;
            }
            if (all_zero) a.zero_blocks++;
            a.blocks++;
            a.elems += static_cast<size_t>(bs);
            a.zero_elems += z;
        }
    }

    for (const auto &kv : acc) {
        const Acc &a = kv.second;
        std::printf("  %-8s %8d %10zu %10zu %8.3f%% %8.2f%%\n",
                    dtype_name(kv.first), a.tensors, a.blocks, a.zero_blocks,
                    100.0 * static_cast<f64>(a.zero_blocks) / static_cast<f64>(a.blocks),
                    100.0 * static_cast<f64>(a.zero_elems) / static_cast<f64>(a.elems));
    }
    return 0;
}