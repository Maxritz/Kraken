// Attention accuracy harness: dumps the raw f32 output of CpuBackend::attention
// over ten shapes so two builds can be compared bit-for-bit.
//
//   KRK_CPU_ATTN_SCALAR=1 ./ta out_scalar.bin
//                          ./ta out_vec.bin
//
// Shapes cover decode at pos 1400, a 64-token prefill, GQA, hd 80/96/256 (the
// tails), a sliding window, the skip_from masked block, sinks and softcap.
#include "krk/backend.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace krk;

static u64 g_rng = 0x9E3779B97F4A7C15ull;
static f32 rnd() {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 7;
    g_rng ^= g_rng << 17;
    return static_cast<f32>((g_rng >> 40) & 0xFFFFFF) / 16777216.0f * 2.0f - 1.0f;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s out.bin\n", argv[0]);
        return 2;
    }
    Backend *be = make_cpu_backend();
    FILE *out = std::fopen(argv[1], "wb");
    if (!out) return 3;

    struct Shape {
        const char *name;
        i64 n_head, n_kv, hd, n_tok, pos0, window, skip_from;
        f32 softcap;
        bool sinks;
    };
    const Shape shapes[] = {
        {"decode@1400", 12, 12, 64, 1, 1400, 0, -1, 0.0f, false},
        {"prefill64", 12, 12, 64, 64, 0, 0, -1, 0.0f, false},
        {"prefill64@pos512", 12, 12, 64, 64, 512, 0, -1, 0.0f, false},
        {"gqa8/2 hd128", 8, 2, 128, 33, 200, 0, -1, 0.0f, false},
        {"hd80", 6, 6, 80, 17, 64, 0, -1, 0.0f, false},
        {"hd96", 6, 6, 96, 17, 64, 0, -1, 0.0f, false},
        {"hd256", 4, 4, 256, 9, 128, 0, -1, 0.0f, false},
        {"window64", 12, 12, 64, 40, 200, 64, -1, 0.0f, false},
        {"skip_from30", 12, 12, 64, 40, 0, 0, 30, 0.0f, false},
        {"sinks+softcap", 12, 12, 64, 24, 80, 0, -1, 50.0f, true},
    };

    for (const Shape &s : shapes) {
        const i64 kv_dim = s.n_kv * s.hd;
        const i64 n_keys = s.pos0 + s.n_tok;
        const i64 layer_stride = n_keys * kv_dim;
        std::vector<f32> kc(static_cast<size_t>(layer_stride));
        std::vector<f32> vc(static_cast<size_t>(layer_stride));
        std::vector<f32> q(static_cast<size_t>(s.n_tok * s.n_head * s.hd));
        std::vector<f32> o(static_cast<size_t>(s.n_tok * s.n_head * s.hd), 0.0f);
        std::vector<f32> sinks(static_cast<size_t>(s.n_head));

        for (f32 &v : kc) v = rnd() * 0.5f;
        for (f32 &v : vc) v = rnd() * 0.5f;
        for (f32 &v : q) v = rnd() * 0.5f;
        for (i64 h = 0; h < s.n_head; h++)
            sinks[static_cast<size_t>(h)] = (h % 3 == 0) ? 0.0f : 1.5f - 0.1f * h;

        AttnDesc d;
        d.n_head = s.n_head;
        d.n_kv = s.n_kv;
        d.hd = s.hd;
        d.n_tok = s.n_tok;
        d.pos0 = s.pos0;
        d.layer = 0;
        d.layer_stride = layer_stride;
        d.pos_stride = kv_dim;
        d.scale = 1.0f / std::sqrt(static_cast<f32>(s.hd));
        d.causal = true;
        d.window = s.window;
        d.skip_from = s.skip_from;
        d.softcap = s.softcap;
        d.sinks = s.sinks ? sinks.data() : nullptr;

        be->attention(o.data(), q.data(), kc.data(), vc.data(), d);
        std::fwrite(o.data(), sizeof(f32), o.size(), out);
        std::fprintf(stderr, "%-18s heads %2lld/%lld hd %3lld tok %3lld pos %4lld "
                             "window %3lld skip %3lld -> %zu floats\n",
                     s.name, (long long)s.n_head, (long long)s.n_kv,
                     (long long)s.hd, (long long)s.n_tok, (long long)s.pos0,
                     (long long)s.window, (long long)s.skip_from, o.size());
    }
    std::fclose(out);
    delete be;
    return 0;
}
