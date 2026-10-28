// Multi-sequence attention must be bit-identical to one sequence at a time.
//
// This is the acceptance test for AttnDesc::row_pos / row_kv_base, the change
// that lets one attention() call carry rows from different sequences. Every
// other kernel in the engine is already row-independent, so this is the piece
// that gates batching several sequences into one forward pass.
//
// The gate is deliberately the strongest one available: a batch of N rows with
// per-row positions and per-row KV bases must produce the SAME BITS as N calls
// of one row each against its own region. No tolerance, no ratio -- `==` on the
// output floats. If a row's key range, its causal bound, or its cache base is
// off by one, the comparison is the thing that notices.
#include "krk/backend.hpp"
#include "krk/common.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace krk;

namespace {

// KV layout for the test: [region][position][head][hd], one layer, f32.
// A "sequence" owns one region, so region bases are what row_kv_base carries.
constexpr i64 kHead = 4;
constexpr i64 kHd = 32;
constexpr i64 kPosStride = kHead * kHd;
constexpr i64 kRegionPos = 64;                       // positions per region
constexpr i64 kRegionElems = kRegionPos * kPosStride;  // elements per region
constexpr i64 kRegions = 4;

struct Case {
    const char *name;
    bool causal;
};

} // namespace

int main() {
    Backend *be = make_cpu_backend();
    if (!be) {
        std::printf("no CPU backend\n");
        return 1;
    }

    std::mt19937 rng(20251028);
    std::uniform_real_distribution<f32> d(-1.0f, 1.0f);

    // One layer, kRegions separate sequences' worth of keys and values.
    std::vector<f32> kc(static_cast<size_t>(kRegionElems * kRegions));
    std::vector<f32> vc(static_cast<size_t>(kRegionElems * kRegions));
    for (auto &v : kc) v = d(rng);
    for (auto &v : vc) v = d(rng);

    // Four rows, one per sequence, each at a DIFFERENT point in its history --
    // the case a single pos0 cannot express.
    const i64 pos[kRegions] = {3, 17, 40, 63};
    std::vector<f32> q(static_cast<size_t>(kRegions * kHead * kHd));
    for (auto &v : q) v = d(rng);

    const Case cases[] = {{"causal", true}, {"non-causal (prefix)", false}};
    int failures = 0;

    for (const Case &cs : cases) {
        // ---- arm A: one call per sequence, from its own base -------------
        std::vector<f32> one(static_cast<size_t>(kRegions * kHead * kHd), 0.0f);
        for (i64 r = 0; r < kRegions; r++) {
            AttnDesc a;
            a.n_head = kHead;
            a.n_kv = kHead;
            a.hd = kHd;
            a.n_tok = 1;
            a.pos0 = pos[r];
            a.layer = 0;
            a.layer_stride = kRegionElems * kRegions;
            a.pos_stride = kPosStride;
            a.scale = 1.0f / std::sqrt(static_cast<f32>(kHd));
            a.causal = cs.causal;
            be->attention(one.data() + r * kHead * kHd,
                          q.data() + r * kHead * kHd,
                          kc.data() + r * kRegionElems,
                          vc.data() + r * kRegionElems, a);
        }

        // ---- arm B: all four rows in ONE call ----------------------------
        std::vector<f32> batched(static_cast<size_t>(kRegions * kHead * kHd), 0.0f);
        std::vector<i64> row_pos(kRegions), row_base(kRegions);
        for (i64 r = 0; r < kRegions; r++) {
            row_pos[r] = pos[r];
            row_base[r] = r * kRegionElems;
        }
        AttnDesc b;
        b.n_head = kHead;
        b.n_kv = kHead;
        b.hd = kHd;
        b.n_tok = kRegions;
        b.pos0 = 0;  // unused when row_pos is set; left at 0 to prove it
        b.layer = 0;
        b.layer_stride = kRegionElems * kRegions;
        b.pos_stride = kPosStride;
        b.scale = 1.0f / std::sqrt(static_cast<f32>(kHd));
        b.causal = cs.causal;
        b.row_pos = row_pos.data();
        b.row_kv_base = row_base.data();
        be->attention(batched.data(), q.data(), kc.data(), vc.data(), b);

        // ---- compare: bit for bit ---------------------------------------
        i64 bad = 0;
        double worst = 0.0;
        for (size_t i = 0; i < one.size(); i++) {
            if (std::memcmp(&one[i], &batched[i], sizeof(f32)) != 0) {
                bad++;
                worst = std::max<double>(
                    worst, std::fabs(static_cast<double>(one[i]) -
                                     static_cast<double>(batched[i])));
            }
        }
        std::printf("%-22s %zu values, %lld differing", cs.name, one.size(),
                    static_cast<long long>(bad));
        if (bad)
            std::printf("  worst |diff| = %.3e  FAIL", worst);
        std::printf("\n");
        if (bad) failures++;
    }

    // ---- and one shape that proves the OLD path still works --------------
    // With no per-row table the descriptor must behave exactly as before: two
    // contiguous rows in one call == the same two rows computed separately.
    std::vector<f32> seq(static_cast<size_t>(2 * kHead * kHd), 0.0f);
    AttnDesc s0;
    s0.n_head = kHead;
    s0.n_kv = kHead;
    s0.hd = kHd;
    s0.n_tok = 2;
    s0.pos0 = 7;
    s0.layer_stride = kRegionElems * kRegions;
    s0.pos_stride = kPosStride;
    s0.scale = 1.0f / std::sqrt(static_cast<f32>(kHd));
    s0.causal = true;
    be->attention(seq.data(), q.data(), kc.data(), vc.data(), s0);

    std::vector<f32> sep(static_cast<size_t>(2 * kHead * kHd), 0.0f);
    for (i64 t = 0; t < 2; t++) {
        AttnDesc a;
        a.n_head = kHead;
        a.n_kv = kHead;
        a.hd = kHd;
        a.n_tok = 1;
        a.pos0 = 7 + t;
        a.layer_stride = kRegionElems * kRegions;
        a.pos_stride = kPosStride;
        a.scale = s0.scale;
        a.causal = true;
        be->attention(sep.data() + t * kHead * kHd, q.data() + t * kHead * kHd,
                      kc.data(), vc.data(), a);
    }
    const bool same = std::memcmp(seq.data(), sep.data(),
                                  seq.size() * sizeof(f32)) == 0;
    std::printf("%-22s %s\n", "legacy contiguous path",
                same ? "unchanged by the new fields" : "CHANGED -- regression");
    if (!same) failures++;

    std::printf(failures ? "MULTI-SEQUENCE ATTENTION FAILED\n"
                         : "MULTI-SEQUENCE ATTENTION IS BIT-EXACT\n");
    return failures ? 1 : 0;
}
