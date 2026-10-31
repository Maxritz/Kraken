// Multi-sequence attention must be bit-identical to one sequence at a time.
//
// This is the acceptance test for AttnDesc::row_pos / row_kv_base -- the change
// that lets one call carry rows from different sequences -- across the three
// ops that are not already row-independent: attention() (which reads the cache),
// kv_append() (which writes it) and rope() (which needs the row's own position,
// via Backend::rope_rows). Everything else in the forward pass is a per-row
// GEMM, norm or FFN and needs no knowledge of the batch.
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

    // ---- kv_append: one batched call must equal N per-sequence calls -------
    // Attention reads the cache through row_kv_base, so the WRITE has to honour
    // it too or a batched decode silently overwrites another sequence's row.
    // Both storage arms are checked: f32 (a memcpy into pos slots) and int8
    // (quantise into the row's own bytes plus its scale block).
    for (int int8 = 0; int8 <= 1; int8++) {
        // Element strides for the descriptor, BYTES for the buffers: the int8
        // row is 144 bytes for 128 values, the f32 row is 128 values of 4.
        const i64 row_elems = int8 ? (kHead * kHd + 4 * ((kHead * kHd) / 32))
                                   : kHead * kHd;
        const i64 region_elems = kRegionPos * row_elems;
        const i64 region_bytes = int8 ? region_elems : region_elems * 4;
        std::vector<f32> k_src(static_cast<size_t>(kRegions * kHead * kHd));
        std::vector<f32> v_src(k_src.size());
        for (auto &v : k_src) v = d(rng);
        for (auto &v : v_src) v = d(rng);

        // arm A: one call per sequence at its own pos0 and base offset
        std::vector<u8> a(static_cast<size_t>(region_bytes * kRegions), 0);
        std::vector<u8> av(a.size(), 0);
        for (i64 r = 0; r < kRegions; r++) {
            AttnDesc one;
            one.n_head = kHead;
            one.n_kv = kHead;
            one.hd = kHd;
            one.n_tok = 1;
            one.pos0 = pos[r];
            one.pos_stride = row_elems;
            one.layer_stride = region_elems * kRegions;
            one.kv_int8 = int8;
            be->kv_append(a.data() + r * region_bytes, av.data() + r * region_bytes,
                          k_src.data() + r * kHead * kHd,
                          v_src.data() + r * kHead * kHd, one);
        }

        // arm B: all four writes in ONE call, addressed per row
        std::vector<u8> b(static_cast<size_t>(region_bytes * kRegions), 0);
        std::vector<u8> bv(b.size(), 0);
        AttnDesc all;
        all.n_head = kHead;
        all.n_kv = kHead;
        all.hd = kHd;
        all.n_tok = kRegions;
        all.pos0 = 0;
        all.pos_stride = row_elems;
        all.layer_stride = region_elems * kRegions;
        all.kv_int8 = int8;
        std::vector<i64> rp(kRegions), rb(kRegions);
        for (i64 r = 0; r < kRegions; r++) {
            rp[r] = pos[r];
            rb[r] = r * region_elems; // this arm's element size, not the f32 one
        }
        all.row_pos = rp.data();
        all.row_kv_base = rb.data();
        be->kv_append(b.data(), bv.data(), k_src.data(), v_src.data(), all);

        const bool same = std::memcmp(a.data(), b.data(), a.size()) == 0 &&
                          std::memcmp(av.data(), bv.data(), av.size()) == 0;
        std::printf("%-22s %s\n", int8 ? "kv_append int8 batch" : "kv_append f32 batch",
                    same ? "same bytes as one call per sequence"
                         : "CHANGED -- the batched write is not the per-sequence write");
        if (!same) failures++;
    }

    // ---- rope: per-row positions must equal one call per row --------------
    {
        const i64 hd = 16, nh = 2, nkv = 2;
        const f32 inv[8] = {1.0f,       0.549306144f, 0.301716626f, 0.165780602f,
                            0.091042796f, 0.05f,       0.027461546f, 0.015077292f};
        const f32 frac = 1.0f, scale = 1.0f;
        std::vector<f32> q0(static_cast<size_t>(kRegions * nh * hd));
        std::vector<f32> k0(static_cast<size_t>(kRegions * nkv * hd));
        for (auto &v : q0) v = d(rng);
        for (auto &v : k0) v = d(rng);
        std::vector<f32> qa = q0, ka = k0, qb = q0, kb = k0;
        for (i64 r = 0; r < kRegions; r++) {
            be->rope(qa.data() + r * nh * hd, ka.data() + r * nkv * hd, nh, nkv, hd,
                     1, pos[r], inv, scale, frac, false);
        }
        std::vector<i64> rp(kRegions);
        for (i64 r = 0; r < kRegions; r++) rp[r] = pos[r];
        const bool have = be->rope_rows(qb.data(), kb.data(), nh, nkv, hd, kRegions,
                                        rp.data(), inv, scale, frac, false);
        if (!have) {
            std::printf("%-22s unsupported on this backend (refused, not guessed)\n",
                        "rope per-row positions");
            failures++;
        } else {
            const bool same =
                std::memcmp(qa.data(), qb.data(), qa.size() * sizeof(f32)) == 0 &&
                std::memcmp(ka.data(), kb.data(), ka.size() * sizeof(f32)) == 0;
            std::printf("%-22s %s\n", "rope per-row positions",
                        same ? "same bits as one call per row"
                             : "CHANGED -- rows rotated at the wrong position");
            if (!same) failures++;
        }
    }

    // ---- attn_sparse_rows: the run-list path must address the same rows ----
    // This one had its own copy of the KV address arithmetic and multiplied the
    // per-row region base by pos_stride as well (harmless only while no caller
    // set row_kv_base). A batch of rows at different positions against one run
    // list is exactly the shape that exposes it.
    {
        std::vector<f32> qa(static_cast<size_t>(kRegions * kHead * kHd));
        for (auto &v : qa) v = d(rng);
        const i64 run[2] = {0, kRegionPos};
        // arm A: one call per sequence, no per-row tables
        std::vector<f32> one(static_cast<size_t>(kRegions * kHead * kHd), 0.0f);
        for (i64 r = 0; r < kRegions; r++) {
            AttnDesc a;
            a.n_head = kHead;
            a.n_kv = kHead;
            a.hd = kHd;
            a.n_tok = 1;
            a.pos0 = pos[r];
            a.layer_stride = kRegionElems * kRegions;
            a.pos_stride = kPosStride;
            a.scale = 1.0f / std::sqrt(static_cast<f32>(kHd));
            a.causal = true;
            a.n_key_runs = 1;
            a.key_runs = run;
            be->attention(one.data() + r * kHead * kHd,
                          qa.data() + r * kHead * kHd,
                          kc.data() + r * kRegionElems,
                          vc.data() + r * kRegionElems, a);
        }
        // arm B: one call, four rows, per-row positions and bases
        std::vector<i64> rp(kRegions), rb(kRegions);
        for (i64 r = 0; r < kRegions; r++) {
            rp[r] = pos[r];
            rb[r] = r * kRegionElems;
        }
        AttnDesc b;
        b.n_head = kHead;
        b.n_kv = kHead;
        b.hd = kHd;
        b.n_tok = kRegions;
        b.layer_stride = kRegionElems * kRegions;
        b.pos_stride = kPosStride;
        b.scale = 1.0f / std::sqrt(static_cast<f32>(kHd));
        b.causal = true;
        b.n_key_runs = 1;
        b.key_runs = run;
        b.row_pos = rp.data();
        b.row_kv_base = rb.data();
        std::vector<f32> batched(one.size(), 0.0f);
        be->attention(batched.data(), qa.data(), kc.data(), vc.data(), b);
        const bool same =
            std::memcmp(one.data(), batched.data(), one.size() * sizeof(f32)) == 0;
        std::printf("%-22s %s\n", "sparse run-list batch",
                    same ? "same bits as one call per sequence"
                         : "CHANGED -- the run-list path addresses the wrong rows");
        if (!same) failures++;
    }

    std::printf(failures ? "MULTI-SEQUENCE BATCHING FAILED\n"
                         : "MULTI-SEQUENCE BATCHING IS BIT-EXACT\n");
    return failures ? 1 : 0;
}
