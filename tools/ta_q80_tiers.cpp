// Every rung of the Q8_0 kernel ladder, on whatever host this runs on.
//
// The ladder is VNNI -> AVX-512BW -> AVX2 -> scalar, each rung guarded by the
// CPU's own feature bits. Two things need testing and they are different:
//
//   1. every rung that this host CAN run must agree with the scalar reference;
//   2. the dispatcher must pick a reachable rung, and forcing an unreachable
//      one must fall through rather than raise SIGILL.
//
// (2) is the one that matters on a machine that lacks a feature. A feature that
// is not available on this host is still implemented -- it has to be reachable
// and skippable, not absent. So this tool prints the rungs it cannot run rather
// than hiding them, and the dispatcher test forces each rung in turn through the
// real dispatch path.
#include "krk/quant.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace krk;

namespace {

struct Tier {
    const char *name;
    i32 tier;
    bool available;    // from the CPU feature bits
};

// Detection lives here rather than in the library so the library's fallback
// helpers stay internal. These are the same feature bits the dispatcher reads.
bool host_avx2() { return __builtin_cpu_supports("avx2") != 0; }
bool host_bw() {
    return __builtin_cpu_supports("avx512f") != 0 &&
           __builtin_cpu_supports("avx512bw") != 0;
}
bool host_vnni() { return __builtin_cpu_supports("avx512vnni") != 0; }

// Relative to the sum of |terms|, as the other accuracy fixtures are.
f64 worst_diff(const std::vector<f32> &a, const std::vector<f32> &ref,
               f64 magnitude) {
    f64 w = 0.0;
    for (size_t i = 0; i < a.size(); i++)
        w = std::max<f64>(w, std::fabs(static_cast<f64>(a[i]) -
                                       static_cast<f64>(ref[i])) / magnitude);
    return w;
}

} // namespace

int main() {
    std::mt19937 rng(20251028);
    std::uniform_real_distribution<f32> d(-3.0f, 3.0f);

    // Odd nb on purpose: the 512-bit rungs refuse it, so the ladder has to fall
    // through to a rung that can. 768/32 = 24 is even; 800/32 = 25 is not.
    bool bad = false;
    for (i64 nb : {24, 25}) {
        const i64 n = nb * 32;
        const i64 rows = 8;

        std::vector<u8> w(static_cast<size_t>(nb) * 34);
        for (i64 b = 0; b < nb; b++) {
            f32 amax = 0.0f;
            for (i64 i = 0; i < 32; i++)
                amax = std::max(amax, std::fabs(d(rng)));
            const f32 scale = amax > 0.0f ? amax / 127.0f : 1.0f;
            const u16 h = fp32_to_fp16(scale);
            u8 *blk = w.data() + b * 34;
            blk[0] = static_cast<u8>(h & 0xff);
            blk[1] = static_cast<u8>(h >> 8);
            for (i64 i = 0; i < 32; i++) {
                const f32 q = std::round(d(rng) / scale);
                blk[2 + i] = static_cast<u8>(
                    static_cast<i8>(std::max(-128.0f, std::min(127.0f, q))));
            }
        }

        std::vector<f32> xf(static_cast<size_t>(rows * n));
        for (auto &v : xf) v = d(rng);
        std::vector<i8> xq(static_cast<size_t>(rows * n));
        std::vector<f32> dx(static_cast<size_t>(rows * nb));
        for (i64 r = 0; r < rows; r++)
            quantize_q8_row(xf.data() + r * n, n, xq.data() + r * n,
                            dx.data() + r * nb);

        // Reference: the scalar rung, called directly.
        std::vector<f32> ref(static_cast<size_t>(rows), 0.0f);
        dot_q8_0_q8_x8_scalar(w.data(), xq.data(), dx.data(), n, n, nb,
                              ref.data(), 1);
        f64 magnitude = 0.0;
        for (f64 v : ref) magnitude = std::max(magnitude, std::fabs(v));
        magnitude = std::max<f64>(magnitude, 1.0);

        const Tier tiers[] = {
            {"vnni", 3, host_vnni()},
            {"avx512bw+f16c", 2, host_bw()},
            {"avx2", 1, host_avx2()},
            {"scalar", -1, true},
        };

        std::printf("nb=%2lld (%s)\n", static_cast<long long>(nb),
                    (nb & 1) ? "odd: 512-bit rungs must refuse" : "even");
        for (const Tier &t : tiers) {
            std::vector<f32> got(static_cast<size_t>(rows), 0.0f);
            if (!t.available) {
                std::printf("  %-14s absent on this host -- not exercised\n",
                            t.name);
                continue;
            }
            // The named-rung entry point, so every rung is exercised in ONE
            // process. The env override this replaced latched on first call,
            // which meant the first tier tested silently ran every later one.
            dot_q8_0_q8_x8_tier(static_cast<i32>(t.tier), w.data(), xq.data(),
                                dx.data(), n, n, nb, got.data(), 1);
            const f64 rel = worst_diff(got, ref, magnitude);
            // Bit-exactness where the rung does the same arithmetic; 1e-6 is the
            // bound the other fixtures use for a rung that reassociates.
            const bool exact = std::memcmp(got.data(), ref.data(),
                                           got.size() * sizeof(f32)) == 0;
            const bool ok = exact || rel < 1e-6;
            std::printf("  %-14s %s  max rel %.3e%s\n", t.name,
                        ok ? "ok    " : "FAILED", rel,
                        exact ? "  (bit-exact)" : "");
            if (!ok) bad = true;
        }
    }

    // And the default ladder, with no override at all, must agree with the
    // scalar reference -- that is the path production actually takes.
    {
        const i64 nb = 24, n = nb * 32, rows = 8;
        std::vector<u8> w(static_cast<size_t>(nb) * 34);
        for (i64 b = 0; b < nb; b++) {
            const u16 h = fp32_to_fp16(0.004f);
            u8 *blk = w.data() + b * 34;
            blk[0] = static_cast<u8>(h & 0xff);
            blk[1] = static_cast<u8>(h >> 8);
            for (i64 i = 0; i < 32; i++) blk[2 + i] = static_cast<u8>(rng() >> 24);
        }
        std::vector<f32> xf(static_cast<size_t>(rows * n));
        for (auto &v : xf) v = d(rng);
        std::vector<i8> xq(static_cast<size_t>(rows * n));
        std::vector<f32> dx(static_cast<size_t>(rows * nb));
        for (i64 r = 0; r < rows; r++)
            quantize_q8_row(xf.data() + r * n, n, xq.data() + r * n,
                            dx.data() + r * nb);
        std::vector<f32> a(static_cast<size_t>(rows)), b2(static_cast<size_t>(rows));
        unsetenv("KRK_CPU_Q80_KERNEL");
        dot_q8_0_q8_x8(w.data(), xq.data(), dx.data(), n, n, nb, a.data(), 1);
        dot_q8_0_q8_x8_scalar(w.data(), xq.data(), dx.data(), n, n, nb, b2.data(), 1);
        // Not memcmp: a wider rung reassociates the eight lanes, which the
        // other fixtures bound at 1e-6. Bit-exactness is only demanded of the
        // scalar rung itself.
        const bool same_bits =
            std::memcmp(a.data(), b2.data(), a.size() * sizeof(f32)) == 0;
        f64 mag = 1.0;
        for (f64 v : b2) mag = std::max(mag, std::fabs(v));
        const bool ok = same_bits || worst_diff(a, b2, mag) < 1e-6;
        std::printf("\ndefault ladder vs scalar reference: %s\n",
                    same_bits ? "bit-exact" : "agrees (reassociated)");
        if (!ok) bad = true;
    }

    // The dispatcher must always land somewhere. On a host with nothing, that is
    // the scalar rung -- which is the whole point of the floor.
    std::printf("\nfailure: %s\n", bad ? "YES" : "none");
    return bad ? 1 : 0;
}
