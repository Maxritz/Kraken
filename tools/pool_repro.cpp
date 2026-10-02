// Temporary diagnostic for the flaky par_pool coverage test. Not part of the
// build; compile by hand on the host that has the pool header:
//   hipcc -O2 -std=c++17 -I include -I src tools/pool_repro.cpp -o /tmp/pool_repro -lpthread
// It replays the exact shape of test_par_pool_covers_every_block() but prints
// which task failed and how the coverage broke down, instead of a bare CHECK.
#include <cstdio>
#include <cstdint>
#include <vector>
#include <algorithm>
#include "../src/par_pool.hpp"

using namespace krk;

int main(int argc, char **argv) {
    const int iters = argc > 1 ? std::atoi(argv[1]) : 400;
    u64 rng = 0x9e3779b97f4a7c15ull;
    auto next = [&](u64 m) {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        return static_cast<i64>(rng % m);
    };
    std::vector<i32> hits(4096, 0);
    for (int task = 0; task < iters; task++) {
        const i64 n = 1 + next(4000);
        const i64 grain = 1 + next(static_cast<u64>(std::min<i64>(64, n)));
        const i64 blocks = (n + grain - 1) / grain;
        std::fill(hits.begin(), hits.begin() + static_cast<size_t>(n), 0);
        Pool::get().run(n, grain, [&](i64 b, i64 e) {
            for (i64 i = b; i < e; i++) hits[static_cast<size_t>(i)]++;
        });
        i64 covered = 0, doubled = 0, zero = 0;
        for (i64 i = 0; i < n; i++) {
            if (hits[static_cast<size_t>(i)] == 1) covered++;
            else if (hits[static_cast<size_t>(i)] > 1) doubled++;
            else zero++;
        }
        if (covered != n || doubled != 0) {
            std::printf("TASK %d BROKE  n=%lld grain=%lld blocks=%lld "
                        "covered=%lld zero=%lld doubled=%lld workers=%lld\n",
                        task, (long long)n, (long long)grain, (long long)blocks,
                        (long long)covered, (long long)zero, (long long)doubled,
                        (long long)Pool::get().workers());
            // Show the first few mis-covered indices and their hit counts.
            int shown = 0;
            for (i64 i = 0; i < n && shown < 12; i++) {
                if (hits[static_cast<size_t>(i)] != 1) {
                    std::printf("   idx %lld hits=%d\n", (long long)i,
                                hits[static_cast<size_t>(i)]);
                    shown++;
                }
            }
            return 1;
        }
    }
    std::printf("all %d tasks clean\n", iters);
    return 0;
}