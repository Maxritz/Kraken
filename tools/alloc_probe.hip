// What does ONE hipMalloc cost the card?
//
// This exists because the engine's expert pool is sized in REQUESTED bytes and
// the driver charges GRANULES: on gfx1201 a pool of 1,155 experts (3 slices
// each, ~1.97 MiB per slice) requested 7,248 MiB while `hipMemGetInfo` reported
// 7,000 MiB more than the allocator's own total. The pool's accounting was
// right to the MiB, so the difference is the driver's rounding, and a budget
// measured in requested bytes therefore cannot know how many experts fit.
//
// Build and run (no project headers: this measures the RUNTIME, not the engine):
//   hipcc -O2 -o /tmp/alloc_probe.exe tools/alloc_probe.hip --offload-arch=gfx1201
//   /tmp/alloc_probe.exe
//
// It prints, per requested size: the charge per allocation, the waste as a
// fraction, and the granularity that explains it. Run it on a quiet card; the
// reading is device-wide.
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

static size_t free_bytes() {
    size_t f = 0, t = 0;
    hipMemGetInfo(&f, &t);
    return f;
}

int main() {
    const int kCount = 64;
    const size_t kMib = 1024 * 1024;
    const size_t sizes[] = {1 * kMib, 1972 * 1024, 3 * kMib, 6 * kMib,
                            8 * kMib, 16 * kMib, 32 * kMib, 64 * kMib};

    std::printf("%-12s %12s %12s %8s\n", "requested", "charged/alloc", "waste",
                "waste %");
    for (size_t want : sizes) {
        const size_t before = free_bytes();
        std::vector<void *> p(kCount, nullptr);
        int got = 0;
        for (int i = 0; i < kCount; i++) {
            if (hipMalloc(&p[i], want) != hipSuccess) {
                p[i] = nullptr;
                break;
            }
            got++;
        }
        if (got == 0) {
            std::printf("%-12zu   (hipMalloc refused)\n", want);
            for (void *q : p)
                if (q) hipFree(q);
            continue;
        }
        // The driver may defer the charge until the memory is touched; a 4-byte
        // write per allocation is enough to commit a page and costs nothing to
        // interpret: an untouched reading would report 0 waste.
        for (int i = 0; i < got; i++)
            hipMemsetAsync(p[i], 0, 4, nullptr);
        hipDeviceSynchronize();
        const size_t after = free_bytes();
        const double charged =
            before > after ? double(before - after) / double(got) : 0.0;
        const double waste = charged - double(want);
        std::printf("%-12zu %12.0f %12.0f %7.1f%%\n", want, charged, waste,
                    want ? 100.0 * waste / double(want) : 0.0);
        for (void *q : p)
            if (q) hipFree(q);
        hipDeviceSynchronize();
    }

    // The shape that actually mattered: the pool's ~1.97 MiB slice, in the
    // count it really uses (three per expert), with every byte written. A
    // reading at 64 buffers and a 4-byte touch said "no waste"; the engine
    // reports 2 MiB per buffer at 3,465 of them. Both cannot be true, so the
    // probe repeats the count and the touch and lets the numbers disagree.
    std::printf("\nslice shape (1,972 KiB, fully written):\n");
    for (int count : {64, 256, 1024, 3465}) {
        const size_t want = 1972 * 1024;
        const size_t before = free_bytes();
        std::vector<void *> p;
        p.reserve(static_cast<size_t>(count));
        for (int i = 0; i < count; i++) {
            void *q = nullptr;
            if (hipMalloc(&q, want) != hipSuccess) break;
            p.push_back(q);
        }
        for (void *q : p) hipMemsetAsync(q, 1, want, nullptr);
        hipDeviceSynchronize();
        const size_t after = free_bytes();
        const double charged =
            p.empty() ? 0.0 : double(before > after ? before - after : 0) /
                                double(p.size());
        std::printf("  %5zu allocations: charged %.0f B each (want %zu), "
                    "waste %.0f B each, total waste %.1f MiB\n",
                    p.size(), charged, want, charged - double(want),
                    double(p.size()) * (charged - double(want)) / 1048576.0);
        for (void *q : p) hipFree(q);
        hipDeviceSynchronize();
        // Does the driver GIVE IT BACK? The engine's pool trims cells while it
        // runs, and a free that is only bookkeeping would leave the card full
        // while the allocator's own total reads empty -- which is exactly the
        // disagreement `unaccounted` reports. Recovery is reported rather than
        // assumed because the next line's `before` is the only place it shows.
        const size_t recovered = free_bytes();
        std::printf("            after freeing: %.1f MiB free, %+.1f MiB vs before\n",
                    double(recovered) / 1048576.0,
                    double(recovered) / 1048576.0 - double(before) / 1048576.0);
    }
    return 0;
}
