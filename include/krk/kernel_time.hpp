// krk/kernel_time.hpp -- per-COMPONENT device time, ranked, in the one format
// that answers "what is fucking it up".
//
// The HIP recorder (src/hip/op_time.hpp) does this with hipEvents, which only
// exist on that backend. This is the same table for every backend: each entry
// point in the backend names itself, and the times are accumulated per name.
//
// What a row means:
//   ops      how many times the component ran
//   dev_us   total wall time inside it, summed across threads that ran in
//            PARALLEL, so it can exceed the run. That is deliberate: the
//            question is which component eats the machine, not which one is on
//            the critical path.
//   us/op    dev_us / ops -- the number to compare a kernel against its own
//            roofline. A row with a large total and a small us/op is called too
//            often; a row with a large us/op is slow.
//
// Turn it on with --kprof (or KRK_KPROF=1). Off, a scope is a clock read that
// the branch predictor eats.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <string>

#include "krk/common.hpp"

namespace krk {

class KernelTime {
public:
    static bool enabled() {
        static const bool on = [] {
            const char *e = std::getenv("KRK_KPROF");
            return e && e[0] != '\0' && e[0] != '0';
        }();
        return on;
    }

    // Bounded so a pathological run cannot allocate without limit. Past the
    // cap a new name is dropped and counted, rather than silently shared with
    // an existing row -- two components in one row is a wrong answer.
    static constexpr int kMax = 160;

    struct Row {
        const char *name = nullptr;
        uint64_t ops = 0;
        double us = 0.0;
    };

    static KernelTime &get() {
        static KernelTime inst;
        return inst;
    }

    inline void add(const char *name, double us) {
        for (int i = 0; i < n_; i++) {
            if (rows_[i].name == name) {
                rows_[i].ops++;
                rows_[i].us += us;
                return;
            }
        }
        if (n_ >= kMax) {
            dropped_++;
            return;
        }
        rows_[n_].name = name;
        rows_[n_].ops = 1;
        rows_[n_].us = us;
        n_++;
    }

    void report(const char *title) const {
        if (n_ == 0) {
            std::fprintf(stderr, "\n[ktime] no component was timed (nothing ran?)\n");
            return;
        }
        // Ranked by total device time: the table reads top-down as "this is
        // eating the run".
        int idx[kMax];
        for (int i = 0; i < n_; i++) idx[i] = i;
        std::sort(idx, idx + n_, [&](int a, int b) { return rows_[a].us > rows_[b].us; });
        double total = 0;
        for (int i = 0; i < n_; i++) total += rows_[i].us;
        std::fprintf(stderr, "\n[ktime] %s -- device time per component, ranked\n", title);
        std::fprintf(stderr, "  %-26s %8s %14s %10s %7s\n", "component", "ops",
                     "dev_us", "us/op", "%run");
        for (int k = 0; k < n_; k++) {
            const Row &r = rows_[idx[k]];
            std::fprintf(stderr, "  %-26s %8llu %14.1f %10.1f %6.1f%%\n", r.name,
                         static_cast<unsigned long long>(r.ops), r.us,
                         r.us / static_cast<double>(r.ops),
                         100.0 * r.us / (total > 0 ? total : 1.0));
        }
        if (dropped_)
            std::fprintf(stderr, "  (%llu call(s) dropped: more than %d distinct names)\n",
                         static_cast<unsigned long long>(dropped_), kMax);
    }

    void reset() { n_ = 0; dropped_ = 0; }

private:
    Row rows_[kMax];
    int n_ = 0;
    uint64_t dropped_ = 0;
};

// RAII. When the recorder is off the name is null and the destructor does
// nothing, so the cost is one predictable branch per backend call.
class KernelScope {
public:
    explicit inline KernelScope(const char *name)
        : name_(KernelTime::enabled() ? name : nullptr) {
        if (name_) t0_ = now_us();
    }
    inline ~KernelScope() {
        if (name_) KernelTime::get().add(name_, static_cast<double>(now_us() - t0_));
    }
    KernelScope(const KernelScope &) = delete;
    KernelScope &operator=(const KernelScope &) = delete;

private:
    static inline uint64_t now_us() {
        timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return static_cast<uint64_t>(ts.tv_sec) * 1000000ull +
               static_cast<uint64_t>(ts.tv_nsec) / 1000ull;
    }
    const char *name_;
    uint64_t t0_ = 0;
};

// The one macro a backend uses. `nm` is a string LITERAL so the row's name
// pointer is stable and the accumulation is a pointer compare.
#define KRK_KTIME(nm) ::krk::KernelScope krk_kscope_(nm)

} // namespace krk
