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
        // Work done, when the caller knows it. A time tells you which component
        // eats the run; only macs tells you whether the kernel that eats the
        // run is slow or merely busy. peak is the same unit per second that
        // this host's ISA can sustain for this kind of work, so the %peak
        // column answers "could a better function do this in less time".
        double macs = 0.0;
        double peak_gmacs = 0.0;   // GMAC/s this host sustains for this class
    };

    static KernelTime &get() {
        static KernelTime inst;
        return inst;
    }

    inline void add(const char *name, double us, double macs = 0.0,
                    double peak_gmacs = 0.0) {
        for (int i = 0; i < n_; i++) {
            if (rows_[i].name == name) {
                rows_[i].ops++;
                rows_[i].us += us;
                rows_[i].macs += macs;
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
        rows_[n_].macs = macs;
        rows_[n_].peak_gmacs = peak_gmacs;
        n_++;
    }

    // Printed under the table so a %peak number can be checked against the
    // measurement it came from rather than a datasheet.
    inline void set_roofline(double int8_gmacs, double f32_gmacs) {
        int8_gmacs_ = int8_gmacs;
        f32_gmacs_ = f32_gmacs;
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
        std::fprintf(stderr, "  %-26s %8s %14s %10s %7s %9s %8s\n", "component",
                     "ops", "dev_us", "us/op", "%run", "GMAC/s", "%peak");
        for (int k = 0; k < n_; k++) {
            const Row &r = rows_[idx[k]];
            const double gm = r.macs > 0 && r.us > 0 ? r.macs / r.us / 1e3 : 0.0;
            char peakbuf[16];
            if (r.peak_gmacs > 0 && gm > 0)
                std::snprintf(peakbuf, sizeof peakbuf, "%7.1f%%",
                              100.0 * gm / r.peak_gmacs);
            else
                std::snprintf(peakbuf, sizeof peakbuf, "%8s", "-");
            std::fprintf(stderr, "  %-26s %8llu %14.1f %10.1f %6.1f%% %9.1f %s\n",
                         r.name, static_cast<unsigned long long>(r.ops), r.us,
                         r.us / static_cast<double>(r.ops),
                         100.0 * r.us / (total > 0 ? total : 1.0), gm, peakbuf);
        }
        std::fprintf(stderr,
                     "  tracked total: %.1f ms across %d component(s) -- model load, "
                     "sampling and tokenisation are outside every row\n",
                     total / 1e3, n_);
        if (int8_gmacs_ > 0)
            std::fprintf(stderr,
                         "  roofline probe: int8 %.0f GMAC/s, f32 %.0f GMAC/s "
                         "(this host, all cores)\n",
                         int8_gmacs_, f32_gmacs_);
        if (dropped_)
            std::fprintf(stderr, "  (%llu call(s) dropped: more than %d distinct names)\n",
                         static_cast<unsigned long long>(dropped_), kMax);
    }

    void reset() { n_ = 0; dropped_ = 0; }

private:
    Row rows_[kMax];
    int n_ = 0;
    uint64_t dropped_ = 0;
    double int8_gmacs_ = 0.0;
    double f32_gmacs_ = 0.0;
};

// RAII. When the recorder is off the name is null and the destructor does
// nothing, so the cost is one predictable branch per backend call.
class KernelScope {
public:
    explicit inline KernelScope(const char *name, double macs = 0.0,
                                double peak_macs = 0.0)
        : name_(KernelTime::enabled() ? name : nullptr),
          macs_(macs),
          peak_(peak_macs) {
        if (name_) t0_ = now_us();
    }
    inline ~KernelScope() {
        if (name_)
            KernelTime::get().add(name_, static_cast<double>(now_us() - t0_),
                                  macs_, peak_);
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
    double macs_;
    double peak_;
    uint64_t t0_ = 0;
};

// The one macro a backend uses. `nm` is a string LITERAL so the row's name
// pointer is stable and the accumulation is a pointer compare. The MAC form
// adds the roofline columns; the plain form is time-only.
#define KRK_KTIME(nm) ::krk::KernelScope krk_kscope_(nm)
#define KRK_KTIME_MAC(nm, macs, peak) \
    ::krk::KernelScope krk_kscope_(nm, (macs), (peak))

} // namespace krk
