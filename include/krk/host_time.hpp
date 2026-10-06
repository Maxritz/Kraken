// host_time.hpp — the host half of the profiler.
//
// The device profiler (src/hip/op_time.hpp) times kernel spans with HIP events,
// but it lives in the HIP translation unit: engine.cpp is portable C++17 and
// cannot include a HIP header. That left whole phases unmeasured -- a
// speculative round, a prefill chunk, an expert promoted out of WARM, a KV
// layer migrated between tiers -- because no backend op delimits them.
//
// This is the same idea for host code, and it shares the switch: `--profile`
// (KRK_PROFILE=1) or KRK_TIME=1 turns it on and the table prints beside the op
// table at the end of the run. It is pure <chrono>, so any translation unit may
// use it, including the CPU-only build.
//
// Reading it: these are *wall* times, so a row that wraps device work includes
// the device time inside it. That is deliberate and is why the scope names say
// what they wrap ("spec.round" is a whole round). The device table answers
// "which kernel"; this one answers "which phase, and how much of the wall clock
// no phase accounts for".
#pragma once

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace krk {

class HostTime {
public:
    struct Row {
        double ms = 0.0;
        long long calls = 0;
        long long bytes = 0;
    };

    static HostTime &get() {
        static HostTime inst;
        return inst;
    }

    bool on() const { return on_; }

    void add(const char *name, double ms) {
        if (!on_) return;
        Row &r = rows_[name ? name : "?"];
        r.ms += ms;
        r.calls += 1;
    }

    void add_bytes(const char *name, long long b) {
        if (!on_ || b <= 0) return;
        rows_[name ? name : "?"].bytes += b;
    }

    // Sorted by total wall time, so the row that matters is first.
    void report(std::FILE *out) const {
        if (!on_ || rows_.empty()) return;
        std::vector<std::pair<const std::string *, const Row *>> order;
        order.reserve(rows_.size());
        double total = 0.0;
        for (const auto &kv : rows_) {
            order.push_back({&kv.first, &kv.second});
            total += kv.second.ms;
        }
        std::sort(order.begin(), order.end(),
                  [](const auto &a, const auto &b) { return a.second->ms > b.second->ms; });
        std::fprintf(out,
                     "\n[stats ] host time  (wall clock per phase; KRK_PROFILE to enable)\n"
                     "    %-22s %10s %8s %10s %9s\n",
                     "phase", "total ms", "calls", "ms/call", "% wall");
        for (const auto &e : order) {
            const Row &r = *e.second;
            const double per = r.calls > 0 ? r.ms / static_cast<double>(r.calls) : 0.0;
            const double pc = total > 0.0 ? 100.0 * r.ms / total : 0.0;
            std::fprintf(out, "    %-22s %10.1f %8lld %10.3f %8.1f%%",
                         e.first->c_str(), r.ms, r.calls, per, pc);
            if (r.bytes > 0) {
                const double mib = static_cast<double>(r.bytes) / 1048576.0;
                const double gbs = r.ms > 0.0 ? mib / 1024.0 / (r.ms / 1000.0) : 0.0;
                std::fprintf(out, "  %8.1f MiB %7.1f GB/s", mib, gbs);
            }
            std::fprintf(out, "\n");
        }
    }

private:
    HostTime() {
        const char *p = std::getenv("KRK_PROFILE");
        const char *t = std::getenv("KRK_TIME");
        on_ = (p && *p && std::strcmp(p, "0") != 0) ||
              (t && *t && std::strcmp(t, "0") != 0);
    }

    bool on_ = false;
    std::map<std::string, Row> rows_;
};

// RAII guard: one per phase. Every exit path (including the `return false` ones
// in the speculative loops) closes its span, which a manual pair of calls would
// have leaked.
class HostScope {
public:
    explicit HostScope(const char *name) : name_(name), on_(HostTime::get().on()) {
        if (on_) t0_ = std::chrono::steady_clock::now();
    }
    ~HostScope() {
        if (!on_) return;
        const double ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - t0_)
                              .count();
        HostTime::get().add(name_, ms);
    }
    HostScope(const HostScope &) = delete;
    HostScope &operator=(const HostScope &) = delete;

    void bytes(long long b) {
        if (on_) HostTime::get().add_bytes(name_, b);
    }

private:
    const char *name_ = nullptr;
    bool on_ = false;
    std::chrono::steady_clock::time_point t0_{};
};

// One per scope. KRK_TIME_HOST uses the default guard name; use KRK_TIME_HOST_N
// when two scopes have to nest in the same block.
#define KRK_TIME_HOST(nm) ::krk::HostScope krk_host_scope(nm)
#define KRK_TIME_HOST_N(nm, var) ::krk::HostScope var(nm)

} // namespace krk
