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
//
// Two tables come out of here, and they answer different questions:
//
//   `host time`     one row per PHASE, sorted by wall time. This is the detail
//                   table: every scope in the engine, the tokenizer, the loader,
//                   the expert tier and the KV tier.
//   `components`    one row per COMPONENT, both lanes together (device ms from
//                   the backend, host ms from these phases). This is the
//                   sherlock-style summary: the parts of a run, ranked.
//
// Where the tables go: stderr by default, or the file named by
// KRK_PROFILE_LOG / `--profile-log PATH`, which is what a comparison run should
// keep.
#pragma once

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "krk/trace.hpp"

namespace krk {

class HostTime {
public:
    // Which part of the run a span belongs to. A component table over the whole
    // run is misleading on a short generation: `load` is 51% of it only because
    // the load happened once, and `routing` looks small because decode is
    // diluted by load and prefill. Every row is stamped, so each phase's
    // components are ranked against that phase's own total.
    enum Phase { kLoad = 0, kPrefill = 1, kDecode = 2, kPhases = 3 };

    static const char *phase_name(int p) {
        return p == kLoad ? "load" : p == kPrefill ? "prefill" : "decode";
    }

    static int &phase() {
        static int p = kDecode; // a backend op before init is not a phase
        return p;
    }
    static void set_phase(int p) { phase() = p; }

    struct Row {
        double ms = 0.0;
        long long calls = 0;
        long long bytes = 0;
        // True when the span can run on a WORKER thread at the same time as the
        // main one (the expert read-ahead is the only one today). Such a row is
        // not additive with the step it overlaps, so it is marked in the table
        // rather than silently inflating the accounted total.
        bool concurrent = false;
        double by_phase[kPhases] = {0.0, 0.0, 0.0};
    };

    // Device components, pushed in by the backend at teardown (it owns the HIP
    // event table). The point is ONE table: a reader asking "what is slow" hit
    // a device table grouped by component and a host table grouped by phase,
    // and had to join them by hand.
    struct DevRow {
        double dev_ms[kPhases] = {0.0, 0.0, 0.0};
        double host_ms[kPhases] = {0.0, 0.0, 0.0};
        long long ops[kPhases] = {0, 0, 0};
    };

    static HostTime &get() {
        static HostTime inst;
        return inst;
    }

    bool on() const { return on_; }

    // Where the profiler tables go. KRK_PROFILE_LOG=<path> (also
    // `--profile-log PATH`) sends them to a file instead of stderr: a run that
    // is being compared against another needs the artifact, and a table on a
    // console scrolls away. The file is opened once, shared by both lanes (the
    // device table writes to the same handle at teardown), and flushed on exit.
    static std::FILE *out(std::FILE *fallback) {
        const char *p = std::getenv("KRK_PROFILE_LOG");
        if (!p || !*p) return fallback;
        static std::FILE *f = nullptr;
        static bool tried = false;
        if (!tried) {
            tried = true;
            f = std::fopen(p, "wb");
            if (!f) {
                std::fprintf(stderr,
                             "[warn ] --profile-log: cannot open %s; the tables "
                             "stay on stderr\n",
                             p);
                return fallback;
            }
            std::fprintf(stderr, "[info ] profile log: %s\n", p);
            std::atexit([] {
                if (f) {
                    std::fflush(f);
                    std::fclose(f);
                    f = nullptr;
                }
            });
        }
        return f ? f : fallback;
    }

    void add(const char *name, double ms, bool concurrent = false,
             int ph = -1) {
        if (!on_) return;
        Row &r = rows_[name ? name : "?"];
        r.ms += ms;
        r.calls += 1;
        if (concurrent) r.concurrent = true;
        const int p = ph >= 0 ? ph : phase();
        if (p >= 0 && p < kPhases) r.by_phase[p] += ms;
    }

    void add_bytes(const char *name, long long b) {
        if (!on_ || b <= 0) return;
        rows_[name ? name : "?"].bytes += b;
    }

    // The backend says it will print the component table itself at teardown
    // (it owns the device numbers, which do not exist yet when the host report
    // is written). Without this the host report printed a components table with
    // every device column at zero and the backend printed a second one --
    // two tables with the same title and different numbers.
    static void note_device_pending() { device_pending() = true; }
    static bool &device_pending() {
        static bool pending = false;
        return pending;
    }

    void add_device(const char *component, double ms, long long ops,
                    int ph = -1) {
        if (!on_ || !component) return;
        DevRow &r = dev_[component];
        const int p = ph >= 0 ? ph : phase();
        if (p < 0 || p >= kPhases) return;
        r.dev_ms[p] += ms;
        r.ops[p] += ops;
    }

    // Which component a host phase belongs to. The device table already groups
    // kernels this way (op_time.hpp `component_of`); this is the same idea for
    // the phases that have no kernel, so both lanes print one row per
    // component and a reader can compare them.
    static const char *component_of(const char *phase) {
        struct Entry { const char *p, *c; };
        static const Entry kMap[] = {
            {"load.", "load"},
            {"tokenizer.", "tokenizer"},
            {"tokenize", "tokenizer"},
            {"decode.logits", "head+sample"},
            {"decode.sample", "head+sample"},
            {"decode.fetch", "head+sample"},
            {"decode.step", "decode"},
            {"decode.forward", "decode"},
            {"prefill.", "prefill"},
            {"moe.route", "routing"},
            {"moe.dispatch", "routing"},
            {"moe.experts", "experts"},
            {"moe.cpu_experts", "experts"},
            {"expert.", "expert-tier"},
            {"kv.", "kv-tier"},
            {"spec.", "speculate"},
            {"mtp.", "speculate"},
            {"server.", "server"},
            {"bench.", "bench"},
        };
        for (const Entry &e : kMap) {
            const size_t n = std::strlen(e.p);
            if (std::strncmp(phase, e.p, n) == 0) return e.c;
        }
        return "other";
    }

    // Sorted by total wall time, so the row that matters is first.
    void report(std::FILE *sink) const {
        if (!on_ || rows_.empty()) return;
        std::FILE *out = HostTime::out(sink);
        std::vector<std::pair<const std::string *, const Row *>> order;
        order.reserve(rows_.size());
        double total = 0.0;
        for (const auto &kv : rows_) {
            order.push_back({&kv.first, &kv.second});
            total += kv.second.ms;
        }
        std::sort(order.begin(), order.end(),
                  [](const auto &a, const auto &b) { return a.second->ms > b.second->ms; });
        // Two percentages: `%acct` is a row's share of the accounted total and
        // `%run` its share of the wall clock actually waited. The first alone
        // always sums to 100 whatever the run took and answered neither
        // question. The run wall is from the first HostTime use to now.
        const double wall = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - t_start_)
                                .count();
        std::fprintf(out,
                     "\n[stats ] host time  (wall clock per phase; KRK_PROFILE to enable)"
                     "  run wall %.1f ms\n"
                     "    %-24s %10s %8s %10s %7s %7s\n",
                     wall, "phase", "total ms", "calls", "ms/call", "%acct",
                     "%run");
        for (const auto &e : order) {
            const Row &r = *e.second;
            const double per = r.calls > 0 ? r.ms / static_cast<double>(r.calls) : 0.0;
            const double pc = total > 0.0 ? 100.0 * r.ms / total : 0.0;
            const double pr = wall > 0.0 ? 100.0 * r.ms / wall : 0.0;
            std::fprintf(out, "    %-24s%c %10.1f %8lld %10.3f %6.1f%% %6.1f%%",
                         e.first->c_str(), r.concurrent ? '*' : ' ', r.ms,
                         r.calls, per, pc, pr);
            if (r.bytes > 0) {
                const double mib = static_cast<double>(r.bytes) / 1048576.0;
                const double gbs = r.ms > 0.0 ? mib / 1024.0 / (r.ms / 1000.0) : 0.0;
                std::fprintf(out, "  %8.1f MiB %7.1f GB/s", mib, gbs);
            }
            std::fprintf(out, "\n");
        }
        // Wall clock no host phase claimed -- the number that decides what to
        // instrument next. Only meaningful when the spans do NOT cover the run:
        // they nest (prefill.chunk contains moe.route contains moe.experts), so
        // the accounted total routinely exceeds the wall and wall - total is a
        // large negative that reads as hundreds of percent. The row flips to
        // `nested excess` rather than print that.
        const double rest = wall - total;
        if (rest >= 0.0) {
            std::fprintf(out,
                         "    %-24s   %10.1f %8s %10s %6s %6.1f%%\n",
                         "unaccounted", rest, "", "", "-",
                         wall > 0.0 ? 100.0 * rest / wall : 0.0);
        } else {
            std::fprintf(out,
                         "    %-24s   %10.1f %8s %10s %6s %6.1f%%"
                         "  (spans nest: the accounted total exceeds the run)\n",
                         "nested excess", -rest, "", "", "-",
                         wall > 0.0 ? -100.0 * rest / wall : 0.0);
        }
        std::fprintf(out,
                     "    a row marked * runs on a WORKER thread and is not "
                     "additive with the main one;\n"
                     "    device time inside a phase is included in that phase.\n");
        // A backend with no device table (the CPU reference, or device timing
        // off) still gets the merged table, host rows only; the HIP backend
        // prints it itself once its component totals are in.
        if (dev_.empty() && !device_pending()) report_components(out);
    }

    // The merged component table: one row per (component, phase), both
    // lanes. Phase matters because it changes the answer: `load` is half a
    // short run by construction, and a component that looks small against
    // the whole run can dominate the phase it lives in. Percentages are of
    // that phase, so every column answers "what is slow HERE".
    void report_components(std::FILE *fallback) const {
        if (!on_) return;
        if (rows_.empty() && dev_.empty()) return;
        std::FILE *out = HostTime::out(fallback);
        const double wall = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - t_start_)
                                .count();
        std::map<std::string, DevRow> comp;
        for (const auto &kv : rows_) {
            DevRow &r = comp[component_of(kv.first.c_str())];
            // MAX, not sum: the phases inside one component NEST (load.model
            // contains load.weights; decode.step contains decode.forward).
            // Summing reported `load` at 3400 ms of a 1952 ms run.
            for (int p = 0; p < kPhases; p++) {
                const double ms = kv.second.by_phase[p];
                if (ms > r.host_ms[p]) r.host_ms[p] = ms;
                if (ms > 0.0) r.ops[p] += kv.second.calls;
            }
        }
        for (const auto &kv : dev_) {
            DevRow &r = comp[kv.first];
            for (int p = 0; p < kPhases; p++) {
                r.dev_ms[p] += kv.second.dev_ms[p];
                r.ops[p] += kv.second.ops[p];
            }
        }
        // Phase totals first: a row shares are of its own phase, and the
        // phase total is the larger of its two lanes (device time sits
        // inside host wall time when a phase waits on a copy, and neither
        // lane alone is the wall clock).
        double ptotal[kPhases] = {0.0, 0.0, 0.0};
        for (const auto &kv : comp)
            for (int p = 0; p < kPhases; p++) {
                const double lane = kv.second.host_ms[p] > kv.second.dev_ms[p]
                                        ? kv.second.host_ms[p]
                                        : kv.second.dev_ms[p];
                ptotal[p] += lane;
            }
        struct Line {
            const std::string *name;
            int phase;
            double dev, host;
            long long ops;
            double ms;
        };
        std::vector<Line> lines;
        for (const auto &kv : comp)
            for (int p = 0; p < kPhases; p++) {
                const double d = kv.second.dev_ms[p];
                const double h = kv.second.host_ms[p];
                if (d <= 0.0 && h <= 0.0) continue;
                Line l;
                l.name = &kv.first;
                l.phase = p;
                l.dev = d;
                l.host = h;
                l.ops = kv.second.ops[p];
                l.ms = d > h ? d : h;
                lines.push_back(l);
            }
        std::sort(lines.begin(), lines.end(),
                  [](const Line &a, const Line &b) { return a.ms > b.ms; });
        std::fprintf(out,
                     "\n[stats ] components  (%run is of the row's own phase; dev ms net of the recorder floor)\n"
                     "    %-14s %-8s %8s %10s %10s %7s %7s\n",
                     "component", "phase", "ops", "dev ms", "host ms", "%phase", "%run");
        for (const Line &l : lines) {
            const double pp = ptotal[l.phase] > 0.0
                                  ? 100.0 * l.ms / ptotal[l.phase]
                                  : 0.0;
            const double pr = wall > 0.0 ? 100.0 * l.ms / wall : 0.0;
            std::fprintf(out,
                         "    %-14s %-8s %8lld %10.1f %10.1f %6.1f%% %6.1f%%\n",
                         l.name->c_str(), phase_name(l.phase), l.ops,
                         l.dev, l.host, pp, pr);
        }
        std::fprintf(out,
                     "    phase totals:");
        for (int p = 0; p < kPhases; p++)
            std::fprintf(out, "  %s %.1f ms (%s%s%.1f%%)",
                         phase_name(p), ptotal[p], "", "",
                         wall > 0.0 ? 100.0 * ptotal[p] / wall : 0.0);
        std::fprintf(out, "\n");
    }

private:
    HostTime() : t_start_(std::chrono::steady_clock::now()) {
        const char *p = std::getenv("KRK_PROFILE");
        const char *t = std::getenv("KRK_TIME");
        // KRK_TIME_HOST is the host table ON ITS OWN. It exists because the two
        // profilers share one switch and the device recorder is not a passive
        // observer: it adds ~22 us per op, so a span that waits on the device
        // (`moe.route` holds a blocking readback) is inflated several times over
        // while a span that only enqueues (`moe.experts`) is not. Comparing those
        // two under `--profile` compares two different multipliers. With this
        // switch the device work is untouched and the row is the real wall time.
        const char *h = std::getenv("KRK_TIME_HOST");
        on_ = (p && *p && std::strcmp(p, "0") != 0) ||
              (t && *t && std::strcmp(t, "0") != 0) ||
              (h && *h && std::strcmp(h, "0") != 0);
    }

    bool on_ = false;
    std::map<std::string, Row> rows_;
    std::map<std::string, DevRow> dev_;
    std::chrono::steady_clock::time_point t_start_;
};

// RAII guard: one per phase. Every exit path (including the `return false` ones
// in the speculative loops) closes its span, which a manual pair of calls would
// have leaked.
class HostScope {
public:
    explicit HostScope(const char *name, bool concurrent = false)
        : name_(name), on_(HostTime::get().on()), concurrent_(concurrent) {
        Trace::get().ensure_on();
        trace_on_ = Trace::get().on();
        if (trace_on_) trace_t0_ = Trace::get().now_us();
        if (on_) t0_ = std::chrono::steady_clock::now();
    }
    ~HostScope() {
        if (trace_on_) {
            const double dur = Trace::get().now_us() - trace_t0_;
            Trace::get().complete("host", name_, trace_t0_, dur);
        }
        if (!on_) return;
        const double ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - t0_)
                              .count();
        HostTime::get().add(name_, ms, concurrent_);
    }
    HostScope(const HostScope &) = delete;
    HostScope &operator=(const HostScope &) = delete;

    void bytes(long long b) {
        if (on_) HostTime::get().add_bytes(name_, b);
    }

private:
    const char *name_ = nullptr;
    bool on_ = false;
    bool concurrent_ = false;
    bool trace_on_ = false;
    double trace_t0_ = 0.0;
    std::chrono::steady_clock::time_point t0_{};
};

// One per scope. KRK_TIME_HOST uses the default guard name; use KRK_TIME_HOST_N
// when two scopes have to nest in the same block.
#define KRK_TIME_HOST(nm) ::krk::HostScope krk_host_scope(nm)
#define KRK_TIME_HOST_N(nm, var) ::krk::HostScope var(nm)
// Use from a thread that runs BESIDE the main one (the expert read-ahead
// worker), so the table marks the row and nobody reads it as serial time.
#define KRK_TIME_HOST_T(nm) ::krk::HostScope krk_host_scope_thr(nm, true)

} // namespace krk
