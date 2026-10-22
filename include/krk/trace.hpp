// trace.hpp — low-overhead Chrome-trace (trace-event JSON) recorder.
//
// Why this exists: KRK_TIME/--profile aggregate per-op device spans and
// host-phase walls, but neither preserves a timeline that Perfetto /
// chrome://tracing can render: which host function ran when, which backend
// op it called, which kernel grid/block that op launched, which transfer
// moved how many bytes, and where the queue waited. This recorder is that
// timeline. It is portable C++17 (no HIP include) so engine.cpp, the CPU
// backend and the HIP backend can all emit into one stream.
//
// Format: Chrome trace-event JSON, viewable in Perfetto (`https://
// ui.perfetto.dev` → Open trace file) or `chrome://tracing`:
//   {"traceEvents":[ {"name":"...","cat":"...","ph":"X","pid":1,"tid":2,
//                     "ts":123.4,"dur":56.7,"args":{...}}, ... ]}
// Timestamps are microseconds since the recorder started (monotonic).
//
// Overhead: one branch when disabled (a single static/atomic load). When
// enabled, one mutex + one buffered string per event; buffers flush once at
// `finish()`. Capture bounded windows (a bench run, not a full session) and
// always compare an untraced wall-time run: tracing perturbs the thing it
// measures, exactly like --profile does.
//
// Switches (all environment so the run itself is untouched):
//   KRK_TRACE=1            enable; implies no other profiler switch.
//   KRK_TRACE_PATH=path    output file (default `kraken-trace.json`).
//   KRK_TRACE_MAX=n        cap buffered events (default 2M; oldest dropped,
//                          drop count reported at finish).
//   KRK_TRACE_TIDS=0       disable tid-name metadata (smaller file).
// CLI `--trace` sets the same enable bit for the run (see main_cli.cpp).
#pragma once

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace krk {

class Trace {
public:
    static Trace &get() {
        static Trace inst;
        return inst;
    }

    bool on() const { return on_.load(std::memory_order_relaxed); }

    // Called once from CLI startup (--trace) or lazily from first use when
    // KRK_TRACE=1. Safe to call twice; first path wins.
    void ensure_on() {
        if (on_.load(std::memory_order_relaxed)) return;
        const char *e = std::getenv("KRK_TRACE");
        const bool env = e && *e && std::strcmp(e, "0") != 0;
        if (!env && !force_) return;
        bool expected = false;
        if (!on_.compare_exchange_strong(expected, true)) return;
        const char *p = std::getenv("KRK_TRACE_PATH");
        path_ = (p && *p) ? p : std::string("kraken-trace.json");
        const char *m = std::getenv("KRK_TRACE_MAX");
        const long want = m ? std::atol(m) : 2000000L;
        max_events_ = want > 0 ? static_cast<size_t>(want) : 2000000u;
        t0_ = std::chrono::steady_clock::now();
        const char *tn = std::getenv("KRK_TRACE_TIDS");
        emit_tids_ = !(tn && std::strcmp(tn, "0") == 0);
    }

    void force_enable(const char *path) {
        force_ = true;
        if (path && *path) {
            // Pre-seed the env so ensure_on() picks the CLI path.
            // (No setenv on Windows CRT here; store directly.)
            cli_path_ = path;
        }
        ensure_on();
        if (!cli_path_.empty()) path_ = cli_path_;
    }

    double now_us() const {
        return std::chrono::duration<double, std::micro>(
                   std::chrono::steady_clock::now() - t0_)
            .count();
    }

    int tid_of(std::thread::id id) {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = tids_.find(id);
        if (it != tids_.end()) return it->second;
        const int n = next_tid_++;
        tids_[id] = n;
        if (emit_tids_) {
            char nm[64];
            std::snprintf(nm, sizeof(nm), "thread-%d", n);
            meta_.push_back(std::string("{\"name\":\"thread_name\",\"ph\":\"M\",") +
                            "\"pid\":1,\"tid\":" + std::to_string(n) +
                            ",\"args\":{\"name\":\"" + nm + "\"}}");
        }
        return n;
    }

    // Complete event (ph X). `args_json` is an already-formatted object body
    // without braces, e.g. `"bytes":123,"grid":4`. Empty is fine.
    void complete(const char *cat, const char *name, double t0_us,
                  double dur_us, const char *args_json = nullptr) {
        if (!on()) return;
        const int tid = tid_of(std::this_thread::get_id());
        std::string ev;
        ev.reserve(160 + (args_json ? std::strlen(args_json) : 0));
        ev += "{\"name\":\"";
        json_escape(ev, name ? name : "?");
        ev += "\",\"cat\":\"";
        json_escape(ev, cat ? cat : "?");
        ev += "\",\"ph\":\"X\",\"pid\":1,\"tid\":";
        ev += std::to_string(tid);
        ev += ",\"ts\":";
        ev += ftoa(t0_us);
        ev += ",\"dur\":";
        ev += ftoa(dur_us < 0.0 ? 0.0 : dur_us);
        if (args_json && *args_json) {
            ev += ",\"args\":{";
            ev += args_json;
            ev += "}";
        }
        ev += "}";
        push(std::move(ev));
    }

    // Instant event (ph I): markers for sync points, evictions, drops.
    void instant(const char *cat, const char *name, const char *args_json = nullptr) {
        if (!on()) return;
        const int tid = tid_of(std::this_thread::get_id());
        std::string ev("{\"name\":\"");
        json_escape(ev, name ? name : "?");
        ev += "\",\"cat\":\"";
        json_escape(ev, cat ? cat : "?");
        ev += "\",\"ph\":\"I\",\"pid\":1,\"tid\":";
        ev += std::to_string(tid);
        ev += ",\"ts\":";
        ev += ftoa(now_us());
        ev += ",\"s\":\"t\"";
        if (args_json && *args_json) {
            ev += ",\"args\":{";
            ev += args_json;
            ev += "}";
        }
        ev += "}";
        push(std::move(ev));
    }

    // Counter event (ph C): sparse resource signals (VRAM free, HOT slots).
    void counter(const char *name, double value) {
        if (!on()) return;
        std::string ev("{\"name\":\"");
        json_escape(ev, name ? name : "?");
        ev += "\",\"ph\":\"C\",\"pid\":1,\"tid\":0,\"ts\":";
        ev += ftoa(now_us());
        ev += ",\"args\":{\"value\":";
        ev += ftoa(value);
        ev += "}}";
        push(std::move(ev));
    }

    // Process + help metadata, then every event, then close. Called once at
    // shutdown (main_cli end, backend destructor guard). Idempotent.
    void finish() {
        bool expected = false;
        if (!done_.compare_exchange_strong(expected, true)) return;
        if (!on()) return;
        std::vector<std::string> events;
        std::vector<std::string> meta;
        size_t dropped = 0;
        {
            std::lock_guard<std::mutex> lk(mu_);
            events.swap(events_);
            meta.swap(meta_);
            dropped = dropped_;
        }
        std::FILE *f = std::fopen(path_.c_str(), "w");
        if (!f) {
            std::fprintf(stderr, "[trace ] cannot write %s; trace dropped\n",
                         path_.c_str());
            return;
        }
        std::fprintf(f, "{\"traceEvents\":[\n");
        std::fprintf(f, "{\"name\":\"process_name\",\"ph\":\"M\",\"pid\":1,"
                        "\"args\":{\"name\":\"kraken\"}}");
        for (const std::string &m : meta) std::fprintf(f, ",\n%s", m.c_str());
        for (const std::string &e : events) std::fprintf(f, ",\n%s", e.c_str());
        std::fprintf(f, "\n]}\n");
        std::fclose(f);
        std::fprintf(stderr,
                     "[trace ] wrote %zu events%s to %s (open in "
                     "Perfetto / chrome://tracing)\n",
                     events.size() + meta.size() + 1,
                     dropped ? (" (" + std::to_string(dropped) + " dropped over cap)").c_str() : "",
                     path_.c_str());
    }

    const char *path() const { return path_.c_str(); }

private:
    Trace() = default;

    static void json_escape(std::string &out, const char *s) {
        for (const char *p = s; p && *p; ++p) {
            const char c = *p;
            if (c == '"' || c == '\\') {
                out += '\\';
                out += c;
            } else if (c >= 0x20) {
                out += c;
            } else {
                char b[8];
                std::snprintf(b, sizeof(b), "\\u%04x", c);
                out += b;
            }
        }
    }
    static std::string ftoa(double v) {
        char b[40];
        std::snprintf(b, sizeof(b), "%.3f", v);
        return std::string(b);
    }
    void push(std::string &&ev) {
        std::lock_guard<std::mutex> lk(mu_);
        if (events_.size() >= max_events_) {
            ++dropped_;
            return;
        }
        events_.push_back(std::move(ev));
    }

    std::atomic<bool> on_{false};
    std::atomic<bool> done_{false};
    bool force_ = false;
    bool emit_tids_ = true;
    std::string path_ = "kraken-trace.json";
    std::string cli_path_;
    size_t max_events_ = 2000000u;
    size_t dropped_ = 0;
    std::chrono::steady_clock::time_point t0_ = std::chrono::steady_clock::now();
    std::mutex mu_;
    std::vector<std::string> events_;
    std::vector<std::string> meta_;
    std::map<std::thread::id, int> tids_;
    int next_tid_ = 1;
};

// RAII host/function span. One per traced scope; every exit path closes it.
// Zero cost when tracing is off beyond the ensure_on() branch.
class TraceSpan {
public:
    explicit TraceSpan(const char *cat, const char *name,
                       const char *args_json = nullptr)
        : cat_(cat), name_(name), args_(args_json) {
        Trace::get().ensure_on();
        on_ = Trace::get().on();
        if (on_) t0_ = Trace::get().now_us();
    }
    ~TraceSpan() {
        if (!on_) return;
        const double dur = Trace::get().now_us() - t0_;
        Trace::get().complete(cat_, name_, t0_, dur, args_);
    }
    TraceSpan(const TraceSpan &) = delete;
    TraceSpan &operator=(const TraceSpan &) = delete;

private:
    const char *cat_ = nullptr;
    const char *name_ = nullptr;
    const char *args_ = nullptr;
    double t0_ = 0.0;
    bool on_ = false;
};

} // namespace krk

// Function/component tracing. KRK_TRACE_FN emits __func__ under cat "host";
// KRK_TRACE_COMP(name) tags engine phases ("prefill", "decode", "moe", ...).
#define KRK_TRACE_FN() ::krk::TraceSpan krk_trace_span("host", __func__)
#define KRK_TRACE_COMP(nm) ::krk::TraceSpan krk_trace_span("component", nm)
#define KRK_TRACE_SCOPE(cat, nm) ::krk::TraceSpan krk_trace_span(cat, nm)
#define KRK_TRACE_SCOPE_ARGS(cat, nm, args) \
    ::krk::TraceSpan krk_trace_span_args(cat, nm, args)
