// op_time.hpp — KRK_TIME=1: where a decode step actually goes.
//
// One event pair per backend op, read back lazily from a pool. Nothing here
// synchronizes the stream, so the numbers describe the real pipeline: an op's
// span is the device time from the previous work finishing to its own
// completion — execution plus any gap where the device sat idle waiting for
// that launch to arrive. Every row therefore prints its host submit time next
// to the device span, so "the kernel is slow" and "the CPU cannot feed the
// queue" stay distinguishable: if the device spans account for the wall clock
// and the host column is small, the kernels own the time.
//
// Bytes are accounted where they are knowable (a GEMM's weight matrix), which
// turns a row into an effective GB/s — the number to compare against the card.
// Reference points measured on the 9070 XT with rocBLAS through torch 2.15:
// a 4096x4096 f16 GEMV reads 33.5 MB in 47 µs (710 GB/s); the same matrix as a
// 32-row f16 GEMM takes 43 µs (772 GB/s, 24.7 TFLOP/s).
//
// Reading the table: the spans and the idle column describe the *same* event
// records, and a record costs ~10-27 µs here, so both are inflated by the
// measurement. They are trustworthy for ops whose span is tens of microseconds
// or more (a 300 KB logits copy, a GEMM), and meaningless for the small ones —
// a filtered run (`KRK_TIME=gemm`) keeps the distortion off everything else,
// and the >= 30 µs rows are the ones to quote. For anything smaller, the honest
// instruments are the wall clock and the GPU-engine busy percentage that
// `--bench` samples.
//
// `--profile` (KRK_PROFILE=1) adds the *timeline*: every timed op in issue
// order with its device-idle gap, for the last complete decode step. The
// aggregate table above answers "which op, in total"; the timeline answers "what
// happens, in what order, and how long each launch waits before its kernel
// starts", which is what a launch-bound step needs — the aggregate cannot tell
// 200 launches of 2 µs apart from one launch of 400 µs. It is the difference
// between guessing where the time goes and reading it off.
//
// Knobs, all environment so the run itself is untouched:
//   KRK_PROFILE=1        the timeline (implies KRK_TIME=all)
//   KRK_PROFILE_STEPS=N  print the last N decode steps, not just one (default 1)
//   KRK_PROFILE_FROM=S   start at step S counted back from the end (0 = last)
//   KRK_PROFILE_OPS=N    fallback window when there is no `embed` marker to
//                        split on (default 700 records)
#pragma once

#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace krk {

class OpTime {
public:
    struct Row {
        double dev_ms = 0.0;   // device spans, summed
        double gap_ms = 0.0;   // device idle between this op and the previous one
        double host_ms = 0.0;  // host submit time, summed
        i64 calls = 0;
        i64 bytes = 0;
        i64 flops = 0;
    };

    // One op, in issue order. `dev` and `gap` partition the device timeline
    // exactly: the device went from the previous op's end-event to this op's
    // end-event, and `gap` is the idle part of that interval. `t_open`/
    // `t_close` are host timestamps, so a step's host wall clock can be
    // compared against the device's own accounting — the difference is the
    // time no instrumented op claims, which is where a launch-bound step
    // actually spends.
    struct Rec {
        const char *name = nullptr;
        double dev = 0.0;
        double gap = 0.0;
        double host = 0.0;
        double t_open = 0.0;
        double t_close = 0.0;
        i64 bytes = 0;
        i64 flops = 0;
    };

    static OpTime &get() {
        static OpTime inst;
        return inst;
    }

    bool on() const { return on_; }

    // Recording a pair of events per op is not free — on a 400-op token it
    // slows the run by more than it measures — so KRK_TIME may name the ops to
    // watch: KRK_TIME=gemm,attention times those two and nothing else. "1" or
    // "all" times everything, for a shape of the whole step rather than a
    // trustworthy absolute number.
    bool wants(const char *name) const {
        if (!on_) return false;
        if (filter_.empty()) return true;
        return filter_.find(name) != std::string::npos;
    }

    void begin(const char *name) {
        host0_ = std::chrono::steady_clock::now();
        Slot &s = pool_[next_];
        // The pool wraps well after the device passed this slot, so reading it
        // costs nothing and never stalls the pipeline.
        if (s.used) resolve(s);
        // The idle gap is measured against the previous *timed* op, which is
        // not the previous slot: under a KRK_TIME filter most slots are skipped
        // and the slot arithmetic would land on a stale event from an op that
        // was not timed at all (its timestamp is whatever it held before), which
        // reads as a plausible-looking wrong gap rather than an obvious error.
        s.prev = last_timed_;
        s.has_prev = last_timed_ != kNoPrev;
        next_ = (next_ + 1) % pool_.size();
        last_timed_ = (next_ + pool_.size() - 1) % pool_.size();
        if (!s.a) {
            hipEventCreate(&s.a);
            hipEventCreate(&s.b);
        }
        s.name = name;
        s.bytes = 0;
        s.flops = 0;
        s.host_ms = 0.0;
        s.t_open = hms(host0_);
        s.used = false;
        hipEventRecord(s.a, 0);
        cur_ = &s;
    }

    // Some ops only know which kernel they take after inspecting the shape
    // (a GEMM's rung). The tag has to be a static string: it is stored, not
    // copied.
    void retag(const char *name) {
        if (cur_) cur_->name = name;
    }

    void add_flops(i64 flops) {
        if (cur_) cur_->flops += flops;
    }

    void end(i64 bytes) {
        if (!cur_) return;
        hipEventRecord(cur_->b, 0);
        const auto now = std::chrono::steady_clock::now();
        cur_->bytes = bytes;
        cur_->host_ms =
            std::chrono::duration<double, std::milli>(now - host0_).count();
        cur_->t_close = hms(now);
        cur_->used = true;
        cur_ = nullptr;
    }

    // Resolves everything still in the pool and prints the table. Called once,
    // from the backend's destructor (before the device goes away).
    void finish() {
        if (!on_) return;
        // Drain first: with a pool that covers the run, nothing has resolved yet.
        for (Slot &s : pool_) resolve(s);
        if (rows_.empty()) return;
        std::vector<std::pair<const std::string *, const Row *>> order;
        double dev_total = 0.0, gap_total = 0.0, wall = 0.0;
        i64 ops = 0;
        for (const auto &kv : rows_) {
            order.push_back({&kv.first, &kv.second});
            dev_total += kv.second.dev_ms;
            gap_total += kv.second.gap_ms;
            ops += kv.second.calls;
        }
        std::sort(order.begin(), order.end(),
                  [](const auto &a, const auto &b) {
                      return a.second->dev_ms > b.second->dev_ms;
                  });
        wall = std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - start_).count();
        if (profile_ && !tl_.empty()) print_timeline();
        std::fprintf(stderr,
                     "\n== KRK_TIME per-op device time (%lld ops)\n"
                     "%-26s %8s %10s %9s %9s %9s %8s %8s\n",
                     static_cast<long long>(ops), "op", "calls", "dev ms",
                     "idle ms", "host ms", "MB", "GB/s", "TFLOP/s");
        for (const auto &e : order) {
            const Row &r = *e.second;
            const double gbs =
                r.bytes && r.dev_ms > 0.0
                    ? static_cast<double>(r.bytes) / (r.dev_ms * 1e6)
                    : 0.0;
            const double tfs = r.flops && r.dev_ms > 0.0
                                   ? static_cast<double>(r.flops) /
                                         (r.dev_ms * 1e9)
                                   : 0.0;
            std::fprintf(stderr,
                         "%-26s %8lld %10.2f %9.2f %9.2f %9.2f %8.1f %8.2f\n",
                         e.first->c_str(), static_cast<long long>(r.calls),
                         r.dev_ms, r.gap_ms, r.host_ms,
                         static_cast<double>(r.bytes) / 1e6, gbs, tfs);
        }
        std::fprintf(stderr,
                     "%-26s %8lld %10.2f %9.2f\n"
                     "device spans are %.1f%% of %.1f ms wall; %.1f ms idle "
                     "between ops (%.1f%%); %.1f ms neither (not ops)\n",
                     "TOTAL", static_cast<long long>(ops), dev_total,
                     gap_total, wall > 0.0 ? 100.0 * dev_total / wall : 0.0,
                     wall, gap_total, wall > 0.0 ? 100.0 * gap_total / wall : 0.0,
                     wall - dev_total - gap_total);
    }

private:
    struct Slot {
        hipEvent_t a = nullptr;
        hipEvent_t b = nullptr;
        const char *name = nullptr;
        double host_ms = 0.0;
        double t_open = 0.0;
        double t_close = 0.0;
        i64 bytes = 0;
        i64 flops = 0;
        size_t prev = 0;
        bool has_prev = false;
        bool used = false;
    };

    static const size_t kNoPrev = static_cast<size_t>(-1);

    OpTime() {
        const char *v = std::getenv("KRK_TIME");
        const char *p = std::getenv("KRK_PROFILE");
        profile_ = p && *p && std::string(p) != "0";
        on_ = profile_ || (v && *v && std::string(v) != "0");
        if (on_) {
            const std::string spec = v ? v : "";
            // --profile needs every op to draw the step, so it overrides a
            // filter rather than being filtered away by one.
            if (profile_) filter_.clear();
            else if (spec != "1" && spec != "all") filter_ = spec;
            // A slot's events must not be re-recorded before they are read, or
            // the timestamp they hold is a later op's. The pool therefore has
            // to cover the whole run: one op per slot. Raise KRK_TIME_POOL for
            // a longer run; the default covers a `--bench`.
            const char *pp = std::getenv("KRK_TIME_POOL");
            const long want = pp ? std::atol(pp) : 131072;
            pool_.resize(want > 0 ? static_cast<size_t>(want) : 131072u);
            if (profile_) tl_.reserve(pool_.size());
            start_ = std::chrono::steady_clock::now();
            base_ = start_;
            // Timing the ops that follow costs tens of microseconds each; the
            // table says so rather than leaving the reader to guess why the
            // instrumented run is several times slower than the real one.
            if (profile_) calibrate();
        }
    }

    // Host milliseconds since the run started, for the step's wall clock.
    double hms(std::chrono::steady_clock::time_point t) const {
        return std::chrono::duration<double, std::milli>(t - base_).count();
    }

    // Which part of the step an op belongs to. The op names are per-backend
// kernels; the components are the things a reader actually asks about ("how
// much of the token is the head worth"). Anything unmatched is reported as
// `other` rather than dropped — a silent omission would make the table look
// complete when it is not.
static const char *component_of(const char *op) {
    struct Entry { const char *op, *comp; };
    static const Entry kMap[] = {
        {"embed", "embed"},
        {"rmsnorm", "norms"},
        {"l2norm", "recurrent"},
        {"conv1d_silu", "recurrent"},
        {"delta_rule", "recurrent"},
        {"softplus_act", "recurrent"},
        {"gemm", "projections"},
        {"gemm_group", "projections"},
        {"gemm_simt", "projections"},
        {"silu_mul", "ffn-activate"},
        {"add_inplace", "residual"},
        {"add_bias_rows", "bias"},
        {"add_bias_cols", "bias"},
        {"sigmoid_act", "attention-mix"},
        {"mul_act", "attention-mix"},
        {"scale_act", "attention-mix"},
        {"scale_cols", "attention-mix"},
        {"scale_rows", "attention-mix"},
        {"q3_split", "attention-mix"},
        {"rope", "attention"},
        {"qk_norm", "attention"},
        {"kv_append", "attention"},
        {"attention", "attention"},
        {"attn_fused", "attention"},
        {"logits_topk", "head+sample"},
        {"download", "transfer"},
        {"upload", "transfer"},
        {"upload_i32", "transfer"},
    };
    // Exact match first, then prefix. A GEMM retags itself to the rung it took
    // ("gemm(gemv)", "gemm(wmma)"), so an exact-match table alone silently
    // files 169 of the step's 541 ops under "other" and the projections row
    // reads as if the model had almost none.
    for (const Entry &e : kMap)
        if (std::strcmp(op, e.op) == 0) return e.comp;
    for (const Entry &e : kMap) {
        const size_t n = std::strlen(e.op);
        if (std::strncmp(op, e.op, n) == 0 &&
            (op[n] == '\0' || op[n] == '(' || op[n] == '_'))
            return e.comp;
    }
    return "other";
}

// ---- the timeline -----------------------------------------------------
    //
    // A decode step is delimited by `embed`: the engine embeds, then runs the
    // layers, then projects and samples. So the step boundary is already in the
    // records and no engine change is needed to find one. Everything after the
    // last embed is the last complete step, and walking back over earlier
    // embeds gives the earlier ones.

    void print_timeline() {
        const char *sev = std::getenv("KRK_PROFILE_STEPS");
        const long steps = sev ? std::atol(sev) : 1;
        const char *fev = std::getenv("KRK_PROFILE_FROM");
        const long from = fev ? std::atol(fev) : 0;

        // Step starts: index of every record named `embed`.
        std::vector<size_t> starts;
        for (size_t i = 0; i < tl_.size(); i++)
            if (tl_[i].name && std::strcmp(tl_[i].name, "embed") == 0)
                starts.push_back(i);
        if (starts.empty()) {
            // No marker (a backend or shape without an embed). Fall back to a
            // fixed-size window at the end rather than printing the whole run,
            // which for a bench is tens of thousands of rows.
            const char *oev = std::getenv("KRK_PROFILE_OPS");
            const long ops = oev ? std::atol(oev) : 700;
            const size_t n = static_cast<size_t>(ops > 0 ? ops : 700);
            const size_t b = tl_.size() > n ? tl_.size() - n : 0;
            std::fprintf(stderr,
                         "\n== KRK_PROFILE timeline — no `embed` marker, "
                         "showing the last %zu of %zu ops\n",
                         tl_.size() - b, tl_.size());
            print_range(b, tl_.size());
            return;
        }
        // starts.back() is the last step; `from` counts back from it.
        long first = static_cast<long>(starts.size()) - 1 - from;
        long last = first - (steps > 0 ? steps : 1) + 1;
        if (last < 0) last = 0;
        if (first < 0) first = 0;
        if (first >= static_cast<long>(starts.size())) first =
            static_cast<long>(starts.size()) - 1;
        const long nsteps = first - last + 1;
        std::fprintf(stderr,
                     "\n== KRK_PROFILE timeline — %ld decode step%s "
                     "(the last of %zu recorded)\n",
                     nsteps, nsteps == 1 ? "" : "s", starts.size());
        for (long s = last; s <= first; s++) {
            const size_t b = starts[static_cast<size_t>(s)];
            const size_t e = (static_cast<size_t>(s) + 1 < starts.size())
                                 ? starts[static_cast<size_t>(s) + 1]
                                 : tl_.size();
            std::fprintf(stderr, "-- step %ld/%ld: ops %zu..%zu\n",
                         static_cast<long>(starts.size()) - s,
                         static_cast<long>(starts.size()), b, e - 1);
            print_range(b, e);
        }
    }

    void print_range(size_t b, size_t e) {
        if (e <= b) return;
        double dev = 0.0, gap = 0.0, host = 0.0;
        for (size_t i = b; i < e; i++) {
            dev += tl_[i].dev;
            gap += tl_[i].gap;
            host += tl_[i].host;
        }
        // The device's own accounting is exact: from the previous op's end to
        // this op's end the device was either running (dev) or idle (gap).
        // What is left over is host-side time no op claimed — the queue
        // draining, the sampler, the launch calls themselves. That residual is
        // the number that decides whether the next thing to fix is a kernel or
        // the op count.
        const double dev_wall = dev + gap;
        const double host_wall = tl_[e - 1].t_close - tl_[b].t_open;
        const double outside = host_wall - dev_wall;
        const auto pc = [&](double v) {
            return host_wall > 0.0 ? 100.0 * v / host_wall : 0.0;
        };
        std::fprintf(stderr,
                     "   %zu ops | host wall %8.3f ms | device spans %7.3f ms"
                     " (%5.1f%%) | device idle %7.3f ms (%5.1f%%)"
                     " | outside every op %7.3f ms (%5.1f%%)\n",
                     e - b, host_wall, dev, pc(dev), gap, pc(gap), outside,
                     pc(outside));
        std::fprintf(stderr,
                     "     %5s %-22s %10s %9s %9s %9s %8s %8s\n", "#", "op",
                     "t+ us", "dev us", "idle us", "host us", "MB", "GB/s");
        // `t+` is the cumulative device position, chained from dev+idle. It
        // needs no reference event: the two numbers partition the device
        // timeline exactly, so the running sum IS the position, and it makes
        // the table a timeline rather than a list — a component's share of the
        // step is read off by where its row starts and ends.
        double t = 0.0;
        for (size_t i = b; i < e; i++) {
            const Rec &r = tl_[i];
            const double gbs = r.bytes > 0 && r.dev > 0.0
                                   ? static_cast<double>(r.bytes) / (r.dev * 1e6)
                                   : 0.0;
            std::fprintf(stderr,
                         "     %5zu %-22s %10.2f %9.2f %9.2f %9.2f %8.2f %8.1f\n",
                         i - b, r.name, t * 1000.0, r.dev * 1000.0,
                         r.gap * 1000.0, r.host * 1000.0,
                         static_cast<double>(r.bytes) / 1e6, gbs);
            t += r.dev + r.gap;
        }
        // Same window, rolled up: the "which op is it" answer to the
        // "what order" answer above, so the slow one is a lookup.
        std::map<std::string, Row> agg;
        for (size_t i = b; i < e; i++) {
            const Rec &r = tl_[i];
            Row &a = agg[r.name ? r.name : "?"];
            a.dev_ms += r.dev;
            a.gap_ms += r.gap;
            a.host_ms += r.host;
            a.calls += 1;
            a.bytes += r.bytes;
            a.flops += r.flops;
        }
        std::vector<std::pair<const std::string *, const Row *>> ord;
        for (const auto &kv : agg) ord.push_back({&kv.first, &kv.second});
        std::sort(ord.begin(), ord.end(),
                  [](const std::pair<const std::string *, const Row *> &a,
                     const std::pair<const std::string *, const Row *> &b) {
                      return a.second->dev_ms > b.second->dev_ms;
                  });
        std::fprintf(stderr,
                     "     %5s %-22s %5s %9s %9s %9s\n", "", "rolled up", "n",
                     "dev us", "idle us", "host us");
        for (const auto &e : ord)
            std::fprintf(stderr, "     %5s %-22s %5lld %9.2f %9.2f %9.2f\n", "",
                         e.first->c_str(),
                         static_cast<long long>(e.second->calls),
                         e.second->dev_ms * 1000.0, e.second->gap_ms * 1000.0,
                         e.second->host_ms * 1000.0);
        // The same window by component — the question "where does a token go"
        // wants the parts of the step, not the 541 kernels inside them.
        std::map<std::string, Row> comp;
        std::map<std::string, i64> comp_n;
        for (size_t i = b; i < e; i++) {
            const Rec &r = tl_[i];
            const char *cn = component_of(r.name ? r.name : "?");
            Row &a = comp[cn];
            a.dev_ms += r.dev;
            a.gap_ms += r.gap;
            a.host_ms += r.host;
            a.calls += 1;
            a.bytes += r.bytes;
            a.flops += r.flops;
            comp_n[cn] += 1;
        }
        std::vector<std::pair<const std::string *, const Row *>> cord;
        for (const auto &kv : comp) cord.push_back({&kv.first, &kv.second});
        std::sort(cord.begin(), cord.end(),
                  [](const std::pair<const std::string *, const Row *> &a,
                     const std::pair<const std::string *, const Row *> &bb) {
                      return a.second->dev_ms > bb.second->dev_ms;
                  });
        std::fprintf(stderr,
                     "     %-16s %6s %6s %11s %11s %11s\n", "component", "ops",
                     "%dev", "dev us", "idle us", "host us");
        for (const auto &c : cord)
            std::fprintf(stderr, "     %-16s %6lld %5.1f%% %11.2f %11.2f %11.2f\n",
                         c.first->c_str(),
                         static_cast<long long>(c.second->calls),
                         dev > 0.0 ? 100.0 * c.second->dev_ms / dev : 0.0,
                         c.second->dev_ms * 1000.0, c.second->gap_ms * 1000.0,
                         c.second->host_ms * 1000.0);
        (void)comp_n;

        std::fprintf(stderr,
                     "   instrumentation floor, %d empty ops timed through the "
                     "same begin/end path: %.2f us host, %.2f us device each.\n"
                     "   Every row pays that, so dev and idle are upper bounds "
                     "and a row sitting on the floor is the recorder, not the "
                     "kernel. The host column carries the same ~%.2f us on "
                     "each of the %zu ops (%.0f us of %.0f us submitted).\n"
                     "   Order, counts and GB/s are exact; use KRK_TIME=<op> or "
                     "a --bench wall delta for an undistorted device time.\n",
                     kCalibN, host_floor_us_, dev_floor_us_, host_floor_us_,
                     e - b, (e - b) * host_floor_us_, host * 1000.0);
    }

    // The instrumentation floor, measured rather than quoted. A begin/end pair
    // over an op that launches nothing is exactly what every tiny row in the
    // timeline also pays, so averaging kCalibN of them through the same code
    // gives the number below which a device span means nothing. Measuring it
    // here (once, at startup) is what keeps the table honest: the alternative —
    // quoting a remembered constant — silently goes stale when the driver does.
    static constexpr int kCalibN = 256;

    void calibrate() {
        calib_ = true;
        // resolve() only runs when the pool wraps back to a slot, so the
        // calibration rows are not in rows_ yet when this returns. Claim the
        // slots they occupy and resolve them here instead of waiting 131072
        // ops for an answer the startup path needs.
        const size_t first = next_;
        auto run_pass = [&]() {
            for (int i = 0; i < kCalibN; i++) {
                begin("calib");
                end(0);
            }
            for (int i = 0; i < kCalibN; i++)
                resolve(pool_[(first + static_cast<size_t>(i)) % pool_.size()]);
        };
        // A hipEvent is allocated lazily on its first record, so a pass over
        // fresh slots measures "create an event" (~39 us) rather than the cost
        // every row of the timeline actually pays (~2 us). Warm the slots, then
        // rewind onto exactly them so the timed pass records onto events that
        // already exist — which is every event the run will use afterwards.
        run_pass();
        rows_.erase("calib");
        next_ = first;
        last_timed_ = kNoPrev;
        run_pass();
        calib_ = false;
        auto it = rows_.find("calib");
        if (it != rows_.end()) {
            host_floor_us_ = it->second.host_ms * 1000.0 / kCalibN;
            dev_floor_us_ = it->second.dev_ms * 1000.0 / kCalibN;
            rows_.erase(it);
        }
    }

    void resolve(Slot &s) {
        if (!s.used) return;
        hipEventSynchronize(s.b);
        float ms = 0.0f;
        hipEventElapsedTime(&ms, s.a, s.b);
        // Two markers a microsecond apart can come back out of order, and the
        // resolution is coarser than the span. A negative "duration" is the
        // measurement failing, not a kernel running backwards; left in, the
        // errors accumulate across hundreds of ops and the step's own totals
        // stop adding up. Clamped, and they read as zero rather than as a
        // silent subtraction from the real ops.
        if (ms < 0.0f) ms = 0.0f;
        Row &r = rows_[s.name ? s.name : "?"];
        r.dev_ms += ms;
        r.host_ms += s.host_ms;
        r.calls += 1;
        r.bytes += s.bytes;
        r.flops += s.flops;
        // The synchronisation trap: the span above starts when this op's work
        // starts, so device idle time between two ops belongs to neither. It is
        // measured here instead, as the distance from the previous op's end to
        // this op's start. A big gap column says the queue is starving — the
        // host or a sync point, never the kernel.
        float gap = 0.0f;
        if (s.has_prev && pool_[s.prev].b) {
            if (hipEventElapsedTime(&gap, pool_[s.prev].b, s.a) != hipSuccess)
                gap = 0.0f;
            // A negative result means the previous slot's events had already
            // been re-recorded for a later op (only possible once the pool has
            // wrapped). It is not a gap; printing it as one would be a lie.
            if (gap < 0.0f) gap = 0.0f;
        }
        r.gap_ms += gap;
        if (profile_ && !calib_) {
            Rec rec;
            rec.name = s.name ? s.name : "?";
            rec.dev = ms;
            rec.gap = gap;
            rec.host = s.host_ms;
            rec.t_open = s.t_open;
            rec.t_close = s.t_close;
            rec.bytes = s.bytes;
            rec.flops = s.flops;
            tl_.push_back(rec);
        }
        s.used = false;
    }

    bool on_ = false;
    bool profile_ = false;
    bool calib_ = false;
    double host_floor_us_ = 0.0;
    double dev_floor_us_ = 0.0;
    std::string filter_; // empty = every op
    std::vector<Slot> pool_;
    std::vector<Rec> tl_; // chronological records, profile mode only
    size_t next_ = 0;
    size_t last_timed_ = kNoPrev;
    Slot *cur_ = nullptr;
    std::map<std::string, Row> rows_;
    std::chrono::steady_clock::time_point host0_;
    std::chrono::steady_clock::time_point start_;
    std::chrono::steady_clock::time_point base_;
};

// RAII guard, one per backend op, so every exit path closes its span.
class OpScope {
public:
    explicit OpScope(const char *name) {
        if (OpTime::get().wants(name)) {
            OpTime::get().begin(name);
            live_ = true;
        }
    }
    ~OpScope() {
        if (live_) OpTime::get().end(bytes_);
    }
    // Weight bytes read by this op, when they are knowable.
    void bytes(i64 n) { bytes_ = n; }
    void retag(const char *nm) {
        if (live_) OpTime::get().retag(nm);
    }
    void flops(i64 f) {
        if (live_) OpTime::get().add_flops(f);
    }

private:
    i64 bytes_ = 0;
    bool live_ = false;
};

// One per scope: every op is its own method body.
#define KRK_TIME_OP(nm) ::krk::OpScope krk_op_scope(nm)

} // namespace krk
