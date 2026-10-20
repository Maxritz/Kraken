// bench_ops — raw device speed per shape, with the launch cost amortized out.
//
// The per-op timer in the engine (KRK_TIME) can only be trusted for ops whose
// span is far above the ~10-27 µs an event record costs, so it cannot answer
// "how fast is this kernel" for a decode GEMM. This tool can, because it times
// N back-to-back calls and one call:
//
//     marginal = (T(N) - T(1)) / (N - 1)
//
// T(1) carries the whole queue-drain and ramp; subtracting it leaves the cost of
// one more call in a saturated queue — launch plus execution, which is exactly
// what the decode loop pays. The launch floor is measured the same way on a
// trivial op, so a kernel that is really just launch-bound says so.
//
// The weights are synthetic blocks built in the right layout per format (the
// float headers are benign, so nothing produces NaN); a short host-side dot
// product over `dequant_row` checks sampled output rows and columns, so a
// broken packer cannot produce a fast wrong answer.
//
//   kraken-bench [--iters N] [--rows R] [--dtype q4_k] [--out N] [--in N]
//   kraken-bench --sweep          # the shapes the local models actually use
//   kraken-bench --gate           # correctness only; non-zero exit on failure
//   kraken-bench --floor          # pure load-store ceiling for the decode shapes
//
// --gate is the build gate (`ninja -C build-hip gate`, also a ctest case).
// It does not print a number and hope: NaN and inf are counted separately
// from "wrong", every sample is compared against a per-dtype tolerance, and
// any failure exits non-zero so `ninja gate` fails. The reason it exists is
// that a NaN-producing Q6_K kernel scored a *perfect* check here -- NaN makes
// every comparison false, so the running maximum never moved -- and a
// hand-written v_perm_b32 helper returned garbage that nothing noticed. Both
// shipped. A fast wrong kernel is the failure mode this box keeps producing,
// and it has to be caught by something that is not a code review.
//
// --floor times `copy_act` over fixed-size buffers so the remaining fold
// candidates (the residual fold, the per-matrix wt[] split, the silu fold on
// the epilogue) rank against the load-store ceiling instead of the launch
// floor. It is a shelf, not a bottleneck: "how fast is pure bandwidth".
//
// Reference, same box, torch 2.15/rocBLAS: f16 GEMV 1x4096x4096 = 47.3 µs
// (710 GB/s), f16 GEMM 32x4096x4096 = 43.4 µs (772 GB/s, 24.7 TFLOP/s).
#include <krk/backend.hpp>
#include <krk/expert_cache.hpp>
#include <krk/hc.hpp>
#include <krk/quant.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iterator>
#include <string>
#include <vector>
#ifndef _WIN32
#include <sys/wait.h>
#endif

using namespace krk;

namespace {

struct Shape {
    const char *what;
    DType t;
    i64 n_out;
    i64 n_in;
    i64 rows;
};

void put_half(u8 *p, f32 v) {
    const u16 h = fp32_to_fp16(v);
    p[0] = static_cast<u8>(h & 0xff);
    p[1] = static_cast<u8>(h >> 8);
}

// Synthetic but *valid* blocks: the scale headers carry small finite values and
// everything else is noise, so the dequant path runs without NaN while the
// bytes stay unpredictable (a real row cannot be re-read more often than a
// synthetic one without showing up in the timing).
std::vector<u8> weights(DType t, size_t row_bytes, i64 n_out) {
    std::vector<u8> w(row_bytes * static_cast<size_t>(n_out));
    u32 rng = 0x1234567u;
    for (u8 &b : w) {
        rng = rng * 1103515245u + 12345u;
        b = static_cast<u8>(rng >> 16);
    }
    if (t == DType::F16) {
        // Random bytes are not a valid f16 tensor: half the patterns are NaN
        // and a chunk are inf, which made the reference row itself NaN and the
        // check report a kernel that is in fact fine. Write real halves.
        for (size_t i = 0; i + 1 < w.size(); i += 2) {
            const f32 v = static_cast<f32>(static_cast<int>(w[i] & 0x1f)) - 16.0f;
            const u16 h = fp32_to_fp16(v * 0.001f);
            w[i] = static_cast<u8>(h & 0xff);
            w[i + 1] = static_cast<u8>(h >> 8);
        }
        return w;
    }
    const size_t blk = static_cast<size_t>(dtype_block_bytes(t));
    const i64 blocks = static_cast<i64>(row_bytes / blk);
    for (i64 r = 0; r < n_out; r++) {
        u8 *row = w.data() + static_cast<size_t>(r) * row_bytes;
        for (i64 b = 0; b < blocks; b++) {
            u8 *p = row + static_cast<size_t>(b) * blk;
            switch (t) {
            case DType::Q4_0:
            case DType::Q8_0:
                put_half(p, 0.02f);
                break;
            case DType::Q4_K:
            case DType::Q5_K:
                put_half(p, 0.02f);
                put_half(p + 2, 0.0f); // dmin, the second f16 header
                break;
            case DType::Q6_K:
                put_half(p + blk - 2, 0.02f);
                break;
            default:
                put_half(p, 0.02f);
                break;
            }
        }
    }
    return w;
}

const char *dtype_tag(DType t) {
    switch (t) {
    case DType::F16: return "f16";
    case DType::Q4_0: return "q4_0";
    case DType::Q4_K: return "q4_k";
    case DType::Q5_K: return "q5_k";
    case DType::Q6_K: return "q6_k";
    case DType::Q8_0: return "q8_0";
    default: return "other";
    }
}

// The result of one numeric check. A struct rather than a double because the
// three failure modes are not the same number: NaN, inf, and "wrong" all have
// to be distinguishable, and the first two used to hide inside the third (a
// NaN makes every comparison false, so `rel > worst` stayed 0 and a kernel
// returning pure NaN printed a perfect score).
struct Check {
    double max_rel = 0.0;  // worst |g-want|/rms over every sampled element
    i64 nan_rows = 0;       // rows where the kernel or the reference was NaN
    i64 inf_rows = 0;       // rows where either side was infinite
    i64 checked_rows = 0;   // rows actually compared (0 means the check did
                            // not run, which is a failure, not a pass)
    i64 bad_rows = 0;       // rows over the dtype's tolerance
    bool ok() const {
        return checked_rows > 0 && nan_rows == 0 && inf_rows == 0 &&
               bad_rows == 0;
    }
};

// Per-dtype tolerance on the RMS-relative scale below. These are not guesses:
// they bracket the measured quantisation error of each format against a host
// dequant dot, with the f16 output rounding included, and sit an order of
// magnitude above the worst value any correct kernel has produced here. A new
// format gets the loosest number here on purpose -- a new quant format should
// be visibly wrong, not subtly wrong.
//
// f16 is the interesting one, because there is no quantisation at all: the
// kernel accumulates in f32 and the *only* error is the single rounding of the
// result on the way out. That rounding is <= 2^-11 (4.9e-4) of the output's
// magnitude, but the denominator here is the RMS of the terms rather than the
// output, and a signed sum of n terms can be ~sqrt(n) times smaller than its
// terms -- so a few ulp of the output is several 1e-3 of the RMS. Measured
// worst case on the two f16 shapes is 8.6e-4; 2e-3 leaves better than 2x
// headroom while still being four orders of magnitude below a wrong kernel.
double tol_for(DType t) {
    switch (t) {
    case DType::F16: return 2e-3;   // pure f16 in, f16 out: output rounding
    case DType::Q8_0: return 2e-3;
    case DType::Q6_K: return 2e-3;
    case DType::Q5_K: return 4e-3;
    case DType::Q4_K: return 1.5e-2;
    case DType::Q4_0: return 4e-2;
    default: return 1e-1;
    }
}

// Host reference dot for sampled rows and sampled columns: catches a packer or
// kernel that is fast because it reads the wrong bytes.
//
// Rows are *spread* across the output rather than taken from the front, and
// each row is checked at several columns. A kernel whose first block is fine
// and whose tail is not -- a span that runs past the end, a block-index
// calculation that only holds for small grids -- passes a first-4-rows check
// and fails this one.
Check check_rows(Backend *be, void *out_dev, DType t, i64 n_out, i64 n_in,
                 i64 rows, const std::vector<u8> &w, size_t row_bytes,
                 const std::vector<f32> &x, double tol) {
    Check c;
    std::vector<u8> dev_out(static_cast<size_t>(rows) * n_out * sizeof(u16));
    be->sync();
    be->download(dev_out.data(), out_dev, dev_out.size());
    std::vector<f32> row(static_cast<size_t>(n_in));
    // Up to 8 output rows, spread over the whole tensor, and up to 4 columns
    // per row, spread over the row. Fewer samples when the tensor is smaller.
    const i64 nchk = std::min<i64>(rows, 8);
    const i64 rstride = nchk > 1 ? (rows - 1) / (nchk - 1) : 1;
    const i64 ncol = std::min<i64>(n_out, 4);
    const i64 cstride = ncol > 1 ? (n_out - 1) / (ncol - 1) : 1;
    for (i64 ri = 0; ri < nchk; ri++) {
        const i64 r = std::min<i64>(rows - 1, ri * rstride);
        for (i64 ci = 0; ci < ncol; ci++) {
            const i64 col = std::min<i64>(n_out - 1, ci * cstride);
            // The reference is weight ROW `col` dotted with the activation row:
            // out[r, col] = sum_k W[col, k] * x[r, k]. Sampling (row, column)
            // pairs is what makes this a shape test rather than a spot check.
            dequant_row(t, w.data() + static_cast<size_t>(col) * row_bytes,
                        row.data(), n_in);
            double want = 0.0, rms = 0.0;
            for (i64 k = 0; k < n_in; k++) {
                const double term =
                    static_cast<double>(row[static_cast<size_t>(k)]) *
                    static_cast<double>(
                        x[static_cast<size_t>(r) * static_cast<size_t>(n_in) +
                          static_cast<size_t>(k)]);
                want += term;
                rms += term * term;
            }
            rms = std::sqrt(rms);
            const u8 *o = dev_out.data() +
                          (static_cast<size_t>(r) * n_out +
                           static_cast<size_t>(col)) * 2;
            const double g = static_cast<double>(
                fp16_to_fp32(static_cast<u16>(o[0] | (o[1] << 8))));
            c.checked_rows++;
            if (std::isnan(g) || std::isnan(want)) {
                c.nan_rows++;
                continue;
            }
            if (std::isinf(g) || std::isinf(want)) {
                c.inf_rows++;
                continue;
            }
            // The sum of |terms| is the scale the error has to be read against:
            // a GEMV output is a signed sum, so `want` itself lands near zero
            // often enough (random test data, n_in=4096) that
            // |g-want|/|want| reports cancellation as a kernel error. RMS of
            // the terms is the honest denominator here.
            const double rel = std::fabs(g - want) / (rms + 1e-6);
            if (rel > c.max_rel) c.max_rel = rel;
            if (rel > tol) c.bad_rows++;
        }
    }
    return c;
}

double time_calls(Backend *be, int iters, const std::function<void()> &call) {
    // One call first: the queue drain, the first-touch of every buffer and the
    // ramp are all inside it and are subtracted out below.
    call();
    be->sync();
    auto t0 = std::chrono::steady_clock::now();
    call();
    be->sync();
    auto t1 = std::chrono::steady_clock::now();
    const double one = std::chrono::duration<double, std::milli>(t1 - t0).count();
    auto t2 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; i++) call();
    be->sync();
    auto t3 = std::chrono::steady_clock::now();
    const double many = std::chrono::duration<double, std::milli>(t3 - t2).count();
    return (many - one) / static_cast<double>(iters - 1);
}

Check run_one(Backend *be, const Shape &s, int iters, double launch_us) {
    const size_t row_bytes = dtype_row_bytes(s.t, s.n_in);
    std::vector<u8> w = weights(s.t, row_bytes, s.n_out);
    std::vector<f32> xf(static_cast<size_t>(s.rows) * s.n_in);
    u32 rng = 99u;
    for (f32 &v : xf) {
        rng = rng * 1103515245u + 12345u;
        v = static_cast<f32>((rng >> 20) & 0xff) / 255.0f - 0.5f;
    }
    std::vector<u16> xh(xf.size());
    for (size_t i = 0; i < xf.size(); i++) xh[i] = fp32_to_fp16(xf[i]);
    void *wd = be->alloc(w.size());
    void *xd = be->alloc(xh.size() * 2);
    void *od = be->alloc(static_cast<size_t>(s.rows) * s.n_out * 2);
    be->upload(wd, w.data(), w.size(), 0);
    be->upload(xd, xh.data(), xh.size() * 2, 0);

    const double ms = time_calls(be, iters, [&] {
        be->gemm(od, xd, wd, s.t, s.n_out, s.n_in, s.rows);
    });
    const double wbytes = static_cast<double>(row_bytes) * s.n_out;
    const double gbs = wbytes / (ms * 1e6);
    const double tfs = 2.0 * s.rows * s.n_out * s.n_in / (ms * 1e9);
    const double tol = tol_for(s.t);
    const Check c = check_rows(be, od, s.t, s.n_out, s.n_in, s.rows, w, row_bytes,
                               xf, tol);
    char chk[48];
    if (!c.checked_rows) std::snprintf(chk, sizeof chk, "NO-CHECK");
    else std::snprintf(chk, sizeof chk, "%.2e/%.0e nan=%lld inf=%lld", c.max_rel,
                       tol, static_cast<long long>(c.nan_rows),
                       static_cast<long long>(c.inf_rows));
    std::printf("%-6s %-6s out=%-7lld in=%-6lld rows=%-3lld %9.2f us  "
                "%8.2f MB  %7.1f GB/s  %6.2f TFLOP/s  %-4s chk=%s\n",
                s.what, dtype_tag(s.t), static_cast<long long>(s.n_out),
                static_cast<long long>(s.n_in), static_cast<long long>(s.rows),
                ms * 1000.0, wbytes / 1e6, gbs, tfs, c.ok() ? "ok" : "FAIL",
                chk);
    be->release(wd);
    be->release(xd);
    be->release(od);
    return c;
}

// The promotion path's per-call bookkeeping, measured the same way the
// kernels are: N back-to-back calls minus one, so the number is one more
// call in a saturated queue rather than a cold first call.
//
// This exists because the engine's own `alloc` row in the expert-path split
// cannot say what it is made of. `alloc_ms_` wraps alloc_pooled, which is a
// free-list lookup, a reuse guard and possibly a wait for the device -- three
// different fixes for one number. The three arms below separate them:
//
//   event create+destroy   what a pool that allocates a guard per release pays
//   record / query / wait  the per-release and per-reuse halves of that guard
//   pool round trip x2     the real pair, with a reader already complete
//                          (pure bookkeeping) and with a reader queued on the
//                          default stream (the wait the guard exists for)
//
// The last arm is the one the decode loop actually pays; the first three say
// whether its cost is the guard or the queue.
// One block of `chunk` calls timed on a queue that already holds `depth`
// copies: the state a decode step is always in, and the only state in which a
// host-side enqueue can be timed on its own. `tail_us` is what the sync AFTER
// the block still had to drain -- the witness that the queue was busy when the
// host returned, which is what makes the reading a queue measurement rather
// than a wall-clock coincidence.
// Every call in this engine has two costs and a table that quotes one number
// per call is quoting half a call:
//   host_us    what the enqueue costs the thread that is issuing the step
//   device_us  what the call costs the DEVICE once it gets there, which is the
//              time the sync after the block still has to drain (the tail)
// `time_calls` cannot see any of this: it subtracts one-call-plus-one-sync from
// many-calls-plus-one-sync, so its quotient is whichever of the two halves
// dominates the queue -- for a legacy-stream record that is the device side
// (~27 us per marker), and for a launch it is the launch itself.
// `depth` copies are pre-issued first, so the block can be timed on a queue
// that is provably still draining when the host returns.
struct QueueArm {
    double host_us = 0.0;   // enqueue, per call
    double device_us = 0.0; // device time left after the block, per call
    double tail_us = 0.0;   // ... and in total
    double host_block_us = 0.0;
};

// kind=false times `copy_act`, kind=true times `record_marker`; the pre-issue is
// always copies, so both arms start from the same queue.
static QueueArm measure_queue_arm(Backend *be, int depth, int chunk, int reps,
                                  bool records) {
    QueueArm r;
    void *a = be->alloc(4096);
    void *b = be->alloc(4096);
    void *m = be->make_marker();
    if (!a || !b || !m) {
        if (a) be->release(a);
        if (b) be->release(b);
        if (m) be->release_marker(m);
        return r;
    }
    be->fill0(a, 4096);
    be->fill0(b, 4096);
    // The pre-issue is the instrument's own red construction: with it skipped,
    // "busy" and "empty" are the same state and the gate's queue checks fail.
    const char *skip = std::getenv("KRK_TEST_QUEUE_NOPREISSUE");
    const bool no_preissue = skip && *skip && std::string(skip) != "0";
    std::vector<double> host(reps, 0.0), tl(reps, 0.0);
    for (int i = 0; i < reps; i++) {
        for (int k = 0; k < depth && !no_preissue; k++) be->copy_act(a, b, 32);
        const auto t0 = std::chrono::steady_clock::now();
        for (int k = 0; k < chunk; k++) {
            if (records) be->record_marker(m);
            else be->copy_act(a, b, 32);
        }
        const auto t1 = std::chrono::steady_clock::now();
        be->sync();
        const auto t2 = std::chrono::steady_clock::now();
        host[i] = std::chrono::duration<double, std::micro>(t1 - t0).count() / chunk;
        tl[i] = std::chrono::duration<double, std::micro>(t2 - t1).count();
    }
    std::sort(host.begin(), host.end());
    std::sort(tl.begin(), tl.end());
    r.host_us = host[reps / 2];
    r.tail_us = tl[reps / 2];
    r.device_us = r.tail_us / chunk;
    r.host_block_us = r.host_us * chunk;
    be->release_marker(m);
    be->release(a);
    be->release(b);
    return r;
}

void bench_pool_primitives(Backend *be, int iters, int depth, double launch_ms) {
    const int n = std::max(50, iters);
    const double create_ms = time_calls(be, n, [&] {
        void *m = be->make_marker();
        be->release_marker(m);
    });
    void *tok = be->make_marker();
    be->record_marker(tok);
    const double record_ms = time_calls(be, n, [&] { be->record_marker(tok); });
    const double query_ms = time_calls(be, n, [&] { be->poll_marker(tok); });
    // Recorded and drained, so the wait is a completed event: this is the
    // cheap case, and the difference against the pool round trip below is the
    // part that is a real wait for the device.
    be->wait_marker(tok);
    const double wait_ms = time_calls(be, n, [&] { be->wait_marker(tok); });
    // The arm the old pool paid on every reuse: ask the device whether the
    // reader has retired. The record makes the query land on a PENDING event
    // every time, which is the state a saturated pool is always in; the
    // difference against the record-only arm above is what the question costs.
    const double recq_ms = time_calls(be, n, [&] {
        be->record_marker(tok);
        be->poll_marker(tok);
    });
    be->release_marker(tok);

    // The pool as the cache drives it: a handful of cells, cycled. KEYS cells
    // of 256 KiB inside a 2 MiB budget -- the same shape as an expert's three
    // slices at a constrained --expert-cache-mb, scaled down so the number is
    // bookkeeping and not bus bandwidth.
    constexpr size_t kCell = 256 * 1024;
    constexpr int kCells = 8;
    be->set_expert_pool_budget(kCell * kCells);
    void *cells[kCells];
    void *src = be->alloc(kCell), *sink = be->alloc(kCell);
    be->fill0(src, kCell);
    for (int i = 0; i < kCells; i++) {
        cells[i] = be->alloc_pooled(kCell);
        be->fill0(cells[i], kCell);
    }
    int slot = 0;
    const double ready_ms = time_calls(be, n, [&] {
        be->release_pooled(cells[slot], kCell);
        cells[slot] = be->alloc_pooled(kCell);
        slot = (slot + 1) % kCells;
    });
    // Same pair, but the cell is released only after a reader has been queued
    // into it on the default stream: reuse must now wait for that reader, and
    // that wait is what the engine's `alloc` row is mostly made of.
    const double reader_ms = time_calls(be, n, [&] {
        be->copy_act(sink, cells[slot], 64);
        be->release_pooled(cells[slot], kCell);
        cells[slot] = be->alloc_pooled(kCell);
        slot = (slot + 1) % kCells;
    });
    // The shape the engine actually drives at a constrained budget: an eviction
    // returns an expert's THREE slices in one call. Both arms below hand back the
    // same three cells with a reader queued on the default stream, so they are
    // one call apart, and the counters say how many records each issued -- the
    // mechanism, not just a time.
    constexpr int kBurst = 3;
    size_t blen[kBurst];
    for (int i = 0; i < kBurst; i++) blen[i] = kCell;
    void *bc[kBurst];
    be->set_expert_pool_budget(kCell * (kCells + kBurst));
    for (int i = 0; i < kBurst; i++) {
        bc[i] = be->alloc_pooled(kCell);
        be->fill0(bc[i], kCell);
    }
    const auto marks0 = be->expert_pool_stats();
    const double burst_ms = time_calls(be, n, [&] {
        be->copy_act(sink, bc[0], 64);
        be->release_pooled_batch(bc, blen, kBurst);
        for (int i = 0; i < kBurst; i++) bc[i] = be->alloc_pooled(kCell);
    });
    const auto burst_st = be->expert_pool_stats();
    const double sep_ms = time_calls(be, n, [&] {
        be->copy_act(sink, bc[0], 64);
        for (int i = 0; i < kBurst; i++) be->release_pooled(bc[i], kCell);
        for (int i = 0; i < kBurst; i++) bc[i] = be->alloc_pooled(kCell);
    });
    const auto sep_st = be->expert_pool_stats();
    for (int i = 0; i < kBurst; i++) be->release_pooled(bc[i], kCell);

    for (int i = 0; i < kCells; i++) be->release_pooled(cells[i], kCell);
    be->release(src);
    be->release(sink);
    const auto st = be->expert_pool_stats();
    std::printf(
        "pool primitives: event create+destroy %.2f us | record %.2f us | "
        "query %.2f us | wait %.2f us | record+query a pending event %.2f us\n",
        create_ms * 1000.0, record_ms * 1000.0, query_ms * 1000.0,
        wait_ms * 1000.0, (recq_ms - record_ms) * 1000.0);
    std::printf(
        "pool round trip (%d cells of %d KiB, budget %d KiB): %.2f us ready | "
        "%.2f us with a queued reader | alloc %llu reuse %llu trim %llu\n",
        kCells, static_cast<int>(kCell / 1024),
        static_cast<int>(kCell * kCells / 1024), ready_ms * 1000.0,
        reader_ms * 1000.0, static_cast<unsigned long long>(st.allocations),
        static_cast<unsigned long long>(st.reuses),
        static_cast<unsigned long long>(st.trims));
    // Sequenced, so this ranks the two arms and does not time the machine: the
    // verdict on the engine comes from the interleaved A/B, and the counter
    // ratio below is the part that is a fact either way.
    std::printf(
        "pool burst (%d slices of %d KiB, a queued reader): %.2f us one mark | "
        "%.2f us one record per slice | %llu vs %llu mark(s) for %llu slice(s)\n",
        kBurst, static_cast<int>(kCell / 1024), burst_ms * 1000.0,
        sep_ms * 1000.0,
        static_cast<unsigned long long>(burst_st.release_marks -
                                        marks0.release_marks),
        static_cast<unsigned long long>(sep_st.release_marks -
                                        burst_st.release_marks),
        static_cast<unsigned long long>(sep_st.release_slices -
                                        marks0.release_slices));
    // Quiet queue vs BUSY queue, host side -- and the busy queue is the only state
    // a decode step is ever in (303 ops in flight on the 135M, 2965 on the 35B).
    // The arms above measure each call with nothing else in flight, and for a
    // legacy-stream record that measures the FLUSH (29.5 us here) rather than the
    // enqueue a step pays (2.56 us: 7252 records removed bought 18.6 ms of the
    // step on Qwen3-MOE-4x0.6B at 16/32 MiB). This arm feeds the queue with real
    // device work and times `chunk` calls issued while it drains, which is why
    // the two figures differ and by how much. It cannot be done with time_calls:
    // that helper subtracts one cold call from a batch mean, and for a host-side
    // enqueue the cold call is the expensive one -- it printed a NEGATIVE
    // marginal record cost (-12.82 us) before this arm replaced it.
    if (depth > 0) {
        constexpr int kChunk = 64; // calls timed per phase
        constexpr int kReps = 8;   // median of these, per state
        const QueueArm bcopy = measure_queue_arm(be, depth, kChunk, kReps, false);
        const QueueArm ecopy = measure_queue_arm(be, 0, kChunk, kReps, false);
        const QueueArm brec = measure_queue_arm(be, depth, kChunk, kReps, true);
        const QueueArm erec = measure_queue_arm(be, 0, kChunk, kReps, true);
        // Both halves of each call, by one method, in both queue states. The
        // time_calls arms above quote whichever half dominates its own loop, so
        // they are printed for reference and never as "the cost".
        std::printf(
            "queue state (%d x copy_act n=32 pre-issued): device left to drain "
            "%.0f us vs %.0f us empty (a launch: %.0f vs %.0f us)\n",
            depth, brec.tail_us, erec.tail_us, bcopy.tail_us, ecopy.tail_us);
        std::printf(
            "primitives, host enqueue per call: record %.2f us | copy %.2f us, "
            "median of %d (empty queue: record %.2f | copy %.2f)\n",
            brec.host_us, bcopy.host_us, kReps, erec.host_us, ecopy.host_us);
        std::printf(
            "primitives, device cost per call: record %.1f us | copy %.1f us "
            "(empty queue: record %.1f | copy %.1f)  [time_calls arms, which "
            "see only the dominant half: record %.2f | copy %.2f]\n",
            brec.device_us, bcopy.device_us, erec.device_us, ecopy.device_us,
            record_ms * 1000.0, launch_ms * 1000.0);
    }
    be->set_expert_pool_budget(0);
}

// The busy-queue arm's own gate. It is a check on the INSTRUMENT, not on the
// engine: the correction the quiet-vs-busy audit rests on says a quiet-queue
// primitive is not the in-step cost, and that claim is worth nothing unless the
// arm can tell a busy queue from an idle one. So the gate constructs both
// states AND the boundary between them, with no environment switch in play:
//   - 512 copies pre-issued: the queue is still draining when the host block
//     returns (witnessed by the tail the following sync has to wait for), and
//     both calls are cheaper than with an empty queue -- measured 88x for the
//     record and ~3x for the copy.
//   - 1 copy pre-issued: nothing is left to drain, and the same block must
//     read like the empty-queue figure. An arm that reported the busy figure
//     here would be reporting a constant, not a queue state -- which is the
//     defect this check exists to catch.
// It runs even when --depth 0 was passed, because `--depth` skips the printed
// arm and a gate check is not optional.
i64 gate_busy_queue(Backend *be, int depth) {
    i64 bad = 0;
    const auto check = [&](bool ok, const char *what) {
        std::printf("  %s busy-queue %s\n", ok ? "ok  " : "FAIL", what);
        if (!ok) ++bad;
    };
    constexpr int kChunk = 64;
    constexpr int kReps = 8;
    const int deep = depth > 1 ? depth : 512;
    const QueueArm bcopy = measure_queue_arm(be, deep, kChunk, kReps, false);
    const QueueArm ecopy = measure_queue_arm(be, 0, kChunk, kReps, false);
    const QueueArm brec = measure_queue_arm(be, deep, kChunk, kReps, true);
    const QueueArm erec = measure_queue_arm(be, 0, kChunk, kReps, true);
    std::printf("    device left to drain: %.0f us deep vs %.0f us empty "
                "(launch %.0f vs %.0f us = %.2fx, needs >2.0x)\n",
                brec.tail_us, erec.tail_us, bcopy.tail_us, ecopy.tail_us,
                ecopy.tail_us > 0 ? bcopy.tail_us / ecopy.tail_us : 0.0);
    std::printf("    host enqueue/call: record %.2f us | copy %.2f us\n",
                brec.host_us, bcopy.host_us);
    std::printf("    device cost/call:  record %.1f us | copy %.1f us\n",
                brec.device_us, bcopy.device_us);
    // The construction the whole arm exists for: with `deep` copies in front of
    // them, the block's own work cannot have been retired by the time the host
    // returns, so the tail must carry it. Launch-only, because a record's own
    // ~24 us device marker swamps the pre-issue's leftovers in the same tail.
    // 2.0x, not 3.0x: measured ratios run 2.9x on a busy machine to 3.7x on a
    // quiet one, while a queue that is NOT fuller reports 1.0x, so 2.0
    // discriminates without flapping with the machine's load. The ratios are
    // printed so the margin is visible in the gate's own output.
    const double tail_ratio = ecopy.tail_us > 0 ? bcopy.tail_us / ecopy.tail_us : 0.0;
    const double dev_ratio = ecopy.device_us > 0 ? bcopy.device_us / ecopy.device_us : 0.0;
    check(tail_ratio > 2.0,
          "a queue holding `depth` calls is measurably fuller than an empty one "
          "(device left to drain: launch arm)");
    check(dev_ratio > 2.0, "and the same difference is visible per call");
    // The defect the arm was built to stop: quoting one half of a call as the
    // call. A legacy-stream record is ~0.3 us of host enqueue and ~28 us of
    // device marker; a table that measures 28 us by T(N)-T(1) has measured the
    // device half and called it the record.
    check(brec.device_us > 4.0 * brec.host_us,
          "a record's device half is >=4x its host half (so the two figures a "
          "table might quote are not the same number)");
    // And the claim this run DISPROVED, inverted into the one it supports: the
    // queue state does not move the host enqueue. The arm used to print a
    // "busy" record of 0.32 us beside a "quiet" 28.2 us from a different
    // method and read the ratio as queue state; both states enqueue in 0.28 us,
    // and the 28.2 was the device half. A future change that really does make
    // the enqueue queue-dependent owes this check an edit.
    check(brec.host_us < 2.0 * erec.host_us && erec.host_us < 2.0 * brec.host_us,
          "the host enqueue is the same in both queue states (the queue changes "
          "what the device has left, not what the enqueue costs)");
    check(ecopy.device_us * 4.0 < erec.device_us,
          "the instrument resolves the device cost per op (an empty queue: a "
          "launch vs a record, >=4x apart)");
    return bad;
}

i64 gate_expert_pool(Backend *be) {
    i64 bad = 0;
    const auto check = [&](bool ok, const char *what) {
        std::printf("  %s expert-pool %s\n", ok ? "ok  " : "FAIL", what);
        if (!ok) ++bad;
    };
    be->set_expert_pool_budget(4096);
    for (size_t bytes : {size_t(257), size_t(769), size_t(1281), size_t(257)}) {
        void *p = be->alloc_pooled(bytes);
        be->fill0(p, bytes);
        be->release_pooled(p, bytes);
        const auto s = be->expert_pool_stats();
        check(s.live == 0 && s.committed == s.reusable && s.committed <= 4096,
              "release accounting and physical bound");
    }
    void *source = be->alloc(1024), *output = be->alloc(1024);
    std::vector<u16> input(512, fp32_to_fp16(1.0f)), got(512);
    be->upload(source, input.data(), 1024);
    for (int i = 0; i < 16; ++i) {
        void *p = be->alloc_pooled(1024);
        be->copy_act(p, source, 512);
        be->copy_act(output, p, 512);
        be->release_pooled(p, 1024);
        void *q = be->alloc_pooled(1024);
        be->fill0(q, 1024);
        be->download(got.data(), output, 1024);
        check(got == input, "queued reader finishes before reuse");
        be->release_pooled(q, 1024);
    }
    {   // The burst's shared mark has to be honored by EVERY writer that reuses
        // one of its cells, not only the first one handed out. Three cells are
        // read on the default stream, released in ONE call, then each handed back
        // out and zeroed before those reads can have completed: every reader must
        // still see its own bytes. This is the burst form of the single-cell check
        // above, and the counter check is the sharing the whole change is for.
        void *bc[3] = {}, *bo[3] = {};
        size_t bl[3] = {1024, 1024, 1024};
        std::vector<u16> bgot[3] = {std::vector<u16>(512), std::vector<u16>(512),
                                    std::vector<u16>(512)};
        for (int i = 0; i < 3; i++) {
            bc[i] = be->alloc_pooled(1024);
            bo[i] = be->alloc(1024);
            be->copy_act(bc[i], source, 512);
            be->copy_act(bo[i], bc[i], 512);
        }
        const auto b0 = be->expert_pool_stats();
        be->release_pooled_batch(bc, bl, 3);
        const auto b1 = be->expert_pool_stats();
        check(b1.release_marks - b0.release_marks == 1 &&
                  b1.release_slices - b0.release_slices == 3,
              "a three-slice burst shares one release mark");
        bool reused[3] = {false, false, false};
        for (int i = 0; i < 3; i++) {
            void *q = be->alloc_pooled(1024);
            for (int j = 0; j < 3; j++)
                if (q == bc[j]) reused[j] = true;
            be->fill0(q, 1024);
            be->release_pooled(q, 1024);
        }
        check(reused[0] && reused[1] && reused[2],
              "every cell of a burst is handed back out");
        for (int i = 0; i < 3; i++) be->download(bgot[i].data(), bo[i], 1024);
        be->sync();
        for (int i = 0; i < 3; i++) be->release(bo[i]);
        check(bgot[0] == input && bgot[1] == input && bgot[2] == input,
              "every cell of a burst keeps its mark: no reader is overwritten");
    }
    void *staged = be->alloc_pooled(1024);
    void *dst[] = {staged};
    const void *src[] = {input.data()};
    const size_t lengths[] = {1024};
    void *ticket = be->upload_expert_staged(dst, src, lengths, 1);
    be->wait_marker(ticket);
    be->release_marker(ticket);
    be->download(got.data(), staged, 1024);
    check(got == input, "staged ticket publishes exact bytes");
    be->release_pooled(staged, 1024);
    be->release(source); be->release(output);
    be->set_expert_pool_budget(0);
    auto s = be->expert_pool_stats();
    check(s.live == 0 && s.reusable == 0 && s.committed == 0, "shrink trims every free cell");
    const int owner_a = 0, owner_b = 0;
    be->set_expert_pool_owner(&owner_a, 1024);
    be->set_expert_pool_owner(&owner_b, 2048);
    check(be->expert_pool_stats().budget == 3072, "target and draft budgets compose");
    be->set_expert_pool_owner(&owner_a, 0);
    be->set_expert_pool_owner(&owner_b, 0);
    be->set_expert_pool_budget(4096);
    void *live = be->alloc_pooled(512);
    be->fill0(live, 512);
    be->set_expert_pool_budget(256);
    check(be->expert_pool_stats().live == 512, "shrink never frees a live cell");
    be->release_pooled(live, 512);
    check(be->expert_pool_stats().committed == 0, "released over-budget cell is reclaimed");
    be->set_expert_pool_budget(0);
    return bad;
}

// ---------------------------------------------------------------------------
// expert-lease gate: the slot lease, proven on the device
// ---------------------------------------------------------------------------
//
// `test_expert_cache_warm_tier` covers the same invariant against the CPU stub
// backend, where "the current expert's bytes are still there" is a host memcmp
// on a buffer nobody else can touch. The device half is the one the lease
// exists for: a staged promotion allocates from the SAME pool the current expert
// is sitting in, so an ignored lease lets `pick_victim` release the current
// expert's cells, `alloc_pooled` hand those exact addresses to the next expert,
// and the current expert's weights quietly become the next expert's. There is no
// crash and no NaN -- just a plausible number from the wrong matrix -- so the
// check has to be the device bytes: after a concurrent staged promotion, the
// current expert's three slices are downloaded and compared byte for byte with
// the source they were promoted from.
//
// The pool's physical limit is what makes this a real test rather than a
// formality: the cache is configured with a two-slot budget, so the staged
// promotion CANNOT get cells without evicting something, and the only reason it
// does not take the current expert's is the lease.
i64 gate_expert_lease(Backend *be) {
    i64 bad = 0;
    const auto check = [&](bool ok, const char *what) {
        std::printf("  %s expert-lease %s\n", ok ? "ok  " : "FAIL", what);
        if (!ok) ++bad;
    };
    // One expert is three 1 KiB slices, so a cell is 1 KiB and a two-slot
    // budget is 6144 bytes -- small enough to be a gate, sized so every cell is
    // one aligned cell and every copy is a real DMA.
    constexpr i64 kPer = 1024;      // bytes per slice, per expert
    constexpr i32 kExperts = 4;
    std::vector<u8> gate_buf(static_cast<size_t>(kPer) * kExperts);
    std::vector<u8> up_buf(gate_buf.size());
    std::vector<u8> down_buf(gate_buf.size());
    for (i32 e = 0; e < kExperts; e++) {
        for (i64 j = 0; j < kPer; j++) {
            gate_buf[static_cast<size_t>(e) * kPer + static_cast<size_t>(j)] =
                static_cast<u8>((e * 0x40 + j) & 0xff);
            up_buf[static_cast<size_t>(e) * kPer + static_cast<size_t>(j)] =
                static_cast<u8>((e * 0x40 + j + 0x10) & 0xff);
            down_buf[static_cast<size_t>(e) * kPer + static_cast<size_t>(j)] =
                static_cast<u8>((e * 0x40 + j + 0x20) & 0xff);
        }
    }
    const auto tensor = [&](std::vector<u8> &buf) {
        GgufTensor t;
        t.type = DType::F16;
        t.dtype_known = true;
        t.n_dims = 2;
        t.ne[0] = static_cast<u64>(kPer / 2);
        t.ne[1] = 1;
        t.n_elements = kPer / 2;
        t.n_bytes = buf.size();
        t.data = buf.data();
        return t;
    };
    const GgufTensor tg = tensor(gate_buf), tu = tensor(up_buf), td = tensor(down_buf);
    ExpertSource src;
    src.gate = &tg;
    src.up = &tu;
    src.down = &td;
    src.n_expert = kExperts;
    src.n_embd = kPer / 2;
    src.n_ff_exp = kPer / 2;

    const auto slice_matches = [&](const QuantTensor &t, const std::vector<u8> &buf,
                                   i32 e, const char *what) {
        std::vector<u8> got(static_cast<size_t>(kPer));
        be->sync();
        be->download(got.data(), t.data, got.size());
        const u8 *want = buf.data() + static_cast<size_t>(e) * kPer;
        const bool ok = std::memcmp(got.data(), want, got.size()) == 0;
        if (!ok)
            std::printf("       lease mismatch: %s slice of expert %d\n", what,
                        static_cast<int>(e));
        return ok;
    };

    {   // A budget of 2.33 slots, so the staged promotion MUST evict to fit.
        //
        // The budget is the whole test. With room for both experts the
        // promotion never contends for anything, the byte checks below pass for
        // the trivial reason that nothing was reused, and ignoring the lease
        // changes nothing (measured: the first version of this gate passed with
        // `pick_victim`'s lease check deleted). The budget is between two and
        // three slots (7 cells for three 3-cell experts): two experts are
        // resident, the staged third needs a slot, so an eviction is mandatory
        // -- and the tie in victim ranking (equal LFU counts, so oldest load
        // order wins) makes the LEASED expert the one a lease-less eviction
        // would take. Exactly three slots would fit and evict nothing.
        ExpertCache cache;
        cache.configure(be, 7 * kPer, 8 * 3 * kPer);
        const ResidentExpert *cur = cache.acquire(src, 0, 0);
        check(cur && cache.in_vram(0, 0), "the current expert is resident");
        // SNAPSHOT of the three device addresses. Holding `cur` is not enough:
        // it points into the cache's own slot, so after a re-promotion it reads
        // the NEW pointers and any comparison against it is a tautology -- and a
        // lease violation is exactly a re-promotion, which restores the right
        // bytes through the wrong addresses. The addresses are what must be
        // checked, and reading a stolen address is safe (it was handed to
        // another expert, not freed).
        const QuantTensor cur_gate = cur->gate, cur_up = cur->up,
                          cur_down = cur->down;
        check(cur && slice_matches(cur_gate, gate_buf, 0, "gate") &&
                  slice_matches(cur_up, up_buf, 0, "up") &&
                  slice_matches(cur_down, down_buf, 0, "down"),
              "its device bytes match the source it was promoted from");
        check(cache.acquire(src, 0, 1) != nullptr && cache.resident_slots() == 2,
              "a second expert is resident beside it");
        const i32 next_ids[] = {2, 3};
        cache.prefetch_layer(src, 0, next_ids, 2);
        check(cache.stage_next(src, 0, 2, 0), "the next routed expert stages concurrently");
        // Non-vacuous: the promotion needed room, so something WAS evicted, and
        // the only reason it was not the current expert is the lease.
        check(cache.evictions() > 0,
              "the staged promotion had to evict a slot to fit");
        check(cache.active_staged() == 1, "the staged promotion is counted");
        // The assertion this gate exists for: the SNAPSHOT addresses, read
        // without re-acquiring, still hold the current expert's bytes. No
        // re-promotion can launder this -- if the promotion had taken those
        // cells, they hold the staged expert's bytes right now.
        check(cache.in_vram(0, 0), "the leased expert is still resident");
        check(slice_matches(cur_gate, gate_buf, 0, "gate") &&
                  slice_matches(cur_up, up_buf, 0, "up") &&
                  slice_matches(cur_down, down_buf, 0, "down"),
              "a concurrent staged promotion did not overwrite the cells the leased expert is using");
        // And the staged expert is the one whose bytes landed, not a copy of the
        // current one: an aliased promotion that wrote the staged expert's cells
        // into the leased expert's addresses would pass the check above only by
        // luck.
        const ResidentExpert *staged = cache.acquire(src, 0, 2);
        check(staged && slice_matches(staged->gate, gate_buf, 2, "gate") &&
                  slice_matches(staged->up, up_buf, 2, "up") &&
                  slice_matches(staged->down, down_buf, 2, "down"),
              "the staged expert's device bytes are its own, not the incumbent's");
        check(cache.active_used() >= 1, "the staged copy was consumed once");
        // The pool never exceeded the cache's own budget while all this ran.
        const auto st = be->expert_pool_stats();
        check(st.live <= 7 * static_cast<size_t>(kPer),
              "physical commitment stayed inside the configured budget");
        cache.clear();
    }
    {   // One slot: staging cannot fit beside the current expert, so it refuses
        // rather than evicting the reader's weights.
        ExpertCache cache;
        cache.configure(be, 3 * kPer, 8 * 3 * kPer);
        const ResidentExpert *cur = cache.acquire(src, 0, 0);
        const i32 next_ids[] = {2};
        cache.prefetch_layer(src, 0, next_ids, 1);
        check(!cache.stage_next(src, 0, 2, 0), "a one-slot budget refuses to stage");
        check(cache.in_vram(0, 0), "and the current consumer keeps its slot");
        const ResidentExpert *after = cache.acquire(src, 0, 0);
        check(after && cur && after->gate.data == cur->gate.data &&
                  slice_matches(after->gate, gate_buf, 0, "gate"),
              "its device bytes survive the refused promotion");
        const ResidentExpert *nxt = cache.acquire(src, 0, 2);
        check(nxt && slice_matches(nxt->gate, gate_buf, 2, "gate"),
              "the refusal falls back to an exact demand load");
        cache.clear();
    }
    // The one-slot block is where cells are provably REUSED (its second acquire
    // evicts the first expert, which returns three cells to the free list), so
    // it is where the guard's mechanism can be asserted rather than the bytes
    // alone: those three cells were handed out carrying their release marks and
    // the promotion ordered itself against them on the device.
    const auto st = be->expert_pool_stats();
    std::printf("       lease reuse guards: device %llu, host %llu, implicit %llu\n",
                static_cast<unsigned long long>(st.order_device),
                static_cast<unsigned long long>(st.order_host),
                static_cast<unsigned long long>(st.order_implicit));
    check(st.order_device > 0, "a reused cell's guard was honored on the device");
    check(st.order_host == 0,
          "the promotion path never fell back to stopping the host for a guard");
    check(st.live == 0, "clearing both caches released every live cell");
    be->set_expert_pool_budget(0);
    return bad;
}

// ---------------------------------------------------------------------------
// logits_topk gate
// ---------------------------------------------------------------------------
//
// The op this tool could not check before: it has no reference in this file,
// and the two bugs it actually shipped with were both invisible to a
// correctness-by-reading pass --
//   * the packed key put the id in the low bits uncomplemented, so equal
//     logits ranked the HIGHER id first and the argmax disagreed with every
//     reference implementation;
//   * stage 4 iterated the whole candidate capacity instead of the count that
//     stage 3 had actually written, so last token's keys came back as this
//     token's when a cut left the buffer short.
// Both produced plausible output. The reference below is written from the
// contract, independently of the kernels, so agreeing with it means something.

// The host rule, spelled out: softcap, then the repetition penalty (divide a
// positive logit, multiply a negative one), then max by (value, lower id).
f32 host_transform(f32 v, i32 i, const std::vector<u32> &mask, f32 rep_pen,
                   f32 softcap) {
    if (softcap > 0.0f)
        v = v > softcap ? softcap : (v < -softcap ? -softcap : v);
    if (!mask.empty() && rep_pen > 0.0f && rep_pen != 1.0f &&
        ((mask[static_cast<size_t>(i) >> 5] >> (static_cast<size_t>(i) & 31)) &
         1u))
        v = v > 0.0f ? v / rep_pen : v * rep_pen;
    return v;
}

// Returns the number of mismatches. `k` ids are expected back, descending.
i64 gate_topk(Backend *be, const char *what, DType store, const f32 *src,
              i64 n, i32 k, const std::vector<u32> &mask, f32 rep_pen,
              f32 softcap, i32 cand_cap) {
    const size_t esz = store == DType::F16 ? 2 : 4;
    std::vector<u8> host(static_cast<size_t>(n) * esz);
    // The reference ranks the numbers the *device* will read, not the numbers
    // this file started with. For an f16 row those differ: 248320 random
    // logits in [-10, 10] collide heavily at f16 resolution, so ranking the
    // f32 source picks a different id than ranking what is stored, and the
    // gate would blame the kernel for the gate's own rounding.
    std::vector<f32> stored(static_cast<size_t>(n));
    for (i64 i = 0; i < n; i++) {
        if (store == DType::F16) {
            const u16 h = fp32_to_fp16(src[i]);
            host[static_cast<size_t>(i) * 2] = static_cast<u8>(h & 0xff);
            host[static_cast<size_t>(i) * 2 + 1] = static_cast<u8>(h >> 8);
            stored[static_cast<size_t>(i)] =
                fp16_to_fp32(static_cast<u16>(h));
        } else {
            std::memcpy(host.data() + static_cast<size_t>(i) * 4, &src[i], 4);
            stored[static_cast<size_t>(i)] = src[i];
        }
    }
    void *ld = be->alloc(host.size());
    void *sd = be->alloc(4096 + static_cast<size_t>(cand_cap) * 8);
    void *od = be->alloc(static_cast<size_t>(k) * 8 + 16);
    void *md = mask.empty() ? nullptr
                            : be->alloc(mask.size() * sizeof(u32));
    be->upload(ld, host.data(), host.size(), 0);
    if (md) be->upload(md, mask.data(), mask.size() * sizeof(u32), 0);
    // ids then values then the count, all inside one allocation so the
    // readback is a single transfer. The count goes after BOTH arrays, not at
    // a fixed offset: at a fixed +8 it lands inside ids[] the moment k > 1,
    // which is a wrong answer rather than a crash. Engine lays it out the same
    // way, and for the same reason.
    char *zone = static_cast<char *>(od);
    const size_t count_off = static_cast<size_t>(k) * 8;
    be->logits_topk(ld, n, k, md, rep_pen, softcap, 1.0f,
                    reinterpret_cast<i32 *>(zone),
                    reinterpret_cast<f32 *>(zone + static_cast<size_t>(k) * 4),
                    nullptr, sd, cand_cap,
                    reinterpret_cast<i64 *>(zone + count_off));
    be->sync();
    const i64 out_bytes = static_cast<i64>(k) * 8 + 16;
    std::vector<u8> back(static_cast<size_t>(out_bytes));
    be->download(back.data(), od, static_cast<size_t>(out_bytes), 0);

    // Reference: rank by (value desc, id asc).
    std::vector<i32> order(static_cast<size_t>(n));
    for (i64 i = 0; i < n; i++) order[static_cast<size_t>(i)] = static_cast<i32>(i);
    std::vector<f32> tv(static_cast<size_t>(n));
    for (i64 i = 0; i < n; i++)
        tv[static_cast<size_t>(i)] = host_transform(
            stored[static_cast<size_t>(i)], static_cast<i32>(i), mask, rep_pen,
            softcap);
    std::stable_sort(order.begin(), order.end(),
                     [&](i32 a, i32 b) { return tv[static_cast<size_t>(a)] >
                                                 tv[static_cast<size_t>(b)]; });
    i64 bad = 0;
    i64 count = -1;
    std::memcpy(&count, back.data() + count_off, 8);
    // The count is checked FIRST because it decides whether the ids mean
    // anything: a count above cand_cap means the candidate buffer was too
    // small for the tie sitting on the cut, and the ids are then undefined by
    // contract. A total tie across a 248k vocabulary legitimately overflows a
    // 4096 buffer -- reporting that is the op working, not failing.
    if (count < 1) {
        std::printf("  FAIL %-22s count=%lld (must be >= 1)\n",
                    what, static_cast<long long>(count));
        ++bad;
    }
    if (count > cand_cap) {
        std::printf("  ok   %-22s n=%-7lld k=%-3d reports truncation "
                    "(count=%lld > cap=%d): ids undefined, as documented\n",
                    what, static_cast<long long>(n), k,
                    static_cast<long long>(count), cand_cap);
        be->release(ld);
        be->release(sd);
        be->release(od);
        if (md) be->release(md);
        return 0;
    }
    for (i32 r = 0; r < k && r < n; r++) {
        i32 gid = -1;
        f32 gv = 0.0f;
        // ids[] then vals[], each contiguous, k of them. Interleaving them
        // (id,val,id,val) is what this gate did first and it read rank 0's
        // value from rank 1's id, which is a wrong answer rather than a crash.
        std::memcpy(&gid, back.data() + static_cast<size_t>(r) * 4, 4);
        std::memcpy(&gv, back.data() + static_cast<size_t>(k) * 4 +
                                  static_cast<size_t>(r) * 4, 4);
        const i32 wid = order[static_cast<size_t>(r)];
        // Both sides read the same stored values now, so the id is a discrete
        // answer and has to match exactly; the value is the same f32 and only
        // has to survive the trip through the kernel unchanged.
        const double dv = std::fabs(static_cast<double>(gv) -
                                    static_cast<double>(tv[static_cast<size_t>(wid)]));
        if (gid != wid || dv > 1e-3) {
            if (bad == 0) {
                // Dump the whole device-side state of the selection: the raw
                // winning key, the cut bin the scan chose, how many candidates
                // were collected, and the first few of them. A wrong id, a
                // wrong cut and an empty candidate buffer are three different
                // bugs, and this distinguishes them without a debugger.
                be->sync();
                u64 raw = 0;
                u32 cutbin = 0;
                u64 got_count = 0;
                u64 cand3[3] = {0, 0, 0};
                be->download(&raw, sd, sizeof(raw), 2048);
                be->download(&cutbin, sd, sizeof(cutbin), 2056);
                be->download(&got_count, sd, sizeof(got_count), 2064);
                be->download(cand3, sd, sizeof(cand3), 4096);
                std::printf("  FAIL %-22s r=%d got id=%d val=%.6f  want id=%d "
                            "val=%.6f\n"
                            "         key=0x%016llx cut_bin=%u collected=%llu "
                            "cand[0..2]=0x%016llx,0x%016llx,0x%016llx\n",
                            what, r, gid, static_cast<double>(gv), wid,
                            static_cast<double>(tv[static_cast<size_t>(wid)]),
                            static_cast<unsigned long long>(raw), cutbin,
                            static_cast<unsigned long long>(got_count),
                            static_cast<unsigned long long>(cand3[0]),
                            static_cast<unsigned long long>(cand3[1]),
                            static_cast<unsigned long long>(cand3[2]));
            }
            ++bad;
        }
    }
    be->release(ld);
    be->release(sd);
    be->release(od);
    if (md) be->release(md);
    std::printf("  %-4s %-22s n=%-7lld k=%-3d %s\n", bad ? "FAIL" : "ok", what,
                static_cast<long long>(n), k,
                bad ? "mismatch above" : "matches host rule");
    return bad;
}

// Every row shape here is adversarial on purpose: a full tie, a tie only at
// the top, a shelf of softcapped saturation, and a penalty that promotes a
// token from outside the naive top set. These are the cases that decide
// whether the ranking is exact or merely plausible.
i64 gate_topk_all(Backend *be) {
    i64 bad = 0;
    const i64 n = 248320; // the Qwen3.5 vocabulary: the shape that is shipped
    std::vector<f32> row(static_cast<size_t>(n));
    u32 rng = 7u;
    // xorshift32, not an LCG: `rng >> 20` from a 32-bit LCG yields 12 bits of
    // entropy, which put every synthetic "random" logit inside a 1.25-wide
    // band and made the row a near-total tie at f16 resolution. A gate whose
    // data is degenerate tests the degenerate case and nothing else.
    auto next = [&]() {
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        return rng;
    };
    auto fill = [&](i32 mode) {
        for (i64 i = 0; i < n; i++) {
            const f32 r = static_cast<f32>(next() >> 8) / 16777216.0f;
            switch (mode) {
            case 0: row[static_cast<size_t>(i)] = r * 20.0f - 10.0f; break;
            case 1: row[static_cast<size_t>(i)] = 1.0f; break;  // total tie
            case 2:                                        // tie only at the top
                row[static_cast<size_t>(i)] =
                    (i % 1000 == 3) ? 7.5f : r * 2.0f - 1.0f;
                break;
            default:                                       // softcap shelf
                row[static_cast<size_t>(i)] =
                    (i % 7 == 0) ? 40.0f : r * 4.0f - 2.0f;
                break;
            }
        }
    };
    // A repetition mask over the ids a real ring buffer would hold.
    std::vector<u32> mask(static_cast<size_t>((n + 31) / 32), 0u);
    for (i64 i = 0; i < 64; i++) mask[static_cast<size_t>(i / 32)] |=
        1u << (i % 32);
    for (i64 i = 0; i < 64; i++) mask[static_cast<size_t>((n - 1 - i) / 32)] |=
        1u << ((n - 1 - i) % 32);

    struct Case { const char *what; i32 mode; f32 pen; f32 cap; i32 k; };
    const Case cases[] = {
        {"plain k=1", 0, 1.0f, 0.0f, 1},
        {"plain k=8", 0, 1.0f, 0.0f, 8},
        {"total tie k=1", 1, 1.0f, 0.0f, 1},
        {"total tie k=8", 1, 1.0f, 0.0f, 8},
        {"top tie k=8", 2, 1.0f, 0.0f, 8},
        {"softcap shelf k=1", 3, 1.0f, 20.0f, 1},
        {"softcap shelf k=8", 3, 1.0f, 20.0f, 8},
        {"repetition penalty k=8", 0, 1.15f, 0.0f, 8},
        {"penalty+tie k=8", 2, 1.15f, 0.0f, 8},
    };
    for (const Case &c : cases) {
        fill(c.mode);
        // The row is stored in the backend's *activation* type, because that is
        // what the op reads: the contract is "the head's output buffer", and
        // there is no dtype argument to say otherwise. Testing an f32 row
        // against an f16 backend therefore measures nothing except that f16
        // bytes read as f32 look like garbage -- which is precisely how this
        // gate wasted its first run.
        bad += gate_topk(be, c.what, be->act_type(), row.data(), n, c.k, mask,
                         c.pen, c.cap, 4096);
    }
    // A small vocabulary, where the whole row fits one block: the span and
    // block-count arithmetic takes a different branch there.
    std::vector<f32> small(1000);
    for (size_t i = 0; i < small.size(); i++)
        small[i] = static_cast<f32>((i * 7919u) % 13u);
    bad += gate_topk(be, "small vocab k=8", be->act_type(), small.data(),
                     static_cast<i64>(small.size()), 8, {}, 1.0f, 0.0f, 4096);
    std::printf("  (rows stored as %s, the backend's activation type)\n",
                dtype_tag(be->act_type()));
    return bad;
}

// A pure load-store ceiling for the decode shapes: one copy_act over a fixed
// buffer so the remaining fold candidates (residual fold, per-matrix wt[], silu
// fold on the epilogue) rank against the bandwidth shelf rather than the launch
// floor. It is a shelf, not a bottleneck: "how fast is pure bandwidth on this
// backend" -- not a claim about any particular matmul.
bool floor_floor(Backend *be, int iters) {
    size_t buf_size = 1024 * 1024;
    void *a = be->alloc(buf_size);
    void *b = be->alloc(buf_size);
    if (!a || !b) {
        std::fprintf(stderr,
                     "kraken-bench: copy/alloc fallback for a small benchmark failed\n");
        be->release(a);
        be->release(b);
        return false;
    }
    be->fill0(a, buf_size);
    be->fill0(b, buf_size);
    const double ms = time_calls(
        be, iters, [&] {
            be->copy_act(b, a, static_cast<i64>(1024 * 1024) /
                                   static_cast<i64>(be->act_size()));
        });
    const i64 bytes = 1024 * 1024;
    double mb = static_cast<double>(bytes) / 1e6;
    // copy_act copies one direction, so throughput here is in bytes/scheduled,
    // not bytes*2.
    double gbs = bytes / (ms * 1e6);
    std::printf("sum load-store overhead (excluding device-side work):  "
                "%12.2f ms / %8.1f gigabyte-sec (%8.1f GB/s, %6.2f us/run)\n",
                ms, static_cast<double>(bytes) / (1e9 * ms), gbs, ms * 1000.0);
    be->release(a);
    be->release(b);
    return true;
}

// ---------------------------------------------------------------------------
// speculation gate: a self-draft must accept tokens AND emit plain's text
// ---------------------------------------------------------------------------
//
// DIAG-decode-cpu.md F20: an 8.9x speculative regression was invisible to the
// gate suite because no arm ever ran `--draft`. The repo's own smallest model
// makes the arm free of new artifacts: `--draft <the same file>` is a
// self-draft, and the assertion a serving stack cares about is exactly two
// facts -- accepted > 0 (the accept/rewind loop works) and the text identical
// to plain (the verify loop changed nothing; at greedy the verifier pursues
// argmax branches, so accepted tokens are bit-identical by construction).
//
// The arm spawns the CLI twice and reads stdout, which carries only the
// banner and the generated text -- every [info]/[stats] line, including the
// acceptance figure (`[stats ] speculation N rounds, A/B draft tokens
// accepted`), is on stderr. A missing binary or model is a FAIL, not a skip:
// this gate already refuses to run without a HIP device, and an arm that
// never ran proves nothing on a box whose model corpus exists.

// std::system encodes the child's status differently per platform: Windows
// hands back the exit code, POSIX a wait status. Read both the same way.
static int sys_exit(int rc) {
#ifdef _WIN32
    return rc;
#else
    if (rc == -1) return -1;
    if (WIFEXITED(rc)) return WEXITSTATUS(rc);
    return 128 + (WIFSIGNALED(rc) ? WTERMSIG(rc) : 0);
#endif
}

static std::string slurp_file(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    return std::string(std::istreambuf_iterator<char>(f),
                       std::istreambuf_iterator<char>());
}

// stdout is the banner line plus the generated text. Drop the banner (both
// arms name the same device, but the comparison should be about the answer)
// and the CRs a Windows pipe leaves behind, so what remains is exactly the
// text a coherence check would compare.
static std::string spec_text_of(std::string s) {
    if (s.rfind("KRAKEN", 0) == 0) {
        const size_t nl = s.find_first_of("\r\n");
        s = nl == std::string::npos ? std::string() : s.substr(nl + 1);
        if (!s.empty() && s[0] == '\n') s.erase(0, 1);
    }
    std::string out;
    out.reserve(s.size());
    for (char c : s)
        if (c != '\r') out.push_back(c);
    return out;
}

static i64 gate_speculation(const char *self) {
    const auto check = [](bool ok, const char *what) {
        std::printf("  %s spec %s\n", ok ? "ok  " : "FAIL", what);
        return ok ? 0 : 1;
    };
    const auto exists = [](const std::string &p) {
        std::ifstream f(p, std::ios::binary);
        return f.good();
    };
    const auto dir_of = [](const std::string &p) {
        const size_t s = p.find_last_of("/\\");
        return s == std::string::npos ? std::string(".") : p.substr(0, s);
    };
    const auto pick = [&](const std::vector<std::string> &c) {
        for (const std::string &p : c)
            if (exists(p)) return p;
        return std::string();
    };

    const std::string sd = dir_of(self && *self ? self : "kraken-bench");
    const std::string bin = pick({sd + "/kraken.exe", sd + "/kraken",
                                  "build-hip/kraken.exe", "build-hip/kraken"});
    const char *rel = "models/SmolLM2-135M-Instruct.Q4_K_M.gguf";
    const std::string model = pick({rel, std::string("../") + rel,
                                    std::string("../../") + rel,
                                    sd + "/../" + rel, sd + "/../../" + rel});
    if (bin.empty()) {
        std::printf("  FAIL spec the kraken CLI is not beside the bench binary "
                    "(searched kraken[.exe] in %s)\n", sd.c_str());
        return 1;
    }
    if (model.empty()) {
        std::printf("  FAIL spec %s not found: the arm cannot run, and an "
                    "untested path is not a passing one\n", rel);
        return 1;
    }

    const std::string tmp = [] {
        const char *e = std::getenv("TMPDIR");
        if (!e || !*e) e = std::getenv("TEMP");
        if (!e || !*e) e = std::getenv("TMP");
        return std::string(e && *e ? e : ".");
    }();
    const std::string stamp = std::to_string(static_cast<long long>(
        std::chrono::steady_clock::now().time_since_epoch().count()));

    const auto run = [&](bool draft, std::string *out, std::string *err) {
        const std::string o = tmp + "/kraken-spec-" + stamp +
                              (draft ? "-d.out" : "-p.out");
        const std::string e = tmp + "/kraken-spec-" + stamp +
                              (draft ? "-d.err" : "-p.err");
        std::string cmd = "\"" + bin + "\" -m \"" + model +
                          "\" -p \"The capital of France\" -n 8 --greedy "
                          "--ctx 512";
        if (draft) cmd += " --draft \"" + model + "\"";
        cmd += " > \"" + o + "\" 2> \"" + e + "\"";
#ifdef _WIN32
        // cmd /c strips the FIRST and LAST quote of a command line that
        // starts with one, which would leave the last redirect target
        // unterminated -- a parse error with both files empty and rc=1.
        // `call` makes the line start with a word instead of a quote.
        const std::string full = "call " + cmd;
#else
        const std::string full = cmd;
#endif
        const int rc = sys_exit(std::system(full.c_str()));
        *out = slurp_file(o);
        *err = slurp_file(e);
        if (rc != 0) {
            std::printf("       spawn rc=%d cmd: %s\n", rc, cmd.c_str());
            if (err->empty()) {
                std::printf("       stderr empty, stdout %zu byte(s)\n",
                            out->size());
            } else {
                const size_t n = err->size() < 300 ? err->size() : 300;
                std::printf("       stderr: %.*s\n", static_cast<int>(n),
                            err->c_str());
            }
            return rc; // keep the files: they are the only evidence
        }
        std::remove(o.c_str());
        std::remove(e.c_str());
        return rc;
    };
    // `[stats ] speculation N rounds, A/B draft tokens accepted (x%)` -- read
    // the A out of the A/B immediately before the phrase, so the check does
    // not depend on the surrounding wording.
    const auto acceptance = [](const std::string &err, i64 *acc, i64 *prop) {
        const size_t at = err.find("draft tokens accepted");
        if (at == std::string::npos) return false;
        const size_t slash = err.rfind('/', at);
        if (slash == std::string::npos || slash == 0) return false;
        size_t b = slash;
        while (b > 0 && err[b - 1] >= '0' && err[b - 1] <= '9') --b;
        size_t a = slash + 1;
        while (a < err.size() && err[a] >= '0' && err[a] <= '9') ++a;
        if (b == slash || a == slash + 1) return false;
        *acc = std::atoll(err.substr(b, slash - b).c_str());
        *prop = std::atoll(err.substr(slash + 1, a - slash - 1).c_str());
        return true;
    };

    std::string po, pe, so, se;
    const int prc = run(false, &po, &pe);
    const int src = run(true, &so, &se);
    i64 bad = 0;
    bad += check(prc == 0, "the plain run exits 0");
    bad += check(src == 0, "the self-draft run exits 0");
    const std::string plain = spec_text_of(po), drafted = spec_text_of(so);
    bad += check(!plain.empty() && !drafted.empty(), "both runs produced text");
    i64 acc = 0, prop = 0;
    if (acceptance(se, &acc, &prop)) {
        char what[160];
        std::snprintf(what, sizeof what, "reports acceptance (%lld/%lld)",
                      static_cast<long long>(acc), static_cast<long long>(prop));
        bad += check(acc > 0, what);
    } else {
        bad += check(false, "the draft run reports its acceptance figure");
    }
    if (!plain.empty() && !drafted.empty()) {
        bad += check(plain == drafted,
                     plain == drafted
                         ? "self-draft text is identical to plain"
                         : "self-draft text differs from plain");
    }
    if (plain != drafted && !plain.empty() && !drafted.empty()) {
        std::printf("       plain: %.70s\n", plain.c_str());
        std::printf("       draft: %.70s\n", drafted.c_str());
    }
    return bad;
}


// ---- rows bit-exactness ---------------------------------------------------
//
// A decode step runs every projection with rows == 1 and a prefill (or a
// speculative verify) runs the same projection with rows > 1. The engine's
// text is only chunk-invariant if those two produce the SAME bytes for the same
// input row. This arm answers that per kernel instead of per model: it runs one
// matrix at rows=1 and at rows=N with the same weights and the same first input
// row, and compares the f16 outputs bit for bit.
//
// It exists because "the text diverges on Tiel after ~29 greedy steps" had no
// smaller reproduction: the model has 40 layers, 30 of them a recurrence, and
// any one of the projections could have been the source. A per-kernel verdict is
// the difference between "a kernel-arithmetic project" and "this kernel".
struct BitExact {
    i64 differ = 0;
    i64 n = 0;
    i64 max_ulp = 0;
    f64 max_abs = 0.0;
};

// f16 as an ordered integer: the ULP distance between two halves is the
// difference of these, which is what "the last bit" means in this storage type.
i32 half_ordered(u16 h) {
    return (h & 0x8000u) ? -static_cast<i32>(h & 0x7fffu)
                         : static_cast<i32>(h);
}

BitExact cmp_half(const std::vector<u16> &a, const std::vector<u16> &b, i64 n) {
    BitExact r;
    r.n = n;
    for (i64 i = 0; i < n; i++) {
        if (a[static_cast<size_t>(i)] == b[static_cast<size_t>(i)]) continue;
        r.differ++;
        const i32 ua = half_ordered(a[static_cast<size_t>(i)]);
        const i32 ub = half_ordered(b[static_cast<size_t>(i)]);
        const i64 du = std::llabs(static_cast<i64>(ua) - static_cast<i64>(ub));
        if (du > r.max_ulp) r.max_ulp = du;
        const f32 fa = fp16_to_fp32(a[static_cast<size_t>(i)]);
        const f32 fb = fp16_to_fp32(b[static_cast<size_t>(i)]);
        const f64 d = std::fabs(static_cast<f64>(fa) - static_cast<f64>(fb));
        if (d > r.max_abs) r.max_abs = d;
    }
    return r;
}

void print_bitexact(const char *what, const char *tag, i64 n_out, i64 n_in,
                    i64 rows, i64 row, const BitExact &r) {
    if (r.differ == 0) {
        std::printf("  %-14s %-5s out=%-6lld in=%-5lld rows=%-3lld row=%-2lld  "
                    "IDENTICAL (%lld values)\n",
                    what, tag, static_cast<long long>(n_out),
                    static_cast<long long>(n_in),
                    static_cast<long long>(rows),
                    static_cast<long long>(row),
                    static_cast<long long>(r.n));
    } else {
        std::printf("  %-14s %-5s out=%-6lld in=%-5lld rows=%-3lld row=%-2lld  "
                    "DIFFERS %lld/%lld  max %lld ulp  max|d|=%.6g\n",
                    what, tag, static_cast<long long>(n_out),
                    static_cast<long long>(n_in),
                    static_cast<long long>(rows),
                    static_cast<long long>(row),
                    static_cast<long long>(r.differ),
                    static_cast<long long>(r.n),
                    static_cast<long long>(r.max_ulp), r.max_abs);
    }
}

// One gemm() at rows=1 and at rows=N, same weights, same input row. `row` is
// which row of the batch to compare: row 0 uses the same x pointer, a later row
// uses x + row*n_in so a wrong row-to-output mapping cannot hide in it.
BitExact gemm_rows_case(Backend *be, DType t, i64 n_out, i64 n_in, i64 rows,
                        i64 row, std::vector<u16> &x, std::vector<u8> &w) {
    const size_t row_bytes = dtype_row_bytes(t, n_in);
    void *wd = be->alloc(w.size());
    void *xd = be->alloc(x.size() * 2);
    void *od1 = be->alloc(static_cast<size_t>(n_out) * 2);
    void *odn = be->alloc(static_cast<size_t>(rows) * static_cast<size_t>(n_out) * 2);
    be->upload(wd, w.data(), w.size(), 0);
    be->upload(xd, x.data(), x.size() * 2, 0);
    const void *xrow = static_cast<const u8 *>(xd) +
                       static_cast<size_t>(row) * static_cast<size_t>(n_in) * 2;
    be->gemm(od1, xrow, wd, t, n_out, n_in, 1);
    be->gemm(odn, xd, wd, t, n_out, n_in, rows);
    std::vector<u16> a(static_cast<size_t>(n_out)), b(static_cast<size_t>(n_out));
    be->download(a.data(), od1, a.size() * 2);
    be->download(b.data(), static_cast<const u8 *>(odn) +
                             static_cast<size_t>(row) * static_cast<size_t>(n_out) * 2,
                 b.size() * 2);
    const BitExact r = cmp_half(a, b, n_out);
    be->release(wd);
    be->release(xd);
    be->release(od1);
    be->release(odn);
    (void)row_bytes;
    return r;
}

// The gate/up pair the MoE path and every dense FFN run: gemm_group with the
// silu fold, which is fused at rows==1 and a fallback (two gemms + silu_mul)
// above it. That is a different expression, so it is the other candidate.
BitExact group_rows_case(Backend *be, DType t, i64 n_ff, i64 n_in, i64 rows,
                         i64 row, std::vector<u16> &x, std::vector<u8> &w) {
    const size_t row_bytes = dtype_row_bytes(t, n_in);
    void *wd = be->alloc(w.size());
    void *xd = be->alloc(x.size() * 2);
    void *g1 = be->alloc(static_cast<size_t>(n_ff) * 2);
    void *u1 = be->alloc(static_cast<size_t>(n_ff) * 2);
    void *gn = be->alloc(static_cast<size_t>(rows) * static_cast<size_t>(n_ff) * 2);
    void *un = be->alloc(static_cast<size_t>(rows) * static_cast<size_t>(n_ff) * 2);
    be->upload(wd, w.data(), w.size(), 0);
    be->upload(xd, x.data(), x.size() * 2, 0);
    const void *xrow = static_cast<const u8 *>(xd) +
                       static_cast<size_t>(row) * static_cast<size_t>(n_in) * 2;

    void *out1[2] = {g1, u1};
    const void *gw[2] = {wd, wd};
    const i64 gno[2] = {n_ff, n_ff};
    bool fused1 = false;
    be->gemm_group(out1, xrow, gw, t, gno, n_in, 2, 1, true, &fused1, false);
    if (!fused1) be->silu_mul(g1, g1, u1, n_ff);

    void *outn[2] = {gn, un};
    bool fusedn = false;
    be->gemm_group(outn, xd, gw, t, gno, n_in, 2, rows, true, &fusedn, false);
    if (!fusedn)
        be->silu_mul(gn, gn, un, rows * n_ff);

    std::vector<u16> a(static_cast<size_t>(n_ff)), b(static_cast<size_t>(n_ff));
    be->download(a.data(), g1, a.size() * 2);
    be->download(b.data(), static_cast<const u8 *>(gn) +
                             static_cast<size_t>(row) * static_cast<size_t>(n_ff) * 2,
                 b.size() * 2);
    const BitExact r = cmp_half(a, b, n_ff);
    be->release(wd);
    be->release(xd);
    be->release(g1);
    be->release(u1);
    be->release(gn);
    be->release(un);
    (void)row_bytes;
    return r;
}

// ---- attention bit-exactness ---------------------------------------------
//
// The first divergence on a chunked prefill is NOT a projection. On Tiel
// (--chunk 1 against --chunk 2, 12-token prompt) the first differing stage is
// `attn.out L03` at position 1, rel 3.8e-5 -- after three recurrent layers that
// are bit-identical, and on the first full-attention layer. So the box is the
// attention kernel: n_tok == 1 takes attention_decode (one block per head,
// warp-strided keys), n_tok > 1 takes the query-tiled prefill kernel. Same
// keys, same query row, a different reduction order over the key range.
//
// This arm is that verdict per call rather than per model: one K/V cache, the
// same query rows, once as N calls of one row (what decode does) and once as a
// single call of N rows (what a prefill chunk does).
struct AttnCase {
    i64 hd;
    i64 n_head;
    i64 n_kv;
    const char *what;
};

BitExact attn_rows_case(Backend *be, const AttnCase &c, i64 n_tok, i64 n_keys) {
    const size_t qsz = static_cast<size_t>(n_tok) *
                       static_cast<size_t>(c.n_head) *
                       static_cast<size_t>(c.hd);
    const size_t ksz = static_cast<size_t>(n_keys) *
                       static_cast<size_t>(c.n_kv) *
                       static_cast<size_t>(c.hd);
    std::vector<u16> q(qsz), k(ksz), v(ksz);
    auto fill = [](std::vector<u16> &dst, u32 seed) {
        u32 rng = seed;
        for (u16 &h : dst) {
            rng = rng * 1103515245u + 12345u;
            const f32 f = static_cast<f32>((rng >> 20) & 0xff) / 255.0f - 0.5f;
            h = fp32_to_fp16(f * 0.5f);
        }
    };
    fill(q, 7u);
    fill(k, 11u);
    fill(v, 13u);

    void *qd = be->alloc(q.size() * 2);
    void *kd = be->alloc(k.size() * 2);
    void *vd = be->alloc(v.size() * 2);
    void *o1 = be->alloc(qsz * 2); // N single-row calls
    void *on = be->alloc(qsz * 2); // one N-row call
    be->upload(qd, q.data(), q.size() * 2, 0);
    be->upload(kd, k.data(), k.size() * 2, 0);
    be->upload(vd, v.data(), v.size() * 2, 0);

    AttnDesc d;
    d.n_head = c.n_head;
    d.n_kv = c.n_kv;
    d.hd = c.hd;
    d.layer = 0;
    d.layer_stride = static_cast<i64>(ksz);
    d.pos_stride = c.n_kv * c.hd;
    d.scale = static_cast<f32>(1.0 / std::sqrt(static_cast<f64>(c.hd)));
    d.causal = true;
    d.window = 0;
    const i64 pos0 = n_keys - n_tok;
    const size_t row_elems = static_cast<size_t>(c.n_head) *
                             static_cast<size_t>(c.hd);
    for (i64 r = 0; r < n_tok; r++) {
        d.n_tok = 1;
        d.pos0 = pos0 + r;
        be->attention(static_cast<u8 *>(o1) + static_cast<size_t>(r) * row_elems * 2,
                      static_cast<const u8 *>(qd) + static_cast<size_t>(r) * row_elems * 2,
                      kd, vd, d);
    }
    d.n_tok = n_tok;
    d.pos0 = pos0;
    be->attention(on, qd, kd, vd, d);

    std::vector<u16> a(qsz), b(qsz);
    be->download(a.data(), o1, a.size() * 2);
    be->download(b.data(), on, b.size() * 2);
    const BitExact r = cmp_half(a, b, static_cast<i64>(qsz));
    be->release(qd);
    be->release(kd);
    be->release(vd);
    be->release(o1);
    be->release(on);
    return r;
}

// Returns how many (kernel, rows) cases differ. It is printed, not enforced:
// a last-bit difference is a property of the arithmetic, not a broken kernel,
// and --gate must keep meaning "wrong or NaN", not "not chunk-invariant".
i64 arm_rows_bitexact(Backend *be) {
    struct Case {
        const char *what;
        DType t;
        i64 n_out;
        i64 n_in;
    };
    const Case cases[] = {
        // the shapes a decode step runs, plus the expert pair's FFN width
        {"gemm", DType::Q4_K, 4096, 4096},
        {"gemm", DType::Q6_K, 4096, 4096},
        {"gemm", DType::Q8_0, 4096, 4096},
        {"gemm", DType::F16, 4096, 4096},
        {"gemm", DType::Q4_K, 512, 2048},
        {"group+silu", DType::Q4_K, 512, 2048},
        {"group+silu", DType::Q8_0, 512, 2048},
        {"group+silu", DType::F16, 512, 2048},
    };
    // 8 is the last row count the decode ladder (`gemv_m_kernel`) covers and 9
    // is the first the tile rung takes, so the boundary is measured rather than
    // assumed: a chunk of 9..15 rows is the narrowest prefill that leaves the
    // exact path.
    const i64 row_counts[] = {2, 4, 8, 9, 12, 16, 32, 64};
    std::printf("\nrows bit-exactness (same weights, same input row, rows=1 "
                "against rows=N):\n");
    i64 differing = 0;
    for (const Case &c : cases) {
        // Size the input for the LARGEST batch, not for one of the entries: a
        // list whose middle element is not its maximum silently skipped every
        // larger row count and reported 0 differences, which is the one answer
        // this arm must never give by accident.
        i64 rows = 1;
        for (i64 rc : row_counts) rows = std::max(rows, rc);
        std::vector<u16> x(static_cast<size_t>(rows) * static_cast<size_t>(c.n_in));
        u32 rng = 12345u;
        std::vector<f32> xf(x.size());
        for (f32 &v : xf) {
            rng = rng * 1103515245u + 12345u;
            v = static_cast<f32>((rng >> 20) & 0xff) / 255.0f - 0.5f;
        }
        for (size_t i = 0; i < xf.size(); i++) x[i] = fp32_to_fp16(xf[i]);
        std::vector<u8> w = weights(c.t, dtype_row_bytes(c.t, c.n_in), c.n_out);
        for (i64 nrows : row_counts) {
            if (nrows > rows) continue;
            const i64 cmp_row = nrows - 1;
            const BitExact r =
                std::strcmp(c.what, "gemm") == 0
                    ? gemm_rows_case(be, c.t, c.n_out, c.n_in, nrows, cmp_row, x, w)
                    : group_rows_case(be, c.t, c.n_out, c.n_in, nrows, cmp_row, x, w);
            if (r.differ) differing++;
            print_bitexact(c.what, dtype_tag(c.t), c.n_out, c.n_in, nrows,
                           cmp_row, r);
        }
    }
    // ---- attention: the kernel the model-level bisect named ---------------
    //
    // Same verdict shape as above, one level up the stack: the whole attention
    // call, decode (one row per call) against prefill (one call of N rows),
    // over the same cache. `n_keys` is 64 so the decode key-range split stays
    // at 1 (attention_decode_splits() only cuts in at ~128 keys), i.e. this
    // compares the two kernels and not the split.
    const AttnCase attn[] = {
        {128, 16, 2, "attn"},
        {256, 16, 2, "attn"},
        {64, 8, 2, "attn"},
    };
    const i64 attn_rows[] = {2, 4, 8, 16};
    // 64 keys is below attention_decode_splits()' threshold and 200 is above
    // it, so the second count compares the unified prefill against the decode
    // path that splits its key range across blocks -- the shape a real decode
    // step takes from ~128 keys up.
    const i64 attn_keys[] = {64, 200};
    std::printf("\nattention bit-exactness (same cache, one row per call "
                "against one call of N rows):\n");
    for (const AttnCase &c : attn) {
        for (i64 n_keys : attn_keys) {
            for (i64 nrows : attn_rows) {
                const BitExact r = attn_rows_case(be, c, nrows, n_keys);
                if (r.differ) differing++;
                char tag[32];
                std::snprintf(tag, sizeof tag, "hd%lld/%lldh",
                              static_cast<long long>(c.hd),
                              static_cast<long long>(c.n_head));
                // out is one row's output width, in is keys attended: the
                // printed pair then reads "one row against N keys".
                print_bitexact(c.what, tag, c.n_head * c.hd, n_keys, nrows,
                               nrows - 1, r);
            }
        }
    }
    std::printf("\n%d of the (kernel, rows) cases are not bit-identical to the "
                "decode row. A case that differs here is a case where a prefill "
                "or a speculative verify writes different bytes than a decode "
                "step for the same input, so the model's text cannot be "
                "chunk-invariant on top of it.\n",
                static_cast<int>(differing));
    return 0; // informational: --gate means wrong/NaN, not "not invariant"
}

// Host->device copy rate, measured on the shapes the expert path actually
// issues. The point is not the card's spec sheet: it is whether THIS path (a
// pageable source, a pinned source, three slices per expert, one big copy)
// reaches the bus, because a promotion that runs at 7 GB/s on a link that does
// 25 is the whole decode budget.
int h2d_bench(Backend *be, int iters) {
    const size_t kExpert = 13219200;  // one gpt-oss expert: three 2880x2880 mxfp4
    const size_t kSlice = kExpert / 3;
    const size_t kBig = 128u << 20;
    struct Arm {
        const char *name;
        void *dst[36];
        const void *src[36];
        size_t bytes[36];
        int n;
        bool pinned;
    };
    void *dev = be->alloc(kBig);
    void *ph = be->alloc_host(kBig);
    void *pp = be->alloc_host_pageable(kBig);
    if (!dev || !ph || !pp) {
        std::printf("h2d: allocation failed\n");
        return 1;
    }
    std::memset(ph, 1, kBig);
    std::memset(pp, 1, kBig);
    auto make = [&](const char *name, int n, size_t bytes, bool pinned,
                    bool slices) {
        Arm a{};
        a.name = name;
        a.n = n;
        a.pinned = pinned;
        for (int i = 0; i < n; i++) {
            void *base = pinned ? ph : pp;
            const size_t span = kBig > bytes ? kBig - bytes : 1;
            a.src[i] = static_cast<const char *>(base) + (slices ? 0 : (i * bytes) % span);
            a.dst[i] = static_cast<char *>(dev) + (i * bytes) % span;
            a.bytes[i] = bytes;
        }
        return a;
    };
    // The three slices of one expert share one 13 MB pinned source, which is
    // what ExpertCache hands over: a promotion is 3 copies, not 1.
    Arm arms[5] = {
        make("pageable  1 x 12.6 MB", 1, kExpert, false, false),
        make("pinned    1 x 12.6 MB", 1, kExpert, true, false),
        make("pinned    3 x  4.4 MB", 3, kSlice, true, true),
        make("pinned    1 x  128 MB", 1, kBig, true, false),
        make("pinned   36 x  4.4 MB", 36, kSlice, true, true),
    };
    std::printf("--- host->device, per call (average of %d)\n", iters);
    for (Arm &a : arms) {
        const double ms = time_calls(be, iters, [&] {
            if (a.pinned)
                be->upload_pinned_batch(a.dst, a.src, a.bytes, a.n);
            else
                be->upload_paged_batch(a.dst, a.src, a.bytes, a.n);
        });
        double mb = 0;
        for (int i = 0; i < a.n; i++) mb += static_cast<double>(a.bytes[i]);
        std::printf("  %-24s %8.3f ms  %7.2f GB/s\n", a.name, ms,
                    ms > 0 ? mb / (ms * 1e6) : 0.0);
    }
    be->release(dev);
    be->release_host(ph);
    be->release_host_pageable(pp);
    return 0;
}

// ---- hyper-connections (qwen4exp): the device kernels against the host ------
//
// The lane-stack ops the qwen4exp forward runs (Backend::hc_init, hc_mix,
// hc_combine, silu_act). `tests/test_kraken.cpp` pins the HOST arithmetic and
// the identical-lane invariant on the CPU backend; this arm is the only thing
// that says the DEVICE kernels implement the same functions, and it exists
// because the default bodies of those ops write nothing at all: a backend with
// no implementation leaves the lane stack at its initial value and the model
// then decodes fluent text that is not this model. No test group opens the
// device, so nothing in the suite can see that.
//
// The oracle is not re-typed here -- it is krk::hc_mix / krk::hc_combine /
// krk::hc_lane_weight, the same library functions the CPU backend answers
// with. Geometry is the REAP-288 file's: four lanes of 2560, three tokens.
// Every input is rounded to f16 before the upload, so both arms see the same
// numbers and the only difference left is the kernel's arithmetic.
struct HcCmp {
    i64 n = 0;
    i64 differ = 0;
    i64 max_ulp = 0;
    f64 max_abs = 0.0;
    f64 scale = 0.0;
    f64 tol = 0.0;
    bool exact = false;
    bool ok() const { return differ == 0; }
};

// f32 oracle against an f16 device result. `exact` demands bit equality, which
// is the only honest bar for a copy and for the identical-lane invariant.
HcCmp cmp_hc_f32(const std::vector<u16> &got, const std::vector<f32> &want,
                 bool exact, f64 tol) {
    HcCmp r;
    r.n = static_cast<i64>(want.size());
    r.exact = exact;
    r.tol = tol;
    // The tolerance is measured against the oracle's own magnitude, so the
    // scale is gathered before anything is judged: a running maximum would let
    // an early small value decide a later large one.
    for (size_t i = 0; i < want.size(); i++) {
        const f64 aw = std::fabs(static_cast<f64>(want[i]));
        if (aw > r.scale) r.scale = aw;
    }
    const f64 limit = tol * std::max(1.0, r.scale);
    for (size_t i = 0; i < want.size(); i++) {
        const f32 g = fp16_to_fp32(got[i]);
        const f64 d = std::fabs(static_cast<f64>(g) - static_cast<f64>(want[i]));
        if (d > r.max_abs) r.max_abs = d;
        if (exact) {
            if (got[i] != fp32_to_fp16(want[i])) r.differ++;
        } else if (d > limit) {
            r.differ++;
        }
    }
    return r;
}

void print_hc(const char *what, const char *detail, const HcCmp &r) {
    if (r.ok()) {
        std::printf("  %-11s %-22s %s (%lld values)", what, detail,
                    r.exact ? "BIT-IDENTICAL" : "PASS",
                    static_cast<long long>(r.n));
        if (!r.exact)
            std::printf("  max|d|=%.4g tol=%.3g", r.max_abs, r.tol);
        std::printf("\n");
    } else {
        std::printf("  %-11s %-22s FAIL %lld/%lld  max|d|=%.6g  max %lld ulp"
                    "  tol=%.3g\n",
                    what, detail, static_cast<long long>(r.differ),
                    static_cast<long long>(r.n), r.max_abs,
                    static_cast<long long>(r.max_ulp), r.tol);
    }
}

// Returns how many cases failed. Every case is a device call whose answer the
// host oracle already knows, so "passed" here means the kernel agreed with the
// arithmetic the suite pins, not that the call did not crash.
i64 arm_hc(Backend *be) {
    std::printf("\nhyper-connections (4 lanes of 2560, 3 tokens):\n");
    if (!be->hc_supported()) {
        std::printf("  this backend does not implement the lane-stack ops "
                    "(hc_supported() is false) -- the qwen4exp forward refuses "
                    "to run on it\n");
        return 1;
    }
    const i64 ne = 2560, hc = 4, nt = 3;
    const i64 nl = nt * hc * ne; // one lane stack [nt, hc, ne]
    const i64 nmix = nt * ne;    // one single-lane stream [nt, ne]
    const i64 ninj = nt * hc;    // one injection matrix [nt, hc]
    HcGeometry g;
    g.n_embd = static_cast<i32>(ne);
    g.hc = static_cast<i32>(hc);
    g.low_rank = 320;

    auto q = [](f64 i) {
        return fp16_to_fp32(
            fp32_to_fp16(static_cast<f32>(std::sin(0.013 * i) * 2.0)));
    };
    std::vector<f32> x(static_cast<size_t>(nmix));
    std::vector<f32> xn(static_cast<size_t>(nl));
    std::vector<f32> gate(static_cast<size_t>(nl));
    std::vector<f32> block(static_cast<size_t>(nmix));
    std::vector<f32> inject(static_cast<size_t>(ninj));
    for (i64 i = 0; i < nmix; i++) x[static_cast<size_t>(i)] = q(i + 1);
    for (i64 i = 0; i < nmix; i++) block[static_cast<size_t>(i)] = q(5 * i + 17);
    for (i64 i = 0; i < nl; i++) xn[static_cast<size_t>(i)] = q(i + 7);
    for (i64 i = 0; i < nl; i++) gate[static_cast<size_t>(i)] = q(3 * i + 11);
    for (i64 i = 0; i < ninj; i++) inject[static_cast<size_t>(i)] = q(9 * i + 23);

    auto up = [](const std::vector<f32> &v) {
        std::vector<u16> h(v.size());
        for (size_t i = 0; i < v.size(); i++) h[i] = fp32_to_fp16(v[i]);
        return h;
    };
    const std::vector<u16> xh = up(x), xnh = up(xn), gh = up(gate),
                           bh = up(block), ih = up(inject);

    void *xd = be->alloc(static_cast<size_t>(nmix) * 2);
    void *xnd = be->alloc(static_cast<size_t>(nl) * 2);
    void *gd = be->alloc(static_cast<size_t>(nl) * 2);
    void *bd = be->alloc(static_cast<size_t>(nmix) * 2);
    void *id = be->alloc(static_cast<size_t>(ninj) * 2);
    void *lanes = be->alloc(static_cast<size_t>(nl) * 2);
    void *mixd = be->alloc(static_cast<size_t>(nmix) * 2);
    void *res = be->alloc(static_cast<size_t>(nl) * 2);
    be->upload(xd, xh.data(), xh.size() * 2, 0);
    be->upload(xnd, xnh.data(), xnh.size() * 2, 0);
    be->upload(gd, gh.data(), gh.size() * 2, 0);
    be->upload(bd, bh.data(), bh.size() * 2, 0);
    be->upload(id, ih.data(), ih.size() * 2, 0);

    auto fetch = [&](const void *src, i64 n) {
        std::vector<u16> h(static_cast<size_t>(n));
        be->download(h.data(), src, h.size() * 2);
        return h;
    };
    i64 failed = 0;

    // (1) hc_init: all hc lanes start as copies of the embedding row. A copy,
    // so bit equality is the bar.
    {
        be->hc_init(lanes, xd, nt, hc, ne);
        const std::vector<u16> got = fetch(lanes, nl);
        std::vector<f32> want(static_cast<size_t>(nl));
        for (i64 t = 0; t < nt; t++)
            for (i64 c = 0; c < hc; c++)
                for (i64 e = 0; e < ne; e++)
                    want[static_cast<size_t>((t * hc + c) * ne + e)] =
                        x[static_cast<size_t>(t * ne + e)];
        const HcCmp r = cmp_hc_f32(got, want, true, 0.0);
        print_hc("hc_init", "lanes <- embedding", r);
        if (!r.ok()) failed++;
    }

    // (2) hc_mix: mean over lanes of xn * sigmoid(gate). The oracle is
    // krk::hc_mix, the same function the CPU backend answers with.
    {
        be->hc_mix(mixd, xnd, gd, nt, hc, ne);
        const std::vector<u16> got = fetch(mixd, nmix);
        std::vector<f32> want(static_cast<size_t>(nmix), 0.0f);
        krk::hc_mix(want.data(), xn.data(), gate.data(), static_cast<i32>(nt), g);
        const HcCmp r = cmp_hc_f32(got, want, false, 2e-3);
        print_hc("hc_mix", "mean(xn*sigmoid(gate))", r);
        if (!r.ok()) failed++;
    }

    // (3) hc_combine: res_lane += block_out * 2*sigmoid(inject/hc). The lane
    // weights are the shared centred rule, so a plain sigmoid here would show.
    {
        be->hc_init(res, xd, nt, hc, ne);
        be->hc_combine(res, bd, id, nt, hc, ne);
        const std::vector<u16> got = fetch(res, nl);
        std::vector<f32> want(static_cast<size_t>(nl), 0.0f);
        for (i64 t = 0; t < nt; t++)
            for (i64 c = 0; c < hc; c++)
                for (i64 e = 0; e < ne; e++)
                    want[static_cast<size_t>((t * hc + c) * ne + e)] =
                        x[static_cast<size_t>(t * ne + e)];
        krk::hc_combine(want.data(), block.data(), inject.data(),
                        static_cast<i32>(nt), g);
        const HcCmp r = cmp_hc_f32(got, want, false, 2e-3);
        print_hc("hc_combine", "res += out*2*sigmoid(inj/hc)", r);
        // The lane weight itself, on the same values: 2*sigmoid(0) is exactly
        // 1, so an un-injected lane gets the block output once and only once.
        // Part of this case, not a case of its own.
        const f32 w0 = krk::hc_lane_weight(0.0f, static_cast<i32>(hc));
        if (!(w0 == 1.0f)) {
            std::printf("  %-11s %-22s FAIL w(0) = %.9g, not exactly 1\n",
                        "hc_combine", "zero injection is identity",
                        static_cast<f64>(w0));
        }
        if (!r.ok() || !(w0 == 1.0f)) failed++;
    }

    // (4) silu_act: x <- x * sigmoid(x), the mixer's low-rank nonlinearity.
    {
        void *sd = be->alloc(static_cast<size_t>(nl) * 2);
        be->upload(sd, gh.data(), gh.size() * 2, 0);
        be->silu_act(sd, nl);
        const std::vector<u16> got = fetch(sd, nl);
        std::vector<f32> want(static_cast<size_t>(nl));
        for (i64 i = 0; i < nl; i++) {
            const f32 v = gate[static_cast<size_t>(i)];
            want[static_cast<size_t>(i)] = v / (1.0f + std::exp(-v));
        }
        const HcCmp r = cmp_hc_f32(got, want, false, 2e-3);
        print_hc("silu_act", "x*sigmoid(x)", r);
        if (!r.ok()) failed++;
        be->release(sd);
    }

    // (5) The identical-lane invariant, on the device. All hc lanes equal and
    // the injection exactly zero: every lane weight is 1, so the lanes stay
    // BIT-identical through a combine, and the mix collapses to
    // xn * sigmoid(gate) on one lane. This is the property the CPU fixture
    // pins; a kernel that summed lanes with the wrong stride, or that read the
    // gate of lane 0 for every lane, fails here and not in the suite.
    {
        std::vector<f32> eq_xn(static_cast<size_t>(nl));
        std::vector<f32> eq_gate(static_cast<size_t>(nl));
        for (i64 t = 0; t < nt; t++)
            for (i64 c = 0; c < hc; c++)
                for (i64 e = 0; e < ne; e++) {
                    eq_xn[static_cast<size_t>((t * hc + c) * ne + e)] =
                        xn[static_cast<size_t>(t * ne + e)];
                    eq_gate[static_cast<size_t>((t * hc + c) * ne + e)] =
                        gate[static_cast<size_t>(t * ne + e)];
                }
        const std::vector<u16> exnh = up(eq_xn), egh = up(eq_gate);
        std::vector<u16> zero_h(static_cast<size_t>(ninj), 0);
        be->upload(xnd, exnh.data(), exnh.size() * 2, 0);
        be->upload(gd, egh.data(), egh.size() * 2, 0);
        be->upload(id, zero_h.data(), zero_h.size() * 2, 0);
        be->hc_init(res, xd, nt, hc, ne);
        be->hc_combine(res, bd, id, nt, hc, ne);
        const std::vector<u16> got = fetch(res, nl);
        i64 differ = 0;
        for (i64 t = 0; t < nt; t++)
            for (i64 c = 1; c < hc; c++)
                for (i64 e = 0; e < ne; e++) {
                    const i64 a = (t * hc) * ne + e;
                    const i64 b = (t * hc + c) * ne + e;
                    if (got[static_cast<size_t>(a)] != got[static_cast<size_t>(b)])
                        differ++;
                }
        std::printf("  %-11s %-22s %s", "lanes-equal", "inject 0 keeps lanes",
                    differ == 0 ? "BIT-IDENTICAL" : "FAIL");
        if (differ == 0) {
            std::printf(" (%lld values)\n",
                        static_cast<long long>(nt * (hc - 1) * ne));
        } else {
            std::printf("  %lld/%lld lanes differ\n",
                        static_cast<long long>(differ),
                        static_cast<long long>(nt * (hc - 1) * ne));
            failed++;
        }

        be->hc_mix(mixd, xnd, gd, nt, hc, ne);
        const std::vector<u16> mg = fetch(mixd, nmix);
        std::vector<f32> want(static_cast<size_t>(nmix), 0.0f);
        krk::hc_mix(want.data(), eq_xn.data(), eq_gate.data(),
                    static_cast<i32>(nt), g);
        // One ulp: the kernel sums hc equal terms and divides once, the oracle
        // does the same, so only the f16 output rounding may differ.
        std::vector<u16> want_h(want.size());
        for (size_t i = 0; i < want.size(); i++) want_h[i] = fp32_to_fp16(want[i]);
        const BitExact r = cmp_half(mg, want_h, nmix);
        std::printf("  %-11s %-22s %s", "lanes-mix", "collapse == one lane",
                    r.max_ulp <= 1 ? (r.differ == 0 ? "BIT-IDENTICAL" : "PASS")
                                   : "FAIL");
        if (r.max_ulp <= 1)
            std::printf(" (%lld values, max %lld ulp)\n",
                        static_cast<long long>(r.n),
                        static_cast<long long>(r.max_ulp));
        else {
            std::printf("  max %lld ulp, %lld values differ\n",
                        static_cast<long long>(r.max_ulp),
                        static_cast<long long>(r.differ));
            failed++;
        }
    }

    // (7) ple_gate: the PLE block's per-lane gate, one launch for the whole
    // block. The output layout is LANE-major on purpose (a lane's column is
    // contiguous), which is what the gated-value broadcast reads next, so the
    // oracle writes it that way too. Inputs are scaled down: with full-scale
    // values the dot product saturates the sigmoid and a kernel that read the
    // wrong lane would still pass.
    {
        const i64 nq = nt * hc * ne;
        std::vector<f32> key(static_cast<size_t>(nq));
        std::vector<f32> qry(static_cast<size_t>(nq));
        const auto r16 = [](f64 v) {
            return fp16_to_fp32(fp32_to_fp16(static_cast<f32>(v)));
        };
        for (i64 i = 0; i < nq; i++)
            key[static_cast<size_t>(i)] = r16(std::sin(0.011 * i) * 0.1);
        for (i64 i = 0; i < nq; i++)
            qry[static_cast<size_t>(i)] = r16(std::cos(0.017 * i) * 0.1);
        const std::vector<u16> kh = up(key), qh = up(qry);
        void *kd = be->alloc(static_cast<size_t>(nq) * 2);
        void *qd = be->alloc(static_cast<size_t>(nq) * 2);
        void *pgd = be->alloc(static_cast<size_t>(nt * hc) * 2);
        be->upload(kd, kh.data(), kh.size() * 2, 0);
        be->upload(qd, qh.data(), qh.size() * 2, 0);
        be->ple_gate(pgd, kd, qd, nt, hc, ne);
        const std::vector<u16> got = fetch(pgd, nt * hc);
        std::vector<f32> want(static_cast<size_t>(nt * hc));
        const f32 inv = 1.0f / std::sqrt(static_cast<f32>(ne));
        for (i64 c = 0; c < hc; c++) {
            for (i64 t = 0; t < nt; t++) {
                f32 s = 0.0f;
                for (i64 e = 0; e < ne; e++) {
                    const i64 k = (t * hc + c) * ne + e;
                    s += key[static_cast<size_t>(k)] *
                         qry[static_cast<size_t>(k)];
                }
                const f32 sc = s * inv;
                const f32 mag =
                    std::sqrt(std::fmax(std::fabs(sc), 1e-6f));
                const f32 sg = sc > 0.0f ? 1.0f : (sc < 0.0f ? -1.0f : 0.0f);
                want[static_cast<size_t>(c * nt + t)] =
                    1.0f / (1.0f + std::exp(-(sg * mag)));
            }
        }
        const HcCmp r = cmp_hc_f32(got, want, false, 2e-3);
        print_hc("ple_gate", "signed-sqrt sigmoid", r);
        if (!r.ok()) failed++;
        be->release(kd);
        be->release(qd);
        be->release(pgd);
    }

    // (8) conv1d_dw: the PLE block's dilated depthwise convolution along the
    // token axis. `x` carries the history rows in front of this pass's, which
    // is the whole reason the window is the caller's rather than a ring inside
    // the kernel.
    {
        const i64 ch = 1024, kern = 4, dil = 2;
        const i64 hist = (kern - 1) * dil; // 6
        const i64 rows = hist + nt;        // 9
        const auto r16 = [](f64 v) {
            return fp16_to_fp32(fp32_to_fp16(static_cast<f32>(v)));
        };
        std::vector<f32> xw(static_cast<size_t>(rows * ch));
        std::vector<f32> w(static_cast<size_t>(kern * ch));
        for (i64 i = 0; i < rows * ch; i++)
            xw[static_cast<size_t>(i)] = r16(std::sin(0.0007 * i) * 1.5);
        for (i64 i = 0; i < kern * ch; i++)
            w[static_cast<size_t>(i)] = static_cast<f32>(std::cos(0.003 * i) * 0.4);
        const std::vector<u16> xwh = up(xw);
        void *xwd = be->alloc(static_cast<size_t>(rows * ch) * 2);
        void *wdev = be->alloc(static_cast<size_t>(kern * ch) * sizeof(f32));
        void *cvd = be->alloc(static_cast<size_t>(nt * ch) * 2);
        be->upload(xwd, xwh.data(), xwh.size() * 2, 0);
        be->upload(wdev, w.data(), w.size() * sizeof(f32), 0);
        be->conv1d_dw(cvd, xwd, wdev, nt, ch, kern, dil);
        const std::vector<u16> got = fetch(cvd, nt * ch);
        std::vector<f32> want(static_cast<size_t>(nt * ch), 0.0f);
        for (i64 t = 0; t < nt; t++) {
            for (i64 c = 0; c < ch; c++) {
                f32 acc = 0.0f;
                for (i64 k = 0; k < kern; k++) {
                    const i64 rr = t + (kern - 1 - k) * dil;
                    acc += w[static_cast<size_t>(k * ch + c)] *
                           xw[static_cast<size_t>(rr * ch + c)];
                }
                want[static_cast<size_t>(t * ch + c)] = acc;
            }
        }
        const HcCmp r = cmp_hc_f32(got, want, false, 2e-3);
        print_hc("conv1d_dw", "dilated depthwise", r);
        if (!r.ok()) failed++;

        // (9) conv-history: the property the op exists for, on device. Three
        // one-token calls, each handed the six history rows the previous one
        // would have produced (a slice of the same buffer, so the values are
        // the f16 ones), must concatenate to the three-token call BIT-for-bit.
        // That is the chunk invariance of the convolution: a chunked prefill
        // and a token-at-a-time decode see the same window.
        void *oned = be->alloc(static_cast<size_t>(ch) * 2);
        std::vector<u16> joined;
        joined.reserve(static_cast<size_t>(nt * ch));
        for (i64 t = 0; t < nt; t++) {
            const void *slice =
                reinterpret_cast<const u8 *>(xwd) + static_cast<size_t>(t * ch) * 2;
            be->conv1d_dw(oned, slice, wdev, 1, ch, kern, dil);
            const std::vector<u16> one = fetch(oned, ch);
            joined.insert(joined.end(), one.begin(), one.end());
        }
        i64 differ = 0;
        for (i64 i = 0; i < nt * ch; i++)
            if (joined[static_cast<size_t>(i)] != got[static_cast<size_t>(i)])
                differ++;
        std::printf("  %-11s %-22s %s", "conv-history", "3x1 == 3 rows",
                    differ == 0 ? "BIT-IDENTICAL" : "FAIL");
        if (differ == 0) {
            std::printf(" (%lld values)\n", static_cast<long long>(nt * ch));
        } else {
            std::printf("  %lld/%lld differ\n", static_cast<long long>(differ),
                        static_cast<long long>(nt * ch));
            failed++;
        }
        be->release(xwd);
        be->release(wdev);
        be->release(cvd);
        be->release(oned);
    }

    // (10) lane_bcast_mul: value[t,e] * g[c*n_tok + t]. The gate is LANE-major
    // (ple_gate's output layout) and the result is token-major, so a kernel that
    // read the gate token-major would pass a symmetric input and fail this one:
    // the two layouts agree only at n_tok == hc.
    {
        const i64 nv = nt * ne;
        const auto r16 = [](f64 v) {
            return fp16_to_fp32(fp32_to_fp16(static_cast<f32>(v)));
        };
        std::vector<f32> val(static_cast<size_t>(nv));
        std::vector<f32> gatev(static_cast<size_t>(nt * hc));
        for (i64 i = 0; i < nv; i++)
            val[static_cast<size_t>(i)] = r16(std::sin(0.009 * i) * 1.5);
        for (i64 i = 0; i < nt * hc; i++)
            gatev[static_cast<size_t>(i)] = r16(std::cos(0.03 * i));
        const std::vector<u16> vh = up(val), gvh = up(gatev);
        void *lbv = be->alloc(static_cast<size_t>(nv) * 2);
        void *lbg = be->alloc(static_cast<size_t>(nt * hc) * 2);
        void *lbo = be->alloc(static_cast<size_t>(nv * hc) * 2);
        be->upload(lbv, vh.data(), vh.size() * 2, 0);
        be->upload(lbg, gvh.data(), gvh.size() * 2, 0);
        be->lane_bcast_mul(lbo, lbv, lbg, nt, hc, ne);
        const std::vector<u16> got = fetch(lbo, nv * hc);
        std::vector<f32> want(static_cast<size_t>(nv * hc), 0.0f);
        for (i64 t = 0; t < nt; t++)
            for (i64 c = 0; c < hc; c++)
                for (i64 e = 0; e < ne; e++)
                    want[static_cast<size_t>((t * hc + c) * ne + e)] =
                        val[static_cast<size_t>(t * ne + e)] *
                        gatev[static_cast<size_t>(c * nt + t)];
        const HcCmp r = cmp_hc_f32(got, want, false, 2e-3);
        print_hc("lane_bcast", "value[t,e]*g[c,t]", r);
        if (!r.ok()) failed++;
        be->release(lbv);
        be->release(lbg);
        be->release(lbo);
    }

    be->release(xd);
    be->release(xnd);
    be->release(gd);
    be->release(bd);
    be->release(id);
    be->release(lanes);
    be->release(mixd);
    be->release(res);
    std::printf("  hc: %lld of 10 cases%s\n", static_cast<long long>(10 - failed),
                failed == 0 ? " identical" : " failed");
    return failed;
}

} // namespace

// ---------------------------------------------------------------------------
// The device dispatch table (KRK_DEV_PTR=1), against the per-expert path.
//
// A real model is the wrong instrument for this: the table is only published
// when the WHOLE routed set is device-resident, which on this box means a model
// small enough that the cache never has to evict -- so "the arm ran on the 12B"
// is a fact about one card's free VRAM, not about the kernel. Here the bank is
// synthetic, its residency is a fact by construction, and the reference is the
// exact sequence the engine's per-expert loop issues: gemm(gate), gemm(up),
// silu_mul at the row length with its row scale, gemm(down), and an f16-rounded
// weighted add per expert in EXPERT-ASCENDING order.
//
// Two checks per case, because they fail differently:
//   * `values` -- every element finite, and within the f16 output tolerance of
//     the reference. Bits are reported (a case can be IDENTICAL, and the k=1
//     cases are expected to be), but the gate does not require them: a token
//     with two experts sums in slot order here and expert order there.
//   * `indirection` -- the same call with two table entries swapped must CHANGE
//     the answer. A kernel that ignored the table and read expert 0 three times
//     would pass `values` on a one-expert case and look perfect; this is the
//     arm that fails when the indirection is fake.
struct MoeCase {
    const char *tag;
    DType t;
    i64 n_embd;
    i64 n_ff;
    i32 n_expert;
    i32 k;
    i64 n_tok;
    // Every slot selects the SAME expert. Two experts summing is order
    // sensitive in f16, so a disagreement there says nothing about the
    // per-slot arithmetic; this case isolates it.
    bool same = false;
};

// MoE-case weights: `weights()` with a smaller block scale.
//
// The shared helper budgets for a GEMV that stops at one projection (a dot over
// 4096 terms of 0.02 * 15 * x stays well inside f16). An expert runs TWO
// projections and a silu product in between, so the up result is ~4 orders
// larger and the product leaves f16 -- both the kernel and the reference then
// compute inf, agree bit-for-bit, and the case proves nothing about either.
// Measured: the q4_k cases came out "diff=0 nan=4096 of 4096", which is a
// harness that produced no finite number to compare.
// `salt` matters more than it looks: `weights()` restarts its RNG at a fixed
// seed every call, so EVERY matrix of the same shape came out identical -- and
// an indirection control that swaps two identical matrices reports "no change"
// no matter how correct the kernel is. That is how this arm first reported
// NOT PROVEN on every case with a kernel that was in fact bit-identical to the
// reference.
std::vector<u8> moe_weights(DType t, size_t row_bytes, i64 n_out, u32 salt) {
    std::vector<u8> w(row_bytes * static_cast<size_t>(n_out));
    u32 rng = 0x1234567u ^ (salt * 2654435761u);
    for (u8 &b : w) {
        rng = rng * 1103515245u + 12345u;
        b = static_cast<u8>(rng >> 16);
    }
    if (t == DType::F16) {
        for (size_t i = 0; i + 1 < w.size(); i += 2) {
            const f32 v = static_cast<f32>(static_cast<int>(w[i] & 0x1f)) - 16.0f;
            const u16 h = fp32_to_fp16(v * 0.001f * static_cast<f32>(1 + (salt & 3)));
            w[i] = static_cast<u8>(h & 0xff);
            w[i + 1] = static_cast<u8>(h >> 8);
        }
        return w;
    }
    const size_t blk = static_cast<size_t>(dtype_block_bytes(t));
    const i64 blocks = static_cast<i64>(row_bytes / blk);
    for (i64 r = 0; r < n_out; r++) {
        u8 *row = w.data() + static_cast<size_t>(r) * row_bytes;
        for (i64 b = 0; b < blocks; b++) {
            u8 *p = row + static_cast<size_t>(b) * blk;
            switch (t) {
            case DType::Q6_K:
                // Q6_K's per-block scales are signed int8 (up to 127) against
                // Q4_K's 6-bit (up to 63), and its dot sums 64 chunks, so the
                // same header scale leaves the silu product four orders higher
                // -- measured as equal NaN counts on both sides, which is a
                // harness with nothing finite to compare.
                put_half(p + blk - 2, 0.0002f);
                break;
            case DType::Q4_K:
            case DType::Q5_K:
                put_half(p, 0.003f);
                put_half(p + 2, 0.0f);
                break;
            default:
                put_half(p, 0.003f);
                break;
            }
        }
    }
    return w;
}

i64 arm_moe_dispatch(Backend *be) {
    const MoeCase cases[] = {
        {"k=1 q4_k", DType::Q4_K, 2048, 1024, 4, 1, 2},
        {"k=2 same", DType::Q4_K, 2048, 1024, 4, 2, 1, true},
        {"k=2 q4_k", DType::Q4_K, 2048, 1024, 4, 2, 2},
        {"k=4 q8_0", DType::Q8_0, 2048, 1024, 8, 4, 3},
        {"k=2 q6_k", DType::Q6_K, 2048, 1024, 3, 2, 1},
        {"k=2 f16", DType::F16, 1024, 512, 2, 2, 2},
        // The shape the models actually run: 4096-wide activations and a
        // 14336-wide expert FFN, which is 45 KiB of shared memory and is the
        // reason the launch checks that bound instead of assuming it.
        {"decode q4_k", DType::Q4_K, 4096, 14336, 2, 2, 1},
    };
    std::printf("\ndevice expert dispatch (table indirection against the "
                "per-expert path):\n");
    i64 failures = 0;
    for (const MoeCase &c : cases) {
        const size_t g_row = dtype_row_bytes(c.t, c.n_embd);
        const size_t d_row = dtype_row_bytes(c.t, c.n_ff);
        const i64 ne = c.n_expert;
        std::vector<std::vector<u8>> wg(static_cast<size_t>(ne)),
            wu(static_cast<size_t>(ne)), wd(static_cast<size_t>(ne));
        std::vector<void *> pg(static_cast<size_t>(ne)), pu(static_cast<size_t>(ne)),
            pd(static_cast<size_t>(ne));
        std::vector<MoeDispatchEntry> table(static_cast<size_t>(ne));
        for (i64 e = 0; e < ne; e++) {
            wg[static_cast<size_t>(e)] =
                moe_weights(c.t, g_row, c.n_ff, static_cast<u32>(e) * 3u + 1u);
            wu[static_cast<size_t>(e)] =
                moe_weights(c.t, g_row, c.n_ff, static_cast<u32>(e) * 3u + 2u);
            wd[static_cast<size_t>(e)] =
                moe_weights(c.t, d_row, c.n_embd, static_cast<u32>(e) * 3u + 3u);
            pg[static_cast<size_t>(e)] = be->alloc(wg[static_cast<size_t>(e)].size());
            pu[static_cast<size_t>(e)] = be->alloc(wu[static_cast<size_t>(e)].size());
            pd[static_cast<size_t>(e)] = be->alloc(wd[static_cast<size_t>(e)].size());
            be->upload(pg[static_cast<size_t>(e)], wg[static_cast<size_t>(e)].data(),
                       wg[static_cast<size_t>(e)].size(), 0);
            be->upload(pu[static_cast<size_t>(e)], wu[static_cast<size_t>(e)].data(),
                       wu[static_cast<size_t>(e)].size(), 0);
            be->upload(pd[static_cast<size_t>(e)], wd[static_cast<size_t>(e)].data(),
                       wd[static_cast<size_t>(e)].size(), 0);
            MoeDispatchEntry &d = table[static_cast<size_t>(e)];
            d.p[kMoeGate] = pg[static_cast<size_t>(e)];
            d.p[kMoeUp] = pu[static_cast<size_t>(e)];
            d.p[kMoeDown] = pd[static_cast<size_t>(e)];
            d.n_out[kMoeGate] = c.n_ff;
            d.n_out[kMoeUp] = c.n_ff;
            d.n_out[kMoeDown] = c.n_embd;
            d.n_in[kMoeGate] = c.n_embd;
            d.n_in[kMoeUp] = c.n_embd;
            d.n_in[kMoeDown] = c.n_ff;
            d.row_bytes[kMoeGate] = static_cast<std::int64_t>(g_row);
            d.row_bytes[kMoeUp] = static_cast<std::int64_t>(g_row);
            d.row_bytes[kMoeDown] = static_cast<std::int64_t>(d_row);
            d.type[kMoeGate] = static_cast<std::int32_t>(c.t);
            d.type[kMoeUp] = static_cast<std::int32_t>(c.t);
            d.type[kMoeDown] = static_cast<std::int32_t>(c.t);
            d.present[kMoeGate] = d.present[kMoeUp] = d.present[kMoeDown] = 1;
        }
        void *table_d = be->alloc(table.size() * sizeof(MoeDispatchEntry));
        be->upload(table_d, table.data(), table.size() * sizeof(MoeDispatchEntry), 0);

        std::vector<u16> x(static_cast<size_t>(c.n_tok) * static_cast<size_t>(c.n_embd));
        u32 rng = 987654321u;
        std::vector<f32> xf(x.size());
        for (f32 &v : xf) {
            rng = rng * 1103515245u + 12345u;
            v = (static_cast<f32>((rng >> 20) & 0xff) / 255.0f - 0.5f) * 0.5f;
        }
        for (size_t i = 0; i < xf.size(); i++) x[i] = fp32_to_fp16(xf[i]);
        void *xn_d = be->alloc(x.size() * 2);
        be->upload(xn_d, x.data(), x.size() * 2, 0);

        std::vector<i32> ids(static_cast<size_t>(c.n_tok) * static_cast<size_t>(c.k));
        std::vector<f32> wts(ids.size(), 0.0f);
        for (i64 t = 0; t < c.n_tok; t++)
            for (i32 s = 0; s < c.k; s++) {
                ids[static_cast<size_t>(t * c.k + s)] =
                    c.same ? static_cast<i32>(t % ne)
                           : static_cast<i32>((t + s) % ne);
                // Half the weight on the first slot and an even split otherwise,
                // so a kernel that dropped the weights entirely would show up as
                // a scale on every element instead of only on a tie.
                wts[static_cast<size_t>(t * c.k + s)] =
                    c.k == 1 ? 1.0f : (s == 0 ? 0.6f : 0.4f / static_cast<f32>(c.k - 1));
            }
        void *ids_d = be->alloc(ids.size() * sizeof(i32));
        void *wts_d = be->alloc(wts.size() * sizeof(f32));
        be->upload(ids_d, ids.data(), ids.size() * sizeof(i32), 0);
        be->upload(wts_d, wts.data(), wts.size() * sizeof(f32), 0);

        // ---- reference: the per-expert loop's own sequence ------------------
        void *g_d = be->alloc(static_cast<size_t>(c.n_ff) * 2);
        void *u_d = be->alloc(static_cast<size_t>(c.n_ff) * 2);
        void *a_d = be->alloc(static_cast<size_t>(c.n_ff) * 2);
        void *o_d = be->alloc(static_cast<size_t>(c.n_embd) * 2);
        void *sc_d = be->alloc(sizeof(f32));
        std::vector<u16> ref(x.size(), 0);
        // One download buffer for both ends of the reference: the down
        // projection's output is n_embd wide and the row scale's copy is n_ff,
        // so it is sized for the larger. Sizing it n_ff overwrote the heap on
        // every case where the model is wider than its expert FFN, which is all
        // of them here.
        std::vector<u16> gv(static_cast<size_t>(std::max(c.n_ff, c.n_embd)));
        for (i64 e = 0; e < ne; e++) {
            for (i64 t = 0; t < c.n_tok; t++) {
                for (i32 s = 0; s < c.k; s++) {
                    if (ids[static_cast<size_t>(t * c.k + s)] != e) continue;
                    const f32 alpha = wts[static_cast<size_t>(t * c.k + s)];
                    // The kernel's own op order (scale, then alpha, then add),
                    // so the comparison tests the path and not the compiler's
                    // freedom to contract a multiply into an add.

                    const void *xrow = static_cast<const u8 *>(xn_d) +
                                       static_cast<size_t>(t) *
                                           static_cast<size_t>(c.n_embd) * 2;
                    be->gemm(g_d, xrow, pg[static_cast<size_t>(e)], c.t, c.n_ff,
                             c.n_embd, 1);
                    be->gemm(u_d, xrow, pu[static_cast<size_t>(e)], c.t, c.n_ff,
                             c.n_embd, 1);
                    be->silu_mul(a_d, g_d, u_d, c.n_ff, c.n_ff,
                                 static_cast<f32 *>(sc_d));
                    be->gemm(o_d, a_d, pd[static_cast<size_t>(e)], c.t, c.n_embd,
                             c.n_ff, 1);
                    be->download(gv.data(), o_d, static_cast<size_t>(c.n_embd) * 2);
                    f32 scale = 1.0f;
                    be->download(&scale, sc_d, sizeof(f32));
                    for (i64 j = 0; j < c.n_embd; j++) {
                        const f32 v = fp16_to_fp32(gv[static_cast<size_t>(j)]);
                        const f32 vs = v * scale;
                        const f32 w = alpha * vs;
                        const f32 cur =
                            fp16_to_fp32(ref[static_cast<size_t>(t * c.n_embd + j)]);
                        ref[static_cast<size_t>(t * c.n_embd + j)] =
                            fp32_to_fp16(cur + w);
                    }
                }
            }
        }

        // ---- the kernel ----------------------------------------------------
        void *out_d = be->alloc(x.size() * 2);
        be->moe_dispatch(table_d, 0, ne, ids_d, wts_d, xn_d, out_d, c.n_tok,
                         c.n_embd, c.n_ff, c.k);
        be->sync();
        std::vector<u16> got(x.size());
        be->download(got.data(), out_d, got.size() * 2);

        i64 diff = 0, nan = 0, inf = 0, over = 0;
        i64 nan_k = 0, nan_r = 0;
        f64 worst = 0.0;
        i64 first = -1;
        f64 sum_sq = 0.0;
        for (size_t i = 0; i < got.size(); i++) {
            const f32 a = fp16_to_fp32(got[i]);
            sum_sq += static_cast<f64>(a) * static_cast<f64>(a);
        }
        const f64 rms = std::sqrt(sum_sq / static_cast<f64>(got.size()) + 1e-30);
        for (size_t i = 0; i < got.size(); i++) {
            const f32 a = fp16_to_fp32(got[i]);
            const f32 b = fp16_to_fp32(ref[i]);
            if (!std::isfinite(a) || !std::isfinite(b)) {
                if (!std::isfinite(a)) nan_k++;
                if (!std::isfinite(b)) nan_r++;
                if (!std::isfinite(a) && !std::isfinite(b)) nan++;
                else inf++;
                continue;
            }
            if (got[i] != ref[i]) {
                diff++;
                if (first < 0) first = static_cast<i64>(i);
            }
            const f64 rel = std::fabs(static_cast<f64>(a) - static_cast<f64>(b)) / rms;
            if (rel > worst) worst = rel;
            if (rel > 4e-3) over++;
        }
        const bool values_ok = nan == 0 && inf == 0 && over == 0;
        if (first >= 0)
            std::printf("  %-11s first difference at %lld: kernel=%.6g ref=%.6g "
                        "(nan kernel=%lld ref=%lld)\n",
                        "", static_cast<long long>(first),
                        static_cast<f64>(fp16_to_fp32(got[static_cast<size_t>(first)])),
                        static_cast<f64>(fp16_to_fp32(ref[static_cast<size_t>(first)])),
                        static_cast<long long>(nan_k), static_cast<long long>(nan_r));

        // ---- the indirection control --------------------------------------
        // Swap two entries that this case actually selects, re-run, and require
        // a different answer. Picking them from `ids` is the point: swapping two
        // unselected experts would prove nothing about what the kernel read.
        // Swapping a selected expert's GATE and UP pointers is the control that
        // always exists: silu(gate) * up is not symmetric in its two operands, so
        // a kernel that ignores the table cannot pass it even with k=1. (The
        // first version swapped two EXPERTS, which a one-expert selection has no
        // second of -- it reported NOT PROVEN for every single-expert case, which
        // is exactly the "control that cannot fail" this box keeps rediscovering.)
        i64 a_e = ids[0];
        bool moved = false;
        {
            MoeDispatchEntry &a = table[static_cast<size_t>(a_e)];
            std::swap(a.p[kMoeGate], a.p[kMoeUp]);
            std::swap(a.type[kMoeGate], a.type[kMoeUp]);
            std::swap(a.row_bytes[kMoeGate], a.row_bytes[kMoeUp]);
            be->upload(table_d, table.data(), table.size() * sizeof(MoeDispatchEntry), 0);
            // Drain before reading it back through the kernel: `upload` may hand
            // the bytes to the pool's staging path, which publishes on its own
            // stream. In the engine the table is published once at init and every
            // use is layers later, so this ordering never bites there; here the
            // dispatch is the very next launch, and without the drain the control
            // read the OLD table and reported "indirection NOT PROVEN" for every
            // case -- a false negative that looks exactly like a dead kernel.
            be->sync();
            // A marker the second dispatch must overwrite: without it a dispatch
            // that silently declined to run leaves the FIRST output in the
            // buffer, and the control reports "the indirection changed nothing"
            // when the truth is "nothing ran".
            be->fill0(out_d, x.size() * 2);
            be->moe_dispatch(table_d, 0, ne, ids_d, wts_d, xn_d, out_d, c.n_tok,
                             c.n_embd, c.n_ff, c.k);
            std::vector<u16> alt(x.size());
            be->download(alt.data(), out_d, alt.size() * 2);
            for (size_t i = 0; i < alt.size(); i++)
                if (alt[i] != got[i]) { moved = true; break; }
            std::swap(a.p[kMoeGate], a.p[kMoeUp]);
            std::swap(a.type[kMoeGate], a.type[kMoeUp]);
            std::swap(a.row_bytes[kMoeGate], a.row_bytes[kMoeUp]);
        }

        std::printf("  %-11s %-5s embd=%-5lld ff=%-6lld experts=%-2lld k=%d "
                    "tokens=%lld  %s  worst=%.2e  indirection=%s\n",
                    c.tag, dtype_tag(c.t), static_cast<long long>(c.n_embd),
                    static_cast<long long>(c.n_ff), static_cast<long long>(ne),
                    static_cast<int>(c.k), static_cast<long long>(c.n_tok),
                    diff == 0 ? "IDENTICAL" : "differs", worst,
                    moved ? "ok" : "NOT PROVEN");
        if (diff) {
            std::printf("  %-11s %lld of %zu element(s) differ from the "
                        "per-expert path (f16 store rounding, reported not "
                        "gated)\n",
                        "", static_cast<long long>(diff), got.size());
        }
        if (!values_ok) {
            std::printf("  %-11s FAIL values: not-finite kernel=%lld ref=%lld "
                        "over-tolerance=%lld worst=%.3e\n",
                        "", static_cast<long long>(nan),
                        static_cast<long long>(inf), static_cast<long long>(over),
                        worst);
            ++failures;
        }
        if (!moved) {
            std::printf("  %-11s FAIL indirection: swapping the selected "
                        "expert's gate and up matrices did not change the output\n",
                        "");
            ++failures;
        }

        be->release(table_d);
        be->release(xn_d);
        be->release(ids_d);
        be->release(wts_d);
        be->release(out_d);
        be->release(g_d);
        be->release(u_d);
        be->release(a_d);
        be->release(o_d);
        be->release(sc_d);
        for (i64 e = 0; e < ne; e++) {
            be->release(pg[static_cast<size_t>(e)]);
            be->release(pu[static_cast<size_t>(e)]);
            be->release(pd[static_cast<size_t>(e)]);
        }
    }
    std::printf("  %lld of %zu case(s) failed\n", static_cast<long long>(failures),
                sizeof(cases) / sizeof(cases[0]));
    return failures;
}

int main(int argc, char **argv) {
    int iters = 100;
    int depth = 512; // calls pre-issued for the busy-queue arm (0 skips it)
    double launch_ms = 0.0;
    bool sweep = false;
    bool gate = false;
    bool bitexact = false;
    bool hc_test = false;
    bool moe_table_test = false;
    bool floor = false;
    bool h2d = false;
    i64 failures = 0;
    Shape one{"one", DType::Q4_K, 4096, 4096, 1};
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto next = [&]() { return std::atoll(argv[++i]); };
        if (a == "--iters") iters = static_cast<int>(next());
        else if (a == "--depth") depth = static_cast<int>(next());
        else if (a == "--rows") one.rows = next();
        else if (a == "--out") one.n_out = next();
        else if (a == "--in") one.n_in = next();
        else if (a == "--sweep") sweep = true;
        else if (a == "--gate") gate = true;
        else if (a == "--bitexact") bitexact = true;
        else if (a == "--hc") hc_test = true;
        else if (a == "--moe-dispatch") moe_table_test = true;
        else if (a == "--floor") floor = true;
        else if (a == "--h2d") h2d = true;
        else if (a.rfind("--dtype", 0) == 0) {
            const std::string d = argv[++i];
            one.t = d == "f16"    ? DType::F16
                    : d == "q4_0" ? DType::Q4_0
                    : d == "q5_k" ? DType::Q5_K
                    : d == "q6_k" ? DType::Q6_K
                    : d == "q8_0" ? DType::Q8_0
                                  : DType::Q4_K;
        } else {
            std::fprintf(stderr, "unknown argument %s\n", a.c_str());
            return 2;
        }
    }
    if (iters < 2) iters = 2;
    // The gate is correctness, not timing: two iterations is the least that
    // keeps time_calls's T(N)-T(1) subtraction defined, and the gate does not
    // read the timing at all.
    if (gate) iters = 2;

    std::string err;
    Backend *be = make_hip_backend(0, &err);
    if (!be) {
        std::fprintf(stderr, "no HIP backend: %s\n", err.c_str());
        return 1;
    }
    std::printf("device: %s\n\n", be->describe().c_str());
    if (h2d) return h2d_bench(be, iters);

    // Launch floor: a real kernel with trivial work, same measurement method.
    {
        void *a = be->alloc(1024);
        void *b = be->alloc(1024);
        const double ms = time_calls(be, iters,
                                     [&] { be->copy_act(a, b, 32); });
        // time_calls subtracts one cold call from the mean, which cannot resolve
        // anything when the switch is 2 -- the gate's setting -- and has printed
        // "-24.00 us per call" there. Quote only what was measured.
        if (ms > 0.0)
            std::printf("launch floor (copy_act n=32): %.2f us per call\n\n",
                        ms * 1000.0);
        else
            std::printf("launch floor (copy_act n=32): unresolvable at "
                        "iters=%d (the per-call subtraction needs a deeper "
                        "switch)\n\n",
                        iters);
        launch_ms = ms;
        be->release(a);
        be->release(b);
    }

    // The promotion path's own bookkeeping, beside the launch floor: both are
    // fixed per-call costs a decode step pays hundreds of times.
    if (!gate) {
        bench_pool_primitives(be, iters, depth, launch_ms);
        std::printf("\n");
    }

    const Shape sweep_shapes[] = {
        // what the local models run every token (rows = 1 is the decode GEMV)
        {"q/o", DType::Q4_K, 4096, 4096, 1},
        {"g/u", DType::Q4_K, 3584, 1024, 1},
        {"down", DType::Q4_K, 1024, 3584, 1},
        {"head", DType::Q4_K, 248320, 1024, 1},
        {"g/u8", DType::Q4_K, 14336, 4096, 1},
        {"k/v", DType::Q4_K, 1024, 4096, 1},
        // the same shapes in the other formats the collections use
        {"q/o", DType::Q6_K, 4096, 4096, 1},
        {"q/o", DType::Q8_0, 4096, 4096, 1},
        {"q/o", DType::Q4_0, 4096, 4096, 1},
        {"q/o", DType::Q5_K, 4096, 4096, 1},
        {"head", DType::Q6_K, 248320, 1024, 1},
        // f16 reference: the format rocBLAS does, for the same shapes
        {"q/o", DType::F16, 4096, 4096, 1},
        {"head", DType::F16, 248320, 1024, 1},
        // prefill rows: the WMMA path
        {"q/o", DType::Q4_K, 4096, 4096, 32},
        {"g/u8", DType::Q4_K, 14336, 4096, 32},
        {"q/o", DType::F16, 4096, 4096, 32},
    };

    if (floor) {
        std::printf("--- load-store floor: one copy_act per shape bucket, "
                    "all shapes times the sum of their own byte footprint");
        std::printf("\n--- (weights+activation+output, f16 for the same "
                    "backend) so a folded path listens at the ceiling\n");
        std::printf("\n");
        floor_floor(be, iters);
    }

    if (sweep) {
        for (const Shape &s : sweep_shapes)
            if (!run_one(be, s, iters, 0.0).ok()) ++failures;
    } else {
        if (!run_one(be, one, iters, 0.0).ok()) ++failures;
    }

    if (bitexact) failures += arm_rows_bitexact(be);

    // Device-only, so it is its own flag rather than part of --gate: the gate
    // must stay runnable on a box where the device is busy.
    if (hc_test) failures += arm_hc(be);

    // The device dispatch table. Its own flag like --hc for the same reason: it
    // is a device arm, and the gate has to stay runnable on a busy box.
    if (moe_table_test) failures += arm_moe_dispatch(be);

    if (gate || sweep) {
        std::printf("\nexpert-lease (device):\n");
        failures += gate_expert_lease(be);
        std::printf("\nlogits_topk:\n");
        failures += gate_topk_all(be);
        failures += gate_expert_pool(be);
        std::printf("\nbusy-queue arm (device) -- the instrument's own control:\n");
        failures += gate_busy_queue(be, depth);
        std::printf("\nspeculation arm (CLI self-draft):\n");
        failures += gate_speculation(argv[0]);
    }

    // clean exit path only when every test group has returned.
    delete be;
    if (gate && failures) {
        std::fprintf(stderr,
                     "\nGATE FAILED: %lld check(s) outside tolerance, or NaN/inf, "
                     "or a logits_topk mismatch\n",
                     static_cast<long long>(failures));
        return 1;
    }
    return 0;
}
