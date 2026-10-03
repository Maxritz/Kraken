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
// Reference, same box, torch 2.15/rocBLAS: f16 GEMV 1x4096x4096 = 47.3 µs
// (710 GB/s), f16 GEMM 32x4096x4096 = 43.4 µs (772 GB/s, 24.7 TFLOP/s).
#include <krk/backend.hpp>
#include <krk/quant.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

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
        std::printf("  FAIL %-22s count=%lld (must be >= 1)\n", what,
                    static_cast<long long>(count));
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

} // namespace

int main(int argc, char **argv) {
    int iters = 100;
    bool sweep = false;
    bool gate = false;
    i64 failures = 0;
    Shape one{"one", DType::Q4_K, 4096, 4096, 1};
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto next = [&]() { return std::atoll(argv[++i]); };
        if (a == "--iters") iters = static_cast<int>(next());
        else if (a == "--rows") one.rows = next();
        else if (a == "--out") one.n_out = next();
        else if (a == "--in") one.n_in = next();
        else if (a == "--sweep") sweep = true;
        else if (a == "--gate") gate = true;
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

    // Launch floor: a real kernel with trivial work, same measurement method.
    {
        void *a = be->alloc(1024);
        void *b = be->alloc(1024);
        const double ms = time_calls(be, iters,
                                     [&] { be->copy_act(a, b, 32); });
        std::printf("launch floor (copy_act n=32): %.2f us per call\n\n",
                    ms * 1000.0);
        be->release(a);
        be->release(b);
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

    if (sweep) {
        for (const Shape &s : sweep_shapes)
            if (!run_one(be, s, iters, 0.0).ok()) ++failures;
    } else {
        if (!run_one(be, one, iters, 0.0).ok()) ++failures;
    }

    if (gate || sweep) {
        std::printf("\nlogits_topk:\n");
        failures += gate_topk_all(be);
    }

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
