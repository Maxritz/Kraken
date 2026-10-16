#!/usr/bin/env python3
"""The first queued arm reported a NEGATIVE marginal record cost: time_calls's
per-arm cold-call subtraction is bigger than a host-side enqueue. Replace it with
a direct measurement of the calls issued into a queue that is already deep."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
p = ROOT / "tools/bench_ops.cpp"
raw = p.read_text(newline="")
eol = "\r\n" if "\r\n" in raw else "\n"
s = raw.replace("\r\n", "\n")

old = """    // The two constants a decode step's folds are sized with, measured with real
    // work queued behind them -- which is the only state a busy step is ever in
    // (303 ops in flight on the 135M, 2965 on the 35B). An empty queue measures
    // the LAST record's completion rather than a record's cost, and it is not a
    // small difference in either direction: on Qwen3-MOE-4x0.6B at 16/32 MiB,
    // 7252 records removed bought 18.6 ms of the step (2.56 us each) against the
    // ~30 us the empty-queue arm above reports, while the same cell's alloc
    // round trip costs MORE in the step (32.8 us against 26-28 us here).
    // `--depth` sets the queue depth; 0 skips the arm.
    if (depth > 0) {
        void *qa = be->alloc(4096), *qb = be->alloc(4096);
        void *qmark = be->make_marker();
        if (qa && qb && qmark) {
            be->fill0(qa, 4096);
            be->fill0(qb, 4096);
            const double qcopies = time_calls(be, n, [&] {
                for (int i = 0; i < depth; i++) be->copy_act(qa, qb, 32);
            });
            const double qbusy = time_calls(be, n, [&] {
                for (int i = 0; i < depth; i++) be->copy_act(qa, qb, 32);
                be->record_marker(qmark);
            });
            std::printf(
                "primitives, queued (%d x copy_act n=32 ahead of every call): "
                "record %.2f us marginal | copy %.2f us marginal  "
                "[empty queue: record %.2f | copy %.2f]\\n",
                depth, (qbusy - qcopies) * 1000.0, qcopies / depth * 1000.0,
                record_ms * 1000.0, launch_ms * 1000.0);
        }
        be->release_marker(qmark);
        be->release(qa);
        be->release(qb);
    }
"""

new = """    // Quiet queue vs BUSY queue, host side -- and the busy queue is the only state
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
        void *ba = be->alloc(4096), *bb = be->alloc(4096);
        void *bm = be->make_marker();
        if (ba && bb && bm) {
            be->fill0(ba, 4096);
            be->fill0(bb, 4096);
            constexpr int kChunk = 64; // calls timed per phase
            constexpr int kReps = 8;
            std::vector<double> bcopy(kReps, 0.0), brec(kReps, 0.0);
            for (int rep = 0; rep < kReps; rep++) {
                // Enough device work that the queue cannot drain under the host.
                for (int i = 0; i < depth; i++) be->copy_act(ba, bb, 32);
                const auto t0 = std::chrono::steady_clock::now();
                for (int i = 0; i < kChunk; i++) be->copy_act(ba, bb, 32);
                const auto t1 = std::chrono::steady_clock::now();
                for (int i = 0; i < kChunk; i++) be->record_marker(bm);
                const auto t2 = std::chrono::steady_clock::now();
                be->sync();
                bcopy[rep] = std::chrono::duration<double, std::milli>(t1 - t0)
                                 .count() / kChunk;
                brec[rep] = std::chrono::duration<double, std::milli>(t2 - t1)
                                .count() / kChunk;
            }
            std::sort(bcopy.begin(), bcopy.end());
            std::sort(brec.begin(), brec.end());
            std::printf(
                "primitives, busy queue (host side, %d x copy_act n=32 already "
                "issued): record %.2f us | copy %.2f us per call, median of %d "
                "  [empty queue: record %.2f | copy %.2f]\\n",
                depth, brec[kReps / 2] * 1000.0, bcopy[kReps / 2] * 1000.0,
                kReps, record_ms * 1000.0, launch_ms * 1000.0);
        }
        be->release_marker(bm);
        be->release(ba);
        be->release(bb);
    }
"""

if s.count(old) != 1:
    sys.exit(f"anchor found {s.count(old)} times")
s = s.replace(old, new)

# default depth: 512 pre-issued copies is ~0.8 ms of device work at ~1.5 us each
old2 = "    int depth = 64; // queue depth for the queued per-op arms (0 skips them)"
new2 = "    int depth = 512; // calls pre-issued for the busy-queue arm (0 skips it)"
if s.count(old2) != 1:
    sys.exit("depth anchor")
s = s.replace(old2, new2)

p.write_text(s.replace("\n", eol), newline="")
print("tools/bench_ops.cpp: busy-queue arm replaced")
