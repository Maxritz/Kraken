#!/usr/bin/env python3
# Make the busy-queue arm prove itself: run it in the gate, construct the
# boundary (deep vs 1 call of pre-issued work), and fail the gate if the arm
# cannot tell a busy queue from an idle one.
import io, sys

def load(path):
    raw = io.open(path, 'r', encoding='utf-8', newline='').read()
    return raw.replace('\r\n', '\n'), raw.count('\r\n') > 0

def save(path, txt, crlf):
    io.open(path, 'w', encoding='utf-8', newline='').write(
        txt.replace('\n', '\r\n') if crlf else txt)

def sub(txt, old, new, count=1, tag=''):
    n = txt.count(old)
    if n != count:
        print('ANCHOR FAIL %s: found %d, expected %d' % (tag, n, count))
        print(old[:500]); sys.exit(1)
    return txt.replace(old, new)

B, crlf = load('tools/bench_ops.cpp')

# --------------------------------------------------------------------------
# 1. the block measurement, factored out so the gate can call it directly
# --------------------------------------------------------------------------
old = """void bench_pool_primitives(Backend *be, int iters, int depth, double launch_ms) {"""
new = """// One block of `chunk` calls timed on a queue that already holds `depth`
// copies: the state a decode step is always in, and the only state in which a
// host-side enqueue can be timed on its own. `tail_us` is what the sync AFTER
// the block still had to drain -- the witness that the queue was busy when the
// host returned, which is what makes the reading a queue measurement rather
// than a wall-clock coincidence.
struct QueueArm {
    double rec_us = 0.0, copy_us = 0.0, host_us = 0.0, tail_us = 0.0;
};

static QueueArm measure_queue_arm(Backend *be, int depth, int chunk, int reps) {
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
    std::vector<double> cp(reps, 0.0), rc(reps, 0.0), tl(reps, 0.0);
    for (int i = 0; i < reps; i++) {
        // Enough device work that the queue cannot drain under the host.
        for (int k = 0; k < depth; k++) be->copy_act(a, b, 32);
        const auto t0 = std::chrono::steady_clock::now();
        for (int k = 0; k < chunk; k++) be->copy_act(a, b, 32);
        const auto t1 = std::chrono::steady_clock::now();
        for (int k = 0; k < chunk; k++) be->record_marker(m);
        const auto t2 = std::chrono::steady_clock::now();
        be->sync();
        const auto t3 = std::chrono::steady_clock::now();
        cp[i] = std::chrono::duration<double, std::micro>(t1 - t0).count() / chunk;
        rc[i] = std::chrono::duration<double, std::micro>(t2 - t1).count() / chunk;
        tl[i] = std::chrono::duration<double, std::micro>(t3 - t2).count();
    }
    std::sort(cp.begin(), cp.end());
    std::sort(rc.begin(), rc.end());
    std::sort(tl.begin(), tl.end());
    r.copy_us = cp[reps / 2];
    r.rec_us = rc[reps / 2];
    r.tail_us = tl[reps / 2];
    r.host_us = (r.copy_us + r.rec_us) * chunk;
    be->release_marker(m);
    be->release(a);
    be->release(b);
    return r;
}

void bench_pool_primitives(Backend *be, int iters, int depth, double launch_ms) {"""
B = sub(B, old, new, tag='helper')

# --------------------------------------------------------------------------
# 2. the print arm measures BOTH states by the same method
# --------------------------------------------------------------------------
old = """    if (depth > 0) {
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
    }"""
new = """    if (depth > 0) {
        constexpr int kChunk = 64; // calls timed per phase
        constexpr int kReps = 8;   // median of these, per state
        const QueueArm busy = measure_queue_arm(be, depth, kChunk, kReps);
        // The SAME block with nothing pre-issued. The two time_calls arms above
        // measure that state as well, but by a different method (T(N)-T(1)),
        // which is what cannot resolve a host enqueue -- so the comparison that
        // matters is this one, both sides by the same block method. `gate`
        // asserts the difference instead of assuming it.
        const QueueArm empty = measure_queue_arm(be, 0, kChunk, kReps);
        std::printf(
            "primitives, busy queue (host side, %d x copy_act n=32 already "
            "issued): record %.2f us | copy %.2f us per call, median of %d "
            "| queue still draining: %.0f us\\n",
            depth, busy.rec_us, busy.copy_us, kReps, busy.tail_us);
        std::printf(
            "primitives, same block, empty queue: record %.2f us | copy %.2f us "
            "| queue still draining: %.0f us  (time_calls arms: record %.2f | "
            "copy %.2f)\\n",
            empty.rec_us, empty.copy_us, empty.tail_us, record_ms * 1000.0,
            launch_ms * 1000.0);
    }"""
B = sub(B, old, new, tag='print arm')

# --------------------------------------------------------------------------
# 3. the gate: the arm must discriminate, with no switch in play
# --------------------------------------------------------------------------
old = """i64 gate_expert_pool(Backend *be) {"""
new = """// The busy-queue arm's own gate. It is a check on the INSTRUMENT, not on the
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
        std::printf("  %s busy-queue %s\\n", ok ? "ok  " : "FAIL", what);
        if (!ok) ++bad;
    };
    constexpr int kChunk = 64;
    constexpr int kReps = 8;
    const int deep = depth > 1 ? depth : 512;
    const QueueArm busy = measure_queue_arm(be, deep, kChunk, kReps);
    const QueueArm shallow = measure_queue_arm(be, 1, kChunk, kReps);
    const QueueArm empty = measure_queue_arm(be, 0, kChunk, kReps);
    std::printf("    %4d deep: record %6.2f us | copy %6.2f us | tail %8.0f us\\n",
                deep, busy.rec_us, busy.copy_us, busy.tail_us);
    std::printf("    1 deep:    record %6.2f us | copy %6.2f us | tail %8.0f us\\n",
                shallow.rec_us, shallow.copy_us, shallow.tail_us);
    std::printf("    0 deep:    record %6.2f us | copy %6.2f us | tail %8.0f us\\n",
                empty.rec_us, empty.copy_us, empty.tail_us);
    check(busy.tail_us > busy.host_us,
          "the device was still working when the host block returned (a queue "
          "was measured, not a coincidence)");
    check(busy.rec_us * 2.0 < empty.rec_us,
          "a record is >=2x cheaper against a deep queue than an empty one");
    check(busy.copy_us * 1.5 < empty.copy_us,
          "a launch is >=1.5x cheaper against a deep queue than an empty one");
    check(shallow.tail_us * 4.0 < busy.tail_us,
          "one call of pre-issued work leaves the queue empty (the tail is the "
          "one a deep queue leaves, divided by the depth)");
    check(shallow.rec_us > busy.rec_us * 2.0 && shallow.copy_us > busy.copy_us * 1.5,
          "the same block at depth 1 reports the EMPTY figures, not the busy "
          "ones (the arm is reading the queue, not a constant)");
    return bad;
}

i64 gate_expert_pool(Backend *be) {"""
B = sub(B, old, new, tag='gate fn')

old = """        std::printf("\\nlogits_topk:\\n");
        failures += gate_topk_all(be);
        failures += gate_expert_pool(be);"""
new = """        std::printf("\\nlogits_topk:\\n");
        failures += gate_topk_all(be);
        failures += gate_expert_pool(be);
        std::printf("\\nbusy-queue arm (device) -- the instrument's own control:\\n");
        failures += gate_busy_queue(be, depth);"""
B = sub(B, old, new, tag='gate call')

save('tools/bench_ops.cpp', B, crlf)
print('patched: tools/bench_ops.cpp (queue-arm helper, both states printed, gate_busy_queue)')
