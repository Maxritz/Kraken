#!/usr/bin/env python3
# Redesign the busy-queue arm around what it can actually prove: the block
# method separates the HOST enqueue from the DEVICE cost (the tail), and the
# gate asserts the queue really is fuller, and that the two halves of a record
# are not one number.
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
# 1. the helper measures one probe kind and returns BOTH halves
# --------------------------------------------------------------------------
old = """struct QueueArm {
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
}"""
new = """// Every call in this engine has two costs and a table that quotes one number
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
}"""
B = sub(B, old, new, tag='helper')

# --------------------------------------------------------------------------
# 2. the print arm: both states, both halves
# --------------------------------------------------------------------------
old = """    if (depth > 0) {
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
new = """    if (depth > 0) {
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
            "%.0f us vs %.0f us empty (a launch: %.0f vs %.0f us)\\n",
            depth, brec.tail_us, erec.tail_us, bcopy.tail_us, ecopy.tail_us);
        std::printf(
            "primitives, host enqueue per call: record %.2f us | copy %.2f us, "
            "median of %d (empty queue: record %.2f | copy %.2f)\\n",
            brec.host_us, bcopy.host_us, kReps, erec.host_us, ecopy.host_us);
        std::printf(
            "primitives, device cost per call: record %.1f us | copy %.1f us "
            "(empty queue: record %.1f | copy %.1f)  [time_calls arms, which "
            "see only the dominant half: record %.2f | copy %.2f]\\n",
            brec.device_us, bcopy.device_us, erec.device_us, ecopy.device_us,
            record_ms * 1000.0, launch_ms * 1000.0);
    }"""
B = sub(B, old, new, tag='print arm')

# --------------------------------------------------------------------------
# 3. the gate: assert the two things the arm can prove
# --------------------------------------------------------------------------
old = """i64 gate_busy_queue(Backend *be, int depth) {
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
}"""
new = """i64 gate_busy_queue(Backend *be, int depth) {
    i64 bad = 0;
    const auto check = [&](bool ok, const char *what) {
        std::printf("  %s busy-queue %s\\n", ok ? "ok  " : "FAIL", what);
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
                "(launch %.0f vs %.0f us)\\n",
                brec.tail_us, erec.tail_us, bcopy.tail_us, ecopy.tail_us);
    std::printf("    host enqueue/call: record %.2f us | copy %.2f us\\n",
                brec.host_us, bcopy.host_us);
    std::printf("    device cost/call:  record %.1f us | copy %.1f us\\n",
                brec.device_us, bcopy.device_us);
    // The construction the whole arm exists for: with `deep` copies in front of
    // them, the block's own work cannot have been retired by the time the host
    // returns, so the tail must carry it. Launch-only, because a record's own
    // ~24 us device marker swamps the pre-issue's leftovers in the same tail.
    check(bcopy.tail_us > 3.0 * ecopy.tail_us,
          "a queue holding `depth` calls is measurably fuller than an empty one "
          "(device left to drain: launch arm)");
    check(bcopy.device_us > 3.0 * ecopy.device_us,
          "and the same difference is visible per call");
    // The defect the arm was built to stop: quoting one half of a call as the
    // call. A legacy-stream record is ~0.3 us of host enqueue and ~24 us of
    // device marker; a table that measures 28 us by T(N)-T(1) has measured the
    // device half and called it the record.
    check(brec.device_us > 4.0 * brec.host_us,
          "a record's device half is >=4x its host half (so the two figures a "
          "table might quote are not the same number)");
    check(bcopy.device_us < 4.0 * bcopy.host_us,
          "a small launch's halves are the SAME order, which is why one number "
          "was good enough for it and was not for the record");
    return bad;
}"""
B = sub(B, old, new, tag='gate fn')

save('tools/bench_ops.cpp', B, crlf)
print('patched: tools/bench_ops.cpp (arm separates host enqueue from device cost; gate asserts fullness + the two-halves split)')
