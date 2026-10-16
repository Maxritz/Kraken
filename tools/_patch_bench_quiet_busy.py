#!/usr/bin/env python3
"""kraken-bench: measure the record and the launch with work queued, and stop
printing a negative launch floor when iters is too low to resolve one."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
p = ROOT / "tools/bench_ops.cpp"
raw = p.read_text(newline="")
eol = "\r\n" if "\r\n" in raw else "\n"
s = raw.replace("\r\n", "\n")

pairs = []

# 1. the queued-queue arm: the two per-op constants the engine ranks on, measured
#    the way a decode step actually issues them.
pairs.append((
    """    be->set_expert_pool_budget(0);
}

i64 gate_expert_pool(Backend *be) {""",
    """    // The two constants a decode step's folds are sized with, measured with real
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
    be->set_expert_pool_budget(0);
}

i64 gate_expert_pool(Backend *be) {"""))

# 2. the launch floor has to reach the queued arm
pairs.append((
    """void bench_pool_primitives(Backend *be, int iters) {
    const int n = std::max(50, iters);""",
    """void bench_pool_primitives(Backend *be, int iters, int depth, double launch_ms) {
    const int n = std::max(50, iters);"""))

# 3. no negative per-call figure when the switch is too low to resolve one
pairs.append((
    """        const double ms = time_calls(be, iters,
                                     [&] { be->copy_act(a, b, 32); });
        std::printf("launch floor (copy_act n=32): %.2f us per call\\n\\n",
                    ms * 1000.0);""",
    """        const double ms = time_calls(be, iters,
                                     [&] { be->copy_act(a, b, 32); });
        // time_calls subtracts one cold call from the mean, which cannot resolve
        // anything when the switch is 2 -- the gate's setting -- and has printed
        // "-24.00 us per call" there. Quote only what was measured.
        if (ms > 0.0)
            std::printf("launch floor (copy_act n=32): %.2f us per call\\n\\n",
                        ms * 1000.0);
        else
            std::printf("launch floor (copy_act n=32): unresolvable at "
                        "iters=%d (the per-call subtraction needs a deeper "
                        "switch)\\n\\n",
                        iters);
        launch_ms = ms;"""))

# 4. thread it through main, with the queue depth as a flag
pairs.append((
    """    int iters = 100;
    bool sweep = false;""",
    """    int iters = 100;
    int depth = 64; // queue depth for the queued per-op arms (0 skips them)
    double launch_ms = 0.0;
    bool sweep = false;"""))
pairs.append((
    """        if (a == "--iters") iters = static_cast<int>(next());""",
    """        if (a == "--iters") iters = static_cast<int>(next());
        else if (a == "--depth") depth = static_cast<int>(next());"""))
pairs.append((
    """    if (!gate) {
        bench_pool_primitives(be, iters);
        std::printf("\\n");
    }""",
    """    if (!gate) {
        bench_pool_primitives(be, iters, depth, launch_ms);
        std::printf("\\n");
    }"""))

for old, new in pairs:
    old, new = old.replace("\r\n", "\n"), new.replace("\r\n", "\n")
    cnt = s.count(old)
    if cnt != 1:
        sys.exit(f"anchor found {cnt} times: {old[:70]!r}")
    s = s.replace(old, new)

p.write_text(s.replace("\n", eol), newline="")
print(f"tools/bench_ops.cpp: {len(pairs)} replacement(s)")
