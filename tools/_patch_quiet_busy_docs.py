#!/usr/bin/env python3
"""Docs: the quiet-vs-busy constants, and the rankings they correct."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def patch(rel, pairs):
    p = ROOT / rel
    raw = p.read_text(newline="")
    eol = "\r\n" if "\r\n" in raw else "\n"
    s = raw.replace("\r\n", "\n")
    for old, new in pairs:
        old, new = old.replace("\r\n", "\n"), new.replace("\r\n", "\n")
        n = s.count(old)
        if n != 1:
            sys.exit(f"{rel}: anchor found {n} times: {old[:70]!r}")
        s = s.replace(old, new)
    p.write_text(s.replace("\n", eol), newline="")
    print(f"{rel}: {len(pairs)} replacement(s)")


patch("AGENTS.md", [
    # 1. the fold bullet: the launch constant, re-measured
    ("""  component rows disappearing (60 = 2 per layer x 30 layers). **The marginal
  launch costs ~1.5 us, not the 3.64 us `kraken-bench` floor** -- 46 us for 30
  launches -- so sizing a fold from the bench floor overstates it ~2.4x; that
  floor is the price of adding a call to a quiet queue, not of holding one in a
  busy step.""",
     """  component rows disappearing (60 = 2 per layer x 30 layers). **Sizing a fold
  from a per-op constant is good to ~2x and nothing better; run the fold.** Re-measured
  on the current build (2026-10-11, 9 interleaved order-balanced pairs each, 0/9
  pairs, md5 identical): `KRK_ACC_GEMM=0` (the residual fold, +30 launches) costs
  **-5.219%** and `KRK_FUSED_SILU=0` (also +30) costs **-4.591%**, i.e. **2.4-2.8 us
  per added call** -- which includes that fold's extra memory pass -- where the
  "1.5 us" this bullet used to quote came from the same fold on an older build and
  reproduces at neither end now. `kraken-bench`'s floor is **2.85-3.05 us** for a
  trivial call in a quiet queue and **0.92 us** for the host-side enqueue of the
  same call into a busy one, so the floor is within ~10% of a call that carries a
  pass and overstates a pure enqueue ~3x."""),
    # 2. the profiler bullet: the 69% band
    ("""  weight traffic differs 6.7x, against a `kraken-bench` launch floor of 3.64 us per
  call, so ~69% of the step is the floor. Time a comparison WITHOUT `--profile`""",
     """  weight traffic differs 6.7x, and the per-op cost is 0.9-2.8 us depending on
  whether the call carries a memory pass, so **17-53% of the step is launch cost**
  (the "~69% at 3.64 us" this used to say is the top of that band, not the middle).
  Time a comparison WITHOUT `--profile`"""),
    # 3. the pool-primitives bullet: the record's in-step figure
    ("""  event create+destroy **0.13 us**, record on the default stream **22.7-30.6 us**,
  query a *completed* event **0.03 us**, wait on a completed event **0.08 us** --
  but **record then query it while pending is +46.9 us over the record**, and the
  real alloc+release pair cost **86.97 us**. 5560 cells x 87 us = 484 ms, the row
  to the millisecond.""",
     """  event create+destroy **0.13 us**, record on the default stream **22.7-30.6 us**,
  query a *completed* event **0.03 us**, wait on a completed event **0.08 us** --
  but **record then query it while pending is +46.9 us over the record**, and the
  real alloc+release pair cost **86.97 us**. 5560 cells x 87 us = 484 ms, the row
  to the millisecond. **The record's own quiet figure is the one that is wrong:
  in-step it costs 2.56 us, not 22-31 -- see the audit bullet below.**"""),
    # 4. the new audit bullet, after the pool-primitives one
    ("""- **An eviction's three slices share ONE release mark, and the row moved by a""",
     """- **A primitive measured on a quiet queue is not the in-step cost, and the
  factor is not a constant.** `kraken-bench` measures one call with an idle queue;
  a decode step issues 243-2965 of them with hundreds already queued, and the two
  figures differ in BOTH directions. Measured (2026-10-11, one session; full table
  and raw transcripts in `docs/AUDIT-quiet-vs-busy-primitives.md` and
  `docs/traces/quiet-vs-busy-primitives-2026-10-11.txt`):

  | primitive | quiet (bench) | busy host enqueue | in-step (switch A/B) | factor |
  |---|---|---|---|---|
  | record on the default stream | 28.2-30.6 us | 0.32 us | **2.56 us** | quiet overstates **11-12x** |
  | trivial launch (`copy_act n=32`) | 2.85-3.05 us | 0.92 us | 2.4-2.8 us per added call *with a pass* | ~1.1x with a pass, ~3.3x for an enqueue |
  | record + query a *pending* event | 24.7-46.9 us | -- | 46.9 us | **1.0x -- the calibrated one** |
  | a *device drain* (`hipDeviceSynchronize`) | not measured | -- | **~129 us** | the 0.07 us wait arm is 1000x off |
  | pool round trip (256 KiB cells) | 18.8-33.7 us | -- | 16.5 us | <=1.6x, either way |
  | `--floor` 1 MiB `copy_act` | 240.9 GB/s | -- | in-step `xfer` is H2D at 12.9 GB/s | wrong shelf: its ceiling is the reader's 15.9 GB/s |

  The arm that makes this checkable is `kraken-bench`'s **busy-queue line**: the
  queue is fed 512 x `copy_act n=32` of real device work and 64 calls are timed
  inside it (median of 8 blocks), so the quiet and busy figures print side by
  side. It cannot use `time_calls` -- that helper's cold-call subtraction is
  bigger than a host enqueue, and the first version of the arm printed a
  **negative** marginal record cost (-12.82 us). `--depth N` sets the pre-issued
  count, `--depth 0` skips the arm, and the gate's floor line now says
  `unresolvable at iters=2` instead of printing `-24.00 us per call` (the gate
  sets iters=2, where the subtraction is the difference of two single
  measurements).
  **Rules:** rank a fold by running the fold (one session, interleaved,
  order-balanced) and treat any constant-times-count estimate as good to ~2x;
  never compare a host-to-device row against `--floor`; and remember that a
  *drain* is 1000x a wait.
- **An eviction's three slices share ONE release mark, and the row moved by a"""),
])

patch("docs/AUDIT-group-type-walls.md", [
    ("""| SmolLM2-135M Q4_K_M | 28 of 243 | 11.5% | ~2.6% at 1.5 us/launch, ~5.4% at 3.1 |""",
     """| SmolLM2-135M Q4_K_M | 28 of 243 | 11.5% | ~4.6% at 2.4 us/launch, ~6.0% at 3.1 |"""),
    ("""The band is this repo's measured marginal launch cost -- 1.5 us from SmolLM2's own
residual fold (30 launches, +2.867%) up to 3.1 us from the GDN fold (54 launches,
+6.084%) -- applied to a count, so treat it as a ranking, not a result.""",
     """The band is this repo's measured marginal launch cost -- **re-measured 2026-10-11
on the current build at 2.4-2.8 us** (SmolLM2's residual fold, 30 launches,
-5.219% and its silu fold, 30 launches, -4.591%, 0/9 interleaved pairs each; the
1.5 us lower end this table used to quote came from the same fold on an older
build) up to 3.1 us from the GDN fold (54 launches, +6.084%) -- applied to a count,
so treat it as a ranking, not a result. See
`docs/AUDIT-quiet-vs-busy-primitives.md` for why the constant moved and which
per-op figures are trustworthy."""),
])

patch("docs/PERF-PLAN.md", [
    ("""## 0c. Harness table (`kraken-bench --sweep`, marginal cost, launch floor 2.65 us)""",
     """## 0c. Harness table (`kraken-bench --sweep`, marginal cost, launch floor 2.65 us)

> **2026-10-11: the launch floor is a band, not a constant, and this constant is
> stale.** Re-measured on the current build: 2.85-3.05 us for `copy_act n=32` in a
> quiet queue, **0.92 us** for the host-side enqueue of the same call into a busy
> one, and **2.4-2.8 us** in-step for a call that also carries a memory pass (the
> fold A/Bs on SmolLM2, 0/9 pairs each). The table below is raw kernel time and is
> unaffected; only the normalization constant moved. `docs/AUDIT-quiet-vs-busy-primitives.md`
> has the table and the method."""),
])

patch("docs/TODO.md", [
    ("""### What is now settled, so it is not re-derived""",
     """### What is now settled, so it is not re-derived

**No per-op constant from `kraken-bench` is the in-step cost, and the factor is
not one number.** `docs/AUDIT-quiet-vs-busy-primitives.md` measures each one three
ways (quiet queue, busy-queue host enqueue, in-step by switch A/B): the record is
**11-12x over** (28-31 us quiet, 2.56 us in-step -- it sized the ~128 ms item this
list used to rank first), a trivial launch is ~1x if the call carries a memory
pass and ~3.3x over for a pure enqueue, a **device drain is ~129 us** where the
bench's wait arm says 0.07 us, and the pending-event query was already calibrated
at 1.0x (484 ms predicted, 481.8 measured). `kraken-bench` now prints a
busy-queue line beside the quiet one, and `--depth 0` skips it. Rule: rank a fold
by running the fold in one session; constants size a plan to ~2x.""")
])
