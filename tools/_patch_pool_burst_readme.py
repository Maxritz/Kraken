#!/usr/bin/env python3
"""README: the pool's release line and how a burst shares one mark."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
p = ROOT / "README.md"
raw = p.read_text(newline="")
eol = "\r\n" if "\r\n" in raw else "\n"
s = raw.replace("\r\n", "\n")

pairs = [
    ("live <l> reusable <r> bytes; alloc <a> reuse <u> trim <t> guard-fallback <g>`.\n",
     "live <l> reusable <r> bytes; alloc <a> reuse <u> trim <t> guard-fallback <g> |\n"
     "release <m> mark(s) for <s> slice(s)>`.\n"),
    ("""change really does move the physical commitment and a shrink trims free cells
without touching one a kernel is still reading. `guard-fallback` counts the times
a release event could not be created and the pool fell back to draining the device
before reuse - it should stay 0.""",
     """change really does move the physical commitment and a shrink trims free cells
without touching one a kernel is still reading. `guard-fallback` counts the times
a release mark could not be created and the pool fell back to draining the device
before reuse - it should stay 0. The `release` tail is the release marks and the
slices they covered: an eviction returns an expert's three slices in one call and
issues **one** record for the burst, so a resident budget reads `3` slices per
mark and `1` means `KRK_POOL_BURST=0` (the per-slice path, kept as the A/B arm).""",),
    ("""it took `alloc` from 618.4 to 182.5 ms and decode up 11.3% at `--expert-cache-mb 16`
(9/9 interleaved pairs, identical text). `KRK_POOL_HOST_WAIT=1` restores the old
query-and-wait behaviour, which is the arm to compare against.""",
     """it took `alloc` from 618.4 to 182.5 ms and decode up 11.3% at `--expert-cache-mb 16`
(9/9 interleaved pairs, identical text). `KRK_POOL_HOST_WAIT=1` restores the old
query-and-wait behaviour, which is the arm to compare against.

The three slices of one eviction are released at the same stream point, so they
share a single mark: measured on Qwen3-MOE-4x0.6B at 16 and 32 MiB, the `room` row
(the release happens inside `make_vram_room`) falls **31% (9/9 interleaved
order-balanced pairs at both budgets)** and the marks go from one per slice to one
per burst. Decode is **NO DIFFERENCE** there - the row is ~18 ms of a ~3.2 s run -
and a record costs ~2.5 us marginal in a busy step against the 29-31 us
`kraken-bench` measures on a quiet queue, so this is a cleanup and not a win. The
sharing is exact, not approximate: the releases in a burst are consecutive with
nothing submitted between them. It must not go further than a burst, because
`paged_publish_` makes the default stream wait on the copy stream while the copy
stream waits on these marks.""",),
]

for old, new in pairs:
    old, new = old.replace("\r\n", "\n"), new.replace("\r\n", "\n")
    n = s.count(old)
    if n != 1:
        sys.exit(f"anchor found {n} times: {old[:80]!r}")
    s = s.replace(old, new)

p.write_text(s.replace("\n", eol), newline="")
print("README.md: 3 replacement(s)")
