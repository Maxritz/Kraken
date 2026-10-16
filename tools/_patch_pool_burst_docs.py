#!/usr/bin/env python3
"""Docs: record the burst-mark result and correct the 128 ms estimate."""
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


patch("AGENTS.md", [(
    """  query-and-wait. What is left of the row is one `hipEventRecord` per release
  (~23 us x 5560 ~= 128 ms) -- an eviction releases three slices at once, so a
  shared mark per burst is the next term.""",
    """  query-and-wait. **The "~128 ms of release records" this bullet used to leave
  as the next term was a quiet-queue estimate and is 10x too big; the next
  bullet has the measurement.**
- **An eviction's three slices share ONE release mark, and the row moved by a
  third -- which is still only ~0.5% of the step.** `Backend::release_pooled_batch`
  releases a burst under one mark: the releases in a burst are consecutive with
  nothing submitted to the default stream between them, so their positions are
  the same point, and the record is the whole cost of a mark. `HipBackend::PoolMark`
  is refcounted -- every cell released under it holds a reference, and so does a
  hand-out whose writer has not consumed the dependency -- and at zero it goes
  back to a free list with its event. `KRK_POOL_BURST=0` restores one record per
  slice, and the engine prints `release %llu mark(s) for %llu slice(s)`.
  Measured on Qwen3-MOE-4x0.6B at `--expert-cache-mb 16` and `32`, n=64, 9
  interleaved order-balanced pairs each: the `room` row (where the release
  happens, inside `make_vram_room`) **58.7 -> 40.6 ms, -31.23% and -31.28%,
  9/9 pairs at both budgets**, with marks 10740 -> 3580 for the same 10740
  slices, i.e. **1.00 -> 3.00 slices per mark**, and stdout md5 unchanged. The
  end-to-end decode verdict is **NO DIFFERENCE** (+1.287% and +2.133%, MAD
  1.6-2.1%): the row is ~18 ms of a ~3.2 s run, and no amount of sharing here
  can be worth more than ~0.5%.
  Two traps, both measured. **A record's cost is queue-dependent**: ~29-31 us on
  the quiet queue `kraken-bench` measures primitives on against ~2.5 us marginal
  in a decode step (7252 records removed = 18 ms), so the old "23 us x 5560 =
  128 ms" was a 10x overstatement -- the same quiet-vs-busy gap as the launch
  floor above, and it is why the whole row is ~13 ms and not ~128. **And the
  sharing must not go past a burst**: `paged_publish_` makes the DEFAULT stream
  wait on the copy stream (`hipStreamWaitEvent(0, paged_ev_[s])`) while the copy
  stream waits on these marks, so a mark re-recorded -- or recycled on a ring --
  while a wait on it is still in flight can land after such a wait and cycle
  with the writer that takes it. One mark per burst, held open while referenced.
  Opening a mark TAKES a reference (`refs = 1`) and `pool_mark_unref_` aborts on
  underflow: without that the same mark reaches two bursts and the destructor
  deletes it twice, which is STATUS_HEAP_CORRUPTION (0xC0000374) with the
  process's buffered stdout lost -- it reads as "prints its banner, exits 127",
  and the raw exit code (python subprocess -> 3221226356) is what names it.
  `kraken-bench --gate`'s three new `expert-pool` checks are the proof, and
  `KRK_POOL_BURST=0 kraken-bench --gate` fails exactly the sharing one (rc=1),
  on a gate whose other two burst checks still pass."""),
])

patch("docs/TODO.md", [(
    """What is left of the row is the release mark itself (~23 us of `hipEventRecord` per
release, ~128 ms).""",
    """The release mark is now shared per eviction burst (`release_pooled_batch`), so a
burst of three slices issues ONE record: marks 10740 -> 3580 for the same 10740
slices. The `room` row (where the release is) fell **-31.23% / -31.28%,
order-balanced, 9/9 pairs** at `--expert-cache-mb` 16 and 32 on
Qwen3-MOE-4x0.6B, n=64; the decode verdict is NO DIFFERENCE (+1.287% / +2.133%),
because the row is ~18 ms of a ~3.2 s run. **The "~128 ms" that was recorded here
was a quiet-queue estimate and is 10x too big**: a record costs ~29-31 us in
`kraken-bench`'s primitives and ~2.5 us marginal in a decode step (7252 records
removed = 18 ms). Evidence:
[traces/expert-pool-burst-marks-2026-10-11.txt](traces/expert-pool-burst-marks-2026-10-11.txt)."""),
])

patch("docs/AUDIT-active-expert-staging.md", [
    ("""What is left of the row is the release mark itself: ~23 us of `hipEventRecord` on
the default stream per release, 5560 of them, i.e. ~128 ms of the remaining
182.5 ms. That is the next term, and it is why it is ranked below rather than
called finished.""",
     """### The release mark is shared per burst now, and the row it lives in moved a third

An eviction returns an expert's three slices in one call, and the three records
were issued at one stream point: nothing is submitted to the default stream
between them, so one record expresses all three. `release_pooled_batch` does
that, with a refcounted `PoolMark` per burst, and `KRK_POOL_BURST=0` restores the
per-slice path.

| budget | `room` ms (old -> new) | order-balanced | pairs | marks old -> new |
|---|---|---|---|---|
| 16 MiB | 58.7 -> 40.6 | **-31.23%** (MAD 1.87%) | 9/9 | 10740 -> 3580 (3.00 slices/mark) |
| 32 MiB | 58.7 -> 40.6 | **-31.28%** (MAD 3.19%) | 9/9 | 10740 -> 3580 (3.00 slices/mark) |

Decode is **NO DIFFERENCE** at both budgets (+1.287%, +2.133%; MAD 1.6-2.1%), and
the stdout md5 is unchanged in every arm.

**The ~128 ms this document used to leave as the next term was a quiet-queue
estimate.** `kraken-bench` measures a record at 29-31 us on an idle queue; in a
busy decode step 7525 records removed bought 18 ms, i.e. **~2.5 us per record**.
The whole row at a constrained budget is ~13 ms of a ~1.7 s run, not ~128 ms, so
this change is a cleanup worth ~0.5% and is recorded as one -- the same
quiet-vs-busy gap that AGENTS.md records for the launch floor."""),
    ("""1. **The release mark, not the allocator** (this pass's leftover). `alloc` is down
   to 182.5 ms at 16 MiB, and ~128 ms of that is one `hipEventRecord` on the
   default stream per release (5560 of them, ~23 us each). A release burst -- the
   cache drops an evicted expert's three slices at once -- needs one mark, not
   three, and `kraken-bench`'s marginal timing is the instrument. The record cost
   is the thing to measure first: if a burst shares a mark, the win is bounded by
   the burstiness of eviction, which the `demotions` counter already reports.""",
     """1. **The release mark is closed** (this pass). One mark per eviction burst, the
   `room` row down a third (9/9 pairs at 16 and 32 MiB), decode NO DIFFERENCE
   because the row is ~18 ms of a ~3.2 s run -- and the ~128 ms it was ranked on
   was a quiet-queue estimate (~2.5 us marginal in the step). See
   `docs/traces/expert-pool-burst-marks-2026-10-11.txt`. What is left in this row
   is the record at all: a cell handed out and released with nothing submitted to
   the default stream in between needs no mark, but in a decode step hundreds of
   launches separate the two, so that case is rare by construction."""),
])
