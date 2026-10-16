#!/usr/bin/env python3
"""Exact medians for the room row, and one record-count typo."""
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


patch("docs/traces/expert-pool-burst-marks-2026-10-11.txt", [(
    """  records removed   10878 - 3626 = 7252 per run (n=64)
  room row saved    58.7 - 40.6 = 18.1 ms at 32 MiB, 18.8 ms at 16 MiB
  -> 2.5 us per record, NOT 23 us""",
    """  records removed   10878 - 3626 = 7252 per run (n=64)
  room row saved    median 58.5 -> 39.7 at 16 MiB (18.8 ms),
                    59.0 -> 40.6 at 32 MiB (18.4 ms)
  -> 18.6 ms / 7252 = 2.56 us per record, NOT 23 us"""),
])

patch("AGENTS.md", [(
    """  interleaved order-balanced pairs each: the `room` row (where the release
  happens, inside `make_vram_room`) **58.7 -> 40.6 ms, -31.23% and -31.28%,
  9/9 pairs at both budgets**, with marks 10740 -> 3580 for the same 10740""",
    """  interleaved order-balanced pairs each: the `room` row (where the release
  happens, inside `make_vram_room`) **58.5 -> 39.7 ms at 16 MiB and 59.0 -> 40.6
  at 32 MiB, -31.23% and -31.28%, 9/9 pairs at both budgets**, with marks
  10740 -> 3580 for the same 10740"""),
])

patch("docs/AUDIT-active-expert-staging.md", [
    ("""| 16 MiB | 58.7 -> 40.6 | **-31.23%** (MAD 1.87%) | 9/9 | 10740 -> 3580 (3.00 slices/mark) |
| 32 MiB | 58.7 -> 40.6 | **-31.28%** (MAD 3.19%) | 9/9 | 10740 -> 3580 (3.00 slices/mark) |""",
     """| 16 MiB | 58.5 -> 39.7 | **-31.23%** (MAD 1.87%) | 9/9 | 10740 -> 3580 (3.00 slices/mark) |
| 32 MiB | 59.0 -> 40.6 | **-31.28%** (MAD 3.19%) | 9/9 | 10740 -> 3580 (3.00 slices/mark) |"""),
    ("""busy decode step 7525 records removed bought 18 ms, i.e. **~2.5 us per record**.""",
     """busy decode step 7252 records removed bought 18.6 ms, i.e. **~2.5 us per record**."""),
])
