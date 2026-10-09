#!/usr/bin/env python3
"""Summarize a KRK_DUMP stage trace: which stages have NaN, in execution order.

  python3 tools/dump_nan_summary.py DUMP [--first N]

The dump is one line per stage per layer ("attn.q L12 n=9 <values...>"), so the
question "where does the first NaN appear" is a scan, not a read. Keep the output
small: this prints a count and the first N affected stages with a few values.
"""
import sys

path = sys.argv[1]
first = 12
if "--first" in sys.argv:
    first = int(sys.argv[sys.argv.index("--first") + 1])

rows = []
with open(path, "r", errors="replace") as f:
    for ln, line in enumerate(f, 1):
        parts = line.split()
        if not parts:
            continue
        stage = parts[0]
        n_nan = sum(1 for p in parts[1:] if "nan" in p.lower() or "inf" in p.lower())
        rows.append((ln, stage, n_nan, len(parts) - 1))

bad = [r for r in rows if r[2] > 0]
print("dump lines=%d  stages with non-finite=%d" % (len(rows), len(bad)))
print("first affected stages, in file order:")
for ln, stage, nbad, n in bad[:first]:
    print("  line %-6d %-16s %d/%d non-finite" % (ln, stage, nbad, n))

# Per-stage-name totals, so a stage that is always poisoned separates from one
# poisoned only at the end of the stack.
tot = {}
for _, stage, nbad, _ in rows:
    name = stage.split("_")[0]
    a, b = tot.get(name, (0, 0))
    tot[name] = (a + (1 if nbad else 0), b + 1)
print("per stage: affected/total")
for name in sorted(tot):
    a, b = tot[name]
    if a:
        print("  %-16s %d/%d" % (name, a, b))
