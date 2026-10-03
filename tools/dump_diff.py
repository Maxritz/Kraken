#!/usr/bin/env python3
"""Compare two KRK_DUMP stage dumps, line by line.

    KRK_DUMP=a.txt ./kraken -m M -p "..." -n 1 ...        # device
    KRK_DUMP=b.txt ./kraken -m M --cpu -p "..." -n 1 ...  # reference
    python tools/dump_diff.py a.txt b.txt

Each line of a dump is one stage of one layer ("attn.gate L03 n=1 v0 v1 ...").
Both runs walk the same stages in the same order, so line i of one dump
corresponds to line i of the other: the first line whose relative error jumps
to O(1) names the stage that broke, and everything after it is downstream
cascade. Small two-decimal-exponent errors (1e-3 and below) are fp16 activation
rounding on the device against the f32 reference and mean agreement.

Lines flagged "<<" exceed 1e-3; the first flag is usually the real one, since a
broken stage perturbs every layer after it. Row order matters: the dumps are
appended, so delete the files before a run (the engine appends).
"""
import sys


def load(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            tok = line.split()
            rows.append((tok[0], tok[1], [float(x) for x in tok[3:]]))
    return rows


def main(argv):
    if len(argv) != 3:
        print(__doc__.strip().splitlines()[0])
        print("usage: dump_diff.py DUMP_A DUMP_B")
        return 2
    a, b = load(argv[1]), load(argv[2])
    print(f"{argv[1]}: {len(a)} lines   {argv[2]}: {len(b)} lines")
    worst = None
    for i, (la, lb) in enumerate(zip(a, b)):
        if la[:2] != lb[:2]:
            print(f"line {i}: stage mismatch {la[:2]} vs {lb[:2]} — dumps are "
                  f"not the same run shape")
            break
        peak = max(abs(v) for v in la[2]) or 1.0
        maxabs = max(abs(x - y) for x, y in zip(la[2], lb[2]))
        rel = maxabs / peak
        if worst is None or rel > worst[1]:
            worst = (la, rel)
        if rel > 1e-3:
            print(f"{i:4d} {la[0]:>12s} {la[1]} rel={rel:.3e} peak={peak:9.3g} <<<")
    if worst:
        print(f"worst: {worst[0][0]} {worst[0][1]} rel={worst[1]:.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
