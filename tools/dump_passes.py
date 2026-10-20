#!/usr/bin/env python3
"""Compare two KRK_DUMP files pass by pass, across DIFFERENT chunk sizes.

    KRK_DUMP=a.txt ./kraken -m M -p "..." -n 1 --chunk 1 ...
    KRK_DUMP=b.txt ./kraken -m M -p "..." -n 1 --chunk 2 ...
    python tools/dump_passes.py a.txt b.txt --chunk-a 1 --chunk-b 2

`tools/dump_diff.py` compares line i of one dump with line i of the other, which
is right when both runs walk the same stages in the same order — the device
against the CPU reference, say. It is wrong for a chunk-size comparison: each
dump line is one stage of one layer for the row that pass predicts from, so a
--chunk 1 run emits one pass per token while --chunk 2 emits one per two tokens.
Line 0 of the two files is then position 0 against position 1 and every line
"differs".

This tool splits both dumps at each `embed L-1` line (one pass per forward
call), works out which absolute position each pass's dumped row belongs to, and
compares only the positions the two runs share. The first position that is not
bit-identical is the answer, and the stage it stops at names the kernel:
`attn.*` is the attention call, `gdn.*` the recurrent block, `moe.*` the routed
FFN.

Values are compared exactly — the dumps are 8 significant digits of an f16
activation, so an exact match means the two paths wrote the same bytes.
"""
import sys


def passes(path):
    """Split a dump into one list of (stage, layer, values) per forward call."""
    out, cur = [], []
    with open(path) as fh:
        for line in fh:
            tok = line.split()
            if len(tok) < 3:
                continue
            if tok[0] == "embed" and tok[1] == "L-1" and cur:
                out.append(cur)
                cur = []
            cur.append((tok[0], tok[1], tok[2], [float(x) for x in tok[3:]]))
    if cur:
        out.append(cur)
    return out


def main(argv):
    if len(argv) < 3:
        print(__doc__.strip().splitlines()[0])
        print("usage: dump_passes.py DUMP_A DUMP_B [--chunk-a N] [--chunk-b N]")
        return 2
    ca = int(argv[argv.index("--chunk-a") + 1]) if "--chunk-a" in argv else 1
    cb = int(argv[argv.index("--chunk-b") + 1]) if "--chunk-b" in argv else 2
    A, B = passes(argv[1]), passes(argv[2])
    print(f"{argv[1]}: {len(A)} passes (chunk {ca})   "
          f"{argv[2]}: {len(B)} passes (chunk {cb})")
    # A pass dumps its LAST row, so pass i of a chunk-c run is position
    # i*c + c - 1. Only positions both runs dumped can be compared.
    pos_a = {i * ca + ca - 1: i for i in range(len(A))}
    pos_b = {i * cb + cb - 1: i for i in range(len(B))}
    shared = sorted(set(pos_a) & set(pos_b))
    if not shared:
        print("no shared position: the two runs dumped disjoint rows")
        return 1
    for pos in shared:
        pa, pb = A[pos_a[pos]], B[pos_b[pos]]
        if len(pa) != len(pb):
            print(f"pos {pos}: pass shape differs ({len(pa)} vs {len(pb)} "
                  f"stages) — the runs are not the same model shape")
            continue
        for i, (x, y) in enumerate(zip(pa, pb)):
            if x[:2] != y[:2]:
                print(f"pos {pos}: stage mismatch at {i}: {x[:2]} vs {y[:2]}")
                break
            if x[3] != y[3]:
                peak = max(abs(v) for v in x[3]) or 1.0
                rel = max(abs(p - q) for p, q in zip(x[3], y[3])) / peak
                tail = " (n differs: %s vs %s)" % (x[2], y[2]) if x[2] != y[2] else ""
                print(f"pos {pos}: first divergence at stage {i} "
                      f"{x[0]} {x[1]} rel={rel:.3e}{tail}")
                return 0
        print(f"pos {pos}: all {len(pa)} stages bit-identical")
    print("every shared position was bit-identical")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
