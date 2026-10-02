#!/usr/bin/env python3
"""Aggregate a rocprofv3 counter CSV into one row per (kernel, grid, wg, vgpr).

Usage:
    rocprofv3 --pmc ... -f csv -d /tmp/prof -- ./app
    python3 tools/prof_agg.py /tmp/prof/<host>/<pid>_counter_collection.csv

Why this exists: rocprofv3 emits one CSV row per (dispatch, counter). A single
GEMM run is thousands of rows and the interesting questions are all ratios
(VALU per cycle, LDS per k-tile, resident waves, bank conflicts per LDS
access). This prints the ratios directly, grouped by the configuration that
produced them, plus a per-dispatch wall span so launch gaps are visible.

Counters are optional: any counter not collected is simply reported as "-".
Derived columns:
    dur_ms      mean per-dispatch span (End_Timestamp - Start_Timestamp)
    val/cyc     SQ_INSTS_VALU / (GRBM_COUNT)
    val/clk/cu  SQ_INSTS_VALU / GRBM_COUNT per CU  (x1000 for readability)
    waves/cu    MeanOccupancyPerCU, else SQ_WAVES / GRBM_COUNT / n_cu
    lds/wave    SQ_INSTS_LDS per wave
    bcf/lds     SQC_LDS_BANK_CONFLICT per LDS instruction
"""
import csv
import sys
from collections import defaultdict

WANT = [
    "GRBM_COUNT", "SQ_WAVES", "SQ_INSTS_VALU", "SQ_INSTS_LDS", "SQ_INSTS_SALU",
    "SQ_WAVE_CYCLES", "SQ_BUSY_CYCLES", "SQC_LDS_BANK_CONFLICT",
    "OccupancyPercent", "MeanOccupancyPerCU", "SQ_WAIT_INST_LDS",
    "ALUStalledByLDS", "SQ_INST_CYCLES_VMEM", "LDSBankConflict",
]


def main(path: str) -> int:
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append(r)
    if not rows:
        print("no rows")
        return 1

    groups = defaultdict(lambda: defaultdict(float))
    spans = defaultdict(lambda: [float("inf"), float("-inf")])
    nruns = defaultdict(set)
    meta = {}

    for r in rows:
        cid = r.get("Dispatch_Id", "?")
        key = (r.get("Kernel_Name", "?"), r.get("Grid_Size", "?"),
               r.get("Workgroup_Size", "?"), r.get("VGPR_Count", "?"),
               r.get("Scratch_Size", "?"))
        g = groups[key]
        name = (r.get("Counter_Name") or "").strip()
        try:
            val = float(r.get("Counter_Value") or "nan")
        except ValueError:
            val = float("nan")
        g[name] += val
        nruns[key].add(cid)
        if cid not in meta:
            meta[cid] = key
        try:
            s = float(r.get("Start_Timestamp", "nan"))
            e = float(r.get("End_Timestamp", "nan"))
            if s == s and e == e:
                spans[cid][0] = min(spans[cid][0], s)
                spans[cid][1] = max(spans[cid][1], e)
        except ValueError:
            pass

    print("per-group totals are sums over all dispatches of that kernel/grid\n"
          "(raw means per dispatch are in the second table; 'n' is dispatches)\n")
    print(f"{'kernel':44s} {'grid':>7s} {'wg':>5s} {'vgpr':>5s} {'scr':>4s} "
          f"{'n':>3s} {'dur_ms':>8s} {'val/cyc':>7s} {'val/clk/cu':>10s} "
          f"{'lds/wave':>8s} {'bcf/lds':>8s} {'waves':>8s} {'busy%':>6s}")
    for key in sorted(groups, key=lambda k: (-float(k[1]), k[0])):
        g = groups[key]
        name, grid, wg, vgpr, scr = key
        n = len(nruns[key])
        dur = sum((spans[c][1] - spans[c][0]) for c in nruns[key]) / n * 1e-6  # ns->ms
        grbm = g.get("GRBM_COUNT", 0.0)
        valu = g.get("SQ_INSTS_VALU", 0.0)
        lds = g.get("SQ_INSTS_LDS", 0.0)
        waves = g.get("SQ_WAVES", 0.0)
        bcf = g.get("SQC_LDS_BANK_CONFLICT", g.get("LDSBankConflict", 0.0))
        busy = g.get("SQ_BUSY_CYCLES", 0.0)
        occ = g.get("MeanOccupancyPerCU", 0.0)

        def r(a, b):
            return f"{a / b:.3g}" if b else "-"
        print(f"{name[:44]:44s} {grid:>7s} {wg:>5s} {vgpr:>5s} {scr:>4s} "
              f"{n:>3d} {dur:>8.3f} {r(valu, grbm):>7s} {r(valu * 1000, grbm):>10s} "
              f"{r(lds, waves):>8s} {r(bcf, lds):>8s} "
              f"{(f'{occ:.1f}' if occ else r(waves, grbm)):>8s} {r(100 * busy, grbm):>6s}")

    want = ["GRBM_COUNT", "SQ_WAVES", "SQ_INSTS_VALU", "SQ_INSTS_LDS",
            "SQC_LDS_BANK_CONFLICT", "LDSBankConflict", "SQ_BUSY_CYCLES",
            "SQ_WAVE_CYCLES", "SQ_LEVEL_WAVES", "SQ_WAIT_INST_LDS",
            "ALUStalledByLDS", "SQ_INST_CYCLES_VMEM", "MeanOccupancyPerCU",
            "OccupancyPercent", "SQ_INSTS_SALU"]
    present = [c for c in want if any(c in groups[k] for k in groups)]
    print("\n--- raw, mean per dispatch ---")
    print(f"{'kernel':36s} {'grid':>7s} " +
          " ".join(f"{c[:9]:>10s}" for c in present))
    for key in sorted(groups, key=lambda k: (-float(k[1]), k[0])):
        g = groups[key]
        if not g.get("SQ_INSTS_VALU"):
            continue
        n = len(nruns[key])
        print(f"{key[0][:36]:36s} {key[1]:>7s} " +
              " ".join(f"{g.get(c, 0.0) / n:>10.4g}" for c in present))
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
