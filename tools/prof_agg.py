#!/usr/bin/env python3
"""Aggregate a rocprofv3 counter CSV into one row per (kernel, grid, wg, vgpr).

Usage:
    rocprofv3 --pmc ... -f csv -d /tmp/prof -- ./app
    python3 tools/prof_agg.py /tmp/prof/<host>/<pid>_counter_collection.csv
    python3 tools/prof_agg.py --guard /tmp/prof/<host>/<pid>_counter_collection.csv

The last form is a REGRESSION GUARD, not a report: it exits non-zero when any
kernel's LDS reads serialise on bank conflicts beyond a budget. That is the
defect class that held the gfx10 tile GEMMs at 2.9 TFLOP/s (fp16) and 6.2 TOP/s
(int8) while every average-based check stayed green -- SQC_LDS_BANK_CONFLICT
over SQ_INSTS_LDS measured 16.4 and 19.4 respectively, against 0.0-2.5 for the
same kernels once their fetch was rotated off the aliased banks. The limit
here is 6.0, comfortably above every clean kernel measured on gfx1031 and
comfortably below the bug. scripts/check_lds.sh drives the whole check.

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


# ---------------------------------------------------------------------------
# the guard
# ---------------------------------------------------------------------------

# Conflicts per LDS instruction. Clean kernels on gfx1031 measure 0.0 (every
# non-GEMM kernel, and the rotated fp16 tile), 2.2 (Q4_K/Q5_K int8 tiles) and
# up to 3.9 (Q6_K's staging); the aliased fetch measured 16.4 (fp16) and 19.4
# (int8). 6.0 therefore sits an order of magnitude below the defect and 1.5x
# above the worst clean kernel, which is what a guard threshold is for.
DEFAULT_CONFLICT_LIMIT = 6.0
# A ratio needs volume to mean anything: a kernel with 500 LDS reads can show
# any ratio at all from one scheduling accident.
MIN_DISPATCH_LDS = 10000.0
MIN_KERNEL_LDS = 100000.0


def guard(path: str, limit: float) -> int:
    """Fail when any kernel's LDS reads serialise on bank conflicts."""
    per = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))
    collected = False
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            c = (r.get("Counter_Name") or "").strip()
            if c not in ("SQ_INSTS_LDS", "SQC_LDS_BANK_CONFLICT"):
                continue
            collected = True
            slot = per[r.get("Kernel_Name", "?")][r.get("Dispatch_Id", "?")]
            slot[0 if c == "SQ_INSTS_LDS" else 1] = float(r.get("Counter_Value") or 0)
    if not collected:
        print("guard: no LDS counters in this CSV; profile with "
              "--pmc SQ_INSTS_LDS SQC_LDS_BANK_CONFLICT")
        return 2

    print(f"LDS bank-conflict guard: limit {limit:.1f} conflicts per LDS "
          f"instruction")
    print(f"{'kernel':40s} {'disp':>5s} {'lds':>12s} {'worst/disp':>11s} "
          f"{'weighted':>9s}  verdict")
    worst_kernel, worst_rate = None, 0.0
    checked = 0
    for kern in sorted(per, key=lambda k: -sum(v[0] for v in per[k].values())):
        total_lds = sum(v[0] for v in per[kern].values())
        total_bcf = sum(v[1] for v in per[kern].values())
        if total_lds < MIN_KERNEL_LDS:
            continue
        # Per-dispatch ratios, because one pathological dispatch must not be
        # able to hide behind a large clean volume.
        rates = [v[1] / v[0] for v in per[kern].values() if v[0] >= MIN_DISPATCH_LDS]
        worst = max(rates) if rates else 0.0
        weighted = total_bcf / total_lds if total_lds else 0.0
        checked += 1
        bad = worst > limit
        if worst > worst_rate:
            worst_kernel, worst_rate = kern, worst
        print(f"{kern[:40]:40s} {len(per[kern]):>5d} {total_lds:>12.3g} "
              f"{worst:>11.3f} {weighted:>9.3f}  "
              f"{'FAIL' if bad else 'ok'}")

    if checked == 0:
        print("guard: no kernel issued enough LDS traffic to judge; is this a "
              "real workload?")
        return 2
    if worst_rate > limit:
        print(f"\n*** {worst_kernel} reaches {worst_rate:.2f} conflicts per LDS "
              f"instruction (limit {limit:.1f}). ***")
        print("    The gfx10 tile kernels measured 16-19 when a tile row's stride "
              "put every lane on one bank;")
        print("    see docs/TODO.md P1b for the rotation that fixed it and "
              "tools/gemm_i8.cpp §3 for the ablation.")
        return 1
    print(f"\nPASS: {checked} LDS-issuing kernels inside the budget "
          f"(worst {worst_rate:.3f}).")
    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    want_guard = "--guard" in args
    limit = DEFAULT_CONFLICT_LIMIT
    if "--limit" in args:
        i = args.index("--limit")
        limit = float(args[i + 1])
        del args[i:i + 2]
    paths = [a for a in args if a != "--guard"]
    if len(paths) != 1:
        print(__doc__)
        raise SystemExit(2)
    if want_guard:
        raise SystemExit(guard(paths[0], limit))
    raise SystemExit(main(paths[0]))
