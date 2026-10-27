#!/usr/bin/env python3
"""trace_stats.py -- summarise a Chrome-trace JSON written by kraken or by VNPU.

Two producers write Chrome trace-event JSON with microsecond timestamps, and
they answer different questions. This tool reads either, detects which one it
has been handed, and reports the numbers that question needs.

  kraken --trace PATH        THE HOST TIMELINE. Spans nest: a decode step
                             contains a layer's components, which contain the
                             backend calls. The question is where the wall time
                             went, so this reports SELF time -- a span's total
                             minus its children's. Without that subtraction
                             `host/decode.step` wins every table simply by
                             being the outermost box.

  KRK_NPU_TRACE=PATH         THE DEVICE TIMELINE. One flat span per command the
                             virtual NPU actually executed, with bytes and FLOPs
                             attached. The question is what the device was doing
                             and how fast, so this reports service time per op,
                             achieved bandwidth and rate, and what fraction of
                             the run's span the device was inside a kernel.

Both clocks are microseconds, but from different epochs: the engine's is
relative to the recorder, VNPU's is CLOCK_MONOTONIC-derived. The two are
therefore NOT joined into one timeline here.

A caveat this tool checks for and will not paper over: VNPU's recorded START
times are quantised to a 10 ms grid, so many commands share a timestamp even
though their durations are exact. (Measured: a 928-command run carries 29
distinct start times, all exactly 10000 us apart, with the durations inside
each 10 ms slot summing to just under 10000 us -- real quantisation, not
precision loss. The platform clock is fine-grained: its own steady_clock
resolves changes 100-200 ns apart, so the coarseness is in what reached the
file.) When that is detected, the timing-derived aggregates -- gaps, idle
time, duty cycle, overlap -- are NOT reported, because a duty cycle computed
from a 10 ms grid would be an artifact of the grid. Per-op service time,
achieved bandwidth and GFLOP/s are measured from durations and remain valid.

A second caveat: VNPU's `tid` is `hash(std::thread::id) % 8`, not a worker
index, so nothing here reports "worker occupancy".

A caveat this tool will not paper over: VNPU's `tid` is
`hash(std::thread::id) % 8`, not a worker index. Equal tids do not mean the
same worker and different tids do not prove parallelism, so nothing here
reports "worker occupancy". What IS measured per event is the duration on the
executing thread, which is why service time is trustworthy and occupancy is
not computed at all.

Usage
-----
    tools/trace_stats.py TRACE.json [TRACE2.json ...] [--top N]
                                  [--gap-us US] [--json OUT] [--label NAME]

Each file gets its own section; all files then share a one-line summary, which
is what makes a `--npu` vs `--npu-cpu` pair readable side by side.

Exit status is 0 when every file parsed and contained at least one span, 1
otherwise -- a trace with no spans is a real failure (the run did not emit),
and a silent empty table would hide it.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import statistics
import sys

# Categories the engine's recorder uses for host spans (include/krk/trace.hpp)
# and the single category VNPU uses for executed commands.
_ENGINE_CATS = {"host", "component", "config", "transfer", "kernel", "wait"}
_DEVICE_CATS = {"compute", "memory", "sync", "dispatch"}


class Trace:
    """One parsed trace file: metadata, spans, and what produced them."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.spans: list[dict] = []
        self.instants = 0
        self.kind = "unknown"
        self.meta: dict[int, str] = {}

    @property
    def label(self) -> str:
        return self._label or os.path.basename(self.path)

    def set_label(self, label: str) -> None:
        self._label = label

    _label = ""


def load(path: str) -> Trace:
    """Read a Chrome-trace JSON. Raises ValueError with the reason if it will not.

    The producers differ in one annoying way: VNPU emits `{"traceEvents":[...],`
    with a trailing `"displayTimeUnit"` and `"otherData"`, and the engine emits
    the same object. Both are plain JSON, so `json.load` takes either -- but a
    run interrupted mid-write leaves a truncated file, and that must be an
    error naming the file rather than a traceback.
    """
    t = Trace(path)
    try:
        with open(path, "rb") as fh:
            raw = fh.read()
    except OSError as exc:
        raise ValueError(f"{path}: cannot read: {exc.strerror}") from exc
    if not raw.strip():
        raise ValueError(f"{path}: file is empty (the run wrote no trace)")
    try:
        doc = json.loads(raw.decode("utf-8", errors="replace"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{path}: not valid JSON at byte {exc.pos} ({exc.msg}) -- "
            "a truncated trace means the process was killed before finish()"
        ) from exc
    if not isinstance(doc, dict) or "traceEvents" not in doc:
        raise ValueError(f"{path}: no 'traceEvents' key; not a Chrome trace")

    cats: collections.Counter[str] = collections.Counter()
    for ev in doc["traceEvents"]:
        ph = ev.get("ph")
        if ph == "M":
            # thread_name metadata: the only place the recorder says which
            # thread a tid is. Kept for the host section's column headers.
            if ev.get("name") == "thread_name" and "args" in ev:
                t.meta[int(ev.get("tid", 0))] = str(ev["args"].get("name", ""))
            continue
        if ph == "I":
            t.instants += 1
            continue
        if ph != "X":
            continue
        try:
            ts = float(ev["ts"])
            dur = float(ev.get("dur", 0.0))
        except (KeyError, TypeError, ValueError):
            continue
        args = ev.get("args") or {}
        t.spans.append(
            {
                "name": str(ev.get("name", "?")),
                "cat": str(ev.get("cat", "")),
                "ts": ts,
                "dur": dur,
                "pid": int(ev.get("pid", 1)),
                "tid": int(ev.get("tid", 0)),
                "args": args,
            }
        )
        cats[str(ev.get("cat", ""))] += 1

    if not t.spans:
        t.kind = "empty"
        return t
    device = sum(n for c, n in cats.items() if c in _DEVICE_CATS)
    engine = sum(n for c, n in cats.items() if c in _ENGINE_CATS)
    if device and not engine:
        t.kind = "device"
    elif engine and not device:
        t.kind = "host"
    elif device >= engine:
        t.kind = "device"
    else:
        # Some other producer's trace, or a merge. Treat it as a host timeline:
        # self time and nesting are the more general reading, and the device
        # section needs the per-op args that only VNPU writes.
        t.kind = "host"
    return t


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------
def _pct(part: float, whole: float) -> float:
    return 100.0 * part / whole if whole > 0 else 0.0


def _q(vals: list[float], q: float) -> float:
    """Nearest-rank quantile. No interpolation: a p95 that is not an observed
    duration would be a fabricated number in a table of measurements."""
    if not vals:
        return 0.0
    s = sorted(vals)
    idx = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[idx]


def _human(v: float, unit: str) -> str:
    """Format a value with a scale prefix, keeping 3 significant figures."""
    for div, pre in ((1e9, "G"), (1e6, "M"), (1e3, "k")):
        if abs(v) >= div:
            return f"{v / div:.1f} {pre}{unit}"
    return f"{v:.1f} {unit}"


def _table(head: list[str], rows: list[list[str]], right: set[int]) -> list[str]:
    """Render an aligned text table. `right` names the right-justified columns."""
    widths = [len(h) for h in head]
    for r in rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(cell))
    out = []
    for r in [head] + rows:
        cells = [
            c.rjust(widths[i]) if i in right else c.ljust(widths[i])
            for i, c in enumerate(r)
        ]
        out.append("  ".join(cells).rstrip())
    if rows:
        rule = ["-" * w for w in widths]
        out.insert(1, "  ".join(rule).rstrip())
    return out


# --------------------------------------------------------------------------
# Device timeline: what the NPU was doing, and how fast
# --------------------------------------------------------------------------
def analyse_device(t: Trace, top: int, gap_us: float) -> dict:
    """Service time and rate per op, plus how much of the span was idle.

    The stream is ordered and single-producer by VNPU's contract, so spans do
    not overlap. That is what makes `busy = sum(dur)` meaningful as a duty
    cycle; if a future producer did overlap, the guard below reports it instead
    of quietly double-counting.
    """
    spans = sorted(t.spans, key=lambda e: e["ts"])
    t0 = spans[0]["ts"]
    t1 = max(e["ts"] + e["dur"] for e in spans)
    span = max(t1 - t0, 1e-9)
    busy = sum(e["dur"] for e in spans)

    # Is the START clock usable? A producer whose timestamps are quantised
    # stamps many commands with the same instant. Durations survive that (each
    # is a difference of two readings inside one command) but every aggregate
    # built from start times -- gaps, idle, duty, overlap -- becomes a
    # measurement of the grid. Detect rather than assume: anything at or above
    # half the commands sharing instants is treated as quantised.
    distinct = sorted({e["ts"] for e in spans})
    shared = len(spans) - len(distinct)
    steps = [distinct[i + 1] - distinct[i] for i in range(len(distinct) - 1)]
    quantum = min(steps) if steps else 0.0
    # `slots` is the honest busy fraction when the grid is fine enough to trust:
    # how full each distinct instant is.
    quantised = len(distinct) <= 0.5 * len(spans)
    slot_busy = 0.0
    if quantised:
        fill: dict[float, float] = collections.defaultdict(float)
        for e in spans:
            fill[e["ts"]] += e["dur"]
        slot_busy = max(fill.values())

    overlap_us = 0.0
    gaps: list[float] = []
    over: list[float] = []
    if not quantised:
        # Overlap detection: walk in start order, track the furthest end seen.
        furthest = 0.0
        for e in spans:
            if e["ts"] < furthest:
                overlap_us += min(furthest, e["ts"] + e["dur"]) - e["ts"]
            furthest = max(furthest, e["ts"] + e["dur"])
        cur_end = spans[0]["ts"] + spans[0]["dur"]
        for e in spans[1:]:
            g = e["ts"] - cur_end
            if g > 0:
                gaps.append(g)
            cur_end = max(cur_end, e["ts"] + e["dur"])
        over = [g for g in gaps if g >= gap_us]

    by_op: dict[str, dict] = {}
    for e in spans:
        op = str(e["args"].get("op") or e["name"].split("#")[0])
        d = by_op.setdefault(
            op,
            {"n": 0, "dur": [], "read": 0, "written": 0, "flops": 0,
             "units": collections.Counter()},
        )
        d["n"] += 1
        d["dur"].append(e["dur"])
        # args values arrive as ints from VNPU; tolerate strings from other
        # producers rather than dropping the whole row on a bad cast.
        for key, dst in (("bytes_read", "read"), ("bytes_written", "written"),
                         ("flops", "flops")):
            try:
                d[dst] += int(e["args"].get(key, 0) or 0)
            except (TypeError, ValueError):
                pass
        if e["args"].get("unit"):
            d["units"][str(e["args"]["unit"])] += 1

    rows: list[list[str]] = []
    ordered = sorted(by_op.items(), key=lambda kv: -sum(kv[1]["dur"]))[:top]
    for op, d in ordered:
        tot = sum(d["dur"])
        moved = d["read"] + d["written"]
        secs = tot * 1e-6
        gbs = moved / 1e9 / secs if secs > 0 else 0.0
        gfs = d["flops"] / 1e9 / secs if secs > 0 else 0.0
        rows.append(
            [
                op,
                str(d["n"]),
                f"{tot / 1e3:.1f} ms",
                f"{_pct(tot, busy):5.1f}%",
                f"{statistics.mean(d['dur']):.1f}",
                f"{_q(d['dur'], 0.50):.1f}",
                f"{_q(d['dur'], 0.95):.1f}",
                f"{max(d['dur']):.1f}",
                # MiB, not MB: a column headed MiB that divides by 1e6 is a
                # unit lie, and it read as 1699.7 where the true figure is 1621.
                f"{moved / 1048576.0:.1f} MiB",
                f"{gbs:.1f}",
                f"{gfs:.1f}",
                ",".join(sorted(d["units"])),
            ]
        )

    return {
        "kind": "device",
        "spans": len(spans),
        "span_us": span,
        "busy_us": busy,
        "duty": _pct(busy, span),
        "overlap_us": overlap_us,
        "timing_valid": not quantised,
        "timestamps": len(distinct),
        "ts_quantum_us": quantum,
        "slot_busy_us": slot_busy,
        "idle_us": max(0.0, span - busy) if not quantised else None,
        "gaps_ge": len(over),
        "gaps_ge_us": gap_us,
        "gaps_total_us": sum(gaps) if not quantised else None,
        "gap_max_us": max(gaps) if gaps else None,
        "gap_median_us": statistics.median(gaps) if gaps else None,
        "ops": [
            {
                "op": op,
                "n": d["n"],
                "total_us": sum(d["dur"]),
                "mean_us": statistics.mean(d["dur"]),
                "p95_us": _q(d["dur"], 0.95),
                "max_us": max(d["dur"]),
                "bytes": d["read"] + d["written"],
                "flops": d["flops"],
            }
            for op, d in sorted(by_op.items(), key=lambda kv: -sum(kv[1]["dur"]))
        ],
        "_rows": rows,
    }


def render_device(t: Trace, r: dict, top: int) -> list[str]:
    out = [f"== DEVICE timeline: {t.label}"]
    out.append(f"   {t.path}")
    out.append(
        f"   {r['spans']} spans, {r['span_us'] / 1e3:.1f} ms end to end"
        + (f", {r['instants']} instant events" if t.instants else "")
    )
    out.append("")
    if not r["timing_valid"]:
        out.append(
            f"   device busy {r['busy_us'] / 1e3:.1f} ms of service time across "
            f"{r['spans']} commands"
        )
        out.append(
            f"   TIMING WITHHELD: {r['spans']} commands share only "
            f"{r['timestamps']} start instants (quantum {r['ts_quantum_us'] / 1e3:g} ms),"
            " so gaps/idle/duty"
        )
        out.append(
            "   would measure the timestamp grid rather than the device. The"
            " service-time columns below are dur-based and unaffected."
        )
        out.append(
            f"   Sanity check on the grid: the fullest single instant holds "
            f"{r['slot_busy_us'] / 1e3:.2f} ms of command time."
        )
    else:
        out.append(
            f"   device busy {r['busy_us'] / 1e3:.1f} ms = {r['duty']:.1f}% of the span"
            f"   |   idle {r['idle_us'] / 1e3:.1f} ms"
        )
        out.append(
            f"   {r['gaps_ge']} gaps >= {r['gaps_ge_us']:.0f} us totalling "
            f"{r['gaps_total_us'] / 1e3:.1f} ms; longest {r['gap_max_us'] / 1e3:.1f} ms, "
            f"median {r['gap_median_us']:.0f} us"
        )
        if r["overlap_us"] > 0.5:
            out.append(
                f"   NOTE: spans overlap by {r['overlap_us']:.0f} us, so 'busy' is a "
                "sum over concurrent work and the duty cycle overstates serialisation"
            )
    out.append("")
    head = ["op", "count", "total", "%busy", "mean us", "p50 us", "p95 us",
            "max us", "moved", "GB/s", "GFLOP/s", "unit"]
    out += ["   " + ln for ln in _table(head, r["_rows"], right=set(range(1, 11)))]
    out.append("")
    out.append(
        "   GB/s and GFLOP/s are achieved rates over the op's own service time,"
        " not the span."
    )
    out.append(
        "   'unit' is what the command asked for; the device may still fan a"
        " single unit's work across its worker threads."
    )
    return out


# --------------------------------------------------------------------------
# Host timeline: where the wall time went, net of nesting
# --------------------------------------------------------------------------
def analyse_host(t: Trace, top: int) -> dict:
    """Self time per span, computed with a nesting stack PER THREAD.

    Nesting must be per (pid, tid): two worker threads each run their own stack
    of spans at overlapping wall times, and treating one thread's span as the
    other's parent would subtract time that was never contained. Spans are
    visited in start order with longer spans first at an equal start, so an
    enclosing span is always on the stack before its children.
    """
    spans = sorted(t.spans, key=lambda e: (e["ts"], -e["dur"]))
    t0 = spans[0]["ts"]
    t1 = max(e["ts"] + e["dur"] for e in spans)
    wall = max(t1 - t0, 1e-9)

    self_us: dict[tuple[str, str], float] = collections.defaultdict(float)
    total_us: dict[tuple[str, str], float] = collections.defaultdict(float)
    counts: collections.Counter[tuple[str, str]] = collections.Counter()
    children: dict[int, float] = collections.defaultdict(float)

    stacks: dict[tuple[int, int], list[int]] = collections.defaultdict(list)
    for i, e in enumerate(spans):
        key = (e["pid"], e["tid"])
        st = stacks[key]
        while st and spans[st[-1]]["ts"] + spans[st[-1]]["dur"] <= e["ts"]:
            st.pop()
        if st:
            children[st[-1]] += e["dur"]
        st.append(i)
        k = (e["cat"], e["name"])
        total_us[k] += e["dur"]
        counts[k] += 1

    for i, e in enumerate(spans):
        self_us[(e["cat"], e["name"])] += e["dur"] - children[i]

    ordered = sorted(self_us.items(), key=lambda kv: -kv[1])
    rows = []
    for (cat, name), s in ordered[:top]:
        rows.append([
            cat or "-",
            name,
            str(counts[(cat, name)]),
            f"{s / 1e3:.1f} ms",
            f"{_pct(s, wall):5.1f}%",
            f"{s / max(counts[(cat, name)], 1):.1f}",
            f"{total_us[(cat, name)] / 1e3:.1f} ms",
        ])

    phases: dict[str, float] = collections.defaultdict(float)
    for (cat, name), s in self_us.items():
        phases[cat] += s

    return {
        "kind": "host",
        "spans": len(spans),
        "wall_us": wall,
        "instants": t.instants,
        "threads": len({(e["pid"], e["tid"]) for e in spans}),
        "by_cat": dict(phases),
        "by_span": [
            {"cat": c, "name": n, "n": counts[(c, n)], "self_us": s,
             "total_us": total_us[(c, n)]}
            for (c, n), s in ordered
        ],
        "_rows": rows,
    }


def render_host(t: Trace, r: dict) -> list[str]:
    out = [f"== HOST timeline: {t.label}"]
    out.append(f"   {t.path}")
    out.append(
        f"   {r['spans']} spans on {r['threads']} thread(s), "
        f"{r['wall_us'] / 1e3:.1f} ms wall"
        + (f", {r['instants']} instant events" if r["instants"] else "")
    )
    out.append("")
    out.append("   self time -- a span's own total minus its children's:")
    head = ["cat", "span", "count", "self", "% wall", "per call us", "total"]
    out += ["   " + ln for ln in _table(head, r["_rows"], right={2, 3, 4, 5, 6})]
    out.append("")
    cat_rows = [
        ["cat " + c, f"{v / 1e3:.1f} ms", f"{_pct(v, r['wall_us']):5.1f}%"]
        for c, v in sorted(r["by_cat"].items(), key=lambda kv: -kv[1])
    ]
    out += ["   " + ln for ln in _table(["group", "self", "% wall"], cat_rows, {1, 2})]
    return out


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description="Summarise a Chrome trace from kraken (--trace) or VNPU "
                    "(KRK_NPU_TRACE). Auto-detects the producer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage\n-----")[1] if "Usage\n-----" in __doc__ else "",
    )
    ap.add_argument("traces", nargs="+", metavar="TRACE.json")
    ap.add_argument("--top", type=int, default=14,
                    help="rows per table (default 14)")
    ap.add_argument("--gap-us", type=float, default=50.0,
                    help="device idle gap counted in the headline (default 50 us)")
    ap.add_argument("--json", metavar="PATH", help="write the analysis as JSON")
    ap.add_argument("--label", action="append", default=[],
                    help="label for the Nth trace, in order (repeatable)")
    ap.add_argument("--quiet", action="store_true",
                    help="parse and check only; print nothing, still set the exit code")
    a = ap.parse_args(argv)

    traces: list[Trace] = []
    failed = 0
    for i, path in enumerate(a.traces):
        try:
            t = load(path)
        except ValueError as exc:
            print(f"trace_stats: {exc}", file=sys.stderr)
            failed += 1
            continue
        if i < len(a.label):
            t.set_label(a.label[i])
        if t.kind == "empty":
            print(f"trace_stats: {path}: parsed, but contains no spans -- the "
                  "run did not emit any (tracing off, or nothing ran)",
                  file=sys.stderr)
            failed += 1
            continue
        traces.append(t)

    results = []
    for t in traces:
        if t.kind == "device":
            r = analyse_device(t, a.top, a.gap_us)
            if not a.quiet:
                print("\n".join(render_device(t, r, a.top)))
        else:
            r = analyse_host(t, a.top)
            if not a.quiet:
                print("\n".join(render_host(t, r)))
        r["path"] = t.path
        r["label"] = t.label
        results.append(r)
        if not a.quiet:
            print()

    # A cross-file line, which is the point of passing a --npu / --npu-cpu pair.
    if len(results) > 1 and not a.quiet:
        print("== SUMMARY")
        head = ["trace", "kind", "spans", "span/wall", "busy", "duty", "idle"]
        rows = []
        for r in results:
            if r["kind"] == "device":
                rows.append([r["label"], r["kind"], str(r["spans"]),
                             f"{r['span_us'] / 1e3:.1f} ms",
                             f"{r['busy_us'] / 1e3:.1f} ms",
                             f"{r['duty']:.1f}%" if r["timing_valid"] else "withheld",
                             f"{r['idle_us'] / 1e3:.1f} ms" if r["timing_valid"] else "-"])
            else:
                rows.append([r["label"], r["kind"], str(r["spans"]),
                             f"{r['wall_us'] / 1e3:.1f} ms", "-", "-", "-"])
        print("\n".join("   " + ln for ln in _table(
            head, rows, right={2, 3, 4, 5, 6})))

    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump({"traces": [{k: v for k, v in r.items()
                                   if not k.startswith("_")} for r in results]},
                      fh, indent=1)
        if not a.quiet:
            print(f"\njson: {a.json}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
