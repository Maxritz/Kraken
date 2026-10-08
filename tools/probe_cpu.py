#!/usr/bin/env python3
"""probe_cpu.py -- sample OUR OWN process CPU time, thread count and RSS while a
kraken run does its work, so "it consumes lots of CPU" becomes a number with a
phase attached to it.

The engine reports device time (`--profile`) and host *wall* time per phase
(`[stats ] host time`), and on Windows it reports GPU-engine utilization
(`engine_usage.hpp`). What none of those answer is how much CPU the process
itself burns and in which phase, which is the question a Task Manager column
raises. Run the target with KRK_PHASE=1 so its stderr carries the phase
timeline, and this prints the sampled CPU% and RSS next to it.

Usage:
    python3 tools/probe_cpu.py -- ./build-hip/kraken.exe -m model.gguf --greedy -n 64
"""
import argparse
import subprocess
import sys
import time

import psutil


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=0.05)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    cmd = args.cmd[1:] if args.cmd and args.cmd[0] == "--" else args.cmd
    if not cmd:
        ap.error("no command given (use: -- ./kraken ...)")

    ncpu = psutil.cpu_count(logical=True) or 1
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd)
    p = psutil.Process(proc.pid)
    # Prime cpu_times(): the first call establishes the baseline.
    try:
        p.cpu_times()
    except psutil.Error:
        pass

    peak_cpu = 0.0
    sum_cpu = 0.0
    samples = 0
    peak_rss = 0
    peak_threads = 0
    rows = []
    last_cpu = 0.0
    last_wall = t0

    while proc.poll() is None:
        time.sleep(args.interval)
        now = time.perf_counter()
        try:
            ct = p.cpu_times()
            rss = p.memory_info().rss
            threads = p.num_threads()
        except psutil.Error:
            break
        total = ct.user + ct.system
        dcpu = total - last_cpu
        dwall = now - last_wall
        last_cpu, last_wall = total, now
        cpu_pct = 100.0 * dcpu / dwall if dwall > 0 else 0.0
        peak_cpu = max(peak_cpu, cpu_pct)
        peak_threads = max(peak_threads, threads)
        peak_rss = max(peak_rss, rss)
        # One full-CPU-equivalent = 100%; of the whole machine = 100/ncpu.
        sum_cpu += dcpu
        samples += 1
        rows.append((now - t0, cpu_pct, rss, threads))

    rc = proc.wait()
    wall = time.perf_counter() - t0
    cputime = last_cpu
    print(f"probe_cpu: wall {wall:.3f} s | cpu time {cputime:.3f} s "
          f"(user+sys) | mean %.1f%% of one core (%.2f%% of %d cpus)"
          % (100.0 * cputime / wall if wall else 0.0,
             100.0 * cputime / wall / ncpu if wall else 0.0, ncpu))
    print(f"probe_cpu: peak sampled CPU {peak_cpu:.1f}% of one core | "
          f"peak threads {peak_threads} | peak RSS {peak_rss / 1048576.0:.1f} MiB "
          f"| samples {samples}")
    if rows:
        # The busiest window, so the phase can be matched against it.
        top = sorted(rows, key=lambda r: -r[1])[:5]
        print("probe_cpu: busiest windows (t s, cpu% of one core, RSS MiB, threads):")
        for t, c, r, th in top:
            print(f"    t={t:8.3f}  cpu={c:7.1f}%  rss={r / 1048576.0:8.1f}  thr={th}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
