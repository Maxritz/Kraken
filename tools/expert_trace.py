#!/usr/bin/env python3
"""expert_trace.py -- what the expert cache counters cannot tell you.

Reads the KRK_TRACE_EXPERTS dump (seq, layer, expert, tier) and reports the
reuse-distance distribution: for every repeat use of an expert, how many
acquires ago it was last used.

That distance is the number the tier sizes hinge on.

  distance <= HOT slots   -> the request is served by HOT, WARM never sees it
  HOT slots < d <= HOT+WARM-> the request is served by WARM, NVMe is skipped
  distance > both         -> the request is a COLD miss whatever we do

So a workload whose distances all sit below the HOT capacity will report
WARM hits near zero no matter how the WARM tier is tuned, and enlarging WARM
cannot change that. The fix there is a bigger HOT tier, not a better WARM
policy.

Usage:  python3 tools/expert_trace.py expert_trace.tsv [--hot N] [--warm N]
"""
import argparse
import collections
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--hot", type=int, default=0,
                    help="HOT slot count, for the served/missed split")
    ap.add_argument("--warm", type=int, default=0,
                    help="WARM expert count, for the served/missed split")
    a = ap.parse_args()

    layer, expert, tier = [], [], []
    with open(a.trace) as f:
        next(f, None)  # header
        for line in f:
            p = line.split()
            if len(p) < 4:
                continue
            layer.append(int(p[1]))
            expert.append(int(p[2]))
            tier.append(int(p[3]))

    n = len(layer)
    if not n:
        print("empty trace")
        return 1

    print("acquires      %d" % n)
    tc = collections.Counter(tier)
    print("  HOT  %6d (%.1f%%)" % (tc[0], 100.0 * tc[0] / n))
    print("  WARM %6d (%.1f%%)" % (tc[1], 100.0 * tc[1] / n))
    print("  COLD %6d (%.1f%%)" % (tc[2], 100.0 * tc[2] / n))

    # Reuse distance over the (layer, expert) key space the cache is keyed by.
    last = {}
    dist = []
    for i in range(n):
        k = (layer[i], expert[i])
        p = last.get(k)
        if p is not None:
            dist.append(i - p)
        last[k] = i

    print("\ndistinct (layer,expert) touched : %d" % len(last))
    print("repeat uses                    : %d" % len(dist))
    if not dist:
        print("no reuse in this run -- every acquire is a first touch")
        return 0

    dist.sort()
    def pct(p):
        return dist[min(len(dist) - 1, int(p / 100.0 * len(dist)))]

    print("\nreuse distance (acquires since last use of the same expert)")
    print("  p10 %8d" % pct(10))
    print("  p25 %8d" % pct(25))
    print("  p50 %8d" % pct(50))
    print("  p75 %8d" % pct(75))
    print("  p90 %8d" % pct(90))
    print("  p99 %8d" % pct(99))
    print("  max %8d" % dist[-1])
    print("  mean %8.1f" % (sum(dist) / float(len(dist))))

    if a.hot:
        print("\nserved-by split at HOT=%d, WARM=%d slots" % (a.hot, a.warm))
        h = sum(1 for d in dist if d <= a.hot)
        w = sum(1 for d in dist if a.hot < d <= a.hot + a.warm)
        c = len(dist) - h - w
        tot = float(len(dist))
        print("  served by HOT  %6d (%5.1f%%)" % (h, 100.0 * h / tot))
        print("  served by WARM %6d (%5.1f%%)" % (w, 100.0 * w / tot))
        print("  COLD miss      %6d (%5.1f%%)" % (c, 100.0 * c / tot))
        cap = a.hot + a.warm
        within = sum(1 for d in dist if d <= cap)
        print("  working set at that capacity: %.1f%% of repeats fit"
              % (100.0 * within / tot))

    # Per-layer cost, so a single expensive layer is visible.
    per = collections.Counter(layer)
    print("\nacquires per layer (top 10)")
    for l, c in per.most_common(10):
        print("  layer %-4d %6d" % (l, c))
    cold = collections.Counter()
    for i in range(n):
        if tier[i] == 2:
            cold[layer[i]] += 1
    if cold:
        print("\nCOLD misses per layer (top 10)")
        for l, c in cold.most_common(10):
            print("  layer %-4d %6d" % (l, c))
    return 0


if __name__ == "__main__":
    sys.exit(main())