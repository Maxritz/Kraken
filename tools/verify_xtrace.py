#!/usr/bin/env python3
"""verify_xtrace.py -- is the recorded tier sequence self-consistent?

The trace claims 6003 HOT + 9 WARM + 4590 COLD over 10602 acquires with 4590
distinct (layer,expert) keys. That is a strong claim, so it gets checked rather
than believed: this replays the access sequence through a plain LRU of --hot
slots backed by a --warm-slot LRU, and compares the simulated tier decisions
against the recorded ones.

If the simulation reproduces the counts, the counters describe a cache that
really did that. If it does not, the trace or the cache is lying.

Usage: python3 tools/verify_xtrace.py expert_trace.tsv [--hot 4340] [--warm 4590]
"""
import argparse
import collections
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--hot", type=int, default=4340)
    ap.add_argument("--warm", type=int, default=8192)
    a = ap.parse_args()

    rec = []
    with open(a.trace) as f:
        next(f, None)
        for line in f:
            p = line.split()
            if len(p) >= 4:
                rec.append((int(p[1]), int(p[2]), int(p[3])))

    n = len(rec)
    keys = {(l, e) for l, e, _ in rec}
    obs = collections.Counter(t for _, _, t in rec)

    print("recorded acquires        %d" % n)
    print("recorded distinct keys   %d" % len(keys))
    print("recorded HOT/WARM/COLD   %d / %d / %d" % (obs[0], obs[1], obs[2]))
    first_touch = sum(1 for _, _, t in rec if t == 2)
    print("recorded COLD == first touches? %s"
          % (first_touch == obs[2],))

    # Replay: HOT is an LRU of --hot keys, WARM an LRU of --warm keys.
    hot, warm = [], collections.OrderedDict()
    sim = collections.Counter()
    for l, e, _ in rec:
        k = (l, e)
        if k in hot:
            hot.remove(k)
            hot.append(k)
            sim[0] += 1
            continue
        if k in warm:
            warm.move_to_end(k)
            sim[1] += 1
            if len(hot) < a.hot:
                hot.append(warm.pop(k))
            continue
        sim[2] += 1
        warm[k] = None
        if len(warm) > a.warm:
            warm.popitem(last=False)
        if len(hot) < a.hot:
            hot.append(warm.pop(k))

    print("\nsimulated HOT/WARM/COLD   %d / %d / %d"
          % (sim[0], sim[1], sim[2]))
    print("match recorded?           HOT %s  WARM %s  COLD %s"
          % (sim[0] == obs[0], sim[1] == obs[1], sim[2] == obs[2]))

    # Where the two disagree, say so per key rather than only in aggregate.
    dis = collections.Counter()
    hot, warm = [], collections.OrderedDict()
    for l, e, t in rec:
        k = (l, e)
        if k in hot:
            got = 0
            hot.remove(k); hot.append(k)
        elif k in warm:
            got = 1
            warm.move_to_end(k)
            if len(hot) < a.hot:
                hot.append(warm.pop(k))
        else:
            got = 2
            warm[k] = None
            if len(warm) > a.warm:
                warm.popitem(last=False)
            if len(hot) < a.hot:
                hot.append(warm.pop(k))
        if got != t:
            dis[(t, got)] += 1
    if dis:
        print("\ndisagreements (recorded -> simulated):")
        for (rt, st), c in dis.most_common():
            print("  recorded tier %d, simulated tier %d : %d"
                  % (rt, st, c))
    else:
        print("\nevery single acquire agrees with the replay")
    return 0


if __name__ == "__main__":
    sys.exit(main())