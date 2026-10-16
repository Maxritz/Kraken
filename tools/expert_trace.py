#!/usr/bin/env python3
"""Analyze KRK_TRACE_EXPERTS TSV without changing model routing.

Inter-access distance counts ALL intervening accesses. LRU stack distance
counts DISTINCT other (layer, expert) keys; only the latter predicts an
empty, equal-sized LRU cache. Neither predicts Kraken's LFU+aging policy.
HOT and WARM replay below are independent inclusive equal-slot LRU tiers,
not a simulation of byte budgets, model warmup, or the live replacement policy.

Usage: python3 tools/expert_trace.py trace.tsv --hot 64 --warm 256
       python3 tools/expert_trace.py --selftest
"""
import argparse
import collections
import sys


def load_trace(path):
    records = []
    with open(path, encoding="utf-8") as stream:
        if stream.readline().strip().split("\t") != ["seq", "layer", "expert", "tier"]:
            raise ValueError("expected TSV header: seq, layer, expert, tier")
        for line_number, line in enumerate(stream, 2):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 4:
                raise ValueError("line %d: expected four integers" % line_number)
            seq, layer, expert, tier = map(int, fields)
            if seq != len(records) or layer < 0 or expert < 0 or tier not in (0, 1, 2):
                raise ValueError("line %d: invalid sequence, key or tier" % line_number)
            records.append((layer, expert, tier))
    return records


def distances(keys):
    """O(n log n) exact distinct-key stack distances via last-use Fenwick tree."""
    tree = [0] * (len(keys) + 1)

    def add(index, delta):
        index += 1
        while index < len(tree):
            tree[index] += delta
            index += index & -index

    def prefix(index):
        total = 0
        while index:
            total += tree[index]
            index -= index & -index
        return total

    last, access, stack = {}, [], []
    for index, key in enumerate(keys):
        previous = last.get(key)
        if previous is not None:
            access.append(index - previous)
            stack.append(len(last) - prefix(previous + 1))
            add(previous, -1)
        add(index, 1)
        last[key] = index
    return access, stack


def replay_lru(keys, hot, warm):
    """Independent inclusive tiers; every demand access touches both."""
    h, w = collections.OrderedDict(), collections.OrderedDict()
    counts = [0, 0, 0]
    for key in keys:
        tier = 0 if key in h else 1 if key in w else 2
        counts[tier] += 1
        for cache, capacity in ((h, hot), (w, warm)):
            if not capacity:
                continue
            cache[key] = None
            cache.move_to_end(key)
            if len(cache) > capacity:
                cache.popitem(last=False)
    return counts


def selftest():
    cases = [([], 1, 2), (list("AAAA"), 1, 1), (list("ABA"), 1, 2),
             (list("ABCABC"), 2, 3), (list("ABBBA"), 2, 0),
             (list("ABCABC"), 3, 1), (list("ABAB"), 0, 0)]
    for keys, hot, warm in cases:
        access, stack = distances(keys)
        naive = []
        last = {}
        for index, key in enumerate(keys):
            if key in last:
                naive.append(len(set(keys[last[key] + 1:index])))
            last[key] = index
        assert stack == naive, (keys, stack, naive)
        expected = [0, 0, len(set(keys))]
        for distance in stack:
            expected[0 if distance < hot else 1 if distance < warm else 2] += 1
        assert replay_lru(keys, hot, warm) == expected
    assert distances(list("ABBBA")) == ([1, 1, 4], [0, 0, 1])
    assert replay_lru(list("ABBBA"), 2, 0) == [3, 0, 2]
    assert distances([(0, 1), (1, 1), (0, 1)])[1] == [1]
    print("expert trace selftest: 10 cases passed (distance and replay)")


def distribution(label, values):
    if not values:
        return
    ordered = sorted(values)
    print("\n%s" % label)
    for p in (10, 25, 50, 75, 90, 99):
        print("  p%-2d %8d" % (p, ordered[min(len(ordered) - 1, p * len(ordered) // 100)]))
    print("  max %8d; mean %.1f" % (ordered[-1], sum(ordered) / len(ordered)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", nargs="?")
    parser.add_argument("--hot", type=int, default=0, help="equal-sized HOT LRU slots")
    parser.add_argument("--warm", type=int, default=0, help="independent inclusive WARM LRU slots")
    parser.add_argument("--top", type=int, default=6, help="top active experts shown per layer")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return 0
    if not args.trace:
        parser.error("a trace is required unless --selftest is used")
    if min(args.hot, args.warm, args.top) < 0:
        parser.error("capacities and --top must be nonnegative")
    try:
        records = load_trace(args.trace)
    except (OSError, ValueError) as error:
        print("invalid trace: %s" % error, file=sys.stderr)
        return 1
    if not records:
        print("empty trace", file=sys.stderr)
        return 1
    keys = [(layer, expert) for layer, expert, _ in records]
    tiers = collections.Counter(tier for _, _, tier in records)
    print("acquires %d; distinct (layer,expert) %d" % (len(keys), len(set(keys))))
    for tier, name in enumerate(("HOT", "WARM", "COLD")):
        print("  observed %-4s %8d (%5.1f%%)" % (name, tiers[tier], 100 * tiers[tier] / len(keys)))
    access, stack = distances(keys)
    distribution("inter-access distance (all accesses since previous use)", access)
    distribution("LRU stack distance (distinct other keys since previous use)", stack)
    if args.hot or args.warm:
        predicted = replay_lru(keys, args.hot, args.warm)
        print("\nEmpty equal-slot LRU replay: HOT=%d, inclusive WARM=%d" % (args.hot, args.warm))
        print("Not the live LFU/byte-budget policy; includes compulsory first touches.")
        for name, count in zip(("HOT", "WARM", "COLD"), predicted):
            print("  replay %-4s %8d (%5.1f%%)" % (name, count, 100 * count / len(keys)))
    layers = collections.defaultdict(collections.Counter)
    misses = collections.defaultdict(collections.Counter)
    for layer, expert, tier in records:
        layers[layer][expert] += 1
        misses[layer][tier] += 1
    print("\nPer-layer active set (ranking is descriptive, not a pin recommendation)")
    print("layer accesses unique HOT WARM COLD top-share top-experts(id:accesses)")
    for layer in sorted(layers):
        counter = layers[layer]
        top = sorted(counter.items(), key=lambda item: (-item[1], item[0]))[:args.top]
        total = sum(counter.values())
        print("%5d %8d %6d %6d %6d %6d %8.1f%% %s" % (
            layer, total, len(counter), misses[layer][0], misses[layer][1], misses[layer][2],
            100 * sum(count for _, count in top) / total,
            " ".join("%d:%d" % pair for pair in top)))
    print("\nTrace lacks request/engine IDs and tensor bytes: do not combine targets/drafters")
    print("or infer byte budgets, per-request pins, prefill/decode cost, or disk bandwidth.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
