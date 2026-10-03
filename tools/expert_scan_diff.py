#!/usr/bin/env python3
"""Compare two kraken expert indexes.

The load-bearing question behind a stubbed router calibration is whether
removing the expert FFNs changes what the routers choose. This answers it with
two numbers instead of an opinion:

  * total-variation distance between the two routing-mass distributions, per
    layer -- how much probability mass moved between experts;
  * top-k agreement -- how often the experts that actually get executed are
    the same set.

It also prints the coverage curve, which is what a loader needs in order to
solve for a hot set under whatever budget the card has.

    python tools/expert_scan_diff.py stub.json full.json [--top-k 2]
"""
import json
import sys


def load(path):
    d = json.load(open(path))
    layers = {}
    for L in d["layers"]:
        layers[L["layer"]] = {e["e"]: e for e in L["experts"]}
    return d, layers


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    top_k = 2
    for a in sys.argv[1:]:
        if a.startswith("--top-k"):
            top_k = int(a.split("=", 1)[1])
    if len(args) != 2:
        print(__doc__)
        return 2

    a, la = load(args[0])
    b, lb = load(args[1])
    print("A  %s   %s   (%d positions, %d experts, %d layers)"
          % (args[0], a["mode"], a["positions_scanned"], a["n_expert"],
             a["n_layer"]))
    print("B  %s   %s   (%d positions, %d experts, %d layers)"
          % (args[1], b["mode"], b["positions_scanned"], b["n_expert"],
             b["n_layer"]))
    for key in ("source_size", "n_expert_used", "expert_bytes"):
        va, vb = a.get(key), b.get(key)
        mark = "" if va == vb else "   <-- MISMATCH"
        print("   %-14s %s vs %s%s" % (key, va, vb, mark))
    print()

    common = sorted(set(la) & set(lb))
    if not common:
        print("no layers in common")
        return 1

    print("per-layer routing-mass distance (total variation, 0 = identical)")
    dvs = []
    for l in common:
        ma = {e: v["mass"] for e, v in la[l].items()}
        mb = {e: v["mass"] for e, v in lb[l].items()}
        num = sum(abs(ma.get(e, 0.0) - mb.get(e, 0.0)) for e in set(ma) | set(mb))
        den = sum(max(ma.get(e, 0.0), mb.get(e, 0.0)) for e in set(ma) | set(mb))
        d = num / den if den else 0.0
        dvs.append(d)
        print("  layer %3d   %.4f" % (l, d))
    print("  mean %.4f   max %.4f" % (sum(dvs) / len(dvs), max(dvs)))
    print()

    agree = tot = 0
    for l in common:
        sa = [e["e"] for e in sorted(la[l].values(), key=lambda x: -x["mass"])[:top_k]]
        sb = [e["e"] for e in sorted(lb[l].values(), key=lambda x: -x["mass"])[:top_k]]
        agree += len(set(sa) & set(sb))
        tot += top_k
    print("top-%d agreement: %d/%d = %.1f%%" % (top_k, agree, tot,
                                                100.0 * agree / tot if tot else 0))
    print()

    print("coverage curve, ranked by mass (what a budget would buy)")
    print("  %-8s %-10s %-10s" % ("top n", "mass A", "mass B"))
    for n in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        if n > a["n_expert"]:
            break
        ca = sum(e["mass"] for L in a["layers"]
                 for e in sorted(L["experts"], key=lambda x: -x["mass"])[:n])
        cb = sum(e["mass"] for L in b["layers"]
                 for e in sorted(L["experts"], key=lambda x: -x["mass"])[:n])
        span = len(a["layers"]) * a["n_expert"]
        print("  %-8d %-10.4f %-10.4f  (per layer: %.4f / %.4f)"
              % (n, ca, cb, ca / span if span else 0, cb / span if span else 0))
    return 0


if __name__ == "__main__":
    sys.exit(main())