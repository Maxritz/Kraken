#!/usr/bin/env python3
"""Which groupable projection sets does a mixed quantisation still split?

The engine fuses a set of projections that share one activation into a single
launch (`Backend::gemm_group`). Several sets qualify, and three sites are still
gated on every matrix having the SAME weight type -- the group used to carry one
`wt`, so a checkpoint that quantises `attn_v` differently from `attn_q`/`attn_k`
paid separate launches even though the data flow allows one:

    set                site                           guard
    attention q/k/v    src/engine.cpp:918             wq.type == wk.type == wv.type
    FFN gate/up        src/engine.cpp:1154            wgate.type == wup.type
    drafter q/k/v      src/dflash.cpp:799             same (only with --draft)
    drafter gate/up    src/dflash.cpp:835             same (only with --draft)
    GDN 4 projections  src/engine.cpp (gdn_forward)   none -- takes wts[]

`FusedLayer` now carries a type per matrix and `gemm_group` takes a nullable
`wts[]`, so those guards are no longer required by the kernel. What this tool
answers is WHICH files trip them, on how many layers, and what that costs.

The guard is evaluated PER LAYER, which is the part a file-level type histogram
hides: llama.cpp's Q4_K_M recipe bumps `attn_v` to a wider type on a subset of
layers, so those checkpoints fuse on most layers and split on the protected ones.

Usage:  python3 tools/projection_type_audit.py [model.gguf ...]
        (default: every *.gguf in models/)

Reads the tensor table from kraken-inspect -- the loader's own view of the file,
not a re-implemented parser -- and never loads weights.
"""

import glob
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INSPECT = os.path.join(ROOT, "build-hip", "kraken-inspect.exe")

# (label, suffixes, split launches, fused launches, needs equal types, site)
SETS = [
    ("attention q/k/v", ["attn_q.weight", "attn_k.weight", "attn_v.weight"],
     3, 1, True, "engine.cpp:918"),
    ("FFN gate/up", ["ffn_gate.weight", "ffn_up.weight"],
     3, 1, True, "engine.cpp:1154 (+silu_mul)"),
    ("drafter q/k/v", ["attn_q.weight", "attn_k.weight", "attn_v.weight"],
     3, 1, True, "dflash.cpp:799 [--draft]"),
    ("drafter gate/up", ["ffn_gate.weight", "ffn_up.weight"],
     3, 1, True, "dflash.cpp:835 [--draft]"),
    ("GDN qkv/z/a/b", ["attn_qkv.weight", "attn_gate.weight", "ssm_alpha.weight",
                       "ssm_beta.weight"],
     4, 1, False, "engine.cpp gdn_forward (wts[])"),
]

LINE = re.compile(r"^\s+(\S+)\s+(\S+)\s+([\dx]+)\s")


def tensor_types(path):
    """(layer, suffix) -> dtype, plus suffix -> histogram, from the file."""
    out = subprocess.run([INSPECT, path], capture_output=True, text=True,
                         timeout=600)
    if out.returncode != 0:
        # A refusal is a fact about the file (or the path), not a verdict about
        # its types: report it and keep the sweep going.
        print("=" * 78)
        print("%s  SKIP (kraken-inspect rc=%d: %s)"
              % (os.path.basename(path), out.returncode,
                 out.stderr.strip().splitlines()[-1] if out.stderr.strip()
                 else "no message"))
        return None, None
    per_layer, hist = {}, {}
    for line in out.stdout.splitlines():
        m = LINE.match(line)
        if not m:
            continue
        name, dtype = m.group(1), m.group(2)
        if not name.startswith("blk."):
            continue
        parts = name.split(".", 2)
        if len(parts) < 3:
            continue
        suffix = parts[2]
        hist.setdefault(suffix, {})
        hist[suffix][dtype] = hist[suffix].get(dtype, 0) + 1
        try:
            per_layer[(int(parts[1]), suffix)] = dtype
        except ValueError:
            pass
    return per_layer, hist


def describe(hist):
    return "+".join("%s x%d" % (d, n) for d, n in sorted(hist.items()))


def report(path):
    per_layer, hist = tensor_types(path)
    if hist is None:
        return 0, 0, 0, 0, 0
    print("=" * 78)
    print(os.path.basename(path))
    print("=" * 78)
    if not hist:
        print("  no blk.* tensors (not a transformer checkpoint?)")
        return 0, 0, 1, 0, 0
    walls, extra, dwalls, dextra = 0, 0, 0, 0
    for label, suffixes, n_split, n_fused, needs_equal, site in SETS:
        present = [s for s in suffixes if s in hist]
        if len(present) < 2:
            continue
        # Layers that carry the whole set -- the layers the engine can group.
        layers = sorted({l for (l, s) in per_layer if s in present})
        layers = [l for l in layers if all((l, s) in per_layer for s in present)]
        if not layers:
            continue
        if not needs_equal:
            print("  %-14s %-7s %s" % (label, "fused", "  ".join(
                "%s: %s" % (s.replace(".weight", ""), describe(hist[s]))
                for s in present)))
            print("  %-14s   %d launch per layer, mixed types expressed by "
                  "wts[]  (%s)" % ("", n_fused, site))
            continue
        split = [l for l in layers
                 if len({per_layer[(l, s)] for s in present}) > 1]
        if not split:
            print("  %-14s %-7s %s" % (label, "fused", "  ".join(
                "%s: %s" % (s.replace(".weight", ""), describe(hist[s]))
                for s in present)))
            continue
        gained = len(split) * (n_split - n_fused)
        # A drafter row is what this file WOULD cost as the --draft model; a
        # different file is loaded there in practice, so it is not a live cost.
        conditional = "[--draft]" in site
        if conditional:
            dwalls, dextra = dwalls + 1, dextra + gained
        else:
            walls, extra = walls + 1, extra + gained
        print("  %-14s %-7s %d of %d layers, +%d launch(es)/step%s"
              % (label, "SPLIT", len(split), len(layers), gained,
                 "  (only if this file is the --draft model)" if conditional
                 else ""))
        print("  %-14s   %s" % ("", "  ".join(
            "%s: %s" % (s.replace(".weight", ""), describe(hist[s]))
            for s in present)))
        for l in split[:3]:
            print("  %-14s   blk.%-3d %s" % ("", l, " ".join(
                "%s=%s" % (s.replace(".weight", ""), per_layer[(l, s)])
                for s in present)))
        if len(split) > 3:
            print("  %-14s   ... %d more layer(s) split the same way"
                  % ("", len(split) - 3))
        print("  %-14s   %d launch(es) instead of %d per split layer  (%s)"
              % ("", n_split, n_fused, site))
    # Sets that share an activation and are NOT grouped at all (no type wall --
    # a separate opportunity, reported so the picture is not half a picture).
    for pair, note in ((("ffn_up_exps.weight", "ffn_gate_exps.weight"),
                        "MoE per-expert up/gate, m rows routed to one expert"),
                       (("ffn_up_shexp.weight", "ffn_gate_shexp.weight"),
                        "MoE shared expert gated pair")):
        if all(k in hist for k in pair):
            kinds = "  ".join("%s: %s" % (k.replace(".weight", ""), describe(hist[k]))
                              for k in pair)
            print("  %-14s %-7s %s" % ("ungrouped", "2 gemms", kinds))
            print("  %-14s   %s" % ("", note))
    moe = {k: v for k, v in hist.items() if k.endswith("_exps.weight")}
    if moe:
        print("  %-14s         %s" % ("expert types", "  ".join(
            "%s: %s" % (k.replace("_exps.weight", ""), describe(v))
            for k, v in sorted(moe.items()))))
    return walls, extra, 1, dwalls, dextra


def main():
    paths = sys.argv[1:]
    if not paths:
        paths = sorted(glob.glob(os.path.join(ROOT, "models", "*.gguf")))
    if not os.path.exists(INSPECT):
        raise SystemExit("no kraken-inspect at %s -- it is in the ninja line on "
                         "purpose; build it before trusting a verdict from "
                         "these tables" % INSPECT)
    total, nwalls, nextra, skipped, ndraft, draftx = 0, 0, 0, 0, 0, 0
    for p in paths:
        w, e, ok, dw, de = report(p)
        nwalls += w
        nextra += e
        ndraft += dw
        draftx += de
        total += ok
        skipped += 1 - ok
    print()
    print("== engine: %d type-equality wall(s) over %d model(s), %d skipped "
          "==" % (nwalls, total, skipped))
    if nwalls:
        print("   %d wall(s) cost +%d launch(es) per decode step"
              % (nwalls, nextra))
    if ndraft:
        print("   the same files would cost +%d launch(es)/step more in the "
              "--draft arms (dflash.cpp),%s" % (draftx,
              " i.e. only if one of them is loaded as the draft model"))
    print("   the marginal launch costs ~1.5-3.1 us here, measured.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
