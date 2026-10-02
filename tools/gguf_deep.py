#!/usr/bin/env python3
"""Deep GGUF dump: arch metadata (arrays expanded), per-layer tensor shapes, MoE expert shapes.

Usage: gguf_deep.py FILE [--layers]
"""
import re
import sys

import gguf


def dec(f):
    if not f.types:
        return None
    if f.types[0] == gguf.GGUFValueType.ARRAY:
        out = []
        for p in f.data:
            v = f.parts[p]
            if len(f.types) > 1 and f.types[1] == gguf.GGUFValueType.STRING:
                out.append(bytes(v).decode(errors="replace"))
            elif hasattr(v, "size"):
                out.append(v.item() if v.size == 1 else v.tolist())
            else:
                out.append(v)
        return out
    v = f.parts[f.data[-1]]
    if f.types[-1] == gguf.GGUFValueType.STRING:
        return bytes(v).decode(errors="replace")
    if hasattr(v, "size"):
        return v.item() if v.size == 1 else v.tolist()
    return v


def main(path):
    r = gguf.GGUFReader(path, "r")
    kv = dict(r.fields)
    arch = dec(kv["general.architecture"])
    print(f"== {path.split('/')[-1]}  arch={arch}  tensors={len(r.tensors)}")
    print("-- arch metadata --")
    for k in sorted(kv):
        if k.startswith(arch + "."):
            print(f"   {k:56s}: {dec(kv[k])}")

    tmap = {t.name: (list(t.shape), t.tensor_type) for t in r.tensors}
    layers = sorted({int(m.group(1)) for n in tmap if (m := re.match(r"blk\.(\d+)\.", n))})
    print(f"-- {len(layers)} layers: {layers[0]}..{layers[-1]} --")

    if "--layers" in sys.argv:
        for L in layers:
            nq = f"blk.{L}.attn_q.weight"
            if nq in tmap:
                print(f"   L{L:2d} attn_q {tmap[nq][0]}")
        for L in layers:
            ng = f"blk.{L}.ssm_conv1d.weight"
            if ng in tmap:
                print(f"   L{L:2d} GDN")
        return

    probe = sorted(set([0, 1, 2, 3, 4, 5, 6, 7, layers[-1]]))
    for L in probe:
        if L not in layers:
            continue
        keys = sorted(n for n in tmap if n.startswith(f"blk.{L}."))
        print(f"  -- blk.{L} ({len(keys)} tensors) --")
        for n in keys:
            sh, tt = tmap[n]
            print(f"     {n:56s} {str(sh):26s} type={tt}")

    print("-- moe expert shapes (first layer that has them) --")
    for L in layers:
        eg = f"blk.{L}.ffn_gate_exps.weight"
        if eg in tmap:
            for nm in ("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight",
                       "ffn_gate_inp.weight", "ffn_gate_shexp.weight", "ffn_up_shexp.weight",
                       "ffn_down_shexp.weight", "ffn_norm.weight", "post_attention_norm.weight"):
                k = f"blk.{L}.{nm}"
                if k in tmap:
                    print(f"   blk.{L}.{nm:28s} {tmap[k][0]}")
            break


if __name__ == "__main__":
    main(sys.argv[1])
