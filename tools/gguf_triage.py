#!/usr/bin/env python3
"""Triage GGUF checkpoints: architecture, MoE params, byte accounting."""
import sys

import gguf


def triage(path):
    r = gguf.GGUFReader(path, "r")
    kv = dict(r.fields)

    def g(name):
        return kv[name].parts[kv[name].data[-1]] if name in kv else None

    arch = g("general.architecture")
    print(f"== {path.split('/')[-1]}")
    print(f"   architecture       : {arch}")
    for k in ("general.size_label",):
        if k in kv:
            print(f"   {k}: {g(k)}")
    for k in ("expert_count", "expert_used_count", "block_count", "embedding_length",
              "attention.head_count", "attention.head_count_kv", "attention.key_length",
              "attention.value_length", "ffn_feed_forward_length",
              "expert_feed_forward_length", "expert_shared_feed_forward_length",
              "ssm.conv_kernel", "ssm.state_size", "ssm.group_count",
              "ssm.time_step_rank", "ssm.inner_size", "full_attention_interval",
              "rope.dimension_count", "rope.freq_base"):
        key = f"{arch}.{k}"
        if key in kv:
            print(f"   {k:32s}: {g(key)}")
    names = [t.name for t in r.tensors]
    has_exps = any(".ffn_gate_exps." in n or n.endswith("ffn_gate_exps.weight") for n in names)
    has_shexp = any("shexp" in n for n in names)
    has_gdn = any(".ssm_conv1d." in n or n.endswith("ssm_conv1d.weight") for n in names)
    print(f"   tensors={len(names)} moe_experts={has_exps} shared_expert={has_shexp} gated_deltanet={has_gdn}")
    # Byte accounting by quantization type.
    by_type = {}
    total = 0
    for t in r.tensors:
        by_type[t.tensor_type] = by_type.get(t.tensor_type, 0) + t.n_bytes
        total += t.n_bytes
    print(f"   total tensor bytes : {total:,} ({total/1e9:.2f} GB)")
    for tt in sorted(by_type):
        print(f"     type {tt:2d}: {by_type[tt]:>14,} bytes")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        triage(p)
