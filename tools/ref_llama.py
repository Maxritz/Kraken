#!/usr/bin/env python3
"""ref_llama.py -- an independent float32 forward pass for a llama GGUF.

Why this exists
---------------
The engine cannot audit itself. When a change makes the int8 weight path
(`KRK_NPU_W8A8`) alter 94% of a greedy continuation, the engine can only say
that its own tokens changed; it cannot say whether that is because the
approximation is bad or because these fixtures have random weights and
therefore near-tied logits where any perturbation flips the argmax. Answering
that needs a reference distribution the engine did not produce.

This is that reference: ~150 lines of numpy, written from the GGUF metadata,
that computes last-position logits for a prompt. It reads no kraken source.

It is validated, not assumed. `--check` compares its argmax against the
engine's own `--prefill-hash first=`, which prints the token the engine would
generate first:

    tools/ref_llama.py --check /tmp/fix/tiny-f16.gguf

Two conventions had to be settled by measurement rather than by reading the
spec, because both are legal llama and only one matches these fixtures:

  * RoPE pairing. These fixtures use the INTERLEAVED pairing -- (2i, 2i+1)
    rotate together. With the NeoX half-split pairing the reference disagreed
    with the engine on two of three models; with this one it agrees on 21/21
    prompt/model pairs.
  * Attention scale is 1/sqrt(head_dim), the usual llama choice.

Supports F32, F16, Q8_0 and Q4_0 tensors, which is what the bench fixtures use.

Usage
-----
    ref_llama.py MODEL.gguf --tokens 1,259,260 [--dump out.npy]
    ref_llama.py MODEL.gguf --prompt "the quick brown fox"   # via the engine
    ref_llama.py MODEL.gguf --check                          # argmax vs engine

`--prompt` and `--check` shell out to the `kraken` binary for tokenization and
for the expected argmax, because the tokenizer is part of the engine and a
second copy of it here would be a second thing to be wrong.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import sys

import numpy as np

# GGML tensor types this loader understands.
GGML_F32, GGML_F16, GGML_Q4_0, GGML_Q8_0 = 0, 1, 2, 8

_KV_SIZE = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
_KV_FMT = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


def read_gguf(path):
    """Parse a GGUF header into (metadata, tensors, raw bytes, data offset).

    `tensors` entries are (name, dims as stored, ggml type, byte offset). GGUF
    stores ne[0] first, and ne[0] is the innermost dimension, so a weight of
    shape [out, in] appears here as [in, out] and must be reversed.
    """
    with open(path, "rb") as fh:
        d = fh.read()
    if d[:4] != b"GGUF":
        raise SystemExit(f"{path}: not a GGUF file")
    off = 4
    version, = struct.unpack_from("<I", d, off); off += 4
    if version < 2:
        raise SystemExit(f"{path}: GGUF v{version} is not supported")
    n_tensors, = struct.unpack_from("<Q", d, off); off += 8
    n_kv, = struct.unpack_from("<Q", d, off); off += 8

    def read_str(o):
        n, = struct.unpack_from("<Q", d, o)
        return d[o + 8:o + 8 + n].decode("utf-8", "replace"), o + 8 + n

    def read_val(o, t):
        if t == 8:
            return read_str(o)
        if t == 9:  # array: element type then length
            at, = struct.unpack_from("<I", d, o); o += 4
            al, = struct.unpack_from("<Q", d, o); o += 8
            vals = []
            for _ in range(al):
                v, o = read_val(o, at)
                vals.append(v)
            return vals, o
        return struct.unpack_from("<" + _KV_FMT[t], d, o)[0], o + _KV_SIZE[t]

    meta = {}
    for _ in range(n_kv):
        key, off = read_str(off)
        t, = struct.unpack_from("<I", d, off); off += 4
        meta[key], off = read_val(off, t)

    tensors = []
    for _ in range(n_tensors):
        name, off = read_str(off)
        nd, = struct.unpack_from("<I", d, off); off += 4
        dims = struct.unpack_from("<" + "Q" * nd, d, off); off += 8 * nd
        ttype, = struct.unpack_from("<I", d, off); off += 4
        toff, = struct.unpack_from("<Q", d, off); off += 8
        tensors.append((name, list(dims), ttype, toff))

    align = meta.get("general.alignment", 32)
    data_off = (off + align - 1) // align * align
    return meta, tensors, d, data_off


def _dequant(d, ttype, count, offset):
    """Decode `count` elements of a GGUF tensor into float32."""
    if ttype == GGML_F16:
        return np.frombuffer(d, dtype="<f2", count=count, offset=offset).astype(np.float32)
    if ttype == GGML_F32:
        return np.frombuffer(d, dtype="<f4", count=count, offset=offset).astype(np.float32)
    if ttype == GGML_Q8_0:
        # 32 values per block: f16 scale, then 32 int8.
        raw = np.frombuffer(d, dtype=np.uint8, count=count // 32 * 34, offset=offset).reshape(-1, 34)
        sc = raw[:, :2].copy().view("<f2").astype(np.float32)
        q = raw[:, 2:].view(np.int8).astype(np.float32)
        return (q * sc).reshape(-1).astype(np.float32)
    if ttype == GGML_Q4_0:
        # 32 values per block: f16 scale, then 16 bytes holding two nibbles each.
        raw = np.frombuffer(d, dtype=np.uint8, count=count // 32 * 18, offset=offset).reshape(-1, 18)
        sc = raw[:, :2].copy().view("<f2").astype(np.float32)
        qs = raw[:, 2:]
        lo = (qs & 0x0F).astype(np.float32) - 8.0
        hi = (qs >> 4).astype(np.float32) - 8.0
        return (np.concatenate([lo, hi], axis=1) * sc).reshape(-1).astype(np.float32)
    raise SystemExit(f"unsupported ggml type {ttype}")


def load(path):
    """Return (metadata, {name: float32 array shaped [out, in]})."""
    meta, tensors, d, data_off = read_gguf(path)
    weights = {}
    for name, dims, ttype, toff in tensors:
        count = int(np.prod(dims)) if dims else 1
        flat = _dequant(d, ttype, count, data_off + toff)
        weights[name] = flat.reshape(dims[::-1])
    return meta, weights


def rmsnorm(x, w, eps):
    return (x / np.sqrt((x * x).mean(-1, keepdims=True) + eps)) * w


def rope(x, base, pos0, scale=1.0):
    """Interleaved RoPE: (2i, 2i+1) rotate together.

    See the module docstring: this pairing is what the fixtures use, and it was
    established by agreement with the engine, not by assumption.
    """
    T, H, D = x.shape
    inv = base ** (-np.arange(0, D, 2, dtype=np.float32) / D)
    ang = np.arange(pos0, pos0 + T, dtype=np.float32)[:, None] * inv[None, :]
    c, s = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
    x0, x1 = x[..., 0::2], x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2] = x0 * c - x1 * s
    out[..., 1::2] = x0 * s + x1 * c
    return out


def forward(meta, w, tokens):
    """Last-position logits for `tokens`; returns the full [T, vocab] matrix."""
    L = meta["llama.block_count"]
    H = meta["llama.attention.head_count"]
    hd = meta["llama.attention.key_length"]
    eps = meta["llama.attention.layer_norm_rms_epsilon"]
    base = meta["llama.rope.freq_base"]
    T = len(tokens)
    mask = np.triu(np.full((T, T), -np.inf, dtype=np.float32), 1)

    x = w["token_embd.weight"][tokens].copy()          # [T, embd]
    scale = 1.0 / np.sqrt(float(hd))
    for l in range(L):
        h = rmsnorm(x, w[f"blk.{l}.attn_norm.weight"], eps)
        q = (h @ w[f"blk.{l}.attn_q.weight"].T).reshape(T, H, hd)
        k = (h @ w[f"blk.{l}.attn_k.weight"].T).reshape(T, H, hd)
        v = (h @ w[f"blk.{l}.attn_v.weight"].T).reshape(T, H, hd)
        q, k = rope(q, base, 0), rope(k, base, 0)
        sc = np.einsum("thd,shd->hts", q, k) * scale + mask
        sc -= sc.max(-1, keepdims=True)
        p = np.exp(sc)
        p /= p.sum(-1, keepdims=True)
        a = np.einsum("hts,shd->thd", p, v).reshape(T, H * hd)
        x = x + a @ w[f"blk.{l}.attn_output.weight"].T

        h = rmsnorm(x, w[f"blk.{l}.ffn_norm.weight"], eps)
        gate = h @ w[f"blk.{l}.ffn_gate.weight"].T
        up = h @ w[f"blk.{l}.ffn_up.weight"].T
        x = x + (gate / (1.0 + np.exp(-gate)) * up) @ w[f"blk.{l}.ffn_down.weight"].T
    x = rmsnorm(x, w["output_norm.weight"], eps)
    return x @ w["output.weight"].T


def _kraken_bin():
    for cand in (os.environ.get("KRAKEN_BIN"),
                 "build-npu/kraken", "build/kraken", "build-cpu/kraken", "kraken"):
        if not cand:
            continue
        if os.sep not in cand or os.path.exists(cand):
            return cand
    raise SystemExit("cannot find the kraken binary; set KRAKEN_BIN")


def engine_tokens(model, prompt):
    out = subprocess.run([_kraken_bin(), "-m", model, "--tokenize", "-p", prompt],
                         capture_output=True, text=True, check=True).stdout
    start, end = out.index("["), out.index("]")
    return [int(t) for t in out[start + 1:end].split(",")]


def engine_first(model, prompt):
    out = subprocess.run([_kraken_bin(), "-m", model, "--cpu", "--prefill-hash",
                          "-p", prompt, "--ctx", "512"],
                         capture_output=True, text=True, check=True).stdout
    for tok in out.split():
        if tok.startswith("first="):
            return int(tok.split("=", 1)[1])
    raise SystemExit("--prefill-hash printed no first= token")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("model")
    ap.add_argument("--tokens", help="comma-separated token ids")
    ap.add_argument("--prompt", help="tokenize this with the engine first")
    ap.add_argument("--check", action="store_true",
                    help="compare argmax against the engine's --prefill-hash")
    ap.add_argument("--check-prompts", default=None,
                    help="JSON list of prompts for --check (default: a built-in ladder)")
    ap.add_argument("--dump", help="write the [T, vocab] logits here as .npy")
    a = ap.parse_args(argv)

    meta, w = load(a.model)
    if a.check:
        prompts = (json.loads(a.check_prompts) if a.check_prompts else
                   ["the", "the quick", "the quick brown", "the quick brown fox",
                    " ".join(["the quick brown fox"] * 8)])
        ok = 0
        for p in prompts:
            tok = engine_tokens(a.model, p)
            ref = int(np.argmax(forward(meta, w, tok)[-1]))
            eng = engine_first(a.model, p)
            ok += (ref == eng)
            print(f"  len={len(tok):3d} reference={ref:6d} engine={eng:6d} "
                  f"{'MATCH' if ref == eng else 'MISMATCH'}")
        print(f"{ok}/{len(prompts)} prompts agree")
        return 0 if ok == len(prompts) else 1

    if a.prompt is not None:
        tokens = engine_tokens(a.model, a.prompt)
    elif a.tokens:
        tokens = [int(t) for t in a.tokens.split(",")]
    else:
        ap.error("need --tokens, --prompt, or --check")

    logits = forward(meta, w, tokens)
    if a.dump:
        np.save(a.dump, logits)
        print(f"wrote {a.dump} {logits.shape}")
    last = logits[-1]
    order = np.argsort(-last)
    print(f"argmax={int(order[0])} top5={[int(i) for i in order[:5]]}")
    print(f"top1-top2 margin {last[order[0]] - last[order[1]]:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
