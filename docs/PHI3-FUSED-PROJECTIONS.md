# Fused projections: Phi-3 / Phi-4 support

`phi3` was refused outright before this change — not "unsupported format", but
"unknown architecture". Phi-3-mini-4k-instruct and Phi-4-reasoning-plus from the
support list could not run at all. The blocker was not the attention math: it was
that two of the family's projections are **fused**, and the only fused-QKV path in
the engine belonged to `gpt2_style`.

## Why the gpt2 bundle could not be reused

`ArchSpec::gpt2_style` bundles four things because on GPT-2 they are one
architecture: LayerNorm-with-bias, learned absolute positions, a fused
`attn_qkv`, and a plain GELU MLP. Phi-3 needs the fused projection with **none**
of the other three — RMSNorm, partial rotary, SwiGLU. Setting `gpt2_style` would
have run Phi's norms as LayerNorm, which is a wrong answer that still decodes
fluently.

So the two fused shapes became their own flags, `fused_qkv` and `fused_gate_up`,
and `gpt2_spec()` sets `fused_qkv` for itself. The flags are appended after
`gpt2_style` in `ArchSpec`, which must stay last: every row in the table is a
positional aggregate initializer, so a member inserted in the middle silently
shifts every field after it (the header records that this has already happened
once).

## The two shapes

* **`attn_qkv`** — output row is `Q | K | V`. Under GQA the K and V blocks are
  `n_head_kv*head_dim` wide, not `n_embd`, so a `3*n_embd` split reads K out of
  V's rows for every head after the first. The loader checks the exact width and
  refuses anything else.
* **`ffn_up`** — output row is `gate | up`, `n_ff` each, and the file carries no
  `ffn_gate` tensor at all. Treating `ffn_up` as the gate (what a missed flag
  does) halves the intermediate width and gates the wrong operand.

## The implementation: row views, not copies

Both are implemented as **row views** of one weight. `QuantTensor` is a plain
descriptor — pointer, `n_in`, `n_out` — and quantisation blocks are row-major, so
a row range is `data + row * dtype_row_bytes(type, n_in)`. That turns each fused
projection into the *same projections the split path already runs*, over the same
bytes:

* QKV becomes one `gemm_group` of three weight views;
* gate|up becomes two views feeding the existing gated path, so the fused silu
  epilogue, `gemm_group` and the GeGLU branch all run unchanged.

This was arrived at after a copy-based split failed the equality test. The copy
downloaded the fused output, re-split it per row and re-uploaded it, and the
result disagreed with the split model. Views remove the entire class of problem:
there is no second buffer, no device sync, and no stride convention that can
differ between the two paths.

## Two gating bugs the equality test caught

The first version produced different logits, and both causes were the same shape
of bug — a sub-block gated on the *split* tensor:

| site | gate | effect on a fused file |
|---|---|---|
| attention block, `forward_core` | `L.wq.present()` | attention skipped entirely; the model ran as FFN-only |
| FFN dispatch, `forward_core` | `L.wgate.present()` | FFN skipped entirely |

Both now also accept the fused tensor. Neither would have crashed or warned: the
model decodes plausibly while ignoring its context, which is the failure mode
this repo treats as worse than a refusal.

## The test that proves it

The fixture generator (`tools/make_bench_gguf.py`) gained `--phi3`,
`--phi3-qkv-only` and `--phi3-ffn-only`. `w_cat` draws the parts in the same
order the split file draws its separate tensors, and because every row is a whole
number of blocks, the quantised bytes are **identical** — verified byte-for-byte
in Python before testing the engine.

So the two files hold the same weights and differ only in layout, and the correct
result is not "close": it is the same logits.

```
split-q8     step 0 top2: 58=1.20879912(-0.684451) 298=1.19132948(-0.701920)
p3-qkvonly   step 0 top2: 58=1.20879912(-0.684451) 298=1.19132948(-0.701920)
p3-ffnonly   step 0 top2: 58=1.20879912(-0.684451) 298=1.19132948(-0.701920)
p3-both      step 0 top2: 58=1.20879912(-0.684451) 298=1.19132948(-0.701920)
```

Four files — arch `llama` with split tensors, and `phi3` with each projection
fused separately and both together — identical to the last printed digit.
Reproduce with `--embd 128 --layers 2 --heads 4 --kv-heads 2 --head-dim 32
--ff 256 --vocab 320 --format q8_0 --seed 7`. GQA is deliberate: with
`n_head_kv == n_head` a wrong `3*n_embd` split would not be caught.

Partial rotary needed no work: `rope_frac` already reads
`rope.dimension_count`, and the header calls it the "Phi partial RoPE" path. The
rope pairing is NORM, per llama.cpp's `llm_arch_rope` table — the one choice here
that a wrong guess turns into fluent nonsense rather than a refusal, which is why
it is named in `phi3_spec()` instead of silently inherited.

## What is still refused, and why

* **`phimoe`** (Phi-3.5-MoE) — the same fused shapes on an MoE stack, plus
  per-expert biases. Not claimed until its tensors are checked against a file.
* **`phi2`** — a fused `attn_qkv` *with a bias*. The loader refuses it explicitly
  rather than ignoring the bias, because an unapplied bias is a wrong answer, not
  a small one.
* **`glm4`**, **`minicpmv`**, **`qwen38`** — still absent from the table; each
  needs its own row and, for `glm4`, a per-layer partial-rotary expression.

## Retesting after a change to either path

```
G="--embd 128 --layers 2 --heads 4 --kv-heads 2 --head-dim 32 --ff 256 \
--vocab 320 --ctx 256 --format q8_0 --seed 7"
python3 tools/make_bench_gguf.py /tmp/split.gguf   $G
python3 tools/make_bench_gguf.py /tmp/qkv.gguf     $G --phi3-qkv-only
python3 tools/make_bench_gguf.py /tmp/ffn.gguf     $G --phi3-ffn-only
python3 tools/make_bench_gguf.py /tmp/both.gguf    $G --phi3
for f in split qkv ffn both; do ./build/kraken -m /tmp/$f.gguf -p hello -n 1 \
    --debug-topk 2 2>&1 | grep '^step'; done
```

All four lines must be identical. If they are not, the fused path and the split
path disagree, and the split path is the reference.
