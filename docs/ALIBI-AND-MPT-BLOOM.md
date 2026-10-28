# ALiBi, and the mpt and bloom families

The GGUF spec lists `llama, mpt, gptneox, gptj, gpt2, bloom, falcon, mamba, rwkv`
as known `general.architecture` values. After `gptneox`/`gptj`/`phi2` were added
as one stack, two more were refused that need not have been: `mpt` and `bloom`
are the *same* pre-norm stack again, with layer norms and a fused QKV and an
ungated GELU — and a different **position signal**.

## ALiBi is not rotary and not learned

GPT-2 adds a learned position vector to the embedding. Everyone else here
rotates q and k. ALiBi does neither: it subtracts a per-head multiple of the
distance between the query and the key from the attention score,

```
score[q][k] -= slope(head) * (pos_q - pos_k)
```

applied after the softmax scale, before the max subtraction. So it is a bias on
the score, it depends on the head index, and it needs no table of any size.

The slope formula is the whole difficulty, and it is *not* the obvious one. From
the reference (`ggml/src/ggml-cpu/ops.cpp`, `soft_max_ext`):

```c
n_head_log2 = 1u << (uint32_t) floor(log2(n_head));
m0 = powf(2.0f, -max_bias / n_head_log2);
m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);
slope(h) = h < n_head_log2 ? powf(m0, h + 1)
                           : powf(m1, 2 * (h - n_head_log2) + 1);
```

Two things there are easy to get wrong in a way that still decodes coherent text:

* the anchor is `floor(log2(n_head))`, **not** `n_head` and not `ceil` — the
  reference itself carries a `TODO` about mpt-7b's `attention.py` using floor
  where BLOOM-derived code used ceil;
* the second half uses **half** the max bias as its base. Drop the `/2` and
  every head past the anchor is penalised twice as hard.

Both produce a slightly wrong attention pattern, so both are spelled out in
`alibi_slope_impl()` in `src/backend_cpu.cpp` and, importantly, the slope is a
function of `(layer, head)` only — it is hoisted out of the key loop rather than
recomputed per position pair.

## Where the bias lives

`AttnDesc` carries `alibi_max_bias`, a **scalar**, not a per-head table: the
table is `n_head` floats and the bias is one multiply-add per score, so shipping
the scalar and computing slopes inside the kernel keeps the descriptor small and
keeps the formula in one place. `alibi_max_bias == 0` — the default for every
other architecture — skips the whole thing.

A family with ALiBi must **not** also run rotary or add a learned position.
`plain_forward` therefore skips `rope()` when `mc.alibi_max_bias > 0`, the same
reason llama.cpp lists these architectures under `LLAMA_ROPE_TYPE_NONE`.

## The two specs

| arch | flags | ALiBi source |
|---|---|---|
| `mpt` | `ln_bias`, `no_gate_ffn`, `fused_qkv` | file key `attention.max_alibi_bias` |
| `bloom` | same + `embd_norm` | **constant 8.0** in the loader |

The asymmetry is not an oversight. MPT states its scale in the file; BLOOM's
reference loader sets `f_max_alibi_bias = 8.0f` in code and never reads a key.
A loader that only reads the key leaves BLOOM at 0, and **ALiBi off still
decodes** — plausible text, wrong answers, no error anywhere.

BLOOM also normalizes the *embedding* before block 0 (`token_embd_norm`, weight
**and** bias), which no other family here does. That is one extra LayerNorm over
the whole `[n, E]` activation block, not per layer.

## Three loader bugs this found

* `position_embd.weight` was **required** under `learned_pos`. MPT ships both
  with and without it — those files were refused while being complete.
* `attn_q_norm`/`attn_k_norm` were loaded and set `cfg_.qk_norm`, but the plain
  path never called `qk_norm()`. A file carrying them loaded, decoded, and was
  silently wrong on both backends. The plain path now applies the per-head norm
  before rotary, exactly where the llama path does.
* `token_embd_norm` is not a rename of any per-layer norm. BLOOM carries both;
  substituting one for the other is a same-shape swap that decodes wrongly.

## Proof

`tools/make_bench_gguf.py` gained `--plain {mpt,bloom}`, `--alibi <float>` and
`--qk-norm`. Fixture:
`--embd 128 --layers 2 --heads 4 --kv-heads 2 --head-dim 32 --ff 256
--vocab 320 --ctx 256 --format f16 --seed 7`.

**Fused QKV is a row view, so fused and split are the same function** — the same
test that pinned the plain family:

```
mpt    fused  42=1.16766608(-0.685854) 70=1.15302539(-0.700494)
       split  42=1.16766608(-0.685854) 70=1.15302539(-0.700494)
bloom  fused  73=1.21770453(-0.606433) 202=1.03603601(-0.788101)
       split  73=1.21770453(-0.606433) 202=1.03603601(-0.788101)
```

**ALiBi is applied, not ignored.** The scale is the only difference between two
MPT files with byte-identical weights:

```
mpt --alibi 8   42=1.16766608(-0.685854) 70=1.15302539(-0.700494)
mpt --alibi 0   42=1.15305948(-0.684558) 70=1.1358068 (-0.701811)
```

`--alibi 0` reproduces `phi2`'s value from the previous change **digit for
digit** — same weights, no position bias, so it collapses onto the shared plain
pre-norm path. That is a check on two features at once.

**BLOOM's embedding norm is applied.** The two fixtures differ only by that
`token_embd_norm` pair, and forcing the norm off (temporary env guard, since
removed) gives

```
bloom             73=1.21770453(-0.606433) 202=1.03603601(-0.788101)
bloom, no embd norm  42=1.16766608(-0.685854) 70=1.15302539(-0.700494)
```

which is the MPT-ALiBi-8 result *exactly*. BLOOM's constant 8.0 therefore equals
MPT's file-supplied 8.0, and BLOOM's graph is MPT's graph plus one norm — both
verified by equality rather than by inspection.

**QK-norm is applied.** `--qk-norm` writes a non-identity ramp for
`attn_q_norm`/`attn_k_norm` (all-ones would be the RMSNorm identity, so presence
alone would be invisible):

```
mpt --qk-norm  140=1.11651647(-0.685522) 16=1.10120773(-0.700831)
mpt            42=1.16766608(-0.685854) 70=1.15302539(-0.700494)
```

Eight tokens of decode on both families: no NaN, no inf, KV cache resident, no
tiering traffic. `kraken-inspect` reports both `supported (dense)` / `runnable`;
`kraken-tests` **5039/5039**.

## falcon: parallel attention with two norms

`falcon` was the other name on the refused list that reuses this stack — but it
is **not** ALiBi. Its loader reads `attention.layer_norm_epsilon` and nothing
else, so `f_max_alibi_bias` stays 0 and the position signal is the NEOX-paired
rotary, the same pairing GPT-NeoX uses. What it does have is a parallel branch
with **two different norms**:

```
x = x + attn(ln1(x)) + mlp(ln2(x))      // Falcon-40B
x = x + attn(ln1(x)) + mlp(ln1(x))      // Falcon-7B -- no attn_norm_2 in the file
```

GPT-NeoX's `parallel_residual` reuses the row attention already normalized.
Falcon normalizes the *same pre-attention residual* a second time with different
weights, so `ArchSpec::parallel_two_norm` exists and the second row
(`ws_xn2_`, allocated only for this family) has to be taken **before** the
attention residual lands in `ws_x_`. Norming after the residual is the one
mistake in this layer that still produces clean, coherent text.

Note which norm the MLP reads: `attn_norm` (ln1), not `attn_norm_2`, even on
40B — the reference's `build_ffn(attn_norm, ...)` is explicit that it is the
attention's *first* norm, not the result's.

### Proof

`--plain falcon` (norm biases, no linear biases), `--attn-norm-2`. Fused vs
split, as everywhere else on this stack:

```
falcon         73=1.18236196(-0.682557) 24=1.16106832(-0.703851)
falcon-split   73=1.18236196(-0.682557) 24=1.16106832(-0.703851)
falcon-ln2     73=1.19939208(-0.675833) 24=1.16445804(-0.710767)
falcon-ln2-split 73=1.19939208(-0.675833) 24=1.16445804(-0.710767)
```

The second norm is live — adding `attn_norm_2` moves the logits. And its
**placement** is pinned by the one file that must be a no-op:
`--attn-norm-2-identity` writes `attn_norm_2` with `attn_norm`'s own weights, so
the two arms normalize identically and the result must equal the one-norm file
*exactly*:

```
falcon           73=1.18236196(-0.682557) 24=1.16106832(-0.703851)
falcon-ln2-id    73=1.18236196(-0.682557) 24=1.16106832(-0.703851)
```

Equal to the last digit. Norming after the attention residual would consume a
different row and could not match. `kraken-inspect`: `supported (dense)` /
`runnable`; `kraken-tests` **5083/5083**.

## Still refused

`mamba` and `rwkv` — not transformer stacks at all: a selective state-space
recurrence and a linear-attention recurrence, neither of which is the gated
delta net this engine already implements.
