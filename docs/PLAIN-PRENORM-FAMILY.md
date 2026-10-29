# The plain pre-norm family: gptneox, gptj, phi2

The GGUF spec's "known values" list for `general.architecture` is
`llama, mpt, gptneox, gptj, gpt2, bloom, falcon, mamba, rwkv`. This engine had
`llama` and `gpt2` and refused the rest outright as unknown architectures. Three
of them are now supported — `gptneox`, `gptj`, `phi2` — because they turned out
to be the *same stack* as GPT-2 with different pieces switched on.

## Why they could not be added before

`ArchSpec::gpt2_style` bundles four things as one architecture:

1. LayerNorm (mean and variance) with a bias beside every weight
2. learned absolute position embeddings
3. a fused `attn_qkv` projection
4. a plain GELU MLP, no gate tensor

On GPT-2 those four really are one architecture. They are **not** one
architecture across the family: GPT-NeoX wants 1, 3 and 4 but RoPE instead of 2.
Phi-2 wants 1, 3 and 4 with *partial* rotary. GPT-J wants 1, 3 and 4 with no
biases anywhere at all. So the bundle had to be split before any of them could
be expressed — exactly the same problem `fused_qkv`/`fused_gate_up` solved for
Phi-3 in the previous change.

Four flags now name the pieces, and `gpt2_spec()` sets all four so GPT-2 reads
exactly as it did:

| flag | meaning |
|---|---|
| `ln_bias` | LayerNorm instead of RMSNorm, bias beside each weight |
| `learned_pos` | `position_embd.weight` summed into the embedding; no rotary |
| `parallel_residual` | attention and FFN both read the SAME normed row |
| `no_gate_ffn` | `up -> gelu -> down`, with no `ffn_gate` tensor |

| arch | flags | notes |
|---|---|---|
| `gpt2` | all four + `fused_qkv` | unchanged |
| `gptneox` | `ln_bias`, `parallel_residual`, `no_gate_ffn`, `fused_qkv`, neoX rope | fused QKV **with** bias |
| `gptj` | `ln_bias`, `no_gate_ffn`, `fused_qkv` | no biases anywhere |
| `phi2` | `ln_bias`, `no_gate_ffn`, `fused_qkv` | fused QKV **with** bias, partial rotary |

## The forward

`gpt2_forward` became `Engine::plain_forward` and covers all four, because they
differ only in which flags are set. `forward_core` dispatches to it on
`gpt2_style || ln_bias`, so no other family's graph is touched.

Two details that are easy to get wrong and are therefore spelled out in the code:

* **The three Q/K/V parts of a fused projection are row VIEWS of one weight**,
  not a copy of the output. `QuantTensor` is a plain descriptor and quantisation
  blocks are row-major, so part 2 starts at `data + lq * dtype_row_bytes(...)`.
  The bias splits at the same offsets, because it is one vector laid out
  `Q | K | V`. This is what makes fused and split bit-identical instead of
  merely close.
* **Parallel residual is a different graph, not an optimisation.** Sequential:
  `x += attn(norm(x)); x += ffn(norm(x))`. Parallel:
  `x += attn(norm(x)) + ffn(norm(x))`. The second reads the same normed row the
  attention read — still live in `ws_xn_` — so the second norm is *skipped*, not
  recomputed. Running the sequential shape on GPT-NeoX loads fine and decodes
  plausible text one graph deeper, which is why the flag exists rather than a
  heuristic.

## Two loader bugs this found

* `output_norm.bias` and the per-layer `attn_norm.bias` / `ffn_norm.bias` were
  **required** whenever the LayerNorm path was on. GPT-J is a LayerNorm with
  weights and *no biases at all*, so demanding them refused a complete file.
  They are now loaded when present and skipped when absent.
* The loader read only `attention.layer_norm_rms_epsilon`. The LayerNorm
  families write `attention.layer_norm_epsilon`. Every real file of the four
  happens to use 1e-5 so the default was right — which is exactly the kind of
  accident that stops being right later. Both keys are read now.

## Proof

`tools/make_bench_gguf.py` gained `--plain {gptneox,gptj,phi2}` and
`--split-qkv`. The bias pattern carries on *across* the Q/K/V parts rather than
restarting per part, so a split file and a fused file hold byte-identical
weights and the same function — which makes the expected result exact equality:

```
gptneox  fused  73=1.19381011(-0.684434) 24=1.17630696(-0.701937)
         split  73=1.19381011(-0.684434) 24=1.17630696(-0.701937)
gptj     fused  42=1.18302655(-0.649242) 70=1.09319985(-0.739069)
         split  42=1.18302655(-0.649242) 70=1.09319985(-0.739069)
phi2     fused  42=1.15305948(-0.684558) 70=1.1358068(-0.701811)
         split  42=1.15305948(-0.684558) 70=1.1358068(-0.701811)
```

All three families identical to the last digit. GPT-J is the weakest of the
three as a *test* — it has no biases, so it cannot exercise the split; gptneox
and phi2 both carry a fused QKV bias and both match, which is the case that
matters.

The `parallel_residual` flag was checked BY HAND to change the graph rather than
being ignored: forcing it off on a gptneox file moved the top logits from
`73=1.19381011 / 24=1.17630696` to `42=1.18604207 / 70=1.15634584`. That record
is not reproducible from the repository -- no gptneox or phi2 fixture exists in
`kraken-tests`, and the plain-family loop covers mpt/bloom/falcon only -- so what
the suite pins today is the ROW (`test_arch_row_pins` asserts gptneox's
`parallel_residual`, `fused_qkv`, `ln_bias`, `no_gate_ffn` and `rope_neox`), not
the graph. A file-level A/B for `parallel_residual` is still owed;
see `docs/TODO-alignment-2026-10-29.md` (B5).

`kraken-inspect` reports all three as `supported (dense)` / `runnable`;
`kraken-tests` **6566/6566** (2026-10-29).

## A fixture bug worth recording

The first attempt produced NaN on every logit, and the cause was in the
*generator*: the `--plain` path emitted no `ffn_up.weight` at all, so the engine
loaded a layer with no FFN (which is legal for nemotron's pure-FFN blocks, so
nothing complained) and then multiplied by an empty `QuantTensor`
(`type = Unknown`, `n_in = 0`). The NaN was the symptom; the empty tensor was
the bug. Instrumenting `L.wup`'s descriptor found it in one run, after three
guesses at the numerical cause were wrong.

## Still refused

`falcon` needs ALiBi plus parallel attention with multiquery — and two
*different* norms in the parallel branch, unlike GPT-NeoX's shared row.
(`mpt` and `bloom` were the other two entries here; they are supported now, see
[ALIBI-AND-MPT-BLOOM.md](ALIBI-AND-MPT-BLOOM.md).) `mamba` and `rwkv` are not transformer
stacks at all: a selective state-space recurrence and a linear-attention
recurrence respectively, and neither is the gated delta net this engine already
implements. Those five are the honest remainder of the spec's list.
