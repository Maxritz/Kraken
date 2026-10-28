# Families that are llama's schema with a piece swapped

Read against llama.cpp master (`src/models/*.cpp`, `src/llama-arch.cpp`,
`conversion/*.py`). Seven architectures were refused before this change for
pieces this engine already had — they are llama's tensor set with one or two
parts exchanged, which is why they cost almost nothing to add and why refusing
them was never a statement about their difficulty.

| arch | what it swaps | where the reference says so |
|---|---|---|
| `chatglm` | one fused `ffn_up` whose row is gate \| up | `chatglm.cpp`: `ffn_up {n_embd, n_ff*2}`, `LLM_FFN_SWIGLU` |
| `glm4` | the same fused MLP **plus** a norm on each branch's output | `glm4.cpp`: `attn_post_norm`, `ffn_post_norm`, NORM rope |
| `mellum` | qwen3-MoE with a 4-layer sliding-window pattern | `mellum.cpp`: `load_swa_pattern(ml, 4)`, softmax gating, `expert_weights_norm` |
| `phimoe` | Phi-3's block with routed experts and two head biases | `phimoe.cpp` reuses `llama_model_phi3::graph` |
| `minicpm` | llama's tensor set; partial rotary | `minicpm.cpp`, NORM rope |
| `qwen2vl` | the Qwen2 text tower of a vision model | `qwen2vl.cpp`, q/k/v biases, NeoX rope |
| `qwen38` | *an alias, not a schema* — see below | `conversion/qwen4exp.py` |

Two of these needed something the engine did not have:

* **`glm4` needs the per-branch post norms** — `x += norm_attn(attn(ln(x)))`,
  not `x += attn(ln(x))`. That graph already existed for gemma2/3/4
  (`ArchSpec::post_branch_norm`), so glm4 sets the flag rather than getting a
  new path. The norm is applied to the branch's own output *before* it joins the
  residual, which is a different function from norming the sum.
* **`phimoe` needs `output.bias`** — one f32 per vocabulary row, added to the
  logits after the head's projection. It rides a new `Engine::logit_head()`,
  which is now the *only* place the vocabulary head is formed; before this, six
  call sites each called `gemm_w(... ws_logits_ ...)` directly, so a bias
  applied at one of them would have been missing at the others. The device
  top-k reads `ws_logits_` without ever touching the host copy, so the bias has
  to land in the buffer.

## Qwen3.8 is `qwen4exp`, and the label cannot decide the schema

llama.cpp has no `qwen38` architecture — 170 model files, no such name. What it
has is:

```python
@ModelBase.register("Qwen4ExpForConditionalGeneration", "Qwen4ExpForCausalLM")
@ModelBase.example("Qwen/Qwen3.8-Flash-Next")
class Qwen4ExpTextModel(...):
    model_arch = gguf.MODEL_ARCH.QWEN4EXP
```

So **Qwen3.8-Flash-Next IS `qwen4exp`**, and this engine already runs it
(`test_qwen4exp_whole_graph`). The earlier row in this table that called the
Qwen3.8 files "REFUSED if the file says qwen38" was reading the name and not the
family.

But the name still cannot decide the schema, because the family has two shapes.
`conversion/qwen4exp.py` builds Qwen3.8-Flash from hyper-connections, a
4-lane residual, a PLE n-gram table, a QSA indexer and a 288-expert MoE — none
of which a *dense* Qwen3.8 has. Reading the tag as qwen4exp unconditionally
demands metadata a dense file does not carry; reading it as qwen3 silently
ignores the delta-net layers, the lanes and the n-gram table of a Flash file.
Both mistakes load.

So `qwen38` is an **alias resolved from the file's tensors** (`kAliases` in
`src/verdict.cpp`):

| probe, in order | resolves to |
|---|---|
| `blk.0.hc_attn_inject.weight`, `blk.0.hc_ffn_inject.weight`, `per_layer_token_embd.weight`, `blk.0.ssm_conv1d.weight`, `blk.0.indexer.q_norm.weight` | `qwen4exp` |
| otherwise | `qwen38`'s own entry (qwen3 geometry) |

The verdict says which way it went, so the choice is never invisible:

```
arch       qwen38 — supported (dense)
verdict    runnable
           . 'qwen38' is an alias: the file carries blk.0.hc_attn_inject.weight,
             so it is read as 'qwen4exp'
```

### The bug the alias found

`Engine::init` sized the query projection with

```cpp
q_proj_ = arch_spec(mc.arch).q_output_gate ? 2 * q_dim_ : q_dim_;
```

— asking the table **by name**. An aliased file's name does not carry the
resolved spec, so a qwen4exp file labelled `qwen38` was sized as though its
query row were unpacked, while the loader had already bound the packed row. The
two disagreed and the engine decoded a different model, with no error anywhere.

The flag is now `ModelConfig::q_output_gate`, copied from the resolved spec, and
the engine reads the config. The general rule this is an instance of: **the
engine must never re-derive geometry from `general.architecture`**, because
after alias resolution or a fallback the name is no longer the authority. That
is what `test_qwen38_is_an_alias` pins — it writes the same bytes under both
labels and requires the logits to be identical to the bit.

## Proof

Seven fixtures, `--schema <arch>`, all run first try:

```
chatglm   58=1.21570027(-0.676298) 298=1.18171251(-0.710285)
glm4      30=1.42149913(-0.591809) 187=1.20738268(-0.805925)
minicpm   58=1.21570027(-0.676298) 298=1.18171251(-0.710285)
qwen2vl  298=1.21223712(-0.652280)  58=1.12876189(-0.735756)
qwen38   187=1.17638159(-0.649242) 139=1.08655381(-0.739069)
mellum    13=1.28044283(-0.625149) 255=1.13948274(-0.766109)
phimoe    78=1.20195365(-0.638804)  77=1.09014356(-0.750614)
```

Each new piece was then checked to be *live* rather than silently skipped, by
A/B on the same weights:

* **chatglm's fused gate\|up** — `--split-ffn` writes `ffn_gate`/`ffn_up`
  separately from the same stream, so the two files hold the same bytes and must
  agree exactly: `58=1.21570027(-0.676298) 298=1.18171251(-0.710285)` on both.
  `minicpm` produces those same logits without fusing at all, which is a third
  independent check on the same equivalence.
* **phimoe's `output.bias`** — `--out-bias 0.5` moves the top two from
  `78=1.20195365 / 77=1.09014356` to `77=1.59014356 / 290=1.58733547`. The first
  attempt at this showed **no change**, because the load had been placed inside
  `if (cfg_.ln_bias)` — and Phi-MoE is an RMSNorm family, so it never ran. The
  bias is now loaded outside that branch, with the A/B that caught it recorded
  in the code.
* **qwen38's alias** — `test_qwen38_is_an_alias`, 11 checks, including the
  bit-identical logits above.

`kraken-tests` **5500/5500**.

## Second pass: `glm4moe`, `phi3moe`, `minicpm3`

Three more, and only one of them was a schema question.

**`glm4moe`** (GLM-4.5 / GLM-4.7-Flash) is `glm4` with routed experts: the same
attention with an `attn_post_norm` on its branch, q/k RMS norms and biases on
q/k/v, then a sigmoid router carrying a per-expert selection bias beside a
shared expert of `n_ff_exp * n_expert_shared` columns. Everything but two
defaults already existed, and both are *defaults for a silent file*, not new
code:

* the router activation — llama.cpp's glm4-moe loader falls back to **sigmoid**
  when `expert_gating_func` is absent, while this engine's generic fallback is
  softmax. On a 128-expert stack those are different *selections*, not just
  different weights;
* `expert_weights_norm` — the reference leaves it false, so a file that says
  nothing must not renormalize. The engine's historical default is to
  renormalize, which is right for Qwen and wrong here.

Both are now `ArchSpec::router_gating` / `ArchSpec::router_w_norm`, and the
file's own key still wins when it has one. The dense/MoE mix needs no code at
all: the layer's own tensors say which it is, and the leading dense blocks are
already read into `cfg_.n_dense_lead`.

**`phi3moe`** is literally the same graph — llama.cpp's phi3 MoE branch is
`llama_model_phi3::graph` with a `build_moe_ffn` call in place of the dense MLP
— so the spec is phi3's two fused-projection flags and nothing else. Its logits
on the fixture match `phimoe`'s to the digit, which is the expected result:
both read the same schema and the fixture's biases are zero.

**`minicpm3`** is the V2-Lite latent attention (which this engine already ran)
with a **low-rank query trunk** over that same uncompressed KV path, plus a
depth scale. The scale is one line: each layer multiplies the ATTENTION
branch's output by `1.4 / sqrt(n_layer)` before the residual add, and only that
branch. It is a constant in the reference graph with no tensor and no metadata
key, so it is divided once at load and applied where the branch's value is
final; at 1.0 the multiply is skipped entirely.

### The bug that found

Building the fixture exposed a real defect, and not in the new code. The lite
MLA forward projected the query **only** through `attn_q` (the direct form):

```cpp
gemm_w(ws_q_, ws_xn_, L.wq_direct, nh * qk, n_embd_, n);
```

A file with a `q_lora_rank` stores `attn_q_a → attn_q_a_norm → attn_q_b`
instead. The loader read and shape-checked that trunk — and the forward never
asked for it, so the projection ran against an absent tensor and produced
queries at 2.8e22 that became NaN one layer later. Nothing errored.

It was not only MiniCPM3: **every V3-class DeepSeek file has that trunk**, so
the `deepseek2` row's "runs" claim covered a file shape nothing had ever
executed end to end. The trunk is now projected (with its norm) and falls back
to the direct path only when the file has no trunk; a layer with neither is a
named refusal instead of a multiply by nothing.

The second half of the same bug: `ws_mlaqn_` — the trunk's intermediate row —
was allocated **only for the shared layout**, so the fixed projection then
segfaulted on the first token. It is now allocated whenever the file declares a
`q_lora_rank`, which is the condition that makes it readable.

Pinned by `test_minicpm3_whole_graph`: the file loads, the resolved scale is
`1.4/sqrt(n_layer)`, the logits are finite, and the same bytes under a label
without the scale produce different logits — so the multiply is known to act
rather than merely to be configured.
