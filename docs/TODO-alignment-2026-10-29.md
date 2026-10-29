# Goal-to-code alignment — work list (2026-10-29)

Source: the `goli-mat-de` audit of `f9aec22` (findings F1–F11), plus what the
fixes themselves turned up. Every item names the check that closes it; an item
is only "done" when that check runs and passes.

Status legend: **todo** · **done** · **blocked (why)**

## Closed in this change

| ID | P | Item | Closing check that now runs |
|----|---|------|------------------------------|
| A1 | P1 | gemma4 attention scale: `ArchSpec::attn_scale` (row = 1.0), `<arch>.attention.scale` read into `ModelConfig::attn_scale` | `test_family_scales` (1): row value; (2): three `llama` files — key absent vs the default value are bit-equal, key 4.0 differs. **The gemma4 file itself is still unbuilt** — see B4. |
| A2 | P1 | MiniCPM embedding scale (`scale_emb = 12.0`, `<arch>.embedding_scale`) | `test_family_scales` (1) row + (3) file A/B (1.0 is a no-op, 12.0 changes the answer); `test_minicpm3_scales` reads 12.0 off a MiniCPM3 file |
| A3 | P1 | mellum window phase flipped to `swa_window_first = true` (dense_first = false) | `test_family_scales` (4): a mellum file with no pattern array asserts `is_swa(0..2)` and `!is_swa(3)`. The old flag fails this arm on `is_swa(0)`. |
| A4 | P2 | `qk_norm_wide` decided once per stack; mixed widths refused | `test_family_scales` (6): mixed q-wide/k-per-head refused naming both tensors; a stack that changes convention at layer 1 refused naming the layer. The uniform olmo2 arm in `test_norm_schema_shapes` still passes. |
| A5 | P2 | row pins for the families that had no test by name | `test_arch_row_pins` (23 checks): gptneox, gptj, phi2, olmoe, qwen3moe, nemotron_h, nemotron_h_moe, qwen35-dflash-draft, minicpm |
| A6 | P3 | dead config state: `has_bias` and `rope_attn_scale` removed (write-only / never assigned), `rope_yarn_factor`/`rope_yarn_log_mul` and `n_dense_lead` kept but documented for what they actually are | build clean; no unreferenced field writes |
| A7 | P3 | `Backend::attention_sink` removed (zero callers; real sinks ride `AttnDesc::sinks`) | `grep attention_sink src include` finds only the HIP kernel the engine calls through `d.sinks` |
| A8 | P2 | baseline re-recorded at `f9aec22` + this change; `AGENTS.md` counts corrected (4521/4528/95 → 6566/125) | `scripts/kraken_tests_baseline_check.sh` rc=0, "0 changed, 0 new" |
| B1 | P1 | MiniCPM3 head-input scale `n_embd_base/n_embd` (256/n_embd), applied in `Engine::head_compute` after the final norm | `test_minicpm3_scales`: config is `256/n_embd` on a real MiniCPM3 file; the run is finite through it |
| B2 | P1 | MiniCPM3 residual scale on **both** branches (`ArchSpec::residual_scale_ffn` → gemm + scale + add in `dense_ffn`) | `test_minicpm3_scales`: flag on the config, `attn_out_scale == 1.4/√n_layer`, 4-layer run finite (a double add would leave f16 range) |
| B3 | P2 | windowed layers rotate with PLAIN rope when the stack is YaRN-scaled (`ArchSpec::swa_plain_rope`, mellum + olmo2) | `test_family_scales` (5): a YaRN mellum file's windowed table differs from its full-layer table; the unscaled control has one table for both |

## Pending-model pass (2026-10-29)

Every row in `src/arch.cpp` that was not `Yes` now has a fixture built from that
family's OWN definition (llama.cpp's declared tensor list plus the metadata its
loader reads), taken through `assess -> Model::load -> Engine::init -> generate`,
and deleted again. Two groups own the claims, so a family is never measured two
ways:

* `test_pending_model_fixtures` (36 checks) — the families that RUN.
* `test_pending_model_refusals` (24 checks) — the families that do not, where the
  assertion is that the refusal is the ROW's and NAMES the missing piece (an
  unknown arch string is refused too, so a bare refusal would prove nothing).

| row | before | fixture result | now |
|-----|--------|----------------|-----|
| `gemma` | Partial (unvalidated) | loads, inits, generates — once the family's QK-norm constant was removed | **Yes** |
| `gemma2` | Partial (softcap) | loads, inits, generates on CPU | **Partial**, reason still true: the attention-logit softcap is applied by the reference attention only, so a HIP build refuses BY CAPABILITY |
| `gemma3` | Partial | loads, inits, generates, `qk_norm=1`, `swa=1024` | **Yes** |
| `qwen2vl` | Partial | text tower runs; a file WITH `v.blk.*`/`mm.*` also runs and the report names the vision tower it cannot read | **Partial**, honest: images are ignored |
| `qwen3vl` | Partial | text tower runs, `qk_norm=1` | **Partial**, same |
| `deepseek41` | Partial | dense-core file runs the whole deepseek4 graph | **Partial**: Engram and the Sinkhorn hyper-connections are real gaps |
| `gpt-oss` | Partial | loads, inits, generates on its own schema | **Partial** (loader schema only) |
| `deepseek4` | Partial | covered by C1 above | **Partial** |
| `k2-horizon` | No | refused, names "routed experts on the attention values (attn_v_exps)" | **No** |
| `minimax-m3` | No | refused, names MiniMax Sparse Attention | **No** |
| `diffuse` | No | refused, names block diffusion | **No** |
| `dspark` | No | refused, names its markov head | **No** |
| `deepseek4-dspark` | No | refused, names the multi-token-prediction head shard | **No** |
| `qwen4-mtp` | No | refused, names the qwen4exp target it belongs to | **No** |

### `Yes` rows with no file behind them (same pass, second half)

A `Partial` with a measured reason is a weaker claim than `Yes` — but a `Yes`
with no file behind it is weaker still. Sweeping the suite and collecting every
arch string the loaders actually saw found **13 `Yes` rows that no group had
ever loaded**:

| row | family definition (llama.cpp) | now |
|-----|-------------------------------|-----|
| `phi3` | one fused `attn_qkv` row, one fused gate|up `ffn_up` (`phi3.cpp:52`) | built, loads, generates |
| `phi3moe`, `phimoe` | `phimoe.cpp`: the same attention, SEPARATE `ffn_gate_exps`/`ffn_up_exps`, biases on every norm | built, loads, generates |
| `chatglm` | llama's split q/k/v with the fused gate|up MLP | built, loads, generates |
| `glm4` | the same MLP plus a norm on each branch's output | built, loads, generates |
| `glm4moe` | `glm4-moe.cpp`: post-norms, per-head QK-norm, q/k/v biases, shared expert | built, loads, generates |
| `qwen3` | llama's block plus a per-head QK-norm | built, loads, generates |
| `laguna` | per-head QK-norm, a separate softplus attention gate, NeoX pairing, routed FFN | built, loads, generates |
| `gpt2` | LayerNorm with biases, learned absolute positions, fused q|k|v, no gate tensor | built, loads, generates |
| `qwen35` | the dense sibling of the gated delta net stack | built, loads, generates |
| `qwen35-dflash-draft` | the third spelling of the dflash head set (own metadata prefix, poolside's underscore tensor names) | built, loads, drafts the same rows as the canonical set |
| `dflash`, `dflash-draft` | already exercised by `test_dflash_*` — the sweep simply does not see them because they load through `src/dflash.cpp`, not `Model::load` | measured: 10 / 1 / 1 drafter loads |

`build_row_fixture` writes the tensor set the **row's own flags** imply
(`fused_qkv`, `qk_norm`, `qkv_bias`, `attn_gate`, `post_branch_norm`,
`fused_gate_up`, `gpt2_style`, `ln_bias`, `norm_bias`) rather than a hand-written
list per name, so this group checks the row against the loader in both
directions: a row that declares a tensor its family does not carry fails just as
loudly as a loader that cannot read one it does.

Re-measured after the pass: **48 `Yes` rows, 0 without a file** (the three draft
rows show up in the dflash loader's own line).

### Defects this pass found

| ID | P | Item | Closing check |
|----|---|------|---------------|
| D1 | **P0** | `gemma_spec` set `qk_norm` for the whole family, so a real Gemma-1 or Gemma-2 file was refused for `attn_q_norm.weight` — a tensor llama.cpp's gemma.cpp/gemma2.cpp never create. Every Gemma 1/2 checkpoint was rejected by a constant, not by a graph. | per-row flag; `test_arch_table` now asserts true for gemma3/gemma4 and false for gemma/gemma2. Control observed: with `qk_norm=true` restored, the gemma2 fixture is refused with "layer 0 is missing attn_q_norm.weight, which arch 'gemma2' declares" |
| D2 | **P0** | A `deepseek41` file carrying V4.1's `hc_attn_fn`/`hc_attn_base`/`hc_attn_scale` tensors LOADED with `model.load=1` and no blocker recorded — the row promised "refused by name, not mis-loaded", and the loader silently dropped the residual the whole stack runs through. Engram had the same hole (`engram_*` was a note, not a gap). | `kGapTensors` entries for the six Sinkhorn stems and `engram_`; `test_pending_model_fixtures` ds41-hc arm asserts the refusal, and `test_load_verdict`'s Engram control now asserts the shape rule is cleared while Engram refuses. Control observed: before the entries, the same arm printed "verdict runnable but expected the opposite; no blocker was recorded; model.load=1" |
| D3 | **P1** | `phimoe`/`phi3moe` never read the bias on their RMSNorms. llama.cpp's phimoe.cpp creates `attn_norm.bias`, `ffn_norm.bias` and `output_norm.bias` with no OPTIONAL flag and its graph adds them after the norm; this engine read every one of them only under `ArchSpec::ln_bias`, which ALSO selects the LayerNorm forward, so the family could not simply take that flag. Every real Phi-MoE checkpoint loaded and decoded with three biases per layer dropped — fluent, wrong, and silent. | `ArchSpec::norm_bias` + `ModelConfig::norm_bias` upload the three vectors, and `Engine::norm_bias_add` applies them after the RMSNorm at the attention pre-norm, the FFN pre-norm (both the fused add_rmsnorm and the pure-MoE branch) and the head. Measured on a twin pair that differs ONLY in those vectors: **md 0.078218743 at peak 2.34401** after the fix, **md exactly 0** before it (control observed — the first run of this very arm printed `md 0 at peak 2.34401`) |
| D4b | **P1** | `lfm2`/`lfm2moe` never read the bias on their RMSNorms either — llama.cpp's lfm2.cpp:60/68 creates `attn_norm.bias` and `ffn_norm.bias` on every layer with no OPTIONAL flag. Found by auditing `norm_bias` (D3) against every family llama.cpp writes a norm bias for: of the twelve, only `phimoe` and `lfm2` are families this engine runs through the RMS path, and `lfm2` was still missing. | `lfm2_spec(bool moe)` sets `norm_bias` for both rows. Twin pair differing only in those two vectors: **md 0.85720319 at peak 3.45463** after, **md exactly 0** before (control observed — the arm printed `md 0 at peak 3.45463`) |
| D4 | P3 | The vision caveat on `qwen2vl`/`qwen3vl` is the ROW's reason, so a text-only file carries it too. | NOT A DEFECT — conservative and never silent; measured and asserted in the qwen2vl arm so it cannot be mistaken for tensor-level detection later |

## Open

Every row below is CLOSED as of this commit; the closing check names the
measurement that closed it and the negative control that was observed, so a row
cannot read as fixed on the strength of a passing group alone.

| ID | P | Item | Status | Closing check |
|----|---|------|--------|---------------|
| B1b | P2 | isolate the MiniCPM3 head-input multiply numerically (config + finiteness is not the same as "the logits move by exactly 8.0") | **fixed** | the head row dumped on both sides of the multiply on a 4-layer minicpm3 fixture: every layer's head input is its pre-multiply row x scale, worst relative error `5.72205e-06` at peak 18.35 (f32 ulp), and a twin stating a different constant moves the logits |
| B2b | P2 | same for the FFN-side residual scale | **fixed** | FFN-branch A/B on the same fixture: all 4 layers' residual contribution equals `ffn.pre * 0.7`, worst `9.53674e-07` at peak 4.27, and the layer count is reported so a run that measured none cannot read as a pass |
| B3b | P2 | linear (non-YaRN) rope scaling on a windowed half: the factor rides one stack-wide `mc.rope_scale`, so a windowed layer cannot carry its own 1.0 | **fixed** | `ModelConfig::rope_scale_layer` + `Model::rope_scale_at(l)`, resolved in `Model::load` for a family whose windowed layers rotate with plain rope (`ArchSpec::swa_plain_rope`: olmo2, mellum) when the file scales linearly and declares a window; the four engine rope sites now pass the layer's own divisor. YaRN leaves the vector empty (its factor is already in the table), as does any file with no window. `test_windowed_rope_scale` (18 checks): the config arm (layers 0..2 -> 1.0, layer 3 -> 2.0), the two "nothing else changes" twins (no window, YaRN), and the rotation itself -- `attn.q` reproduces a hand rotation at position/1.0 (md 9.5e-7) and NOT at position/2.0 (md 3.76). Negative control observed: with the stack-wide value restored the match inverts (pos/2 md 8.3e-7, pos/1 md 3.76), 16/18 |
| B4 | P2 | a real `gemma4` fixture (per-layer embeddings + shared-KV + two-branch MoE) so the row is graph-tested, not row-tested | **fixed** | `build_gemma4_model` (16-wide, 4 layers, 2 shared-KV, per-layer embedding table, routed two-branch variant) runs end to end in `test_gemma4_forward`; the attention scale is now pinned on the real file rather than on `llama` lookalikes: `mc.attn_scale == 1.0` from the row, and a twin stating 1/sqrt(head_dim) = 0.3536 moves the three-token logits row by **0.3058**. Negative control observed: row constant removed -> both arms fail (42/44) |
| B5 | P2 | forward fixtures for `minicpm`, `minicpm`-v1 scales, `olmoe`, `qwen3moe`, `nemotron_h*`, and the `parallel_residual` graph A/B for gptneox/phi2 | **fixed** | one group per family (7 groups, 49 checks); every A/B moves the logits — minicpm 0.59714, olmoe 0.4228, qwen3moe 0.7258, gptneox 2.11504, phi2 3.46209, nemotron_h 0.0245, nemotron_h_moe 0.0535 — and each collapse to 0 was observed with the behaviour removed |
| B6 | P2 | gemma2/gemma3 **27B** use `1/√(n_embd/n_head)` rather than `1/√head_dim`; only the model type says which, and metadata does not carry it | **fixed** | the size-keyed rule is on the row (`gemma2` 46 layers, `gemma3` 62, 0 everywhere else) and `test_gemma_27b_scale` pins it on fixtures: a 46-layer gemma2 gets `1/√(n_embd/n_head)`, a 62-layer gemma3 the same, and a twin declaring `head_dim` 16 against the same `n_embd/n_head` of 8 separates the two formulas (they coincide in the miniature, which is why the second fixture exists) |
| B7 | P1 | DeepSeek's YaRN `mscale²` attention scale (`attn_factor*(1+0.1*log_mul*ln(factor))`, squared, over √key_length; the file's `yarn_log_multiplier` is divided by 0.1 first) | **fixed** | `test_deepseek_yarn_mscale` on a YaRN fixture: `mscale(factor 40, log_mul 0.1) = 1.36889`, giving `qk = 1.87385x` the plain `1/√key_length`; a twin carrying `log_mul 0.05` moves the logits by `0.101924` at peak 1.57344, and a plain-rope twin by `0.239383` |
| C1 | P1 | `deepseek4`/`deepseek41` compressed attention: the engine still refuses a non-zero `compress_ratio` (`mla_shared_forward_compressed` exists test-only) | **fixed** | end-to-end V4 file with compression runs: a CSA (ratio 4 + indexer) layer and an HCA (ratio 128) layer load and generate, and the same weights with the ratios off differ by md 0.06096 (CSA, 9 tokens) and 0.000433 (HCA, 134 tokens); control observed, dispatch disabled -> md 0 on both |

## Notes

- The reference's `minicpm.cpp` reads `residual_scale` and `logit_scale` into
  hparams and never applies them; matching that is correct, so only the
  embedding scale was a defect there (`f5222f25`).
- `minicpm3.cpp` applies all three (`scale_embd`, `scale_res` on both branches,
  `scale_lmhead = 256/n_embd`), so all three were defects for that row.
- A row that becomes `Yes` needs a fixture before it can be called verified;
  A5 pins flags, not graphs.
- **Negative control, run once on 2026-10-29:** with the two defects put back
  (mellum's `swa_window_first` reverted to false, and the `.attention.scale`
  read removed) `test_family_scales` reports 29/32 -- `is_swa(0)` fails, the
  row pin fails, and "a stated attention scale of 4.0 changes the answer" fails.
  Reverted; the suite is 6566/6566 with the fixes in place. These arms were
  observed to fail, not assumed to.

## BitNet / ternary / 2-bit pass (2026-10-29, later same day)

| id | pri | item | state | evidence |
|---|---|---|---|---|
| D1 | P1 | `I2_S` (ggml id 36) — microsoft/BitNet's own 2-bit format, i.e. every official `ggml-model-i2_s.gguf` — was not in `DType`, so those files were refused as "unknown type 36" | **fixed** | `DType::I2_S`, 128/32 geometry, `deq_i2_s`, fused `dot_i2_s_q8[_x8]`, tensor-tail scale -> `QuantTensor::gs`; `test_i2_s_layout` + `test_i2_s_model` |
| D2 | P1 | TQ2_0/PQ2_0/Q2_0/Q2_0_64 had no fused int8 kernel | **fixed** | `q2code_dot_1/_x8` with code ladder {-1,0,+1,+2}; `test_q2code_dot` |
| D3 | P1 | **`_mm256_sign_epi8(x, v)` is not `x*v` at `x = -128`** (no int8 negation), so the TQ1_0/PTQ1_0 kernels that were already shipping flipped that element's contribution — and the same trick was in the first draft of the 2-bit kernels | **fixed** | exact positive/negative split (`maddubs(+1 where v>0, x) - maddubs(+1 where v<0, x)`); `test_trit_sign_extreme` fails 6/7 before the fix and passes 7/7 after |
| D4 | P1 | `bitnet-b1.58` had **no arch row**: it fell through to the llama defaults, so no SubLN and (in the dense FFN) no squared ReLU anywhere | **fixed** | `ArchSpec::ffn_relu_sqr` + a `bitnet` row + a `dense_ffn` arm; `test_bitnet_family` |
| D5 | P2 | I2_S **embeddings** (BitNet-embedding-0.6B): `token_embd.weight` in I2_S is a lookup table, so its per-tensor scale has no projection to ride | **fixed** by materializing the table at load (scale folded in), which also covers `per_layer_token_embd` and a tied head | `test_i2_s_embedding`; the 1.0-instead-of-0.25 control was observed to fail |
| D7 | P3 | `dot16_signed_lanes` (Q3_K/Q4_K hot path) keeps the `sign_epi8` trick and is exact only while the quantiser clamps activations to [-127, 127] | **pinned** | `test_q8_activation_domain` asserts the clamp; the reason is written at the kernel |
| D6 | P2 | TL1 (38) / the fork's TL2 are not named: 38 falls out as "UNKNOWN (type id 38)" and 42 is resolved as Q2_0 by tensor geometry, so a fork TL2 file would decode as Q2_0 | todo | an explicit by-name refusal |

### Activation of BitNet's FFN — the one deliberate contradiction

llama.cpp's CPU port (`src/models/bitnet.cpp` @ `f5222f25`) builds the FFN with
`LLM_FFN_SILU`; microsoft/BitNet's GPU reference (`gpu/model.py`) computes
`ffn_sub_norm(squared_relu(gate) * up)`. Both are real and they disagree. The
row follows the paper and the reference the weights were trained against
(squared ReLU), recorded here because a later reader will find the llama.cpp
line and think the flag is a mistake.

### RESOLVED 2026-10-29: the synthetic fixtures' logits were scale-blind

Measured while testing I2_S: for the synthetic 2-layer BitNet fixture, the
two-token logits were **bitwise identical** across (a) I2_S weights with
`gs = 0.125`, (b) the same weights as F16 at 0.125, and (c) the same weights as
F16 at 1.0 -- `max|a-b| = max|a-c| = max|b-c| = 0` over all 260 logits, which
made a whole-model logits A/B unable to validate a weight scale in it. That was
recorded with an unverified hypothesis ("the response is dominated by the tied
head and the layer contribution rounds away").

**The cause was the fixture, not the engine.** `build_bitnet_file`'s ternary
cutoff was `t > 0.15f`, and its value generator emits at most
`(11 mod 11 - 5) * 0.03f`: the largest magnitude is `5 * 0.03f`, which rounds to
`0.14999999` -- *below* `0.15f`. So neither branch of the cutoff ever fired and
**every matrix in the fixture was all zero**. The blocks were inert; the logits
came from the tied head over `token_embd` alone, and no weight scale could move
them. The hypothesis is therefore refuted: nothing "rounded away", nothing was
ever there.

Fixed at the fixture (cutoff 0.14, so ~18% of entries are +-1) and the arm now
asserts the two things it could not before:

| pair | measured `max|a-b|` | meaning |
|---|---|---|
| I2_S 0.125 vs F16 0.125 (same weights) | **0.0138** on a peak of 2.03 | the decode is exact, the PATHS are not: a 256-wide I2_S matmul takes the fused int8 kernel, which quantises the activation, while F16 runs f16 math |
| I2_S 0.125 vs F16 1.0 (8x the scale) | **0.0301** | a weight scale now moves the logits by more than the encoding difference, so the fixture can distinguish one |

Consequence kept: any "the scale is applied" claim that rests on a logits
comparison in a synthetic fixture has to show the fixture moves for a change
that is known to be a change. `test_i2_s_model` now carries that control.

### Audit pass over the BitNet/2-bit work (goli-mat-de, before the commit)

| id | pri | class | item | state |
|---|---|---|---|---|
| D8 | P3 | CONFIRMED (code path), MEASURED (range), device-side OPEN | the dense relu^2 arm does not apply `row_scaled`: there is no relu_sqr+mul+row-max kernel, so `relu(gate)^2 * up` is stored unscaled. `test_relu_sqr_row_scale` measures what that costs: on a 256-wide row of ordinary size the product peaks at **2.4e7**, 366x f16's 65504, and every one of those lanes clamps in an f16 store; a power-of-two row scale brings all of them in, moves only the exponent (the stored mantissa is the unscaled one -- pinned, with a non-power-of-two control that fails), and the `ffn_sub_norm` that follows is an RMSNorm, so `norm(x) == norm(x*2^-k)` to md 0. The arm's OUTPUT therefore cannot depend on the guard. | pinned by `test_relu_sqr_row_scale` (7 checks). What is still open is the device store itself: the CPU reference runs f32 activations and cannot saturate, and HIP is not available in this environment, so a kernel is neither needed-not-written here nor verifiable. `[~]` in the routed list for that reason |
| D9 | P2 | CONFIRMED, **fixed** | a *ternary* file whose `general.architecture` is `llama` (the HF1BitLLM Llama3-8B-1.58 and Falcon3-*-1.58bit exports) got SubLN from the tensor set but **silu**, not relu^2, because the activation is keyed on the arch row: half the family's graph. Fixed by a narrow rule in `Model::load` -- generic `llama` name + a SubLN pair on the first block + ternary weights (`dtype_is_ternary`: TQ1_0/TQ2_0/I2_S/Q1_0, NOT Q2_0/PQ2_0 whose ladder reaches +2) -- which sets `ffn_relu_sqr`, exactly as the `bitnet-b1.58` row does. | `test_ternary_llama_arch`: the llama-named file's logits are **bit-identical** to the same weights under `bitnet-b1.58` (md 0); SubLN alone (f16) and ternary alone (no SubLN) both stay silu. Negative control observed: rule disabled -> the flag and the bit-identity both fail (14/16) |

Two integrity guards added while auditing:
- `gguf.cpp` refuses an I2_S tensor whose 32-byte tail falls outside the shard
  (the alternative is reading past the mapping and getting a scale of 4.6e-41,
  i.e. an all-zero model that still runs).
- the loader refuses an I2_S matmul whose inner dimension is not a multiple of
  128: the payload is one flat run, so such a row shares its block with the
  next one and a row-strided read decodes the padding as weights.

## A real GGUF on disk (2026-10-29, third pass)

Unit tests only ever proved the fixtures. This pass writes an actual file with
`tools/make_bitnet_gguf.py` and runs the actual binaries on it.

| check | command | result |
|---|---|---|
| the container is well formed | stock `gguf` Python reader on the F16 twin | 46 tensors, 22 KVs, `bitnet-b1.58` |
| I2_S payload + tail abut exactly | manual parse of the tensor table | 28 I2_S tensors, each `n/4 + 32` bytes, every tensor starting exactly where the previous one ends |
| the arch is recognised | `kraken-inspect` | `arch bitnet-b1.58 -- supported (dense)`, `verdict runnable`, `I2_S 28 tensors 640.0 KiB` |
| it runs | `kraken --prompt "the cat" --max-tokens 12 --greedy` | rc=0, prefill 1411 tok/s, decode 1296 tok/s on 4x256 |
| **the decode is right** | `--prefill-hash --cpu` on I2_S vs the same weights as F16 | **identical**: `prefill-hash 46e6872153233e0e logits 7f4d5428a5edbef0` on a 4x384 file (width chosen so both sides take the dequant path; the codes are +-1 and the scale is a power of two, so every product is exact and the sums run in the same order) |
| the scale is applied | `--prefill-hash` at 0.0625 vs 0.125 | different (`c21ca3e3...` / first token 57 vs `79bab1f7...` / first token 204) |

That last row also closes the "fixtures are scale-blind" question from the
previous PR: on a 4-layer file the whole-model logits DO move with the weight
scale. The blindness was a property of the 2-layer fixture (its logits are
dominated by the tied head), not of the engine.

### New defect found by running a file rather than a fixture

| id | pri | class | item | state |
|---|---|---|---|---|
| D11 | **P1** | CONFIRMED | **I2_S has two dialects.** `convert-hf-to-gguf-bitnet.py` writes code = value + 1 and zero-fills the 28 tail bytes after its f32 scale; `convert-ms-to-gguf-bitnet.py` writes code = value + 2 and permutes the payload into that build's tile layout (`preprocess_weights`, constants from its own `build/kcfg.ini`). Decoding the second as the first is silently wrong -- measured: the ms file loaded with verdict `runnable` and produced `prefill-hash 28eaece7b3a5a08f` against `c21ca3e304853383` for the same weights, with no warning | **fixed**: `i2s_tail_is_ms_converter()` (one predicate, used by the loader AND the verdict) refuses it by name; `test_i2_s_ms_dialect_refused` pins both tails and the three non-signatures (zero, short, unknown) |

## Remaining work — routed 2026-10-29 (branch `arena/7e9ab449-kraken-x`)

Every open item from the two ledgers above, in execution order. `[x]` means the
closing check runs and passes; `[~]` means classified with evidence and left
open on purpose.

- [x] **B7 (P1)** DeepSeek V2/V3 YaRN `mscale^2` attention scale.
      `deepseek_yarn_mscale`/`deepseek_kq_scale` + `ModelConfig::mla_kq_mscale`,
      consumed by `mla_forward`. The old code's argument -- "the reference's
      mscale is built from `rope.freq_scale`, and when that is 1.0 the whole
      correction is 1.0" -- was false: on a YaRN file `rope_freq_scale` IS
      1/factor (llama-model.cpp:1516-1518). `test_deepseek_yarn_mscale` pins the
      closed form against the reference's own chain; `test_mla_yarn_file` pins
      the two keys, the multiplier on the config, and an A/B where two files
      differ **only** in `yarn_log_multiplier` (negative control: maxdiff was
      exactly 0 before the fix, 0.1019 after).
- [x] **B6 (P2)** gemma2/gemma3 27B use `1/sqrt(n_embd/n_head)`, not
      `1/sqrt(head_dim)`. `ArchSpec::attn_scale_embd_per_head_layers` (46 for
      gemma2, 62 for gemma3) resolved in the loader; a stated `.attention.scale`
      still wins. `test_gemma_27b_scale` (14 checks) separates the two formulas
      with a 16-wide-head fixture, pins the off-by-one, and both cross-counts
      (gemma3 at 46 is not gemma3's 27B). Negative control observed: 3 arms
      fail with the rule disabled.
- [x] **B1b/B2b (P2)** isolate the MiniCPM3 scales numerically instead of
      asserting "config read + the run is finite". `head_compute` dumps
      `head.pre` (the normed row) and `head.in` (what the head reads) around
      `lm_head_scale`; `dense_ffn`'s residual-scale arm dumps `ffn.pre` (the
      down projection as computed) and `attn.down` (after the multiply).
      `test_minicpm3_scale_arithmetic` (11 checks) asserts
      `head.in == head.pre * 8` and `attn.down == ffn.pre * 0.7` on all four
      layers to f32 precision, and that the head factor is applied ONCE (a
      `head.pre * 64` arm is rejected). Negative control observed: with both
      multiplies removed the two arms fail (head worst 11.45 vs peak 13.09,
      FFN worst 1.88 vs peak 4.38), 9/11; restored, 11/11.
- [x] **D6 (P2)** TL2 vs Q2_0 on ggml id 42: refuse the fork dialect by name
      instead of decoding it as Q2_0. `Gguf::parse_shard` now resolves the id
      per file from the tensors' own strides (34 B per 128 values = Q2_0,
      18 B per 64 = Q2_0_64, so an upstream-made file declaring 42 is retyped
      rather than mis-strided), tolerates `general.alignment` slack, and
      refuses a file that votes both ways or whose stride fits neither -- that
      refusal names TL2 and both layouts. `test_q2_0_dialect` (19 checks,
      5 files). Negative control observed: with the resolver disabled 8 arms
      fail, 11/19. TL1 (id 38) was already named and refused by
      `test_refused_formats`.
- [x] **D9 (P2)** a ternary file whose `general.architecture` is `llama` gets
      SubLN from its tensors but silu, not relu^2. Fixed in `Model::load`
      (generic name + SubLN pair + `dtype_is_ternary`), which is the narrow
      rule the ledger asked for: either half alone is not evidence.
      `test_ternary_llama_arch` (16 checks) pins the predicate's boundary
      (Q2_0/PQ2_0 excluded), a bit-identical logits match against the
      `bitnet-b1.58` row, and both negatives. Negative control observed: rule
      disabled -> 2 arms fail, 14/16. While building it, the fixture that
      backed the "logits are scale-blind" open item was found to be all-zero
      (its ternary cutoff sat on the generator's largest magnitude); the
      section above records the cause and the new measurement.
- [~] **D8 (P3)** the dense relu^2 arm skips the row-scale range guard.
      Classified, not "fixed": `test_relu_sqr_row_scale` (7 checks) measures
      the product's peak (2.4e7 on a 256-wide row of ordinary size, vs f16's
      65504 -- every lane clamps unscaled), shows a power-of-two row scale
      brings all of them in and moves only the exponent, and shows the
      `ffn_sub_norm` behind it is scale-invariant (md 0), so the guard is a
      range transform and not part of the function. The guard itself still
      needs a relu_sqr+mul+row-max kernel, and the only backend where it can
      matter (f16 activations) is HIP, which is not verifiable here.
- [x] **B3b (P2)** a windowed layer cannot carry its own linear rope scale.
      `ModelConfig::rope_scale_layer` + `Model::rope_scale_at(l)`, resolved for
      `swa_plain_rope` families with a window and a linear factor; the four rope
      call sites consume it. `test_windowed_rope_scale` (18 checks) pins the
      resolution, both "nothing else changes" twins, and the rotation itself
      (hand rotation at pos/1.0 matches to 9.5e-7; pos/2.0 misses by 3.76).
      Negative control observed: stack-wide value restored -> the match
      inverts, 16/18.
- [x] **B4 (P2)** a real gemma4 fixture (per-layer embeddings + shared KV +
      two-branch MoE). Already built; what was missing was A1's evidence on the
      family itself. `test_gemma4_forward` now pins `attn_scale == 1.0` on the
      real file and shows a twin stating 1/sqrt(head_dim) moves the logits by
      0.3058. Negative control observed: row constant removed -> 2 arms fail,
      42/44.
      Fixture trap recorded at the arm: a ONE-token prompt cannot see the
      attention scale at all (softmax over one key cancels it exactly), which
      is why the A/B uses three tokens.
- [x] **B5 (P2)** forward fixtures for minicpm / olmoe / qwen3moe /
      nemotron_h*, and the `parallel_residual` graph A/B. One test group per
      family (`test_minicpm_embedding_scale_forward`,
      `test_olmoe_expert_weight_forward`, `test_qwen3moe_forward`,
      `test_gptneox_forward`, `test_phi2_forward`, `test_nemotron_h_forward`,
      `test_nemotron_h_moe_forward`; 49 checks). Measured: 12x on the minicpm
      embedding vs the same weights as `llama` md 0.59714; olmoe's top-k rule
      forced on md 0.4228; qwen3moe's per-head QK norm off md 0.7258 (the
      `qwen2moe` twin was REJECTED as evidence: the two rows also differ in the
      top-k rule, and with the norm forced off the pair still differs by
      0.5562, so a new one-file `KRK_QK_NORM` hatch replaced it); gptneox
      residual 0 vs 1 md 2.11504, phi2 3.46209; nemotron_h A=0 twin md
      0.0245, nemotron_h_moe 0.0535, and chunk 1 vs 8 md 0 for both. Negative
      controls observed: with the minicpm scale forced to 1.0 md=0, with
      `KRK_EXPERT_W_NORM` ignored md=0, with `KRK_QK_NORM` ignored md=0, with
      `KRK_PARALLEL_RESIDUAL` ignored md=0 on BOTH plain families, and with the
      nemotron twin built at the same A md=0 -- each collapse turns the arm red.
- [x] **C1 (P1)** DeepSeek-V4 compressed attention. The loader reads the
      per-layer `attn_compressor_{wkv,wgate,ape,norm}` tables and, on ratio-4
      layers, the indexer's (`indexer_proj`, `indexer_attn_q_b`,
      `indexer_compressor_*`); `Engine::mla_shared_forward_compressed` keeps the
      two-block state ring, pools a block when its last token arrives (AFTER
      that row's attention, so a pooled key never serves its own block), and
      attends raw + pooled rows through the new
      `Backend::mla_shared_attention_compressed`. Closing check measured on a
      two-layer fixture (layer 0 CSA ratio 4 + indexer, layer 1 HCA ratio 128):
      the file initialises and runs, and the same weights with every ratio 0
      give md 0.06096 at 9 tokens (CSA) and md 0.000433 at 134 tokens (HCA,
      past a completed 128-token block). Negative control observed: with the
      dispatch to the compressed path disabled both differences collapse to
      exactly 0 and the two arms fail. A third arm replays the same prompt twice
      on one engine: md 0 (a pooled key is taken once, when its block ends, so
      without `Engine::compressed_reset()` the second run attends the first
      run's summaries) -- control observed, reset disabled -> md 1.45795. The op is host f32 (the oracle is the
      definition), so it is gated on `mla_shared_compressed_supported()` -- the
      CPU reference answers true, a device backend is refused BY CAPABILITY.

