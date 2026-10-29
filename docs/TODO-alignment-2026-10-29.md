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

## Open

| ID | P | Item | Status | Closing check |
|----|---|------|--------|---------------|
| B1b | P2 | isolate the MiniCPM3 head-input multiply numerically (config + finiteness is not the same as "the logits move by exactly 8.0") | todo | two files of one arch that differ only in the constant |
| B2b | P2 | same for the FFN-side residual scale | todo | FFN-branch A/B on the minicpm3 fixture |
| B3b | P2 | linear (non-YaRN) rope scaling on a windowed half: the factor rides one stack-wide `mc.rope_scale`, so a windowed layer cannot carry its own 1.0 | todo | needs a per-layer rope scale; no family in the table pairs linear scaling with a window today |
| B4 | P2 | a real `gemma4` fixture (per-layer embeddings + shared-KV + two-branch MoE) so the row is graph-tested, not row-tested | todo | `test_gemma4_*` runs the file end to end and pins the 1.0 attention scale |
| B5 | P2 | forward fixtures for `minicpm`, `minicpm`-v1 scales, `olmoe`, `qwen3moe`, `nemotron_h*`, and the `parallel_residual` graph A/B for gptneox/phi2 | todo | one group per family; the A/B moves the logits |
| B6 | P2 | gemma2/gemma3 **27B** use `1/√(n_embd/n_head)` rather than `1/√head_dim`; only the model type says which, and metadata does not carry it | todo | either a size-keyed rule with a fixture, or a documented limitation |
| B7 | P1 | DeepSeek's YaRN `mscale²` attention scale (`attn_factor*(1+0.1*log_mul*ln(factor))`, squared, over √key_length; the file's `yarn_log_multiplier` is divided by 0.1 first). The inputs are read (`rope_yarn_factor`/`rope_yarn_log_mul`) and not applied | todo | a deepseek2/V4 YaRN fixture whose scores match the formula |
| C1 | P1 | `deepseek4`/`deepseek41` compressed attention: the engine still refuses a non-zero `compress_ratio` (`mla_shared_forward_compressed` exists test-only) | **blocked** (a feature, not a defect; the rows are `Partial`) | end-to-end V4 file with compression runs |

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
