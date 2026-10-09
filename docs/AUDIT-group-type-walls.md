# AUDIT: which projection sets a mixed quantisation still splits

`Backend::gemm_group` fuses a set of projections that share one activation into a
single launch. `FusedLayer` now carries a type per matrix and `gemm_group` takes a
nullable `wts[]`, but three call sites still guard on every matrix having the
**same** weight type -- the group used to carry one `wt`. The question this audit
answers: which checkpoints actually trip those guards today, on how many layers,
and what do they pay for it.

Tool: `python3 tools/projection_type_audit.py [model.gguf ...]` (default: every
`*.gguf` under `models/`). It reads the tensor table from **kraken-inspect** --
the loader's own view of the file, not a re-implemented parser -- evaluates each
guard **per layer**, and loads no weights.

## The sites

| set | site | guard | status |
|---|---|---|---|
| attention q/k/v | `src/engine.cpp:918` | `wq.type == wk.type == wv.type` | live cost |
| FFN gate/up | `src/engine.cpp:1154` | `wgate.type == wup.type` | live cost (group also carries the silu) |
| drafter q/k/v | `src/dflash.cpp:799` | same as attention | only when that file is the `--draft` model |
| drafter gate/up | `src/dflash.cpp:835` | same as FFN | only when that file is the `--draft` model |
| GDN qkv/z/a/b | `src/engine.cpp` (`gdn_forward`) | none -- already passes `wts[]` | fused, mixed types expressed |

The guard is per layer, which is what a file-level type histogram hides. The
llama.cpp recipes bump `attn_v` on a **subset** of layers, so these checkpoints
fuse where the bump is absent and split where it lands.

## What the four models on this box do

| model | set | layers | types | verdict |
|---|---|---|---|---|
| SmolLM2-135M **Q4_K_M** | attn q/k/v | **14 of 30** | q=k=Q5_0, v=Q8_0 | **SPLIT, +28 launches/step** |
| SmolLM2-135M **Q4_K_M** | FFN gate/up | 30 of 30 | Q5_0 both | fused |
| SmolLM2-135M **IQ4_XS** | all four sets | uniform | IQ4_NL | fused |
| Qwen3.5-0.8B **Q4_K_M** | attn q/k/v | **4 of 6** | q=k=Q4_K, v=Q6_K | **SPLIT, +8 launches/step** |
| Qwen3.5-0.8B **Q4_K_M** | FFN gate/up | 24 of 24 | Q4_K both | fused |
| Qwen3.5-0.8B **Q4_K_M** | GDN qkv/z/a/b | 18 of 18 | Q5_K/Q4_K/Q4_K/Q4_K | **fused** -- `wts[]` expresses it |
| Qwen3-MoE-4x0.6B **Q4_K_M** | attn q/k/v | **14 of 28** | q=k=Q4_K, v=Q6_K | **SPLIT, +28 launches/step** |

The split layers are not contiguous -- the recipe protects the early ones and
spreads the rest:

| model | layers whose `attn_v` differs from `attn_q`/`attn_k` |
|---|---|
| SmolLM2-135M Q4_K_M | 0, 1, 2, 5, 8, 11, 14, 17, 20, 23, 26, 27, 28, 29 (of 30) |
| Qwen3-MoE-4x0.6B Q4_K_M | 0, 1, 2, 5, 8, 11, 14, 17, 20, 23, 24, 25, 26, 27 (of 28) |
| Qwen3.5-0.8B Q4_K_M | 3, 7, 11, 23 (of its 6 attention-carrying layers) |

The reported per-layer types are the guard's own inputs, so the verdict is a
restatement of the site's condition rather than a model of it -- which is why the
run below matters.

## The run agrees with the table

`KRK_PROFILE=1` distorts time by ~10x, not order or counts, so the op *counts* in
one decode step are ground truth for which path the engine took:

| model | ops/step | `gemm_group` | predicted | `gemm(gemv)` | predicted |
|---|---|---|---|---|---|
| SmolLM2-135M Q4_K_M | 243 | **46** | 16 fused qkv + 30 FFN | **73** | 14x3 split + 30 o_proj + 1 |
| Qwen3.5-0.8B Q4_K_M | 397 | **44** | 2 fused qkv + 18 GDN + 24 FFN | **19** | 12 split + 6 o_proj + 1 |
| Qwen3-MoE-4x0.6B Q4_K_M | 817 | -- | -- | **267** | (per-expert path dominates) |

Both columns close exactly, and the trace's first layer of SmolLM2 shows the three
separate `gemm(gemv)` calls the table predicts for `blk.0` (`attn_v=Q8_0`) before
`attn_fused`. The GDN row on Qwen3.5 -- 18 group calls on a layer set whose four
matrices are three different types -- is the same measurement for the fix that
already landed: mixed types travel through `wts[]`, and no guard is needed.

## What removing the guards would buy (estimated, not measured)

The fix at the three live sites is mechanical: build a `wts[]` from the layer's
own `wq/wk/wv` (or `wgate/wup`) types and drop the equality test. There is no
per-matrix **scale** to carry: only the MoE expert and shared-expert tensors have
`.scale` sidecars (`src/model.cpp:1073-1124`), so for q/k/v and dense gate/up the
dtype is the whole question.

| model | launches/step saved | share of ops | est. decode gain |
|---|---|---|---|
| SmolLM2-135M Q4_K_M | 28 of 243 | 11.5% | ~2.6% at 1.5 us/launch, ~5.4% at 3.1 |
| Qwen3-MoE-4x0.6B Q4_K_M | 28 of 817 | 3.4% | ~0.4-1.0% |
| Qwen3.5-0.8B Q4_K_M | 8 of 397 | 2.0% | ~0.4-0.9% |

The band is this repo's measured marginal launch cost -- 1.5 us from SmolLM2's own
residual fold (30 launches, +2.867%) up to 3.1 us from the GDN fold (54 launches,
+6.084%) -- applied to a count, so treat it as a ranking, not a result. SmolLM2
Q4_K_M is the one worth doing first: it is the largest share of a step and the
band's lower end comes from that same model.

## Not a type wall, and bigger than one

Two more sets share an activation and were ungrouped for reasons that have nothing
to do with dtypes, so `wts[]` did not enter into either -- one is folded now, the
other is latent:

- **MoE per-expert up/gate -- FOLDED, see `sherlock_report.md` section 11.** Both
  are Q4_K in `Qwen3-MoE-4x0.6B-Q4_K_M`, and the step paid 56 `gemm(gemv)` + 56
  `silu_mul` for 56 expert visits: 3 launches per visit where the FFN group's
  shape (2 matrices, one input, silu in the epilogue) needs 1 -- ~112 of 817
  ops/step, ~13.7% of the launches, on a model whose routed rows are 1 in decode,
  exactly the shape the group's fused path takes. Measured after the fold:
  **817 -> 705 ops/step** with all 56 visits folded, A/B **order-balanced median
  +7.835%** (8/9 pairs), stdout md5 unchanged and equal to this repo's recorded
  acceptance hash. Prefill falls back per-matrix, which is correct but not faster.
- **MoE shared-expert gate/up** (`src/engine.cpp:2861-2896`): 2 gemms + a
  `silu_mul`, and the only place where the pair is *not* groupable as written --
  when the two sidecar scales differ the input is rescaled *between* the two
  gemms, which a one-input group cannot express. Equal scales (including both 1.0
  on every non-NVFP4 file) are fine. No model on this box carries
  `ffn_*_shexp` tensors, so this one is latent.

## Limitations

- Four models. The box has SmolLM2-135M x2, Qwen3.5-0.8B and Qwen3-MoE-4x0.6B; a
  wider sample needs the other model dirs (`G:/More-models/`).
- The drafter rows describe the file *if it were loaded as the `--draft` model*;
  a real draft setup loads a different file, so those rows are conditional and
  are excluded from the totals.
- The tool's per-layer verdict is a restatement of the guard's own predicate, so
  it inherits the guard: if a site's condition changes, the tool's `SETS` table
  has to change with it. The cross-check above (predicted counts vs the op table)
  is what keeps the restatement honest.
- Gains in the third section are estimates from a launch-count band. Only an A/B
  on the same binary pair, interleaved, with stdout md5 unchanged, settles one.
