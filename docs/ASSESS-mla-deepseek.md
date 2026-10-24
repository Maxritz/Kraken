# What MLA support would cost in kraken, and what it would buy

> **Status 2026-10-26 -- deepseek4 RUNS, dense core only, and the arch row is
> `Partial`.** `Engine::mla_shared_forward` wires the oracle's graph into the
> layer loop (latent -> `mla_shared_latent` -> query trunk -> `mla_shared_attention`
> -> `mla_shared_output` straight into the residual's buffer, so the ordinary
> `attn_output` projection is skipped for this layout), `MlaLayout::Shared` is no
> longer refused at init, and `kv_dim_at` sizes V4's cache row as the latent
> ALONE -- the rope is its tail, and the lite rule's `+ qk_rope` had every plane
> 64 columns wide. `test_dsv4_whole_graph` (21 checks) pins the fixture end to
> end: the whole logits row by FNV hash, its argmax, the greedy sequence, a second
> prompt, chunk 1 vs 8, a shifted hash table, and an out-of-range id that must
> refuse. Two wiring defects it caught: the hash route was indexed by ROW instead
> of by vocab id (chunk 1 and chunk 8 disagreed), and the red arm that rotates the
> leading slice instead of the tail fails 3 of the 21 while `test_mla_shared`
> stays 109/109 -- the per-piece gate compares two implementations that share the
> premise, so only the sentence can see it. Refused by construction, not by luck:
> a non-zero `attention.compress_ratios` (the compressors are absent), any
> `attn_sinks`, and the Split layout. The indexer, the compressors and sinks are
> what still separate this from `Yes` on a real V4 file.
>
> **Historical status 2026-10-25 -- the V4 SHARED-LATENT attention has a host oracle and a
> gate; the graph still does not run.** `include/krk/mla_shared.hpp` +
> `src/mla_shared.cpp` implement one V4 layer of the read path: q_a -> q_a_norm
> -> q_b, the single `attn_kv` latent (normed whole, rope on the TAIL),
> sink-in-denominator softmax, the positional window, and the grouped low-rank
> output (per-group `wo_a` column blocks -> `wo_b`). Two orders -- per-key
> accumulation and weights-then-gather -- and `test_mla_shared` pins them:
> **109 checks**, including the one-key reduction (softmax 1 means the head
> output IS the latent, which pins the q path, the latent norm and the grouped
> output at once), the rope tail offset, a saturating sink taking all the mass,
> group isolation (zero group 0's block and rewire heads 0..hpg-1: the answer
> must not move), and the window excluding keys by POSITION. Red construction:
> rotating the leading slice instead of the tail and always reading output group
> 0 fails **34** of the 109, and reverting returns to 109/109. Suite total
> 3833 (was 3724). The oracle's ONE unconfirmable assumption is stated in the
> header: V reads the UNROTATED latent (one tensor feeds K and V, so no file on
> this box can settle it; it is one function, `build_scores`, if a reference
> comparison says otherwise).
>
> **Also landed: the hash route (same session).** `include/krk/dsv4_route.hpp` +
> `src/dsv4_route.cpp`: `dsv4_hash_route` resolves the first
> `hash_layer_count` layers from `ffn_gate_tid2eid.weight` -- TOKEN-major
> (`tid2eid[token * per_token + slot]`), every id range-checked, nothing
> written on a refusal, duplicates preserved because the table is authority --
> and `dsv4_hash_gather` reads the selected ids' router scores in TABLE order.
> `test_dsv4_hash_route` is **25 checks**; red construction (a column-major
> read, plus a gather that pairs each weight with the neighbouring id) fails
> **10 of 25**. Suite total 3858. Two things this deliberately does not decide:
> the mixing weights stay with the engine's existing gating path (hash layers
> still carry `ffn_gate_inp`, and the file names the transform), and the engine
> call site is NOT wired, because `deepseek4` refuses at `Engine::init` today
> and an unreachable call would read as a live gate.
>
> **Why the arch row is still `No`:** the oracle is the *dense* attention of the
> three non-compressed layers. Still missing, in dependency order: the CSA/HCA
> compressors (`attn_compressor_{ape,gate,kv,norm}` -- not read at all today),
> the lightning indexer (`indexer_*`, top-512 sparse selection), the `tid2eid`
> hash route (loaded, never used by the forward), the V4 Sinkhorn hyper-connector
> (`hc_*` is wired for qwen4exp's DIFFERENT mixer), YaRN rope (absent from the
> engine), wiring the oracle into the CPU backend and `Engine::mla_forward`, then
> the HIP kernels. The REAP-160 file also cannot be used as its own oracle:
> both llama.cpp builds on this box refuse it (`ffn_gate_inp.weight` expects 256
> experts, the file is pruned to 160 per `reap.layer.expert_count`).

> **Status 2026-10-22 (later) -- the legacy lite layout RUNS.** A 576-wide cache
> row (kv_lora 512 + the decoupled rope key 64), the absorbed score reduction
> and the rope-on-the-decoupled-key path are implemented twice: on the device
> (`src/hip/kernels/mla.hpp`, `Backend::mla_kv_write` / `mla_attention`) and as
> the scalar reference in `src/backend_cpu.cpp`. `Engine::mla_forward` wires
> them into the layer loop and `Model::kv_dim_at` / `layer_has_kv` give the KV
> tier the row it needs. Measured on `DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M`
> (the local V2-Lite file): loads, and greedy text matches llama.cpp's ROCm
> build token-for-token over 64 tokens (` calculate_salary(name, salary,
> bonus):` ... `def k(x`), device and CPU both.
>
> Two semantic bugs had to be fixed for that, and neither was in the kernel:
> `expert_weights_norm` is FALSE for this family (the top-6 of a 64-way softmax
> sum well under 1, and renormalizing them multiplied every layer's expert
> contribution by ~2 -- the arch row now says so, like gpt-oss), and the
> attention scale is plain `1/sqrt(key_length)` on this file even though
> llama.cpp's deepseek2 reads `mscale^2/sqrt(...)`: with the mscale factor the
> first greedy token already differed, without it the run agrees to 64 tokens.
>
> Gates: `kraken-bench --mla` runs the layer at the V2-Lite geometry against the
> `src/mla.cpp` oracle (rows 1/2/4/8/17, a nonzero position, and a TRANSPOSED
> table as the negative control that must fail -- 13 of 13 checks),
> `test_mla_backend` pins the CPU reference against the same oracle plus head
> isolation and 1-row-vs-n-row identity (10 checks), and `test_mla_loader`
> (80) covers the layouts. Still refused BY NAME at `Engine::init`: the split
> (V3 class) and shared-latent (V4) layouts, whose graphs are different
> computations -- and `deepseek4` keeps `attention_implemented = false`.
>
> **Historical status 2026-10-22 (earlier) -- the loader reads all three
> layouts, the engine still refused to generate.** The ranks come from the keys the exporters actually
> write (`key_length` / `value_length` / `kv_lora_rank`, with nope =
> `key_length - rope.dimension_count`; the `qk_nope_head_dim` names the
> estimate below implies are written by nobody and refused every real file).
> The layout is now decided by the TENSORS a layer carries -- `Legacy`
> (`attn_q` / `attn_kv_a_mqa` / `attn_kv_b`, V2-Lite), `Split`
> (`attn_q_a` / `attn_q_b` / `attn_kv_a` / `attn_k_b` / `attn_v_b`, the V3
> class) and `Shared` (`attn_q_a` / `attn_q_b` / `attn_kv` /
> `attn_output_a` / `attn_output_b`, V4) -- and every layer has to agree on
> it. Each layout's geometry is checked against the file's own metadata, and
> V4's metadata (grouped output, window, hash depth, compress_ratios) and
> hash route are read. `test_mla_loader` (66 checks) pins the three layouts,
> the mix, the geometry refusals and the engine's refusal; `test_mla_lite`
> (57) pins the host oracle in two orders. So item 1 of the estimate is done
> in the shape the files actually have; items 2-4 (latent decode kernel,
> latent prefill, coherence fixture) are open, and the arch verdict is
> unchanged: a run is refused AFTER the load by
> `ArchSpec::attention_implemented`, which `KRK_ARCH_ALLOW_EXPERIMENTAL=1`
> deliberately does not lift. V4's quant type #26 turned out not to be a
> blocker at all -- it is I32, and `ffn_gate_tid2eid` is a table of expert
> ids read as data. What is still unread on V4: `attn_sinks`, Sinkhorn
> `hc_*`, the per-layer compressor tensors and the QSA indexer.

> 2026-10-21: DeepSeek open-sourced DeepInfer, a serving runtime for
> DeepSeek-V4.1-Flash whose design implements exactly this path (absorbed
> latent KV, sparse indexer, FP4 cache). `docs/AUDIT-deepinfer.md` reads it
> against this file's estimate and splits what is portable from what is
> SM100-only.

Assessed 2026-10-10 from the tree, not from the papers. MLA (multi-head
latent attention, DeepSeek-V2/V3) is refused today by the tensor-gap map:
`attn_kv_a` / `attn_kv_b` / `attn_kv` name "latent KV compression" in
`src/arch.cpp:310-312`, so a file carrying them is refused with that
reason whether or not its arch string was ever added to the table. There
is no `deepseek` entry in `include/krk/arch.hpp`.

## What MLA changes, in kraken's terms

Per token per KV-carrying layer, a standard-GQA model stores per-head K
and V (`n_kv_heads x head_dim x 2`); MLA stores one compressed latent
`c_KV` (512 dims for V3, 4096->512 compression via `attn_kv_a`) plus a
small decoupled RoPE key `k^R` (64 dims), and the per-head V projection
is absorbed at load into `W_UV` (512 -> 128 per head). Scores become
`q_c_h . c_KV + q_r_h . k^R` (512 + 64 dims per head instead of a
per-head K row), and the output is the softmax-weighted `c_KV` sum
projected through the absorbed `W_UV`. Less cache traffic (576 dims
against V3-GQA's 2048), more flops per token, no per-head KV cache at
all.

## The pieces kraken already has

* **The KV tier is ready.** `KvTierCache` was generalized to per-layer
  row widths (`Model::kv_dim_at(l)` / `layer_has_kv(l)`, the F2-F4 work),
  so an MLA layer is one more width: 576 floats (V3) instead of
  `n_heads x head_dim x 2`. HOT/WARM/COLD paging, the plane tags and the
  admission rules are width-agnostic. This is the piece that would have
  been a redesign before that work landed.
* **MoE is ready** (V3's routed experts are the same top-k shape as the
  256-expert models already run), as is the mixed-per-layer-geometry KV
  accounting the Nemotron fixture exercises.

## The pieces that are new work

1. **Loader + arch entry**: a `deepseek` (llama.cpp spells it
   `deepseek2`/`deepseek3`) `ArchSpec` row, the `attn_kv_a`/`attn_kv_b`/
   `attn_q_b`/`attn_output_b` tensor classes, qk-norm and rope-64-dim
   conventions, and the absorb at load (fold `W_UV` per head). ~2 days;
   the arch table and the gap map were designed for exactly this.
2. **A latent-space attention decode kernel**: one reduction per head
   over 512+64 dims with the absorbed output projection in the epilogue.
   The split-kernel shape in `src/hip/kernels/attention.hpp` carries
   over (the score reduction is the same softmax machinery on a wider
   row), but the row is one wide "head" per token, so this is a new
   kernel, not a flag. ~3-5 days.
3. **A prefill path** (rows > 1, WMMA-capable): absorbed `W_UK`/`W_UV`
   make the GEMM shapes different from every dense/MoE prefill today.
   ~2-3 days.
4. **Tests + coherence fixture** on a real MLA file. ~2 days.

Rough total: **1.5-2 weeks** of focused work, most of it in the two
attention paths.

## What it buys on THIS box — the honest answer

Every DeepSeek-V3-class target is 671B total parameters; even Q2 is
~170 GB and this box has 15.9 GiB VRAM + 96 GB RAM, so the flagship
models do not fit here at any quant. The MLA file that DOES fit is
**DeepSeek-V2-Lite (16B-A2.4B, MLA + MoE, Q4 ~9-10 GB)**, which is the
realistic testbed: it exercises the latent cache, the absorbed kernels
and the MoE path at a size this machine runs. DeepSeek distills
(R1-Distill-Qwen, plain Qwen arch) already load — no MLA involved.

So MLA support here buys: a correctness/performance reference on
V2-Lite, and the infrastructure for boxes with more memory. It is NOT
the blocker for "DFlash on DeepSeek" that it first appears to be — the
blockers stack as: MLA (targets) -> DFlash2 head (conv+selector,
see DFLASH-DRAFTER.md) -> either one alone is a week, both is a project.

## Recommendation

Do it only with V2-Lite as the standing fixture and with the absorb
decided up front (decode without absorption is 10x the cache traffic
and is not worth shipping). Otherwise the right near-term DeepSeek move
is the one already ranked in DFLASH-DRAFTER.md: DFlash2 support against
a Qwen3.8 target, which is the same drafter family and a target this
box already runs.
