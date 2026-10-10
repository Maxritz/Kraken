# What MLA support would cost in kraken, and what it would buy

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
