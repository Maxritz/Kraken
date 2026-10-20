
---

## Reference-engine research (2026-10-20): how others run these stacks

Researched from llama.cpp, SGLang, vLLM and friends for *reference*, not copying —
every plan below is adapted to kraken's own machinery (expert cache, KV tier,
existing `--mtp-spec` loop). Sources cited inline; the researchers' full reports
are in the session transcript.

### DeepSeek-V4 (deepseek4) — confirmed math

- **Shared-KV attention is simple**: `attn_kv` [4096→512] + `attn_kv_a_norm`,
  one 512-wide row used as both K and V for all 64 query heads (head_dim 512 →
  dimension match is exact). RoPE rotates the **tail 64** of the latent
  (`rope_set_offset(head_dim - n_rot)`). This is kraken's existing GQA path with
  n_kv=1, hd=512 — the tiled kernel covers it; no per-head up-projection.
  (llama.cpp `src/models/deepseek4.cpp`, `load_arch_tensors`.)
- **Grouped low-rank output**: `attn_output_a` is 8 groups × (8 heads×512=4096 →
  1024); `attn_output_b` [8192→4096]. New output shape for the engine.
- **CSA/HCA compressors**: per-token `wkv`/`wgate` projections feed rolling KV +
  SCORE states (F32 rows, position-indexed). Compression = per-channel softmax
  over the block's scores × values, sum, RMS norm, then compression-position
  RoPE (base 160000, yarn). CSA ratio 4 uses **overlapping windows** (prev 4 +
  current 4 rows, 2×head-width state); HCA ratio 128 compresses 128 rows per
  block. Scores get an APE term at `pos % ratio`.
- **Lightning indexer** (top-512): `indexer_attn_q_b` from the q-lora row,
  rope'd indexer queries; keys live in a **separate LID cache**; score =
  `sum_heads(relu(q·k)) × weights`, + mask, top-k → block-expanded KQ mask over
  compressed + raw window keys. Dense until `top_k + ratio − 1` tokens cached.
- **MoE scoring is sqrt-softplus** (`expert_gating_func=4`), with
  `expert_weights_scale=1.5` + `expert_weights_norm`; first 3 layers route by
  **token-id hash** (`ffn_gate_tid2eid` [6×129280], the type-#26 I32 tensors),
  remaining layers carry `exp_probs_b` bias. Gate logits clamp (swiglu 10.0).
- **Hyper-connections**: hc_mult 4, full-rank mix [16384→24], scale[3]/base[24]
  affine → sigmoid pre (×(1+eps)), post (×2), comb → **Sinkhorn 20 iterations**
  (softmax over dst, eps added, alternating col/row normalization), plus a final
  `hc_head` mixer before the LM head. n_embd_out = 4×4096.
- **Yarn rope** (factor 16, orig 65536, beta 32/1) — engine has no yarn yet.

### Qwen3.8-Flash-Next (qwen4exp) — confirmed math

- **QSA**: indexer keys are **mean-pooled raw keys** (pool before norm/rope) in
  a third cache sharing the attention cache's slot layout; score =
  Σ_heads relu(q·k); select blocks, expand to token positions; dense below
  `top_k + ratio − 1` (2051) tokens. (llama.cpp PR #27742.)
- **PLE**: host-side splitmix64 hash of (token, prev ngram−1) → I32 row indices →
  `get_rows` on the giant table; EOS cuts history strictly-before; a depthwise
  conv over PLE output is recurrent state (chunk-invariant when carried).
- **HC here ≠ DeepSeek's**: low-rank down(320) → SiLU → up → sigmoid gate →
  mean collapse, with 1/hc_count mix scale and ×2 combine scale; the final
  mixer's norm replaces output_norm. Do not reuse the sinkhorn mixer.
- **attn_qkv** packs [q(6144) | gate(6144) | k(512) | v(512)]; gate is
  **sigmoid**, per head, applied after attention, before wo. Rope is interleaved
  MRoPE with 64 rotary dims of 256 (sections [11,11,10,0] = 32 pairs).
- GDN layers use a sigmoid output gate here (Qwen3.5 used SiLU).

### Speculative decoding for these stacks

- Outer loop matches kraken's `--mtp-spec` (draft chain → batched verify →
  keep prefix → commit accepted state). SGLang recipes: DeepSeek V3/V4
  **2–4 draft tokens, top-k 1**; Qwen3.8 NEXTN 3/1/4 with **acceptance length
  3.3 including the bonus token** — always report bonus-inclusive length, plus
  proposed/accepted counts and end-to-end tok/s.
- **Recurrent rollback**: SGLang's documented options are per-step state cache
  (memory ×draft length) vs **ReplaySSM** (store draft inputs, replay accepted
  prefix — 114 MB → 26 MB per request in their Qwen3.5 numbers). Kraken's
  snapshot+replay (`rec_state_` copy) is the Replay path already; the
  improvement to steal: skip the full 63 MiB copy by keeping the pre-round state
  and folding accepted inputs forward.
- V4 caution (SGLang launch): hybrid-attention speculative writes go into ring
  buffers **before acceptance is known** — size rings with headroom so a reject
  cannot clobber the live window. Kraken's KV tier needs the same check.
- EAGLE-3 is a separate trained drafter (feature fusion, tree drafts,
  llama.cpp PR #18039: 28–93% acceptance, 0.89–3.28× — MoE can lose). Not
  drop-in; only worth it if a checkpoint ships the head.
- llama.cpp Qwen3-Next MTP: 83–97% draft acceptance at `--spec-draft-n-max 2`;
  draft count must come from checkpoint metadata, not assumptions.

### Consequence for the audit order

Pieces 2–6 of the DeepSeek plan stand, now with exact formulas. The qwen4exp
plan gains: sigmoid (not softplus) GDN gate check, MRoPE, ReplaySSM-style state
folding, and the "HC mixer is a different function" warning. The type-#26
refusal is the `ffn_gate_tid2eid` I32 hash table (host-side routing data, never
a GPU weight).
