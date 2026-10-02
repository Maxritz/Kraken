# LLaDA (diffusion language models) — research and plan

This is the working note for the `diffuse` architecture: LLaDA2.2-flash in the
model folders (`G:/More-models/dllm/llada/`), the LLaDA family in general, and
what in kraken already carries it. Everything below was read off the file itself
(`tools/gguf_keys.py`, `kraken-inspect`) and off the reference implementation
(ML-GSAI/LLaDA `generate.py`, InclusionAI LLaDA2.X, SGLang's dLLM page) — not
from memory.

## What the model is

A **masked discrete diffusion** language model (dLLM): the sequence starts as
`[prompt | MASK × gen_length]` and is denoised in parallel — each step the model
sees the whole sequence at once and commits the tokens it is most confident
about. There is no left-to-right order and **no causal mask**: attention is
bidirectional, so a token can depend on tokens after it (that is what lets a
whole block be predicted at once). Generation is not a decode loop with a KV
cache; it is a repeated full-sequence forward with the committed tokens fixed.

LLaDA2.x adds: a **block** schedule (`block_length`, semi-autoregressive — a
block is denoised to completion before the next one starts), **Block Routing**
(MoE expert activation bounded at the diffusion-block level, `moe_block_size`),
group-limited expert routing, and — in 2.1/2.2 — token *editing* (Levenshtein
edits with dedicated DELETE/INSERT tokens).

## The file's own schema (`LLaDA2.2-flash-Q4_K_S.gguf`)

Metadata, verbatim:

| key | value |
| --- | --- |
| `general.architecture` | `diffuse` |
| `diffuse.model_type` | `llada2_moe` |
| `diffuse.block_count` | 32 |
| `diffuse.embedding_length` | 4096 |
| `diffuse.attention.head_count` | 32 |
| `diffuse.attention.head_count_kv` | 4 |
| `diffuse.head_dim` | 128 |
| `diffuse.rotary_dim` | 64 (partial RoPE) |
| `diffuse.rope.freq_base` | 3e6 |
| `diffuse.attention.layer_norm_rms_epsilon` | 1e-6 |
| `diffuse.use_qk_norm` | 1 |
| `diffuse.context_length` | 131072 |
| `diffuse.vocab_size` | 157184 |
| `diffuse.mask_token_id` | 156895 |
| `diffuse.eos_token_id` | 156892 |
| `diffuse.delete_token_id` | 156930 |
| `diffuse.split_token_id` | 156931 |
| `diffuse.block_length` | 32 |
| `diffuse.moe_block_size` | 32 |
| `diffuse.expert_count` | 256 |
| `diffuse.expert_used_count` | 8 |
| `diffuse.expert_shared_count` | 1 |
| `diffuse.expert_feed_forward_length` | 1024 |
| `diffuse.first_k_dense_replace` | 1 (layer 0 is dense) |
| `diffuse.n_group` / `topk_group` | 8 / 4 |
| `diffuse.norm_topk_prob` | 1 |
| `diffuse.routed_scaling_factor` | 2.5 |
| `diffuse.feed_forward_length` | 9216 (the dense layer's FFN) |

Tensors (dims are ggml `[in, out]`):

| tensor | dims | notes |
| --- | --- | --- |
| `blk.N.attn_norm.weight` | [4096] | pre-attention RMSNorm |
| `blk.N.attn_qkv.weight` | [4096, 5120] | **fused** q\|k\|v; 4096 + 512 + 512 rows (q first) |
| `blk.N.attn_q_norm.weight` | [128] | per-head QK-norm |
| `blk.N.attn_k_norm.weight` | [128] | per-head QK-norm |
| `blk.N.attn_output.weight` | [4096, 4096] | |
| `blk.N.post_attn_norm.weight` | [4096] | this is the **pre-FFN** norm |
| `blk.0.ffn_{gate,up,down}.weight` | [4096,9216] / [9216,4096] | dense layer 0 |
| `blk.N.moe_gate.weight` (+ `.bias`) | [4096,256] + [256] | router and its routing bias |
| `blk.N.moe_experts_{gate,up,down}.weight` | [4096,1024,256] / [1024,4096,256] | expert count in the **last** dim |
| `blk.N.moe_shared_{gate,up,down}.weight` | [4096,1024] / [1024,4096] | one shared expert, dense |

Tokenizer is `gpt2`/BPE; `add_bos_token` and `add_eos_token` are false.

## The reference decoding loop (ML-GSAI/LLaDA `generate.py`)

```
x = [prompt | mask_id * gen_length]
num_blocks   = gen_length // block_length
steps_block  = steps // num_blocks
for block in range(num_blocks):
    transfer = get_num_transfer_tokens(masks_in_block, steps_block)   # base + remainder spread
    for step in range(steps_block):
        logits = model(x)                      # full sequence, BIDIRECTIONAL
        x0     = argmax(logits)                # per position
        p      = softmax(logits)
        conf   = p[x0]
        conf[:, beyond this block] = -inf      # candidates are this block only
        x0     = where(is_mask(x), x0, x)      # never overwrite committed tokens
        conf   = where(is_mask(x), conf, -inf) # ...so candidates are masked positions only
        sel    = topk(conf, transfer[step])
        x[sel] = x0[sel]
```

So: exactly `transfer[step]` masked positions are committed per step, chosen by
confidence; `remasking='low_confidence'` refers to *which masked positions* get
committed, not to re-masking committed ones (the base loop never re-masks —
`confidence` is `-inf` off the mask). LLaDA2.2's Levenshtein editing and
SGLang's threshold variants (M2T threshold, then a T2T edit phase with
`max_post_edit_steps`) are refinements on top of this; the block schedule and the
confidence selection are the load-bearing parts.

## What kraken already has

* **Bidirectional attention is already implemented, in both backends.** The
  attention op takes `AttnDesc::causal`, and both the CPU and the HIP prefill
  kernels compute `n_keys = causal ? pos + 1 : pos0 + n_tok`
  (`src/backend_cpu.cpp`, `src/hip/kernels/attention.hpp`). Nothing upstream
  passes `causal = false` today — a diffusion forward is exactly that call with
  the whole sequence as one chunk.
* **MoE**: routed experts with `[n_embd, n_ff_exp, n_expert]` layout, lazily
  paged in through `ExpertCache`, grouped GEMMs per expert, shared experts,
  router on the host (`src/engine.cpp`, `moe_ffn`). This is the right machinery
  for the `moe_*` tensors; the routing *rule* is what differs (see below).
* **A GQA attention core with per-head QK-norm, partial RoPE and RMSNorm**, and
  a tokenizer that already reads `gpt2`/BPE with byte fallback.
* **The expert-cache budget** (`--expert-cache-mb`), which is what makes a 59 GB
  file discussable on a 16 GB card at all.

## What is missing (in the order it has to be built)

1. **Loader schema for `diffuse`** — read `diffuse.*` keys (`mask_token_id`,
   `block_length`, `first_k_dense_replace`, `n_group`, `topk_group`,
   `norm_topk_prob`, `routed_scaling_factor`, `use_qk_norm`, `head_dim`), split
   the fused `attn_qkv` by output rows into the existing q/k/v slots (row slicing
   of a quantized matrix is `dtype_row_bytes` arithmetic, no dequantize), map
   `post_attn_norm` to the pre-FFN norm, and map `moe_gate` /
   `moe_experts_{gate,up,down}` / `moe_shared_*` onto the existing MoE slots.
2. **Routing rule**: group-limited selection (`n_group`/`topk_group`, group score
   = sum of its top-2 expert scores), plus `moe_gate_bias` added to the score
   *for selection only*, `norm_topk_prob` renormalization, and
   `routed_scaling_factor` applied to the gate weights. The host router loop is
   the only place this changes. A file that carries `moe_gate_bias.weight` is
   currently refused by the gap map rather than routed with the Qwen rule.
3. **A cacheless bidirectional forward** — one full-sequence pass with
   `causal = false`, no KV reuse across steps (LLaDA2 gains a block-level KV
   cache by *finalizing* blocks; that is an optimization, not a correctness
   requirement).
4. **The block-diffusion sampler** in the engine, with `steps`,
   `gen_length`, `block_length`, temperature 0 by default, and the EOS
   handling from the reference (`logits_eos_inf`).
5. **Metrics and CLI**: `pp` for a dLLM is the prompt forward pass; there is no
   per-token TG — the useful numbers are *tokens per step* (block commit rate)
   and *steps per second*. `--diffusion-steps`, `--diffusion-block`,
   `--diffusion-gen-length`.

## Memory reality on this box

`LLaDA2.2-flash-Q4_K_S.gguf` is ~59 GB. The local box has 15.9 GiB of VRAM, so
the resident part is ~16 GB and the expert spill (~43 GB) has to stream through
the expert cache — the same situation as the 35B-A3B MoE models, only larger. A
`diffuse` benchmark on this machine therefore measures *streaming*, not the
model's speed; the CPU backend is the correctness oracle, and PP/TG for the
models that fit (up to ~14 GB) are the numbers to quote.

## Status

* `diffuse` is in the architecture table (`src/arch.cpp`) with the schema
  recorded above, and it is **refused** with a reason that names the missing
  pieces — the honest state until (1)–(4) land. The research above is what those
  pieces are built from.
* Verified while researching: bidirectional attention already works in both
  backends, so the first real commit is the loader schema plus the routing rule,
  and the sampler is the third.
