# LFM2.5 (`lfm2moe`) — what the file is, and what a port needs

Measured from `G:/More-models/LFM2.5-24B-A2B-QAD-Q4_0.gguf` with `kraken-inspect`
(428 tensors, 12.64 GiB, Q4_0 experts + Q6_K embedding) and its DSpark drafter
`LFM2.5-24B-A2B-DSpark-F16.gguf` (625 MiB, arch `dflash`, already refused-correct
as a draft file).

**Status: refused by name** (`src/arch.cpp`, `lfm2moe` → `ArchSupport::No`).
Before that row existed the file fell through to llama defaults and died on
`output_norm.weight is missing` before reading a single layer — a message about
a tensor, for a mixer the engine has no shape for. The tensor gap map carries
`shortconv.*` as well, so an unlisted arch string still refuses for the piece it
actually carries.

## Shape

- 40 blocks, `n_embd` 2048, vocab 128000, rope base 1e7, eps 1e-5, tokenizer
  `lfm2` (gpt2 BPE with the `lfm2` pre-tokenizer).
- **Layer kinds alternate 3 conv : 1 attention**, attention at indices
  2, 6, 10, … 38 (10 of 40), conv at the rest (30 of 40). Derive the kind from
  the tensors (`shortconv.in_proj.weight` present ⇒ conv), the way
  `nemotron_h_moe` derives SSM from `ssm_in.weight`; the file carries no
  `attention.recurrent_layers` array, and the existing
  `(l + 1) % full_attention_interval` rule is off by one for this stack.
- **Attention** (10 layers): 32 query heads × hd 64 over 8 KV heads, with
  `attn_q_norm`/`attn_k_norm` of 64, no q/k/v bias, `attn_output` 2048×2048.
- **Conv** (30 layers): `shortconv.in_proj` 6144×2048 (three 2048-wide outputs),
  `shortconv.conv` F32 2048×3 (depthwise, `l_cache` = 3), `shortconv.out_proj`
  2048×2048. Graph: `B, C, X = in_proj(x)`;
  `y[t] = Σ_{k<3} w[:,k] ⊙ X[t-2+k]` (causal, state = the last 2 rows);
  `z = B ⊙ y + C ⊙ X`; `out = out_proj(z)`. `src/hip/kernels/hc.hpp`'s
  `conv1d_dw` and the PLE block's per-layer window (`ple_state_`) are the
  precedent for the state keeping; chunk invariance needs the same treatment.
- **FFN**: the first 2 blocks are dense SwiGLU at 11776; blocks 2..39 are
  64 experts × 1536, top-4, with `exp_probs_b.bias` (64) — the laguna-style
  selection bias this engine already loads as `L.exp_gate_bias` — and
  `expert_gating_func` 2, which llama.cpp reads as a **sigmoid** over the
  probabilities (`mc.router_sigmoid`).
- **Final norm**: there is no `output_norm`; the file carries
  `token_embd_norm.weight` (2048) and uses it. `ArchSpec::no_out_norm` is the
  flag for that shape (qwen4exp uses it for a different reason).

## What a port costs

1. `Layer`: `sc_in`, `sc_conv`, `sc_out` (+ upload of the F32 conv), and a
   ring of the last `l_cache - 1` rows per layer.
2. A `shortconv` kernel: depthwise causal conv over `[n, 2048]` with the state,
   then `B ⊙ y + C ⊙ X`. n is 512 on a prefill chunk, so it wants to be one
   launch, and its state write must be chunk-invariant (n rows at once and n
   one-row calls must produce the same bytes).
3. Loader: accept a layer with no `attn_*` at all, pick `n_kv` per layer off the
   `head_count_kv` array (or simply refuse a conv layer that carries one),
   fall back to `token_embd_norm` when `output_norm` is absent, and set
   `router_sigmoid`.
4. Gate: a fixture in the style of `test_qwen4exp_whole_graph` — the last prompt
   row's whole logits f16 FNV hash plus its argmax, chunk 1 reproducing chunk 8,
   and a red construction that disables the mixer. Cross-check the sentence
   against `H:/LLAMA-bins/rocm10/llama-completion.exe` on the real 24B file,
   24 greedy tokens.

The drafter side is not part of this: `dflash` is implemented (`src/dflash.cpp`,
`--draft`), and the DSpark file declares 5 blocks, `block_size` 9,
`target_layers [5]`, `sample_from_anchor` and a confidence head.
