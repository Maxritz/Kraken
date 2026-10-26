# LFM2.5 (`lfm2moe`) — what the file is, and what a port needs

Measured from `G:/More-models/LFM2.5-24B-A2B-QAD-Q4_0.gguf` with `kraken-inspect`
(428 tensors, 12.64 GiB, Q4_0 experts + Q6_K embedding) and its DSpark drafter
`LFM2.5-24B-A2B-DSpark-F16.gguf` (625 MiB, arch `dflash`, already refused-correct
as a draft file).

**Status: SUPPORTED** (`src/arch.cpp`, `lfm2moe` → `ArchSupport::Yes`). The row
started as a refusal-by-name because the file fell through to llama defaults and
died on `output_norm.weight is missing` before reading a single layer — a message
about a tensor, for a mixer the engine had no shape for. It now loads, runs and
is gated by `test_lfm2_shortconv`.

What the port measured on the 24B-A2B file, after it was running:

| prompt (BOS included) | this engine | llama.cpp (`-ngl 0`, temp 0) |
|---|---|---|
| `The capital of France is` | ` Paris. The capital of Italy is Rome. The capital of` | ` Paris. The capital of Germany is Berlin. The capital of` |

The one word that differs is a near-tie between two numerically different
pipelines: f16 GPU activations here against llama's CPU q8 dot-products. An
independent f32 numpy reference of the same graph (`tools/lfm2_full.py`) agrees
with this engine word for word on that prompt, so the engine is faithful to the
graph and the sentence is not a pin — the FIRST token is (4741 on the file), plus
the logits row's own consistency across chunk sizes.

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
- **Conv** (30 layers):  `shortconv.in_proj` 6144×2048 (three 2048-wide outputs),
  `shortconv.conv` F32 2048×3 (depthwise, `l_cache` = 3), `shortconv.out_proj`
  2048×2048. Graph — **the convolution runs on `B ⊙ X`, and `C` gates the
  convolved row**, which is the one place this block is easy to get wrong:

  ```
  B, C, X = in_proj(x)
  bx[t] = B[t] ⊙ X[t]
  y[t]  = Σ_{k<3} w[:,k] ⊙ bx[t-2+k]      (causal; state = the last 2 rows)
  out   = out_proj(C ⊙ y)
  ```

  `hidden_states = B * x` immediately before the conv in transformers'
  `Lfm2ShortConv`, and the same two lines in llama.cpp's graph. The other reading
  (`B ⊙ conv(X) + C ⊙ X`) has the same shape and op count and produces a fluent,
  wrong model. The state is a ring keyed by ABSOLUTE position, holding `bx` as
  f16 — and a row the same call produced must be rounded to f16 before its tap,
  or `--chunk 1` carries one extra bit and the prefill stops being
  chunk-invariant (`src/hip/kernels/sc.hpp`).
- **FFN**: the first 2 blocks are dense SwiGLU at 11776; blocks 2..39 are
  64 experts × 1536, top-4, with `exp_probs_b.bias` (64) — the laguna-style
  selection bias this engine already loads as `L.exp_gate_bias` — and
  `expert_gating_func` 2, which llama.cpp reads as a **sigmoid** over the
  probabilities (`mc.router_sigmoid`).
- **Final norm**: there is no `output_norm`; the file carries
  `token_embd_norm.weight` (2048) and uses it. `ArchSpec::no_out_norm` is the
  flag for that shape (qwen4exp uses it for a different reason).

## What the port turned out to cost

1. `Layer`: `sc_in`, `sc_conv`, `sc_out` (+ upload of the F32 conv), and a ring
   of the last `l_cache - 1` rows per layer (`Engine::sc_state_`).
2. `Backend::shortconv`: one launch, depthwise causal conv over `B ⊙ X` with the
   ring, then `C ⊙`. Implemented in `src/hip/kernels/sc.hpp` and the CPU
   reference in `backend_cpu.cpp`; its state write is chunk-invariant (n rows at
   once and n one-row calls produce the same bytes).
3. Loader: a layer with no `attn_*` at all, `n_kv` per layer off the
   `head_count_kv` ARRAY, `token_embd_norm` when `output_norm` is absent,
   `router_sigmoid`, and `expert_w_norm` 1.
4. Gate: `test_lfm2_shortconv` — a 3-layer fixture (2 conv + 1 attention + MoE),
   the arch row's claim (loads with no experimental hatch), the per-layer KV and
   conv counts, and chunk 1/2/4 agreeing on both backends.

Two further answers that are not in the file and had to be measured:

- **RoPE pairs NeoX half-split.** llama.cpp prints `rope type = 2` for the file,
  so the GGUF keeps the HF layout unpermuted and `(j, j + n_rot/2)` is the
  pairing. `ArchSpec::rope_neox` was false for this row at first and the model
  decoded fluent nonsense.
- **The top-k sigmoid weights are sum-normalized** (`expert_w_norm` 1). Without
  it the same prompt says ` "The" and "the" are the same`, with it ` Paris.`; the
  two differ by the sigmoid sum, about 2x on every expert contribution.

One thing outside the family: `tokenizer.ggml.add_bos_token` is absent from this
file, and its absence means TRUE (llama.cpp's rule). The tokenizer defaulted it
to true for SPM only, so this gpt2-style file got no BOS and the model — which is
trained with one — answered in whitespace. See `src/tokenizer.cpp`.

## The expert tier: what warmup and prefetch are worth

This file is 12.64 GiB against a **12.0 GiB expert budget** (clamped 11964 MiB
once the KV cache and workspaces exist), so the routed set does not fit and 25.7%
of the first acquires are COLD reads. Measured on this card, greedy, one process
per arm, sidecar ranked from a 73-position scan of the same paragraph
(`--expert-scan`, 103 KiB):

| arm | prefill 58 tok | decode tok/s | COLD misses | wall, 24 out |
|---|---|---|---|---|
| old default (`warmup -1` → skipped) | 2410 ms (24.1) | 39.5 / 39.9 | 1213 (25.7%) | 4.77 s |
| `--expert-warmup 1` (now the default) | **111 ms (523)** | **82.7 / 84.1** | **0** | 6.66 s |
| `--expert-warm-prefetch 1` | 546 ms | 25.5 | 0 | — |
| both | 109 ms | 82.8 / 83.9 | 0 | — |
| `--expert-warmup 0` (control) | 2410 ms | 39.5 / 39.7 | 1213 | — |

So the ranked warmup is worth **2.1x on decode and 22x on prefill** and it is the
only flag of the three that helps. Three things worth keeping:

- **Prefetch ALONE is a loss (25.5 against 39.7).** `--expert-warm-prefetch`
  fills WARM at load but does not stage VRAM, so the hot set sits in host RAM and
  every decode step pays a promotion. It is only free on top of warmup, where it
  changes nothing (82.8 against 82.7 — the ranked pass already filled both
  tiers).
- **The warmup is a 3.87 s up-front tax** (12312 MiB read in 1526 ms = 7.9 GB/s,
  alloc 2687 ms of which 769 overlapped, VRAM +2357 experts). At 24 output tokens
  that tax is why its wall is *worse*; at 128 tokens the arms converge (7.85 s
  against 7.21 s) because the saving is 2.30 s of prefill plus 1.19 s of decode.
  Decode's per-token saving is 20.7 → 11.5 ms, so a single cold run breaks even at
  roughly 165 output tokens — and any second request in the same process pays
  none of the staging.
- **The old auto rule skipped this file, and its premise was wrong.** It fired
  only when the expert CORPUS exceeded the PLAN, on the reasoning that a set
  which fits is "resident anyway, so staging is pure wall time". The misses here
  are compulsory first touches (2432 distinct cells over the prefill), not
  evictions, and the plan is 12312 MiB against a 11964 MiB budget earned after
  the KV tier and the workspaces exist. The default is now ON for every MoE with
  a device tier; the corpus-vs-budget comparison survives only on the CPU
  backend. Two further files settle it, both on this card:

  | model | corpus vs budget | decode | prefill | COLD |
  |---|---|---|---|---|
  | Qwen3-30B-A3B i1-Q2_K | 10008 vs 10008 (exact fit) | 9.7 → 47.5 | — | 41.8% → 0 |
  | llada-moe-7b-a1b | 3720 vs ~11.9 GiB (3x slack) | 19.2 → 107.4 | 541 → 34 ms | 36.4% → 0 |

  The 3x-slack file is the one that proves the point: room in the pool does not
  make a first touch free, so "does it fit" was never the right question.

## The DSpark drafter does not load yet

`--draft /g/More-models/LFM2.5-24B-A2B-DSpark-F16.gguf` exits 1 with
`draft model: DFlash head set has neither enc.aux_norm nor aux_hidden_norm.0`.
That is the right answer for the file as it stands and not a wiring gap: the
head set is a **different input contract**, not a renamed one. 62 tensors,
625 MiB, `arch dflash`, 5 blocks, `block_size` 9, hd 64 over 8 KV heads,
`target_layers [5 i32]`, `sample_from_anchor`, `has_confidence_head`:

| tensors | what they are | state here |
|---|---|---|
| `fc.weight` [10240, 2048] | the input transform: the **5** target hidden layers (5 x 2048) concatenated and projected to 2048, in place of the per-layer `aux_hidden_norm.N` norms the loader requires | missing |
| `blk.0..4.*` (attn q/k/v + q/k norms, attn_output, ffn gate/up/down, 2 norms) | the 5-block decoder, 6144-wide FFN | read by name already |
| `enc.output_norm`, `output_norm` | the two norms around it | read by name already |
| `markov_w1`/`markov_w2` [256, 128000] | DFlash's markov head: a rank-256 additive logit bias taken from the anchor token | missing |
| `conf_proj` [2304, 1] + `conf_proj.bias` | confidence over `[hidden 2048 | markov 256]`, gated by `sample_from_anchor` | missing |

Dropping the markov head is not a shortcut: it adds to the **proposal logits**, so
a head set run without it proposes different tokens and any acceptance it records
is a number for a drafter nobody shipped. So the port is `fc` as an alternative
input transform (and a 5-slot hidden capture through the engine), the markov
bias, and the confidence gate — then acceptance is measurable.

## The recorded row

`test_lfm2_shortconv` pins the fixture's last prompt row by FNV-1a: **row
`0x399373273e5a2b60`, argmax 2** (CPU backend, prompt `the`, chunk 8), with four
arms that keep the pin from being vacuous — another prompt moves it, chunk 1
reproduces it, a fixture whose **one** conv coefficient is moved gives
`0xd8b9a0cef214f8fc`, and the arithmetic red construction sets it:

- break the mixer to the plausible-wrong reading (convolve `X` instead of `B*X`
in `backend_cpu.cpp`) → **32/35**, failing the row, the argmax and chunk 1 == 8;
- revert → 35/35 with the same recorded value.

One measurement trap, now in the code: the first version of the moved-tap arm
changed `w[0]`, the tap a ONE-row prompt never reads (its taps address p < 0 and
are skipped), so the row stayed bit-identical and the arm reported "the pin is
blind" when the pin was fine. It moves the last tap row now.
