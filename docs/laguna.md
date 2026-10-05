The Laguna family, what kraken implements of it, and where the wins are
======================================================================

Sources
-------
* "Laguna M.1/XS.2 Technical Report", arXiv:2605.27605 (26 May 2026) — the
  architecture section is the normative description.
* poolside/Laguna-S-2.1 and poolside/Laguna-XS-2.1 model cards (HF).
* the reference implementation: upstream src/models/laguna.cpp
  (ggml-org/llama.cpp#25165) and, for speculative decoding, Poolside's own
  fork `poolsideai/llama.cpp`, branch `laguna` (src/models/dflash.cpp).

Family
------
| model    | total/active | layers | attention                  | experts            |
|----------|--------------|--------|----------------------------|--------------------|
| XS.2     | 33.4B / 3B   | 40     | 1:3 full:SWA, window 512   | 256 routed, top-8  |
| XS 2.1   | 33B / 3B     | 40     | same recipe                | 256 routed, top-8  |
| S 2.1    | 118B / 8B    | 48     | 12 full / 36 SWA, window512| 256 routed, top-10 |
| M.1      | 225.8B/23.4B | 70     | all full attention         | 256 routed, top-10 |

All four: one shared expert summed in parallel, a linear+sigmoid router
with a per-expert selection bias, sum-normalization of the top-k scores,
and the routed output modulated by 2.5 before it meets the shared expert.
First transfer layer is dense (XS.2: 1, M.1: 3). GQA with 8 KV heads and
head dim 128 in every variant. Vocabulary 100,352, BPE, pre-tokenizer
"laguna" (qwen2-style rules — see docs/traces/laguna-chat-tokenizer.txt).

Attention, per layer type
-------------------------
* Full layers: 48 Q-heads (XS.2) / 48 (S 2.1), theta 500,000, YaRN with
  factor 32, and partial RoPE — only the first 64 of the 128 head dims
  rotate.
* Windowed layers: 64 Q-heads (XS.2) / 72 (S 2.1), plain RoPE, theta
  10,000, all 128 dims.
* Both: QK-RMSNorm between projection and rotation, `kq_scale` =
  1/sqrt(128), and a softplus output gate — one scalar per query head
  computed from the *pre-attention* norm output, applied before `wo`.
  (M.1 gates per element instead of per head; the tensor's stored width
  says which.)

The head counts are per layer and they are in the GGUF as a
`attention.head_count` array of `block_count` entries. Two things follow,
and kraken got both wrong at some point in its history:

  * the query projection, qk_norm, rope and gate widths are per layer;
  * so is the GQA grouping the attention op derives (`n_head / n_kv` is 6
    on an XS.2 full layer, 8 on a windowed one) — see
    docs/traces/laguna-per-layer-heads.txt.

Where the performance is
------------------------
1. **The window is most of the model.** 3 of every 4 layers never look
   further back than 512 tokens. A per-layer ring buffer for those layers
   turns their attention cost and their KV footprint from O(context) into
   O(1) — at 1M context that is the difference between usable and not.
   kraken still allocates `kv_cap_` slots for every layer.
2. **Per-layer head counts matter to attention cost**: an XS.2 full layer
   is 48 heads, a windowed one 64 — sizing any of them at the stack
   maximum wastes 25% of the work (and, worse, changes the answer, as the
   trace above shows).
3. **Top-8/10 of 256** means ~3% of the experts per token; the expert
   cache hit rate is the decode bottleneck on a 16 GiB card, which is what
   `.krakenexperts.json` (routing mass per expert) is for.
4. **Quantization**: Poolside ships FP8, INT4, NVFP4 and MXFP8 variants of
   the weights plus an FP8 KV cache; the GGUF conversions on this machine
   are UD-Q4_K_M. kraken's MXFP4/ROCmFP4 dequant (docs/traces/
   mxfp4-rocmfpx-dequant.txt) is the other half of that story.
5. **DFlash**. Both local draft files are real and small (~0.5-0.65 GiB
   for a 118B target), and the reference flow is: capture the target's
   hidden state at 6 (`target_layers`) layers, fuse them (per-feature RMS
   norm, `fc` down to 3072, RMS norm) into one vector per position, inject
   that vector's K/V into the drafter's cache through its own `wk`/`wv`
   (+k_norm, +rope, from the *normalized* input), then run one masked block
   of 16 positions and keep the tokens that verify. One forward pass
   yields up to 16 candidates, which is why the shipping recipe is
   `--spec-draft-n-max 7`.
6. **Reasoning**: the chat template carries `enable_thinking`; the model
   reasons before and between tool calls, and wants previous thinking kept
   in the history ("preserved thinking"). kraken's ChatML template is the
   plain one — worth keeping in mind when comparing outputs against a
   reference run that enables thinking.

Files on this machine
---------------------
| file | what | state |
|------|------|-------|
| G:/More-models/laguna-xs2-Q4_K_M.gguf | XS.2 target | runnable, coherent |
| G:/More-models/Laguna-S-2.1-UD-Q4_K_M-*.gguf (3 shards, 73 GiB) | S 2.1 target | multi-part, not yet loadable |
| G:/More-models/laguna-s-2.1-DFlash-Q4_K_M.gguf | S 2.1 drafter, 6 blocks, 72 heads | refused (dflash.*) |
| G:/More-models/laguna-xs21-dflash-q8.gguf | XS 2.1 drafter, 5 blocks, 64 heads | refused (dflash.*) |

Setup notes for reference runs
------------------------------
* llama.cpp must be told the template explicitly (`--no-jinja
  --chat-template chatml`): the GGUF's `chat_template` is
  `{% include 'chat_template.jinja' %}`, which cannot resolve from inside
  the file, and raw `-p` mode without a template fails outright.
* tokenizer.ggml.pre is "laguna"; anything that maps it to the GPT-2
  rules mangles `<|im_start|>` (which is five BPE pieces here, not one
  special token).
