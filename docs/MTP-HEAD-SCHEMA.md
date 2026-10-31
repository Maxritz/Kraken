# The qwen4exp MTP head: 31 tensors, their shapes, and who reads them

Plan step **1.1** of [PLAN-mtp-and-prefill.md](PLAN-mtp-and-prefill.md). This is
the tensor table the loader is written against: every row names a shape, the
format the artifact stores it in, and the code that consumes it. A row without a
consumer is a tensor the port would silently drop, and a consumer without a row
is a guess.

Read [AUDIT-mtp-drafter.md](AUDIT-mtp-drafter.md) first for what the head is and
the three traps (the pairing, the per-stream hidden norm, the trunk's own LM
head).

## Where the numbers come from

| symbol | value | source |
|---|---|---|
| `n_embd` | 2560 | the artifact's `fc_embedding` / any projection's `ne[0]`, checked against the trunk |
| `hc` (lanes) | 4 | `hnorm` width / `n_embd` = 10240 / 2560 |
| `hc_dim` | 10240 | `hnorm` element count |
| `hc_rank` | 320 | `hyper_connection_mixer.*_weight_down` last dim |
| `n_expert` | 512 | `mlp.experts.gate_up_proj` `ne[2]` |
| `n_ff_exp` | 640 | `gate_up_proj` `ne[1] / 2`, cross-checked against `down_proj` `ne[0]` |
| `n_ff_shexp` | 640 | `shared_expert.up_proj` `ne[1]` |
| `index_dim` | 128 | `indexer.k_proj` `ne[1]` |
| `n_index_head` | `indexer.q_proj ne[1] / index_dim` | read, not assumed |

Every one of those is **read off the tensors** and then checked against the
trunk, so a head from a differently-pruned checkpoint is refused by tensor name
rather than loaded into workspaces sized for another model. The artifact ships
almost no metadata of its own (`block_count`, `embedding_length`,
`expert_count`, `hyper_connection.low_rank` — see `tools/mtp_extract.py`); the
geometry above is the file's, the attention geometry is the trunk's.

## Names

Two converters write this head and they disagree about spelling. The loader
accepts both, in this order, and reports which one it used:

- the **HF/vLLM shape** (`mtp.hyper_connection_mixer.input_mix_weight_down.weight`)
  — what the checkpoint's own module names look like and what the q2_0 repack
  carries;
- the **flattened shape** (`mtp.layers.0.hc_attn_down.weight`) — the llama.cpp
  `blk.N.*` naming pushed through an `mtp.` prefix.

`%s` below is the layer prefix, `mtp.layers.0.` except at the head's top level.

## The 31

Format counts are from the q2_0 repack: **20 BF16, 9 F32, 2 q2_0**. The two
q2_0 tensors are the routed bank and its down projection — 637.5 MiB of the
810 MiB payload — and they stay in the file for `ExpertCache` to page.

### The dense prefix (2 norms + 1 fused projection, or 2 norms + 2 halves)

| # | name (preferred) | ggml shape | fmt | consumer |
|---|---|---|---|---|
| 1 | `mtp.pre_fc_norm_embedding.weight` | `[2560]` | F32 | `MtpBlock::enorm`, step 1.5 |
| 2 | `mtp.pre_fc_norm_hidden.weight` | `[10240]` | F32 | `MtpBlock::hnorm`, step 1.5 (both modes) |
| 3 | `mtp.eh_proj.weight` | `[2560, 5120]` | BF16 | `MtpBlock::eh_proj`, step 1.5 |

`eh_proj` maps `cat([e_norm, h_norm_lane])` — 5120 wide — to one lane of 2560.
It is applied **per lane**, so the block's input is `[2560, hc]`.

The EXL3 export instead stores the same function as two matrices over the two
halves (they are the fused matrix's column blocks):

| 3a | `mtp.fc_embedding.weight` | `[2560, 2560]` | BF16 | `MtpBlock::fc_embedding` |
| 3b | `mtp.fc_hidden.weight` | `[2560, 2560]` | BF16 | `MtpBlock::fc_hidden` |

A file has the fused form or the split pair, never both, and a file with
`fc_embedding` but no `fc_hidden` is refused: `fc([e|h]) == fc_embedding(e) +
fc_hidden(h)` only holds over both halves, and half of it computes a prefix that
looks plausible and drafts nothing. `MtpBlock::eh_proj_split` records which
form the file used, because the forward has to know.

### The two block mixers and the head mixer (11 tensors)

Hyper-connections: the residual is `hc` parallel lanes, and each mixer turns
that stack into one 2560-wide vector for its sub-layer, then folds the result
back with a per-lane weight `2*sigmoid(inject/hc)` (see `src/hc.cpp`,
`hc_mix`/`hc_combine`).

| # | name | ggml shape | fmt | consumer |
|---|---|---|---|---|
| 4 | `%shyper_connection_mixer.input_mix_norm_weight` | `[10240]` | F32 | `MtpBlock::hc_attn.norm` |
| 5 | `%shyper_connection_mixer.input_mix_weight_down.weight` | `[10240, 320]` | BF16 | `MtpBlock::hc_attn.down` |
| 6 | `%shyper_connection_mixer.input_mix_weight_up.weight` | `[320, 10240]` | BF16 | `MtpBlock::hc_attn.up` |
| 7 | `%shyper_connection_mixer.input_mix_inject_weight` | `[10240, 4]` | BF16 | `MtpBlock::hc_attn.inject` |
| 8 | `%shyper_connection_mixer.output_mix_norm_weight` | `[10240]` | F32 | `MtpBlock::hc_ffn.norm` |
| 9 | `%shyper_connection_mixer.output_mix_weight_down.weight` | `[10240, 320]` | BF16 | `MtpBlock::hc_ffn.down` |
| 10 | `%shyper_connection_mixer.output_mix_weight_up.weight` | `[320, 10240]` | BF16 | `MtpBlock::hc_ffn.up` |
| 11 | `%shyper_connection_mixer.output_mix_inject_weight` | `[10240, 4]` | BF16 | `MtpBlock::hc_ffn.inject` |
| 12 | `mtp.final_mix_norm_weight` | `[10240]` | F32 | `MtpBlock::hc_head.norm` |
| 13 | `mtp.final_mix_weight_down.weight` | `[10240, 320]` | BF16 | `MtpBlock::hc_head.down` |
| 14 | `mtp.final_mix_weight_up.weight` | `[320, 10240]` | BF16 | `MtpBlock::hc_head.up` |

The head mixer has no inject: it is followed by the LM head, not by another
residual. The loader checks all three mixers are the same rank — one block, one
bottleneck.

**The inject is a projection, not a 4-long vector.** It maps the normalized
10240-wide lane stack onto one weight per lane (`[hc_dim, hc]`), which is what
`Engine::hc_enter` consumes: `gemm_w(ws_hc_inj_, ws_hc_n_, inject, hc, wide, n)`.
The first version of this loader validated it as four numbers, and the mixer then
read 128 rows out of a 4-element tensor — garbage lanes rather than an error,
found by the block's own negative control (`test_mtp_hf_block_qwen4`, which
produced NaN until the shape was fixed). An export that stores the transpose
(`[hc, hc_dim]`) is the same matrix and is re-oriented at load.

### Attention (6 tensors)

| # | name | ggml shape | fmt | consumer |
|---|---|---|---|---|
| 15 | `%sself_attn.q_proj.weight` | `[2560, n_head*head_dim]` | BF16 | `MtpBlock::wq` |
| 16 | `%sself_attn.k_proj.weight` | `[2560, n_kv*head_dim]` | BF16 | `MtpBlock::wk` |
| 17 | `%sself_attn.v_proj.weight` | `[2560, n_kv*head_dim]` | BF16 | `MtpBlock::wv` |
| 18 | `%sself_attn.o_proj.weight` | `[n_head*head_dim, 2560]` | BF16 | `MtpBlock::wo` |
| 19 | `%sself_attn.q_norm.weight` | `[head_dim]` | F32 | `MtpBlock::q_norm` |
| 20 | `%sself_attn.k_norm.weight` | `[head_dim]` | F32 | `MtpBlock::k_norm` |

`q_proj` may be twice as wide as `n_head*head_dim` when the block gates its
output (the trunk's convention); the loader accepts either and records the
attention geometry from the trunk, checking it against these two widths.

### The QSA indexer (4 tensors)

| # | name | ggml shape | fmt | consumer |
|---|---|---|---|---|
| 21 | `%sself_attn.indexer.q_proj.weight` | `[2560, 128*n_index_head]` | BF16 | `MtpBlock::index_q` |
| 22 | `%sself_attn.indexer.k_proj.weight` | `[2560, 128]` | BF16 | `MtpBlock::index_k` |
| 23 | `%sself_attn.indexer.q_norm.weight` | `[128]` | F32 | `MtpBlock::index_q_norm` |
| 24 | `%sself_attn.indexer.k_norm.weight` | `[128]` | F32 | `MtpBlock::index_k_norm` |

All four or none — a partial indexer cannot score a block, and a block that
cannot score cannot select. Absent, the block attends densely (the older native
exports have no indexer and keep their dense path).

### The MoE (7 tensors, 2 of them the whole file's bulk)

| # | name | ggml shape | fmt | consumer |
|---|---|---|---|---|
| 25 | `%smlp.gate.weight` | `[2560, 512]` | BF16 | `MtpBlock::router` |
| 26 | `%smlp.experts.gate_up_proj` | `[2560, 1280, 512]` | q2_0 | `MtpBlock::exps_gate` / `exps_up` — **two views, no copy** |
| 27 | `%smlp.experts.down_proj` | `[640, 2560, 512]` | q2_0 | `MtpBlock::exps_down` |
| 28 | `%smlp.shared_expert_gate.weight` | `[2560]` | F32 | `MtpBlock::shexp_inp_gate` |
| 29 | `%smlp.shared_expert.gate_proj.weight` | `[2560, 640]` | BF16 | `MtpBlock::shexp_gate` |
| 30 | `%smlp.shared_expert.up_proj.weight` | `[2560, 640]` | BF16 | `MtpBlock::shexp_up` |
| 31 | `%smlp.shared_expert.down_proj.weight` | `[640, 2560]` | BF16 | `MtpBlock::shexp_down` |

**`gate_up_proj` is one tensor, not two.** `dim0` is the contiguous inner
dimension and `dim1` the output rows, so expert `e`'s plane is 1280 rows of
2560: rows `[0, 640)` are its **gate** rows and `[640, 1280)` its **up** rows.
`Gguf::add_expert_plane_slice` turns those into two directory entries that share
the mapping — `ne = [2560, 640, 512]` each, with `plane_stride` the full 1280-row
plane — so `ExpertCache` pages the 512-expert bank exactly once and both halves
stay in the file. Step 1.4's gate is that the views read back byte-identical to
the source rows, per expert, and `test_mtp_head_qwen4` checks that on every
expert of a synthetic bank rather than on a sample.

## What the loader refuses, and why it matters

Each of these produces a head that loads and drafts plausible garbage if it is
not caught, so each is a named refusal:

| condition | refusal names |
|---|---|
| the bank's inner width is not the trunk's `n_embd` | `gate_up_proj` |
| `down_proj` disagrees with the bank's expert/FF geometry | `down_proj` + both shapes |
| `fc_embedding` without `fc_hidden` | `fc_hidden` |
| neither `eh_proj` nor the `fc_*` pair | both spellings |
| 1–3 of the 4 indexer tensors | the count and all four names |
| the attention widths disagree with the trunk's heads | `q_proj` / `k_proj` + both numbers |
| the router's expert count is not the bank's | `mlp.gate.weight` + both shapes |
| the three mixers have different ranks | both ranks |
| a required tensor is absent | the tensor's role and preferred name |

## Not in the 31, on purpose

- **No `output.weight` and no `output_norm`.** The block ends in its final mixer
  and reuses the trunk's LM head, so its vocabulary *is* the trunk's — there is
  no `d2t` map, unlike the EAGLE head. Loader: `MtpBlock` never looks for one.
- **No token embedding.** `mtp.*` has no `embed_tokens` in this artifact; the
  proposed token's embedding comes from the trunk. A newer export may carry
  `nextn.embed_tokens`/`shared_head_head` as optional overrides; the loader
  accepts their absence as the normal case.
- **No routing bias** for the MoE (`ffn_exp_probs_b` is the *trunk's* qwen4exp
  tensor, not the head's).

## Status

Steps 1.1–1.7 are done: this table, the extended `MtpBlock`, the loader in
`src/mtp_head.cpp`, the fused-bank views, the host prefix
(`mtp_dense_prefix`), the block itself on the device
(`Engine::mtp_hf_block_forward` — hyper-connection around attention and around
the MoE, its own KV, the trunk's head), and the chained proposal:
`Engine::mtp_chain_` hands the block's own `[n_lane * n_embd]` lane stack back
to its next step, so `--mtp-spec` drafts a chain instead of refusing. Group
`test_mtp_hf_spec_chain` holds it to plain greedy's text, token for token, with
a live-attention control that accepts 0 of 36 and still verifies the same text.

The arch row stays `Partial` for the one thing left: no real checkpoint has been
run through it, so acceptance and tokens/s are unmeasured (plan 1.8).
