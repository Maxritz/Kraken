# Supporting qwen4exp: the measured inventory, the pieces, and the gates

Status, 2026-10-09. `qwen4exp` (Qwen3.8-Flash-Next and its REAP expert-reduced
repacks) is now a *named* family in the architecture table — refused, as before, but
"not in the architecture table; read against llama defaults" is gone, which was the
wrong claim: the engine cannot run it, and now it says so and lists the pieces.

Everything below comes from the file itself
(`H:\OLLAMA-Models\GGUF\qwen3.8-flash-next-reap-288-Q4_K_M.gguf`, 1224 tensors,
78.04 GiB) and from the pack index the reference engine generated for it
(`H:\OLLAMA-Models\strata-pack-reap288\index.txt`), not from a paper or a summary.

## What the file declares

| key | value | what it pins |
|---|---|---|
| `block_count` | 48 | the stack |
| `embedding_length` | 2560 | the block I/O width |
| `attention.head_count` / `head_count_kv` | 24 / 2 | 12 query heads per KV head |
| `attention.key_length` / `value_length` | 256 / 256 | head dim 256, so q is 6144 wide |
| `rope.dimension_count` / `dimension_sections` | 64 / `[4 i32]` | an unusual rope; **not read** |
| `ssm.conv_kernel` / `state_size` / `group_count` / `time_step_rank` / `inner_size` | 4 / 128 / 16 / 48 / 6144 | a Mamba-style SSM block |
| `full_attention_interval` | 4 | full attention every 4th layer |
| `hyper_connection.count` / `low_rank` | 4 / 320 | a 4-lane, 10240-wide stream through a 320 bottleneck |
| `attention.indexer.*` | 4 heads, key_length 128, `top_k 2048` | a QSA indexer |
| `attention.compress_ratios` | `[48 i32]` | per-layer KV compression |
| `expert_count` / `expert_used_count` / `expert_feed_forward_length` | 288 / 10 / 640 | the MoE this engine already has a shape for |
| `ple.*` | `layers [1]`, ngram 3, heads_per_ngram 8, conv_kernel 4, `embedding_length_per_layer_input 160` | the PLE n-gram table |

The MTP head set extracted in `docs/MTP-DRAFTER.md` is a *block of this same
family*: gated attention, a QSA indexer, hyper-connections, and a 512-expert MoE.

## The five pieces, with shapes and the nearest existing machinery

| piece | tensors (per block) | nearest thing in this tree | status |
|---|---|---|---|
| attention output gate | `attn_gate.weight` [2560, **6144**] = 24 heads x 256 | **the per-head variant is already implemented and verified** (see below); `laguna` runs through it | **per-element variant missing** |
| fused QKV | `attn_qkv.weight` [2560, **10240**] Q6_K | the gated delta net's own fused `attn_qkv` is implemented (`ArchShape::Recurrent`); dense `gemm_group` with per-matrix `wts[]` is the launching machinery | **unpack unresolved** |
| SSM block | `ssm_conv1d` [4, 10240], `ssm_alpha`/`ssm_beta` [2560, 48], `ssm_a` [48], `ssm_dt.bias` [48], `ssm_norm` [128], `ssm_out` [6144, 2560] | `nemotron_h_moe` runs Mamba-2 SSD blocks already (`Engine::mamba2_forward`, `src/hip/kernels/gdn.hpp`), with its own `ssm_*` names | family match, names differ |
| hyper-connections | `hc_attn_{down,up,inject,norm}`, `hc_ffn_{...}`, `output_hc_{down,up,norm}` — down [10240, 320], up [320, 10240], inject [10240, 4], norm [10240] | nothing | not implemented |
| QSA indexer + PLE | `per_layer_token_embd.weight` [160, **320001536**] Q5_0 | nothing; the "lookup table that stays on the SSD" in the cross-engine notes is this table, read a few rows per token | not implemented |

Two facts that decide the order of work:

- **The MoE is already this engine's shape.** 288 experts, top-10, `ffn_gate/up_exps`
  Q4_K, `ffn_down_exps` Q8_0, a shared expert with its own gate, `ff 640` on both
  sides. Routing, grouping and the expert tier port.
- **The fused QKV's row split is not derivable from the metadata.** q (24x256 =
  6144) + k (2x256 = 512) + v (2x256 = 512) = 7168 of the 10240 output rows; the
  remaining 3072 rows are unaccounted for, and `compress_ratios` (per-layer, `[4, ...]`)
  plus the indexer are the candidates. Guessing that split is how a file "loads and
  decodes plausible nonsense", so it is the first thing the reference has to settle.
- **It cannot be read off the source either — that search is done.** The llama.cpp
  fork on this box (`H:\llamadx\llama.cpp`, the one the AGENTS notes call llama-dx)
  has **no `qwen4exp` at all**: `grep -rln qwen4exp` returns nothing, and its
  `attn_qkv` hits are the tensor-name table, a quantiser regex, a MIMO2 MTP assert
  and the qwen35 delta-net's `linear_attn_qkv_mixed` — all different schemas. The
  binary that *runs* the file (`llama-completion.exe`, version `0.5.0-dev`, build
  11146) is a newer build than any source tree here, so the split has to come from
  a measurement, not a grep.
- **How to settle it by measurement:** row ablation. Copy the GGUF, zero one
  contiguous band of `attn_qkv`'s 10240 output rows, and see which part of the
  reference's output changes: the q band changes the query stream (and so everything
  downstream), the k band changes only scores, the v band changes only the mixed
  values. The boundaries show up as band edges, and 3072 rows of "something else"
  announce themselves as the one band whose ablation does *not* look like any of
  q/k/v. Costs one model copy and one reference run per band (10 s load, ~6 tok/s
  decode, CPU-side), which is cheap next to a wrong split.

## The reference engine runs it, and that is the enabler

`H:\LLAMA-bins\rocm10\llama-completion.exe` loads the same file and generates from
it. Measured on this box (`export PATH="/g/ROCM10RT-gfx1201/bin:$PATH"` first, as
always, or it exits 127 through a pipeline):

```
load time =  9963.07 ms
prompt eval time = 3471.38 ms / 4 tokens (867.84 ms per token, 1.15 tok/s)
eval time = 2502.93 ms / 15 runs  (166.86 ms per token, 5.99 tok/s)
```

`llama-cli --list-devices` reports `ROCm0: AMD Radeon RX 9070 XT`, so the runtime is
present — but the run's own log says expert tensors were overridden to CPU with mmap
enabled, and 1.15 tok/s prompt eval is not a GPU number. **Treat it as a correctness
reference, not a speed one.** Kraken already beats it by orders of magnitude on the
models it *can* run (558.8 tok/s prefill / 35.8 tok/s decode on the 256-expert
35B-A3B, `docs/STRATA-VS-KRAKEN.md`), which is the case for this work at all: the
reference is slow and correct, and it is the only thing on the box that can say what
the right answer is.

### What the reference can and cannot give you (measured, build 11146)

It **cannot give per-position logits**. `llama-completion.exe --help` (version
`0.5.0-dev`, build 11146, commit `7fe450e19`) has no logits-dump flag — the only
match for "logits|dump" is `--grammar-file`. So there is no per-stage oracle to
compare a half-built block against, and the gate has to be built the way the rest of
this repo builds one: a **CPU reference arm inside the engine**, bisected against the
device arm with `KRK_DUMP` stage by stage, with the reference's *text* as the final
acceptance test rather than as a per-layer telescope.

What it does give is a recorded, reproducible end-to-end answer for a fixed prompt.
Protocol: stdout only, stderr separated, greedy, exact paths.

```
PATH=/g/ROCM10RT-gfx1201/bin:$PATH
llama-completion.exe --model H:/OLLAMA-Models/GGUF/qwen3.8-flash-next-reap-288-Q4_K_M.gguf \
  -no-cnv --no-jinja --temp 0 --seed 1 -n 16 \
  -p "The history of computing is a history of"

stdout:
  The history of computing is a history of abstraction, to wit: the process of
  hiding complexity behind simpler interfaces.

  —

stdout md5  d4381153e8f9b03dffe2192a48dcb6a4
rc=0
prompt eval 3348.46 ms / 8 tokens (418.56 ms per token, 2.39 tok/s)
eval        2428.78 ms / 15 runs  (161.92 ms per token, 6.18 tok/s)
```

The text is coherent and on-topic for the prompt this repo's own acceptance runs
use, which is what makes it usable as the target: a qwen4exp implementation is
correct when Kraken's greedy output matches that string, token for token, with the
same hash discipline every other measurement here uses (stdout only, the protocol
written down beside the hash — see the `cf0408eb02` case in `docs/test-results.md`
§10 for why an unstated protocol makes a stored hash worthless).

## The gate each piece has to pass before it is enabled

The requirement is the repo's own: **a piece is enabled only after a CPU reference
for it exists and agrees** — device against `--cpu`, `scripts/coherence_check.sh`
style — and the family stays `ArchSupport::No` until every piece is in, because a
half-implemented block does not fail, it produces fluent wrong text.

| stage | what is built | how it is verified |
|---|---|---|
| 1. fused QKV unpack | row split decided by the reference (logits on a fixed prompt at a fixed position), then q/k/v views | the reference's own first-token logits; a fixture pair (fused vs separate q/k/v with identical weights) must agree bit for bit |
| 2. attention output gate | **per-element** variant (width `n_head * head_dim`, as this file has) and the confirmation of its activation | the per-head variant is done: `src/engine.cpp:1025-1032` projects one scalar per head, `softplus_act`s it, and `mul_head_broadcast`s it over that head's whole row before `wo`; `src/arch.cpp:239-247` records that calling it a gap was a false blocker; the loader enforces `n_out == head_count` (`src/model.cpp:1238`); and the broadcast op is now pinned by tests with exact per-(token, head) products (`test_gdn_ops`) |
| 3. SSM block | `ssm_*` mapped onto the Mamba-2 path with the oracle's recurrence (conv 4, state 128, 16 groups, dt_rank 48, inner 6144) | `nemotron_h_moe`'s existing CPU arm plus a `KRK_DUMP` stage bisection against the reference |
| 4. hyper-connections | the 4-lane 10240 stream, norms over 10240, the 320 bottleneck and the 4-wide inject | per-stage `KRK_DUMP` against the reference; the MTP head reuses the same block, so this is a shared cost |
| 5. QSA indexer + PLE | sparse attention selection over compressed KV, and the 320M-row table read | last, because nothing else depends on it and the table is 29 GB of I/O |

## What this turn did and did not do

**Did:** the family entry, with the inventory above in the comment; the shape
decision, pinned by tests — declared `Moe` and not `RecurrentMoe`, because the gap
map exempts `attn_qkv`/`attn_gate`/`ssm_*` for recurrent shapes (the gated delta net
owns those names), so a recurrent declaration would have hidden exactly the four
tensors that block this family and a later "Yes" would have loaded it into nonsense.
Build rc=0, 0 errors; suite **2838/2838 checks, rc=0**. The real file's verdict is now:

```
arch       qwen4exp — not supported
verdict    refused
           - a hyper-connection stack with a gated attention output, a fused
             query/key/value projection, Mamba-style SSM blocks, a QSA indexer and a
             PLE n-gram table, none of which this engine implements
           - uses an attention output gate (attn_gate), which this engine does not implement
           - uses a fused query/key/value projection (attn_qkv), which this engine does not implement
           - uses a Mamba-style SSM block (ssm_conv1d), which this engine does not implement
           - uses a Mamba-style SSM block (ssm_out), which this engine does not implement
```

### Correction to the first version of this table

The first draft of this document called the attention output gate missing "for
laguna too", quoting the older laguna comment. That was wrong, and the tree says so
in three places: the engine path exists (`src/engine.cpp:1025`), the loader validates
and uploads it (`src/model.cpp:1238`), the arch entry is `ArchSupport::Yes`, and
`src/arch.cpp:239-247` had already corrected the comment as a *false blocker* —
"a wrong diagnosis costs the next person a kernel to write". Worth recording as a
pattern: a stale comment about a gap is itself a defect, because it is what a later
reader budgets against.

What the gate *did* need was a test, and it had none: `mul_head_broadcast` (the op
that decides which head's scalar multiplies which row) was uncovered while `laguna`
was already a supported arch. It is now pinned — softplus's closed forms plus exact
per-(token, head) products over distinguishable rows, so a transposed read or a
per-element read fails instead of scaling the model quietly.

**Did not:** no piece of the five is implemented, and nothing about the family has
been executed by this engine — there is no CPU arm to compare a device arm against,
which is precisely why stage 1 starts there. The rope (`dimension_sections [4 i32]`,
`dimension_count 64`) is unread and its convention is unverified. The reference
engine is used here only as a *loads-and-generates* fact plus its timings; its
per-stage internals were not inspected.

## The reference implementation, found locally (2026-10-20)

`G:\Stratauild-hip-win\_deps\strata_llamacpp-src` is a full llama.cpp tree
that **runs this file**, with a dedicated `src/models/qwen4exp.cpp` (1294 lines)
and `conversion/qwen4exp.py`. The QKV split this document called "not derivable
from the metadata" is stated there outright, and it is not a guess:

```
head_k_dim = ssm.state_size        = 128
n_k_heads  = ssm.group_count       = 16     -> key_dim   = 2048
n_v_heads  = ssm.time_step_rank    = 48     -> value_dim = 6144
conv_dim   = key_dim*2 + value_dim = 10240  ==  attn_qkv's output width
```

so `attn_qkv` is **[q(2048) | k(2048) | v(6144)]**, and `attn_gate` [2560, 6144]
is the z gate over the value stream. Both are **already the shapes
`LayerWeights::wqkv` / `wqkv_gate` carry** for the gated delta net, and
`Engine::gdn_forward` already runs conv -> L2 norm -> softplus/dt -> delta rule ->
grouped norm -> `ssm_out` for exactly these tensors. Two real differences:

| piece | reference | kraken today |
|---|---|---|
| output gate activation | `sigmoid(z)` (`build_norm_gated`) | `silu(z)` (`silu_mul`) |
| q/k head count vs v | 16 vs 48, q/k repeated 3x | supported by the kernel already |

The full-attention layers (`(il+1) % 4 == 0`, `compress_ratios[il] == 4`) use
**separate** `attn_q` [2560, 12288] with `[q | gate]` interleaved *per head*,
`attn_k`/`attn_v` [2560, 512], per-head `attn_q_norm`/`attn_k_norm` [256], a QSA
indexer (`indexer.q_proj` [2560,512], `.k_proj` [2560,128], q/k norm [128],
`top_k 2048`, keys pooled over blocks of `compress_ratio`), and a per-element
sigmoid gate before `attn_output`. That is Qwen3.5's packed-query-gate shape
(`q_output_gate`) plus an indexer this engine has no code for.

The residual is **not** `x + f(norm(x))`: four hyper-connection lanes
(`[n_embd, hc=4]`, 10240 wide) are mixed by
`xn = rms_norm(per lane) * gamma; lo = silu(down*xn / 4); gate = sigmoid(up*lo);
mix = mean_lane(xn * gate); inject = w_inject*xn`, and the block output returns
as `residual_lane += block_out * 2*sigmoid(inject_lane / 4)`. There is **no
`output_norm`**: the final `hc_head_*` mixer is the last norm.

### Order of work, with what each stage can be verified against

1. **Arch + loader** (flags `hc`, `ple`, `qsa`, `attn_gate_elem`, `no_out_norm`;
   upload `hc_*`, `ple_*`, `indexer_*`, `hc_head_*`; read `compress_ratios`).
   Verified by `kraken-inspect` naming every tensor it read.
2. **Hyper-connections** (`hc_mix` / `hc_combine`): two small kernels
   (`mean_lanes`, `lane_scatter`) plus the existing gemms/norms. Verified by a
   CPU fixture whose lanes start identical (the reference starts them as `hc`
   copies of the embedding, so lane 0..3 must stay equal until the first
   combine, which the test pins). **DONE, and wired into `forward_core`:** the
   lane stack is `ws_hc_` (`[n, hc, n_embd]`, 10240 wide on this file), each
   sub-block is bracketed by `hc_enter` / `hc_leave`, and `hc_head_mix` replaces
   the final norm. The per-lane norm is the existing `rmsnorm_grouped` (the
   Mamba-2 op is exactly "one weight vector per group"), and the only new
   backend ops are `hc_init`, `hc_mix`, `hc_combine` and `silu_act`.
   Two things are worth keeping from building it:
   (a) the block is handed the MIXED row and adds its own output onto it, so
   `hc_leave` recovers the block output by subtracting the saved mix -- the
   residual sum must never enter the weights the combine applies;
   (b) `hc_leave` scatters with the mixer's per-lane `inject` OUTPUT
   (`ws_hc_inj_`), not with the `[wide, hc]` weight tensor. Passing the weight
   read f16 bytes as f32 lanes: it ran, produced finite text, and kept the four
   lanes equal -- the lane-invariance check is what caught it.
   Evidence: `test_hc_lanes` 24/24 (arithmetic at 4 x 2560 through 320),
   `test_qwen4exp_forward` 17/17 (a real 4-lane-of-2560 file through
   `Engine::init` and `generate`, the lane stack read back at 10240 wide, a
   lane-varying injection that must separate the lanes and a lane-symmetric one
   whose lanes must stay **bit-equal**, plus the PLE/QSA refusal), and
   `kraken-bench --hc` **6 of 6** on the device against the host oracle in
   `src/hc.cpp`. Removing the scatter from `hc_leave` fails the separation
   check (16/17), so the invariant is shown able to fail.
3. **PLE**: host-side n-gram hash -> 16 row indices per token, gathered straight
   out of the mmap (the table is 320,001,536 x 160, i.e. tens of GB and must
   never be uploaded), then key/value projections, per-lane dot gate, and a
   dilated depthwise conv with per-layer history. Verified by a fixture whose
   hashes and rows are computable in the test. **DONE and wired.** The row is
   `n_embd` wide (`ple_key` widens it to hc*n_embd, `ple_value` keeps it narrow),
   the conv's window is per-layer state, and `test_qwen4exp_ple` recomputes the
   rows independently from the tokenized prompt and requires chunk 8 and chunk 1
   to decode the same tokens. `kraken-bench --hc` proves the three device pieces
   (10 of 10, with the convolution's chunk invariance bit-identical).
4. **Recurrent block**: `gdn_forward` plus the `sigmoid` gate switch.
5. **Full-attention block**: packed `[q|gate]`, per-head norms, and *dense*
   causal attention first (a superset of the QSA selection, so text is coherent),
   then the indexer exactly: pooled keys per block of `ratio`, one score per
   block, top-k over blocks, mask everything else.
6. **End to end**: greedy text against `H:\LLAMA-binsocm10\llama-completion.exe`
   on the same prompt, plus a logits comparison on the first token.

## Where the graph diverged, measured (2026-10-20)

The whole-graph gate in step 6 is green now, and the divergence was found by
walking the graph host-side from the first token instead of comparing sentences:

1. **Everything up to layer 1 matched.** Layer 0 of the REAP-288 file, token 4 of
   `"The capital of France is"`, recomputed from the file's own weights against a
   `KRK_DUMP` trace: the hyper-connection mixer (`hc_n`, `hc_gate`, `hc_mix`,
   `hc_inject`) 1.7e-4 median relative, the delta net (`gdn_conv`, `gdn_delta`,
   `gdn_norm`, `gdn_gated`) 3e-4..7e-4, and the lane scatter after the block
   reproduced `hc_attn_norm` of the next layer's mix at corr 1.000000. The
   routed MoE was checked the same way on **all 48 layers**: `moe.sum` equals
   `sum_k w_k * expert_k(ffn.pre)` plus the gated shared expert at corr 1.0000
   for layers 0-34, 36-38 and 40-47 and 0.983 for L35/L39, with the routing
   weights identical to the trace's top-10. Attention was cleared separately:
   kraken's own dumped q/k/v, put through a dense softmax with the 12:1 GQA
   grouping, reproduce `attn.out` bit-exactly at position 0 and to 1.7e-4 after
   it, so the kernel and the KV mapping are right.
2. **The first disagreement was the PLE residual at layer 1.** The reference
   (`build_ple`) returns `hidden + gated + silu(conv(grouped_norm(gated)))`:
   the grouped norm exists to feed the convolution, and the *raw* `value * gate`
   is what reaches the lane stack. `ple_block` normalized in place and added the
   same buffer, so the residual received a per-lane unit-RMS row. Measured on
   token 0 (where the conv's history is one real row, so the whole block is
   computable from the trace): `ple_gate` matched at 3.5e-4 and the normalized
   row matched at 4.7e-4, while the resulting lane stack scored **corr 0.373**
   under the reference's rule against **0.9999** under kraken's. That single
   term is the difference between `(n,)))` and the reference's answer.
3. **The rope pairing was the second, smaller one.** `llm_arch_rope` answers
   `LLAMA_ROPE_TYPE_IMROPE` for this arch, and IMROPE pairs channel j with
   j + n_rot/2 (the half-split pairing) -- the table said `rope_neox=false`
   (adjacent channels). The `[11,11,10,0]` section table does not change that
   for text, because `ggml_mrope_cache_init` advances one theta per pair and only
   selects the axis position per sector, and all four positions are equal.

With both fixed, `kraken.exe -m <REAP-288> -p "The capital of France is" -n 24
--greedy` prints the same 24 tokens as `llama-completion.exe` (`-no-cnv
--no-jinja`, `--temp 0`): "
<think>
The user is asking about the capital of
France. This is a straightforward factual question. The capital of France". The
8-token arm was not enough to see the rope term: both pairings emit the
reference text at 8 tokens, and the adjacent pairing only flips at token 20
(`of France` -> `(city`).

Gates for both: `test_qwen4exp_ple` now 16/16 (it reads the block's three terms
through `ple_debug_terms` and fails 14/16 when they are re-aliased), and the
arch stays behind `KRK_ARCH_ALLOW_EXPERIMENTAL` because no suite gate pins a
whole-graph output -- the blocks are pinned, the sentence is not.
