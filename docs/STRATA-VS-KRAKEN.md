# Strata and kraken: where the designs already agree, and where they don't

Sources for everything on the Strata side: `Maxritz/Strata-rocm` `README.md`
(fetched, quoted below). The `docs/paper/Strata-Paper.pdf` text was **not**
extracted, so nothing here claims to represent the paper's internals beyond what
its README states. Everything on the kraken side is measured on this machine
(RX 9070 XT, gfx1201, 15.92 GiB VRAM) during the session that landed the
nemotron_h_moe expert-tier fixes.

## What the README actually claims

| Claim | README wording |
|---|---|
| Target | "Run a 125-billion-parameter AI model on a normal gaming PC … one NVIDIA card (12-24 GB) + 64 GB of RAM" |
| Model | Qwen3.8-Flash-Next, sharded. "Shard 1 is the part of the model that gets loaded when it starts: its experts go into your **RAM**, the rest onto your graphics card (the second shard, a 29 GB lookup table, stays on the SSD)" |
| Speed, 12 GB card | Q2_0: 90 tok/s short chat, 67 tok/s at 128K context, 1,310 tok/s prompt |
| AMD/ROCm | "A **ROCm/HIP port** runs the same models on AMD cards (`gfx1201` RDNA4, `gfx1031` RDNA2)"; on **RX 9070 XT (16 GB)**: pinned experts 34.2 GB host RAM → 14-17 tok/s answers, 380-580 tok/s prompt; `--expert-ram-gb 22` bounded → **23.2 GB** host RAM → 11-14 tok/s, "Output is coherent and byte-identical to the pinned mode" |
| Speculation | short-chat numbers are "speculative decoding" |
| Not yet supported | `qwen35moe` (Qwen3.5-35B-A3B and friends) is listed as "new architecture — see `docs/MODEL_SUPPORT.md`" |

Two things worth noting immediately. First, Strata's ROCm target is the *same
GPU as this box*, so its AMD row is comparable hardware. Second, its bounded
mode is an LRU ring over a fixed RAM expert pool that is **byte-identical** to
the pinned mode — the same invariant kraken enforces with
`scripts/coherence_check.sh` (device arm vs `--cpu` arm, diffed).

## The four mechanisms, mapped

| Strata mechanism | kraken's counterpart | Measured here |
|---|---|---|
| GPU holds the part used for every word **plus the hottest experts** | HOT tier: VRAM expert budget, LFU+aging, pins | 2074 experts resident of 2944 in the routed set; budget 11102 MiB |
| RAM holds all experts | WARM tier: pageable host pages, read-through on miss, eviction is free (the file is canonical) | 15758 MiB capacity, 2944 held (100%), 0 rejects |
| SSD holds a large lookup table; a few rows read per word | weights are read from the mapping/file on demand (`Backend::read_host`, batched `read_host_batch`) | **0.00 MiB/token** once tiering worked; was 918.63 MiB/token when it didn't |
| CPU computes the few experts the GPU doesn't have, concurrently with the GPU | `CpuExpertPool` / `--hybrid-experts` | **−3.1×** when it engages (per-layer merge barrier; see AGENTS.md "Known open items" and docs/test-results.md §9) |
| A small in-model helper guesses the next words; several guesses verified at once | draft/MTP path (`--draft`, `spec_steps`, device top-k verification) | was a 2.7× net loss, caused by a call-site pulling the whole logits row per proposal; both call sites now use `fetch_logits(true)` / `topk_argmax` |

Kraken already runs `qwen35moe`, which Strata's README lists as an unsupported
architecture — and kraken runs it with a CPU scalar reference arm that must
match the GPU token-for-token. That is the one axis where this engine is ahead,
and it is a correctness axis, not a speed one.

## The honest comparison to the AMD numbers

They are **not** comparable, and nothing here should be read as one:

- Strata's 14-17 tok/s is a **125B** model with ~33 GiB of experts, complete and
  coherent, on the same card.
- kraken's 88-92 tok/s is a **30B-A3B** nemotron whose 23 mamba-2 blocks
  currently contribute **nothing** to the residual (the SSD scan is
  unimplemented), so its output is degraded and its speed is a *lower bound on
  cost*, not a useful throughput figure. Its routed set is 15.4 GiB against a
  10.8 GiB device budget (70%), with the remainder served from RAM.

What the fix did prove is that this engine's tier design behaves as specified
when the bookkeeping is right: 92% HOT / 8% WARM / 0% COLD per token, zero file
reads, and the device ceiling respected at 2074/2074 slots.

## What is worth taking from Strata

1. **An explicit, bounded expert-RAM knob with the trade documented.** Strata's
   `--expert-ram-gb` is what lets a 12 GB / 48 GB-RAM machine run at all, and the
   README states both arms (34.2 GB at 14-17 tok/s vs 23.2 GB at 11-14 tok/s,
   byte-identical output). Kraken's WARM budget is automatic (half of free RAM);
   an explicit cap plus a published trade table would make the same promise.
2. **An SSD tier that is a design input, not a fallback.** For a 24,576-expert
   corpus the WARM tier cannot hold the routed set, and AGENTS.md already records
   the read-path ceiling that matters then (593 MiB/s at queue depth 1 against
   3400 MiB/s sequential). Strata treats the second shard as permanently
   SSD-resident and reads rows per word.
3. **Byte-identical output across residency modes.** Strata's README asserts it;
   kraken can prove it per change with `coherence_check.sh`.
4. **Re-examine the CPU-overlap path — but measure, don't copy.** The user's
   description has the CPU computing the experts the GPU lacks *concurrently*;
   this repo measured that arm losing 3.1× because a per-layer merge barrier
   forbids the overlap the paper's ceiling assumes. Either the barrier can be
   removed (a real design change) or the claim does not transfer to this
   architecture. That question is open and is the highest-value one on this list.

## Measured head-to-head: the same card, both engines (2026-10-09)

This box is the card both sides target (RX 9070 XT, gfx1201, 15.92 GiB VRAM), so
the AMD rows are comparable hardware. One command each:

| engine | model | prefill | decode | arms on |
|---|---|---|---|---|
| kraken | `Qwen3.8-Distill-35B-A3B-Coder-Abliterated-Q2KXL` (12.3 GB, 40 layers x 256 experts) | **558.8 tok/s** (1000 tok / 1789.7 ms) | **35.8 tok/s** (27.9 ms/step) | `--expert-warmup 1`, fp16 KV, greedy, no speculation |
| strata | Coder IQ1_M, RX 9070 XT row of `docs/AMD_HIP.md` (validated 2026-09-30) | 782 tok/s @4K, 1235 @16K | 30.8 tok/s @4K, 35.4 @16K | `--spec 4 --spec-min-p 0.5`, MTP, `--kv int8`, `--max-context 32768` |
| strata | 125B model, README ROCm row | 380-580 tok/s | 14-17 tok/s pinned / 11-14 bounded | `--expert-ram-gb` arms |

The kraken command, for reproduction:

```
kraken.exe --model G:/More-models/Qwen3.8-Distill-35B-A3B-Coder-Abliterated-Q2KXL_ROCMFPX.gguf \
  -p "<the abstraction sentence x50>" -n 16 --greedy --ctx 2048 --expert-warmup 1
```

Three things this settles.

1. **Decode is at parity, not behind.** kraken's 35.8 tok/s carries no speculative
   decoding, no MTP and no int8 KV; Strata's 30.8-35.4 carries all three. The
   14-17 tok/s row is a *125B* model on the same card and is not this comparison.
2. **The ~3x gap that reads as "Strata is so much better" is the cold tier, not the
   kernel.** On the same file and command kraken reads 16.5 tok/s while the expert
   corpus fills on first touch -- `[stats] expert-path read 438.8 | room 0.1 |
   alloc 92.2 | xfer 675.4 ms`, i.e. 29.4 ms of a 60.6 ms step is promotion, and the
   corpus *fits* the 9600 MiB budget (10240 slots x 0.937 MiB) -- against 35.0/35.8
   tok/s once the same run starts warm (`--expert-warmup 1`: HOT 100%, 0
   promotions, 0 alloc, 0 COLD). Same binary, same prompt, same output.
3. **Where Strata is genuinely ahead, named precisely:** long-prompt prefill (782
   @4K and 1235 @16K against 558.8 @1K here -- prefill per token *rises* with
   prompt length there, so its batched prompt path is the thing to match), MTP +
   spec 4 + int8 KV already wired, and a two-GPU layer split (R9700 alone
   1016-1794 tok/s prompt, 38-40 tok/s decode).

### The MTP head is an artifact, not a training project

`C:\Strata-HIP-data\mtp\` is the record, and it is worth more than the speed table:

- `mtp-inventory.md`: **"MTP block in the BF16 checkpoint -- 31 tensors, 5.214 GB,
  in 28 shards."** The head ships *in the model's own checkpoint*. The tensors are
  `mtp.fc_embedding` / `mtp.fc_hidden` (2560x2560), `mtp.hyper_connection_mixer.*`
  (a 320-dim down/up bottleneck), `mtp.layers.0.*` with its own attention
  (`q_proj` 12288x2560 = the 2x gated query, `k/v_proj` 512x2560, an indexer) and a
  512-expert MoE (`gate_up_proj` 512x1280x2560), plus `block_inject_weight`
  (4x10240) and the `pre_fc_norm_*` weights.
- `mtp-q2_0.gguf` -- 889,014,272 bytes, with `mtp-q2_0.gguf.report.json` naming the
  format of every tensor: the 5.214 GB BF16 block becomes an 889 MB drafter.
- Strata's config reaches it with `--mtp G:\Strata-data\mtp\rt` beside
  `--spec 4 --spec-min-p 0.5`.

kraken's side of that gap is one function: `src/arch.cpp:113` **recognizes and
refuses** an `mtp.*` shard ("multi-token-prediction head shard rather than a
model"), and there is no `--mtp` and no draft-head loader. So "can we generate an
MTP" has a better answer than yes: there is nothing to train -- the tensors are in
the checkpoint these GGUFs are converted from, and the missing pieces are an
extractor (checkpoint -> small GGUF), a loader, and the draft/verify loop the
`--draft` path already has.

### The same-file test, and what it actually found

The user's own Strata config points at a file that is on this box: the<br/>
REAP-reduced 288-expert Flash-Next `qwen3.8-flash-next-reap-288-Q4_K_M.gguf`, with a
1.4 GB pack beside it (`dense.bin`, `index.txt`, `native_experts.txt`, `tokenizer`)
and the live config `G:\Strata\strata-reap_288.json`. Strata's own log for it
(`G:\Strata\strata-reap_288.log`, 2026-10-08) reads: **42.19 GiB loaded at 5.01
GiB/s**, 2443 expert slots (7.45 GiB VRAM, ranked by routing frequency, no eviction,
pre-filled from the profile in 1.2 s at 6747 MB/s), `--kv int8 --kv-resident 16384`
(20,480 of 65,536 cells per layer in VRAM, the K/V in 0.77 GiB of pinned RAM),
`--spec 2`, prompt chunk 8,192 tokens, and a 45.3 GB pinned arena.

kraken on that same file, exit 1, in its own words:

```
[warn ] architecture 'qwen4exp': not in the architecture table
kraken: cannot run this file ('qwen4exp'). uses an attention output gate
(attn_gate), which this engine does not implement. uses a fused query/key/value
projection (attn_qkv), which this engine does not implement. uses a Mamba-style
SSM block (ssm_conv1d), which this engine does not implement. uses a Mamba-style
SSM block (ssm_out), which this engine does not implement
```

So for **this** model the comparison is not "slower", it is **not runnable**, and the
gap is a model family rather than a kernel: gated attention + fused QKV + an SSM
block. It is also the failure the repo wants -- loud, per-tensor, before a weight
load, which is `src/verdict.cpp` doing its job. The throughput table above is
therefore only meaningful on files both engines load, which is why it uses the
256-expert Qwen3.8-Distill-35B-A3B: Strata's README lists `qwen35moe` as
unsupported there and this engine runs it against a CPU oracle.

One further measured datum from Strata's own startup, directly on the RDNA4
question: it prints that `STRATA_HIP_WMMA=1` "reads prompts about **30% faster** on
this card with `--kv int8` (prompt attention on the matrix cores); it is off by
default because the output bits change (last-place rounding)". A 30% prompt win
from an instruction-class switch, priced at bit-level divergence -- the same trade
this repo's md5 discipline exists to make explicit.

### What supporting `qwen4exp` would actually need

The file's own tensor table (1224 tensors, `general.size_label 288x56B`, 48 blocks,
`full_attention_interval 4`) names five things this engine has no code path for, and
they are not refinements of the ones that exist:

| tensor class | what it is | closest thing here |
|---|---|---|
| `blk.*.attn_gate.weight` [2560, 6144] | a gate multiplying the attention output (6144 = 24 heads x 256) | `qwen35`'s gated attention rides inside `q_proj`; this is a separate tensor, and nothing reads it |
| `blk.*.attn_qkv.weight` [2560, 10240] | one fused QKV projection | the q/k/v `gemm_group` with per-matrix `wts[]` is the launching machinery; the *unpack* is missing |
| `blk.*.ssm_conv1d` [4, 10240], `ssm_alpha/beta` [2560, 48], `ssm_a` [48], `ssm_dt.bias` [48], `ssm_norm` [128], `ssm_out` [6144, 2560] | a Mamba-style SSM block: conv kernel 4, state 128, 16 groups, dt_rank 48, inner 6144 | `Engine::gdn_forward` is the same *family* (a gated recurrent state); none of these tensor names appear |
| `blk.*.hc_{attn,ffn}_down/up/inject/norm`, `output_hc_*` | hyper-connections: a 320-dim low-rank bottleneck mixing the layer input with the block output (`hyper_connection.low_rank 320`) | nothing -- and Strata stages the down-projection read with `cp.async`, which is where its cost lives |
| `per_layer_token_embd.weight` [160, 320001536] Q5_0 | the PLE n-gram table: 320,001,536 rows x 160 dims -- the ~29 GB "lookup table that stays on the SSD" in Strata's README, read a few rows per token (`--ple-io direct`) | nothing; the KV tier pages a per-layer cache, not rows of a 300M-row table |

Plus `qwen4exp.attention.indexer.*` (4 heads x 128, `top_k 2048`) and
`attention.compress_ratios` -- a sparse-attention indexer over compressed KV, which
is the `HiSparse`/`OasisKV` shape from the research digest, and the reason Strata's
`--kv-resident` sizes the **QSA layers only**.

The good news in the same table: the MoE side is already this engine's shape -- 288
experts, top-10, `ffn_gate/up_exps` Q4_K, `ffn_down_exps` Q8_0, a shared expert with
its own gate and the same feed-forward length -- so routing, grouping and the expert
tier would port. The gated attention, hyper-connections, SSM block and PLE are new
arms, not new tuning.

## Not verified here

- The paper's own text. Only the README was read.
- **Strata was not executed here.** Its CLI is `strata generate --pack DIR
  --tokens "1,2,3"` and its weights arrive as a per-model *pack* (configs
  `G:\Strata\strata-*.json`, data `C:\Strata-HIP-data\`), so a same-file A/B needs
  a pack build first; the Strata rows above are its published numbers, not a run of
  it. `strata.exe` also exits 127 with an empty-looking error until `hipblas.dll` is
  on PATH (`G:\ROCM10RT-gfx1201\bin`).
- Strata's expert tiering internals, its speculative scheme, and its SSD row
  layout: not inspected.
- Any kraken throughput number for a *complete* nemotron run: the mamba-2
  forward is still missing.
