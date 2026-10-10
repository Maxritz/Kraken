# Pending-work analysis — everything open, with a fix path for each

Companion to [TODO.md](TODO.md): that file is the live list; this one is the
analysis pass over it — every open item inventoried, its measured evidence
named, and the cheapest concrete fix path (files, switches, falsifier)
written down so the next session starts executing instead of re-deriving.
Prepared 2026-10-11 after the fused-attention-chain commit (`b36af97`).

---

## 0. Corrections to TODO.md found during this pass

Three entries are stale relative to what has since landed. Do not treat them
as open:

1. **T8/G4 "tiered KV cache: zero implementation" is DONE.** TODO cites
   `engine.cpp:275` "one flat alloc"; the tree now has `KvTierCache`
   (VRAM HOT / host WARM / `--kv-cold-dir` COLD), per-layer packed KV
   geometry, and `scripts/kv_tier_geometry_check.sh` checking the
   reservation against each file's own header — 49 models ≤ 8 GiB pass
   arm B, 0 failed, mixed KV widths load (the per-layer geometry work,
   2026-10-10). The README also documents `--kv-hot-mb/--kv-warm-mb/
   --kv-cold-dir`. What remains of T8 is only the spec's *radix prefix
   reuse* and *persistence*, which are server features, not engine ones.
2. **G2 "CPU expert fallback: the 6-10x lever" is BUILT and measured a 3.1x
   LOSS** (TODO updated in one place; keep the correction traveling): the
   budget that holds the routed set makes it inert; a 16 MiB budget runs
   all experts on the host at 5.392 ms/expert (8 threads) vs ~2.0 ms of
   read+promote — the host arm is a per-layer merge barrier. It is a
   memory feature, not a speed feature. Do not re-attempt without removing
   the barrier (see §3, item M1).
3. **T11 quant ids: closed.** Everything in 100..143 is implemented host and
   device except the arch-gated refusals; the only open quant-adjacent item
   is the `qwen4exp` registry entry (below).

Also stale-forward: the README's "Fusing the rest of that chain is the next
move" claim was replaced on `e611575` with the landed one-launch chain
(+10.2% decode on SmolLM2, order-balanced).

---

## 1. P0 — correctness (the open ones)

### C1. Fused attention chain diverges at NeoX + qk_norm — OPEN, gated

Recorded 2026-10-11 in TODO. Evidence on disk:
`/tmp/de0.txt` (separate arm), `/tmp/de1.txt` (fused arm), `/tmp/dk0.txt`,
`/tmp/dk1.txt` (with the `attn.kc` cache-row dump rows). Shape: Qwen3-8B
(NeoX, freq_base 1e6), first divergence `attn.out L07` head 9 only,
39/4096 elements, 1–6 f16 ULP.

Fix path (in order of cost):
1. **Reproduce the head-9-only signature with a micro-probe**
   (`tools/probe_attn_qtile.hip` is the template for hosting a kernel-side
   dump): one block, hd=128, NeoX, and a synthetic q row whose norm-scale /
   rotation exercises the same lane arithmetic. 30 minutes, and it turns a
   4 GB model load into a daemon-less reproducible.
2. Check the **f16 double-narrow at the rope store**: the fused NeoX arm
   computes `o` in f32 and stores `(_Float16)o` per channel, while the
   separate chain stores f16 AFTER the norm in one place and reads it back
   for rotation — if the fused arm rotates
   `f32(norm-f16) * cos` while the separate chain rotates
   `f32(f16(norm)) * cos` with a DIFFERENT f16 rounding placement, that is
   exactly a 1-2 ULP, single-lane-slice signature. The reconcile experiment
   (rotate arm1's `attn.q` residue, compare against arm0's) failing
   arm1→arm0 in BOTH directions at L07 while `attn.kc` agrees is consistent
   with this.
3. If it is the rounding placement, the fix is to make the fused arm store
   the normed f16 row to `ws_q_` FIRST (as the separate chain does), then
   rotate from global — still one launch, no extra round-trip vs today
   (the kernel already reads the row once), and it collapses the
   divergence by construction rather than by argument.
4. After any change: the three hash contracts (SmolLM2
   `d6e2b2b3b00b3`, Qwen3-MoE `995c6b60e743e`, 8B both arms
   `c70cb85baa1fc`) plus 9-pair interleaved A/B on 8B and SmolLM2.

Cost: small. Value: unlocks the fused chain for every NeoX qk_norm model
(Qwen3 family of 4B/8B/30B shapes) — that is the majority of the
collection's dense models.

### C2. Spark_one.Q6_K drift — revisit AFTER three stale assumptions re-run

The table in TODO documents five eliminated hypotheses, but the entry
itself flags the fatal one: the "deterministic ⇒ not a race" row ran on a
build **with the split-K race in it** and never launched the split path.
The race is fixed now. The exact re-run list, in order:
1. 5-prompt drift table on the CURRENT build (3/5 passed before; does it
   still split 3/2?).
2. GPU x5 md5 at `-n 96 --ctx 1024` (cross the 128-key split threshold the
   old probe never reached).
3. If still drifting: the discriminating test the entry names — device-vs-
   host Q6_K dequant comparison over a REAL tensor (whole-tensor agreement
   is currently unverified; the suite only checks one synthetic block).
4. Only then the fp16-vs-f32 tie argument (top-2 gap 0.52) gets to be
   invoked.

Cost: an hour of runs on the current build; no code until step 3 says a
kernel is wrong.

### C3. G13 — olmoe-1b-7b wrong text, shared path, both arms agree

Ruled out: tokenizer, rope (NeoX norm matches fallback), MoE shapes,
routing, numerics (finite, plausible-but-wrong). The device-vs-cpu gate
cannot find it because both arms walk the same path.

Fix path: the TODO-names it — compare against **llama.cpp HIP**, not
against kraken's own CPU arm (T25). `H:/LLAMA-bins/rocm10/
llama-completion.exe --no-cnv` is on this box (remember
`G:\ROCM10RT-gfx1201\bin` on PATH or it silently runs CPU-only), and
`olmoe-1b-7b-0924-q4_k_m-imat.gguf` is on maclin. Note maclin's state must
be re-checked (it was down at the last TODO update; a `/tmp/q9b.gguf`
tmpfs artifact was lost there).

Also required, cheaply, first: **add an `olmoe` arch-table entry (G15)** —
geometry matching "by luck" means the key builder may be reading the wrong
keys. One table row + the load-time warn stops; that alone can change what
the loader thinks the model is.

### C4. G14 — linking the HIP backend changes host FP results

2152/2154 vs 2154/2154, Provision FTZ/DAZ ruled out. Mechanism unknown;
it can move the reference the coherence gate judges against, so it is a
correctness item, not hygiene.

Fix path: bisect by symbol, not by guess. The linking order means some
object file's static initializer or CRT linkage differs. Candidates:
DLL-injected CRT state, `atexit` handlers in the HIP runtime, or a
different `libm` resolution. Concrete probe: link a CPU-only test binary
against one extracted `.o` at a time from `backend_hip.hip`'s object set
(they are few — `krk_hip`, `op_time`, `dequant`). 1–2 hours. If it is an
initializer, the fix is to make the offending object's global state
explicitly initialized, which is mechanical.

### C5. DFlash proposer still diffuse (acceptance 0% matching the
reference's 0.0% on the official pair)

Open P0 only in the sense that proposer feature semantics/head weights are
unverified. With F8 (self-draft ceiling) now FIXED (~100% acceptance), the
loop is dead innocent — so the remaining 0% is genuinely the proposer.
Fix path: `--debug-topk` on a drafter-vs-target same-position comparison
MUST be re-run on the post-F8 build (the MTP/DFlash losses predate it);
`docs/DFLASH-DRAFTER.md` protocol. The DFlash2 loader work (conv1d block +
candidate selector) is the separate capability item below.

---

## 2. Races and diagnostics

- **`src/http.cpp:456` listen_fd_ data race** — make it atomic or give
  ownership to `run()`. Mechanical, 20 minutes, then rerun the TSAN build
  (1383→2905-era baseline) to confirm the count drops.
- **`tests/test_kraken.cpp:1303` heap-use-after-free in `test_gdn_ops`** —
 -test-lifetime suspicion: `CpuBackend` outliving the engine whose buffers
  it dereferences, or a captured view into Tokenizer memory freed by a
  later test's engine teardown. Fix is a test-ownership fix (own the
  backend in the test, or deep-copy at capture). Do NOT dismiss without
  confirming, per house rules.
- **TSAN baseline is stale** (1383-era counts vs the 2905 suite): re-run
  `build-tsan` once on the current tree and update the entry whatever the
  count becomes.

---

## 3. Performance — ranked by measured size, with fix paths

### M0. The 28%-of-decode promotion stall — the direct (Strata-shaped) fix

Evidence: 1,010 WARM→VRAM promotions of a ~1.86 MiB DMA at ~700 us,
serialized inside the per-expert loop against 63 steps' worth of GEMMs,
`read 0 | room 264 | alloc 263 | xfer 184 ms` per decode. The tier cannot
hold the routed set (12.7 of 19.43 GiB; VRAM free is 1.48 GiB), so this is
structural, not a budget misconfiguration.

Fix path, in expected size order:
1. **Route-ahead prediction for the NEXT layer** (Strata's look-ahead
   thread shape): the current layer's hidden state is already in VRAM
   before the layer runs; its router row is one 1.34 KB read (`n_expert`
   wide) and computing it against the router is a `n_embd × 256` GEMV on
   a small host copy (or a second tiny device launch). Prefetch the
   *next* layer's routed set into a DOUBLE-BUFFERED staging area with one
   `prefetch_layer`-style async batch while this layer's attention runs.
   This is the direct successor to `ExpertCache::prefetch_layer`, which
   already exists for file reads (`read_host_batch`, 593→1446 MiB/s) but
   has no VRAM-promotion twin — TODO names exactly this as "the untried
   shape".
   Falsifier: decode tok/s with `KRK_ROUTE_AHEAD=1` vs 0, 9 interleaved
   pairs on laguna-xs2. Expected ≥ +10% if the promotions actually
   overlap; the 710 ms/2550 ms stall share bounds the max win at ~28%.
2. **Reuse-distance is 1/2/6 tokens** on the 35B file (p10/50/75) —
   a STAGE window two layers deep needs no predictor at all: prefetch
   layer l+1's experts UNCONDITIONALLY two layers early, sized by the
   stage window. No router needed for the first version; the predictor is
   a cold-start tool only while the corpus fits (AGENTS.md records this
   for the 256-expert file; keep it as the fallback when routing-ahead
   hurts).
3. Do NOT re-seed ranked warmup into eviction counters (measured, +38%
   promotions, −13%); `KRK_EXPERT_SEED=1` stays the off-by-default arm.
4. Gate every arm behind `KRK_ROUTE_AHEAD` and keep the same binary in
   both arms of the A/B (one process apart is the rule here).

### M1. MoE decode is still host-orchestrated (69 ms/token, 7.3% compute-
busy, 13 GB/s)

The next lever after M0 is the **device-side routing plan** (T4): 40
blocking router downloads/token at 512 bytes each, plan+alpha re-uploaded
per expert, O(n_exp·k) host scan per layer. Fix = one per-step device
plan (top-k on device, one upload of the batched gather/scatter indices),
which also removes the explicit router sync today (`KRK_ROUTER_SYNC`
restores it). This is also HIP-graph capture blocking (T5) — a graph
bakes in weight pointers that move with evictions, so fixed expert slots
come with it; that is why T4 first.

### M2. Dense decode launch count (the P1c line 3) + per-op trace

The `attn_fused` chain landed (+10.2% SmolLM2, commit `63eadae`). The NEXT
folds in the same shape, ranked by op table share on SmolLM2's post-fuse
243 ops/step:
- **norm+silu+residual chains** into the GEMV epilogues where the fused
  kernel's own `gemm_group` already does residual + silu (the silu fold is
  measured +6.1%, the residual fold +2.9%, the pair +8.3% — on the same
  binary, one switch apart).
- The remaining `rope`-shaped launches that a fused chain refused (NeoX
  mostly, C1 above).

### M3. download_f32 = 21% of the 8B step

Device-side softcap + top-K candidates, copy back 8–128 pairs
(P1c item 4), with the tail-mass guard the entry names (host filters must
provably never need a candidate they were not offered). This is a P1c
kernel work item, not MoE.

### M4. gfx1201 Q6_K/Q5_K decode (P1c item 1)

Q6_K at 143 GB/s on a cache-fitting matrix = instruction-bound. Fix is
already named: packed-f16 decode (`v_pk_mul_f16`/`v_pk_fma_f16`) with a
branch-free 6-bit unpack. Falsifier: 4096×4096 Q6_K > 400 GB/s. This is a
self-contained kernel task with a bench that already exists
(`kraken-bench` dequant path).

### M5. gfx1201 prefill dequant staging (P1c item 2, T2)

22% of the memory-bound ceiling, 55–97% of time before WMMA sees data.
Double-buffered LDS → producer/consumer split → square decode into VGPRs.
Target > 40 TFLOP/s at rows=32. Largest singlegfx1201 headroom, self-
contained.

### M6. gfx1031 leftover (P1b): Q6_K dp4a budget fixed, dispatch time
missing

`kDp4aEnableQ6K` stays false until a measurement on maclin; the maclin
state must be re-verified (it was down; /tmp/q9b.gguf lost). If maclin is
gone, this becomes a Windows-side `hipcc --offload-arch=gfx1031` ->
`-S` + emulation-free instruction count + one dispatch on the bench — the
harness already exists (`kraken-i8k`).
Also: `kraken-i8k` on gfx1201 reads back all-zero codes with a nonzero
output (open); `KRK_I8K_DEBUG=1` exists for it.

---

## 4. Capability gaps (each a named refusal today)

| Gap | Unlocks | Fix path (concrete) |
|---|---|---|
| **`qwen4exp` arch entry** | The 288-expert REAP GGUF + the whole Qwen3.8-Flash family; needs `attn_gate`, fused `attn_qkv`, `ssm_conv1d`, `ssm_out` — i.e. MTP-drafter work too | Table row + the four named classes; `docs/AUDIT-plan-vs-code.md` is the per-piece inventory |
| **MLA (deepseek2/deepseek4)** | GLM-4.7-Flash, DeepSeek-V4-Flash-files refused today | See §5 — this is the paper path |
| **Sliding-window attention** | gemma2/gemma3-4B/30B-A4B files | `AttnDesc::window` exists (laguna uses it); what is missing is per-layer window patterns and the four HIP kernels' bound in the OWN kernels at decode — C1's chain gate interacts |
| **gemma4 geometry** | gemma-4-E4B-it, gemma4-coding | Per-layer input embeddings + shared-KV 18-layer shape |
| **MTP / dflash / dspark draft-head loading** | Speculative on every model that ships a head set | The MTP artifact exists (`C:\Strata-HIP-data\mtp\mtp-qwen4-mtp-bf16.gguf`); loader/extractor + `--draft` verify loop is the gap (AGENTS.md F8 note) — NOW actionable since F8 is fixed |
| **Radix prefix cache**, concurrent batching | server throughput | RadixAttention; needs Tokenizer trie + recurrent rollback (Tail-Replay paper is the named first read) |
| **LLaDA loader** | diffusion model files | Schema already sketched in docs/LLaDA.md |
| **Architecture pass 44/87** | named dequantizers for the rest | Each refusal needs a named reason or a quant path |

---

## 5. The papers the user supplied — what each enables HERE

Mapped to this engine's actual refused-file list, not to research generally.

### DeepSeek-V2 (arXiv 2405.04434) — MLA
**This is the direct unlock for the `deepseek2`/`deepseek4` refusal rows.**
Per-token cache is one 512-dim latent `c_KV` plus one shared 64-dim
rope key `k_pe` (576 scalars/token/layer, vs MHA's 2×heads×head_dim).
Decode uses *absorbed* attention: fold each head's `W_UKᵀ` into the query
side (`q̃ = W_UKᵀ q_C`), attend against `[c_KV, k_pe]`, and up-project the
value side after the softmax via `W_UV`. What kraken would need: a latent
KV layout in `KvTierCache` (it already packs per-layer widths — MLA's
per-layer width visibility is by construction), two new W-matmul shapes
(absorbed q-side, value-side up-proj), and the attention kernels reading
the two-plane cache. RoPE's decoupled slice is what makes absorption
legal, and the repo's NeoX/adjacent work is directly relevant. Est. 2–3
kernel files + loader schema. **Do this before chasing V4.1-Flash; MLA
is the prerequisite for both deepseek refusal families and it is a
bigger scorer than any single next paper.**

### DeepSeek-V4.1-Flash (arXiv 2609.19969) — FP4 KV + CSA2 reuse
Two pieces are directly liftable: 
1. **FP4/FP8 KV** on the main (non-SWA) tier with one `UE4M3` scale per
   16 channels, quantized after RoPE. For kraken this is a NEW invocation
   of the expert-cache's existing block-scale machinery, applied to the
   KV tier's HOT/WARM bytes: 4× the KV the tier holds. The repo already
   has precedent for narrow stores with per-row exponents
   (`add_residual`'s per-row pow-2 exponent rides storage exactly this
   way) — a KV analogue is a per-block (not per-row) version at
   a different site. Est: one dequant step in the attention kernels and
   a quantizer on append. **Gate it behind `--kv-fp4 0/1`**; it IS lossy
   and the repo's house rule (bit-identical vs reference before enabling)
   applies to the ENABLE check, not to the flag being off by default.
2. **CSA2's Reuse/Reindex layer sharing** is architecturally costly and
   the SWA/FP8 boundary cut conflicts with the laguna window handling.
   Take the FP4 idea; leave CSA2's layer-sharing until a real
   deepseek-v4-family file is runnable (i.e. after MLA).

### TransMLA (arXiv 2502.07864) — GQA→MLA conversion
**A research note for this repo, not a feature.** It converts GQA models
post-training, so the work is in a fine-tuning tool, not an inference
engine — but its *mechanism* (RoRoPE concentrating positional info into
a small key subspace, BKV rescaling, then absorbed attention) means a
fine-tuned GQA model could be shipped as an MLA GGUF and run with the
MLA machinery from the V2 bullet. Sequencing implication: build the MLA
runtime first; conversion tooling is a separate artifact, and until a
converted Qwen/Llama file exists on this box, this paper buys no
kilogram-free win here.

### PiKV (arXiv 2508.06526) — MoE KV sharding / scheduling
**Mostly not applicable (multi-GPU sharding).** Two pieces are:
1. The **page-level utility-evaluation menu** it catalogues
   (H2O, LRU+freq, AdaKV multi-feature scoring, layer-summed Duo) —
   kraken already has LFU+aging; PiKV is a menu of what to try next if
   M0's route-ahead prefetch does not close the gap. Treat as a policy
   research menu, not a port.
2. **Cache-aware routing** (`−λ·log(1+miss_e)` penalty) literally
   changes which experts the model selects — in this repo's house rules
   that means it changes the output and it cannot be default. It would
   be `--moe-route-penalty λ` opt-in, and it competes with the free
   scheduling wins (M0/M1) that do NOT touch outputs. **Prefer M0/M1;
   keep PiKV's penalty idea on the shelf unless the cache misses
   survive them.**

### Strata's repo (Niko1221/Strata) — the direct competitor shape
Note: the paper PDF would not extract; the mechanism below is from
DETAILS.md/HOW_IT_WORKS.md/README on the repo and is vendor-reported,
not independently measured.

**Kraken already has the memory handler shape Strata has**: VRAM HOT
(experts the router selected) + host WARM (page-locked demotions) +
file-back COLD fallback + a ranked warmup index + adapt-to-conversation
swaps. Where Strata is AHEAD of kraken, by its own docs:
1. **Look-ahead routing for prefetch** — applies the *next* layer's
   router to the current layer's input while CPU work proceeds, and
   warms OS pages / issues whole-blob reads for predicted experts. This
   is exactly kraken's M0 item 1 and TODO's "the untried shape"; build
   it first, it is the highest-expected-value next work and the repo
   already has `prefetch_layer` + `read_host_batch` as building blocks.
2. **Async swapping / buffer-ownership rotation** — Strata's docs spell
   out rotating buffers so the DMA is NOT serialized against the layer's
   compute. That is the second half of the M0 fix (the first half is
   prediction; the second half is overlapping it). Kraken's promotions
   today are issued inside the per-expert loop on the default stream —
   the exact shape Strata finished.
3. **Overlap of expert streaming with attention during PREFILL** —
   Strata attributes its 0.1.13 prefill jump (572→1290 tok/s Q2_0@32K)
   mostly to this. Kraken's prefill expert path has no equivalent
   overlap and TODO's missing item "a prefill `[stats]` expert-path
   line" is the prerequisite diagnostic — build the diagnostic first (an
   hour), then the overlap.
4. **MTP speculative decoding** — Strata reports 1.6–1.8× with the MTP
   layer as a drafter. Kraken has the MTP artifact extracted but no
   loop; with F8 (self-draft ceiling) now fixed, the drafter loop is the
   only remaining gap.

Where kraken is ahead of Strata: a real multi-format dequant-matrix
(GGML quant zoo on both host and device), a correctness gate culture
(md5 contracts, coherence checks, baseline guards), and a much larger
measured-model corpus.

**"Can we tweak the MoE selection?" — the precise answer**: the *router*
(which experts the model picks for a token) is the model's own weights;
changing it changes outputs and must stay default-off. The *selection
pipeline around it* is all engine: what is resident (ExpertCache policy),
what is prefetched (route-ahead + stage window), when it is overlapped
(async swaps), and what is computed where (CPU fallback, measured a loss
on this workload — do not re-enable for speed). That is exactly where
Strata is ahead, and it is the plan in §3 M0–M1.

---

## 6. Sequencing (what to start next, and why this order)

1. **C2's re-runs** (Spark_one on the current build, split-threshold
   probe) — an hour, closes the stalest correctness claim.
2. **C1 micro-probe** (fused NeoX ULP source) — unlocks the fused chain
   for the Qwen3 dense family; the repo now has the dump rows to verify
   with.
3. **M0 route-ahead prefetch (Strata-shaped)** — the top measured
   performance item, two building blocks already exist, falsifier is one
   switch apart (`KRK_ROUTE_AHEAD`).
4. **The prefill expert-path `[stats]` line** (an hour) then **M1
   device-side routing plan**.
5. **TSAN items** (http.cpp race is 20 minutes; test lifetime fix).
6. **Then MLA (DeepSeek-V2 path)** — the biggest capability unlock per
   unit of kernel work, with the FP4-KV idea as its follow-on.
7. G13/G15 (olmoe entry + llama.cpp reference) whenever a shared-host
   window opens, since it needs llama.cpp on the same box.

Anything marked CLOSED/negative in TODO (re-seeding eviction counters,
zero-block skipping, self-draft-as-speed, CPU-fallback-default-on) is
excluded by measurement, and the tests are ahead of what the entries
said.
