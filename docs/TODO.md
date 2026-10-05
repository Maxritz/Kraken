# KRAKEN — pending work and open issues

Live list of what is unfinished, what is broken, and what is blocked. Ordered
by severity, not by age. Last updated after commit `7337f3b`.

Status of each area is in [docs/STATUS.md](STATUS.md); this file is the
"what still needs doing" view of the same project.

---

## Laguna (poolside) — runs end to end; what is left is verification and paging

Read off `G:/More-models/Laguna-S-2.1-UD-Q4_K_M-000{1,2,3}-of-00003.gguf` and
cross-checked against llama.cpp's `src/models/laguna.cpp`, which is the
reference for every piece below. Laguna-S is 48 layers, 3072 embedding,
262144 context, 256 experts top-10 of 1024-wide, one shared expert.

**Runs now.** The engine side landed: per-layer query width, the softplus
per-head output gate, the sigmoid + selection-bias router with the
sum-norm and 2.5 scale, and the 512-token window in all four attention
kernels (`AttnDesc::window`). The arch entry is `Yes`. `laguna-xs2` answers
`The capital of France is Paris.` on the device, matching llama.cpp, and
`Laguna-S-2.1` parses as an 814-tensor shard set.

The last blocker was not a laguna feature at all: **RoPE's pair convention
was wrong engine-wide** (adjacent channels everywhere, while the NeoX family
— qwen2/qwen3/qwen3moe/gemma/laguna — pairs channel `j` with `j + n_rot/2`).
The write-up is [STATUS.md](STATUS.md) §10d; the fix is `ArchSpec::rope_neox`
plus `KRK_ROPE_NEOX=0/1`.

**Landed.** The architecture is out of the gap map (`attn_gate`/`exp_probs_b`
are exempted for `laguna`), the schema is read, and `laguna` refuses with a
reason that names the real remainder instead of a flat "not implemented":

- `attention.head_count` is an **array** — 48 on the full-attention layers and
  72 on the windowed ones, which is why `attn_q` is `[3072, 6144]` on one layer
  kind and `[3072, 9216]` on the other, and why the loader no longer requires
  `n_embd % n_head == 0`. `Model::n_head_at(l)` / `q_dim_at(l)` give the
  per-layer width; `q_dim()` is the maximum, which is what a shared workspace
  sizes to.
- `rope.dimension_count` is now read for **every** arch, not only recurrent
  ones. It was a latent bug: a non-recurrent partial-rotary checkpoint kept its
  key and the table ignored it. Default is head_dim, the same default llama.cpp
  uses.
- YaRN is folded into `inv_freq` at load time, matching ggml's `rope_yarn` +
  `rope_yarn_corr_dims` exactly: extrapolated frequency fading into interpolated
  over the pair range `[low, high]` with `low`/`high` computed in dim units and
  compared against pair indices, exactly as both references do. It also resets
  `rope_scale` to 1.0, because the 1/factor is now in the frequency and the
  kernel divides the *position* by `rope_scale`. The magnitude part of YaRN is
  folded away rather than implemented: the reference driver pre-divides
  `yarn_attn_factor` by `1 + 0.1*ln(factor)`, so a checkpoint that asks for 1.0
  cancels and the attention scale is plain `1/sqrt(head_dim)`.
- The hybrid geometry (`attention.sliding_window`, `rope.freq_base_swa`,
  `rope.dimension_count_swa`) is parsed; the windowed layers get their own
  plain-RoPE table at base 10000 over all 128 dims, which is what llama.cpp
  passes them (`ext_factor` 0).
- `attn_gate.weight`, `ffn_exp_probs_b.bias` and the `expert_weights_*` /
  `expert_gating_func` keys are loaded and carried on `LayerWeights` /
  `ModelConfig`, unused until the engine side lands.

**Remaining, in the order they have to happen.**

1. Engine, per-layer geometry. The layer loop uses `mc.n_head` and `q_dim_` for
   the query width; both have to become `model_.n_head_at(l)` / `q_dim_at(l)`,
   with `ws_q_`, `ws_attn_` and `ws_agate_` sized to the maximum. The two rope
   call sites take `inv_freq_at(l)` / `rope_frac_at(l)`.
2. The output gate. `ws_agate_` gets a `gemm` from `ws_xn_` (the attention-norm
   output, which is the same hidden state q/k/v read — not the attention
   result), then `softplus_act`, then a **per-head broadcast** multiply into
   `ws_attn_` before `wo`. The broadcast is the one new backend op: it does not
   exist in either backend today, and `mul_act` is elementwise and cannot stand
   in for it.
3. The router. `moe_ffn` is softmax over all experts then top-k, then an
   unconditional sum-normalize. Laguna needs sigmoid, a per-expert bias added to
   the *probabilities* for selection but **not** to the weights, and the
   normalize/scale pair. Keep the unconditional normalize as the default when
   `expert_weights_norm` is absent, or every Qwen MoE changes behaviour.
4. Sliding window. `AttnDesc` needs the bound and all four HIP attention paths
   need it: `attention_kernel` (mask the tile-start keys below the bound and
   start the tile loop at the bound's tile), `attention_qtile_kernel` (already
   wrong, leave it out), `attention_decode_split_kernel` (clamp `j0`), and the
   CPU reference. The subtlety is the online softmax: starting the first tile at
   a tile boundary *below* the bound is what keeps `tmax` finite, since a tile
   with every key masked gives `exp(0) = 1`. Gate `attn_fused_chain` off when a
   bound is set.
5. Then flip `{"laguna", ...}` to `ArchSupport::Yes` with an empty `why`, and
   check coherence on a real laguna file. There is no small laguna in the
   collection: `laguna-xs2-Q4_K_M.gguf` is 18.9 GiB and Laguna-S is a 3-shard
   73 GB set, so this is the slow step.

The `dflash` draft shard of the same family (`laguna-s-2.1-DFlash-Q4_K_M.gguf`)
is refused as a draft; `attn_gate` still blocks it there, which is right, since
a draft file is not a target.

---

**Left on this path.**

- `laguna-xs2` is *not* a clean coherence entry: device and CPU pick the same
  first token, then part company at a 0.35-logit near-tie (see §10d). It is
  fp16-vs-f32 on a small chaotic model, and the gate's size cap now excludes
  it by default rather than hanging the machine on a scalar pass.
- `Laguna-S-2.1` is a **3-shard split set**, so the `read()`-based weight pull
  is inactive there (STATUS.md §10a) and every tensor falls back to a cold
  mapping. Until the per-shard pull table exists, Laguna-S is the worst case
  for the paging work below, not a test of it.
- The expert tier has no host half yet: `--expert-l2-mb` defaults to 0, so a
  miss re-reads from the file instead of a pinned host copy.

## Closed / negative results

### Block-level zero skipping for MoE expert paging — MEASURED, NO YIELD

The idea: ternary (and low-bit) expert weights are mostly zero, so an all-zero
quant block could be skipped without reading it, reducing page-in traffic.

Measured with `tools/probe_zero_blocks.cpp` on the real files named in the
request, decoding every block with the engine's own reference decoder and
testing for all values exactly zero:

| file | dtype | blocks scanned | all-zero blocks |
|---|---|---:|---:|
| Qwen3-MoE-4x0.6B-2.4B Q4_K_M | Q4_K | 4,129,678 | **0 (0.000%)** |
| Qwen3-MoE-4x0.6B-2.4B Q4_K_M | Q6_K | 656,519 | **0 (0.000%)** |
| probe_TQ2_0 (BitNet-style) | TQ2_0 | 408,576 | **0 (0.000%)** |

Zeros are plentiful per *element* (0.01% Q4_K, 2.68% Q6_K, 88-98% TQ2_0) but
never fill a whole block at 256- or 32-value granularity. The TQ2_0 case is
decisive: ~90% of its weights are zero and not one of its 408,576 blocks is
entirely zero, because a 256-value block needs all 256 values zero at once.

**Not implemented, deliberately.** The policy would add a per-block test to
every page-in and save zero bytes. Do not re-attempt this shape of the idea.
Element-wise sparsity is real but needs a kernel that skips zero *work* (e.g.
a presence-bitmap representation), which is a different feature.

### The RoPE pair convention — FIXED 2026-10-04 (was silently wrong for six archs)

The engine paired adjacent head channels for every architecture. llama.cpp's
`llm_arch_rope` table says that is the llama family only; qwen2, qwen3,
qwen3moe, qwen3vl, the gemma family and laguna use the NeoX pairing (channel
`j` with `j + n_rot/2`). Both conventions are finite and in range, and only
the relative rotation between a query and a key reaches the score, so the
wrong one produced fluent-looking text instead of an error — which is why it
survived every self-consistency check (device and CPU were wrong together).
It surfaced by comparing against llama.cpp, not against ourselves.
Details and evidence: [STATUS.md](STATUS.md) §10d.

### TQ1_0 / TQ2_0 — DONE (was an open refusal)

Both ternary formats are now decoded on host and device; `probe_TQ2_0.gguf`
loads and runs. See [STATUS.md](STATUS.md) §10b for the validation and the two
packing bugs the round-trip probe caught.

---

## P0 — correctness

### Spark_one.Q6_K: one prompt of five diverges from the CPU reference — NO DEFECT FOUND

`FAIL (drift step 12 (474 vs 11694))` in the full sweep; it is the only
failure of 31 runnable models. **This entry records five dead ends so they are
not repeated.** It is NOT fixed, because no cause was found — not because it
was judged unimportant.

Spark_one is Qwen2.5-3B architecture (qwen2, 36 layers, embd 2048, 16 heads /
2 KV heads = GQA 8:1, hd 128, rope_freq_base 1e6) quantised **Q6_K**. It is
the only qwen2-arch Q6_K model in the corpus; the other Q6_K models are qwen3
and qwen35, and `qwen2.5-coder-3b-instruct-q8_0` — same architecture, Q8_0 —
passes.

| Hypothesis | Test | Result |
|---|---|---|
| Race / nondeterminism | GPU x3, CPU x2, md5 of text | Byte-identical every run — deterministic, so not a race |
| GEMM bug | `KRK_GEMM_PATH=wmma` / `simt` / `dp4a` | **All three bit-identical**, same drift step. Three independent GEMM implementations agreeing exactly rules out the GEMM |
| Prefill attention rewrite | `KRK_ATTN_QTILE` unset / `0` / `1` | Identical |
| Decode key-range split | `KRK_ATTN_SPLIT=0` | No-op here — `attention_decode_splits` keeps `n_split == 1` until 128 keys |
| KV cache stride | `--ctx 256` / `512` / `1024` / `2048` | Drift stays at step 12, so not `pos_stride` |
| Fixed absolute position | prompt lengths 9 / 14 / 15 / 18 / 26 | Drift **moves with prompt length** (plen 15 -> step 12 = pos 27; plen 26 -> step 2 = pos 28), so not a fixed position |
| Model-specific kernel bug | 5 prompts on the same model | **3 PASS** (`Hello`, `Write a Python function that reverses a list.`, `The quick brown fox jumps over`), 2 drift. Anything model-blind would fail on all five |

What the stage dump shows: the per-forward end-of-model error is flat in the
**2e-3 – 1.3e-2** band for prefill and the first 11 decode steps — the normal
fp16 activation band, and the same band `qwen2.5-coder-3b` produces. It then
jumps to 8.3e-2 on one forward and 3.6e-1 on the next. The last of those is
meaningless (the two backends had already fed different tokens by then); the
8.3e-2 one is the first divergence and has an **identical input token** on both
sides, which is the part that is not yet explained.

Most likely remaining explanation is fp16-vs-f32 rather than a kernel defect:
the device keeps activations and the logits row in fp16 while the reference is
f32 throughout, and at the divergence step the reference's own top-2 gap is
0.52 — close, but not a coin flip, which is why the gate does not accept it as
a tie. The `qwen2`-at-`Q6_K` combination is the one cell of the architecture x
quantisation matrix with no passing model in it, so the gap in coverage and the
gap in behaviour coincide and neither can be separated yet.

**Do not "fix" this by loosening the gate.** The next step that would actually
discriminate is a device-vs-host Q6_K dequant comparison over a real tensor
(`tools/probe_attn_qtile.hip` shows how to host a kernel-side dump); the test
suite only checks Q6_K dequant on a synthetic block, so whole-tensor agreement
is currently unverified.

### The qwen35 GPU path is not reproducible (FIXED 2026-10-03 on gfx1201; gfx1031 confirmation outstanding)

**Root cause: the packed-query split raced its own source.** Qwen3.5 packs a
per-head output gate behind the query, and the engine unpacked it in place —
`qwen3_next_split(q = ws_q_, gate = ws_agate_, packed = ws_q_)`. A head's
source row is twice the width of the row it writes, so head `h` writes over
`[h*hd, (h+1)*hd)`, which is exactly the source window of heads `2h` (its
query half) and `2h+1` (its gate half) — for every token, and across tokens
too, since a token's compacted row lands inside the packed row of the token
before it. With one block per head, whichever block stores first decides what
its neighbours read, so the query and gate came back different from run to
run.

Measured on the 9070 XT with `-p "A" -n 1`: the layer dump put the first
one-order-of-magnitude break in `attn.gate L03` at rel 1.5, with heads 0..7
wrong and 8..15 correct — exactly the heads whose source lies inside the write
range — while the query half of the *same kernel* was correct; three identical
command lines produced two identical dumps and one different one. With a
9-token prompt the race corrupted the layer *after* the first attention layer
(`gdn.conv L04`, rel 0.105) instead, which is what made it look like a
recurrent-state bug at first.

The fix is structural: the packed projection lands in its own workspace
(`Engine::ws_qpack_`) and the split is a pure gather from a buffer it never
writes. `Backend::qwen3_next_split` now documents the no-alias contract, the
scalar backend walks tokens in order when a caller aliases anyway (it can be
exact there; the device cannot), and the HIP kernel's "in place is safe by
index order" comment — the assumption that produced the bug — is gone.

**Evidence after the fix:** the CPU/HIP stage dumps agree to ≤ 8.6e-3 across
all 24 layers of Qwen3.5-0.8B (fp16 rounding, no structural break), the HIP
completion of `"The capital of France is"` is byte-identical to the CPU
reference, `Qwen3.5-9b-Sushi-Coder-RL.Q4_K_M` is byte-identical across five
runs with bit-identical step-0 logits, and `scripts/coherence_check.sh`
reports 9 of 9 local runnable models printing the same readable text as the
reference.

**Still to confirm on gfx1031** when maclin is back: the same five-run
reproducibility test on `Qwen3.5-9B-Q4_K_M.gguf`, whose reference completion is
`' Paris.\nThe capital of France is Paris.\nThe capital of France is'`.

What follows is the original report, kept because its negative results are
still worth not repeating:

`kraken --model Qwen3.5-9B-Q4_K_M.gguf --prompt "The capital of France is"`
on the 6700 XT produces a different 16-token greedy completion on almost every
run. The llama.cpp reference for that prompt is exactly:

```
' Paris.\nThe capital of France is Paris.\nThe capital of France is'
```

Observed kraken GPU variants include that exact string (1 run in ~5), a
near-miss that lowercases one word, `What is the capital of France?`, and
purely degenerate loops (`The|The|The`, `|-|-|-`). The **CPU f32 backend
reproduces the reference byte for byte**, so the model, the tokenizer, the
GDN math and the sampler are all correct — this is specific to the HIP path.

**What has been ruled out** (each measured, not assumed):

| Hypothesis | Result |
|---|---|
| Float atomics / split-K reductions | No `atomicAdd` anywhere in `src/hip/` |
| Multiple streams / missing sync | Single default stream; `sync()` is `hipDeviceSynchronize()` |
| VRAM pressure | Killed three stale processes holding 6.44 GiB; VRAM down to 86 MiB, still varies |
| Filesystem / mmap (model on NTFS vs tmpfs) | Copied to `/tmp` (tmpfs); same md5s from both mounts, still varies |
| The quantizer | Dense SmolLM2-135M **Q4_K_M** on the same build: 4/4 byte-identical |
| The sampler | `--greedy --temp 0 --top-k 1 --top-p 1 --min-p 0 --repeat-penalty 1 --seed 0` still varies |
| Uninitialized device memory | `KRK_ZERO_ALLOC` memset on every `alloc()` did not stabilize it |
| Kernel serialization flags | `HIP_LAUNCH_BLOCKING`, `AMD_SERIALIZE_KERNEL`, `AMD_SERIALIZE_MEMORY` all still vary |
| compute-sanitizer | **Not installed on maclin** — cannot get an authoritative race report |

**What the new `--debug-topk` flag established** — this is the key finding.
The divergence is *drift present in the very first step*, not a race that
accumulates over tokens:

```
run A: step 0 pos 5  11751=16.437500 (-0.194323)  198=13.750000 (-2.881823)
run B: step 0 pos 5  11751=15.960938 (-0.187116)  198=13.203125 (-2.944928)
```

Step 0 is the last prefill row — no generated token has been fed back yet. The
top-1 logit differs by ~0.5 and the runner-up by ~0.5, with **zero NaN and
zero Inf**. A same-binary, same-input run that disagrees on its *first*
forward pass is reading memory that differs between processes, or running a
kernel whose result depends on uninitialised or mis-launched state.

The CPU f32 path over the same prompt and the same build is bit-identical
step for step, which is the control that makes this a GPU-path bug.

**Next steps, in order of expected yield** (mostly overtaken by the fix above;
the ablation route is what actually found it — a per-layer stage dump of the
same 5-token prompt on both backends, `KRK_DUMP` + `tools/dump_diff.py`):

1. Dump and compare a *single* GDN op's output across two GPU processes using
   `tools/oracle_probe.cpp --gdn` with the real 9B weights — the per-op suite
   passes on synthetic inputs, so the fault may be weight-scale dependent.
2. Bisect by ablation: neutralise one op at a time in `gdn_forward`
   ([src/engine.cpp](../../src/engine.cpp)) behind a `KRK_SKIP` env var and
   find which removal makes step 0 stable.
3. Prime suspects, in order: the `extern __shared__` `red_kv`/`red_o`
   reduction buffers and their barrier schedule in
   [gdn_delta_rule_kernel](../../src/hip/kernels/gdn.hpp) (three
   `__syncthreads()` per token, delicate at `n_tok` > 1); the tap-staging
   cache keyed on `(ptr, type, n)` in `stage_gdn_taps`
   ([backend_hip.hip](../../src/hip/backend_hip.hip)), where a recycled
   weight pointer could alias a previous entry; and the f16 conversion path
   for `ssm_conv1d`.
4. Install compute-sanitizer on maclin if a package is available — it would
   settle this in one run.

### Text that is not text: the coherence gate

Coherence is checked as *readable output*, not as a logit distance, because a
backend can match every activation and still print numbers, degenerate loops,
or an answer to a different question. `scripts/coherence_check.sh` runs greedy
decoding on the CPU reference and on the device for each runnable model and
reports:

- **PASS** — byte-identical text.
- **PASS (tie)** — the text differs, but the first differing step is a coin
  flip at activation precision: both backends rank the same two tokens inside
  `KRK_TOL` (0.05) of each other. Example: SmolLM2-135M Q4_K_M at step 6 ranks
  tokens 338 and 351; fp16 puts them at exactly 27.265625, the f32 reference
  separates them by 0.006. Either answer is correct rounding behaviour, and
  both continuations are fluent.
- **FAIL (drift)** — a real gap at the first differing step.
- **FAIL (text)** — the continuation is >25% digits or has no letters.

Current sweep 2026-10-03 (9070 XT, 32 tokens, `--greedy`): 9 models coherent —
SmolLM2-135M (tie), Qwen3-MoE-4x0.6B, Qwen3.5-0.8B, Qwen3-4B-2507-Q6_K,
Qwen3-4B-Q8_0, Qwen3-8B-Q4_K_M, Qwen2.5-Coder-7B-Q5_K_M, Qwen3.5-9b-Sushi-Coder,
Q3.5-9B-GLM-5.1-DA.

**Drive instruct models the way they were trained, or the gate lies to you.**
The first version of the sweep asked a raw sentence — `The capital of France is`
— and every Qwen3.5 model answered by *repeating the question* until the token
budget ran out: `the capital of France. The capital of France is the capital of
France.` Both backends produced it byte for byte, so the sweep called it a pass
while the output was useless to a human. It is the model behaving as trained: an
instruct model handed a bare sentence has no reason to answer it, and the
llama.cpp reference recorded earlier in this file repeats the same prompt too. The sweep now wraps the prompt in the ChatML turn markers the models
expect (`--chat`, `KRK_CHAT=0` for base models) and asks a real question; the
same four models then print `The capital of France is **Paris**. It serves as the
country's seat…` and agree byte for byte. A coherence result on an untemplated
prompt is not evidence about a backend, and "the text is degenerate" is worth
checking against the *reference* before it is treated as a kernel bug.

---

## P1 — known races ThreadSanitizer still reports

`build-tsan/kraken-tests` is **1383/1383 passing with 5 warnings**. Two of
the original seven were the `par_pool` race fixed in `7337f3b`. The rest:

- **`src/http.cpp:456` — data race in `HttpServer::stop()`.** The `stop()`
  thread writes `listen_fd_` while the `run()` thread reads it in the accept
  error path. `running_` is already an atomic; `listen_fd_` is a plain `int`.
  Make it an atomic, or have `stop()` only signal and let `run()` own the
  descriptor.
- **`tests/test_kraken.cpp:1303-1304` — heap-use-after-free in
  `test_gdn_ops`.** A read in `CpuBackend::qwen3_next_split` lands in memory
  freed by `Tokenizer::~Tokenizer` via `Engine::~Engine` in a *later* test.
  Almost certainly a test-lifetime bug (a `CpuBackend` outliving its engine,
  or a dangling vector), not a library defect — but it needs confirming
  before it is dismissed.

---

## P1b — prefill throughput on gfx1031: it was an LDS bank-conflict bug (FIXED)

**The previous entry here was wrong and is retracted.** It reported a
"measured ceiling" of 0.81 TFLOP/s from `tools/probe_gemm_tile.hip` and
concluded that the 6700 XT "simply cannot do dense GEMM quickly" and that
"there is no structural prefill bug on this part". All of that came from a
probe that was broken three ways: it no longer compiled against the current
`krk_hip.hpp` (it borrowed a `HIP_CHECK` the header had since dropped), it
sized its grid from `multiProcessorCount` (20 on a 40-CU part), and it timed
0.084 GFLOP per launch — tens of microseconds of work whose measurement is
launch ramp and drain, not pipe throughput. The file is fixed: it carries its
own `HIP_CHECK`, takes the CU count from argv, runs ~21 GFLOP per launch, and
prints a flatness check. Its section-2 table is flat at ~24.1 TFLOP/s from
NACC 4 up, against a 25.3 TFLOP/s register-only reference
(`tools/dot2_ceiling.cpp`).

### The actual bug: the SIMT tile's LDS fetch aliased every lane onto one bank

`gemm_simt_kernel` (the gfx10 prefill path) fetched each k-pair as two
`_Float16` loads, and re-read `k+1` inside the `(i,j)` loop on top. Its tiles
are padded to `BK+8` halves = 80 B = 20 words per row, so consecutive `tx`
lanes are 80 B apart and the bank of a fetch is `(row*20 + q) % 32`: `tx`
enters only through `tx & 1`. All 16 `tx` lanes of a wave therefore landed on
2 banks — an 8-way serialization on every `ws` fetch. rocprofv3 on the
standalone 128x128x32 tile: **`SQC_LDS_BANK_CONFLICT / SQ_INSTS_LDS = 16.4`**,
`MeanOccupancyPerCU = 15.9` of 32 (VGPR=120), `GRBM_COUNT` = 94% of a 49 ms
run. The kernel was never arithmetic- or occupancy-bound: roughly 43 of those
49 ms were spent serialized in LDS. That is also why the earlier attempts
could not work — a wider tile changes the LDS ops per dot, not the aliasing,
and split-K changes nothing at all.

**The fix, applied to `gemm_simt_kernel`:** fetch the k-pair as ONE u32, and
rotate the pair index by the thread's `tx` (`q -> q ^ tx`). The rotation is a
bijection over the 16 pairs — every thread still sums all of k, in its own
order, so the result stays deterministic run to run — and it spreads the 16
lanes over 16 banks. Measured:

| workload | before | after |
|---|---|---|
| tile harness 128x128x16 s3 TM8 TN8, 4096^3 (verified maxerr 0) | 2.98 TFLOP/s | **8.28 TFLOP/s** |
| tile harness 128x128x32 s2 TM8 TN8 | 2.87 TFLOP/s | 7.35 TFLOP/s |
| `--bench` SmolLM2-135M Q4_K_M, 51-token prefill | 584 tok/s | **1406 tok/s** |
| `--bench` Qwen3.5-9B Q4_K_M, 55-token prefill | 27.9 tok/s | **180-190 tok/s** |
| `--bench` Qwen3.5-9B, decode 256 | 21.3 tok/s | 21.3 tok/s (decode is gemv, untouched) |

The old 27.9 tok/s *was* the llama.cpp reference for this model (28.0), which
is where the "we are at the ground truth, so nothing big is left" reading came
from. There was: 6.5x.

**Correctness:** `kraken-tests` 1383/1383 on the HIP build; the oracle's
per-layer divergence profile (including the pre-existing layer-7 blowup in the
qwen35 test model) is identical before and after the change; every tile-harness
configuration verifies at `maxerr 0.0e+00`.

### What was tried and why it failed

* **128x128 block tile with 8x8 register tiles.** The textbook fix for a
  2:1 FMA:LDS ratio, and the oracle still agreed. Prefill got **2x worse**
  (3245 -> 6882 ms over 801 tokens). Now explained: the wider tile does not
  touch the bank aliasing, and it cuts the grid from ~19 waves to ~5 over 40
  CUs on top. Reverted, correctly.
* **split-K.** Proposed as the fix, never justified. With the tile at 12% of
  the pipe, neither occupancy nor tile shape was ever the constraint.

### What is actually left on this path

1. **The tile is now at 8.28 of the 25.3 TFLOP/s register-only ceiling (33%).**
   The ablation on the fixed kernel attributes 59.5% of the remaining time to
   the global->LDS staging path (latency, not bandwidth: ~93 GB/s of a
   ~384 GB/s card) and 43.7% to barriers. More stages (`BK=16` s4/s5), fewer
   registers, or `global_load_lds` are the plausible next ~1.5-2x.
2. **The int8 DP4A tile had the same aliasing (fixed, ported, K-quants
   added).** `tools/gemm_i8.cpp` fetched `Bs[(tx*TN+j)*BK + q*4]` with a
   32-byte row, so `tx` dropped out of the bank address entirely (16-way).
   Rotating the k-quad by `tx` fixed it: **6.20 -> 9.27 TOP/s** at 4096^3
   (11.26 with a 128x128x16 s4 tile), `maxerr 0` against the CPU reference in
   every configuration, and `SQC_LDS_BANK_CONFLICT/SQ_INSTS_LDS` 19.4 -> 7.4.

   *The wide-read rotation was tried and is a dead end.* Rotating at 16-byte
   GROUP granularity (so clang's 128-bit LDS merge should come back) emits the
   **same ISA** as the narrow form: the wide source merges into 64
   `ds_read2_b32`, the narrow into 32 `ds_read_b128`, and the two run at
   identical time (9.28 vs 9.29; 11.27 vs 11.28 TOP/s). The merge is therefore
   not a lever here — at 2.25 conflicts per LDS instruction the tile is no
   longer LDS-limited, which is also why equal-work variants are equal-time.

   *The port* is `src/hip/kernels/gemm_dp4a.hpp`, behind `KRK_GEMM_INT8=1`
   (off by default, like the other switches but for a second reason): weights
   are consumed as int8 codes, activations are quantized per 32-value block,
   and each k-tile's int32 partial is folded into the fp32 accumulator with
   that block's scale (and, for the K-quants, its affine min term). It is
   **lossy** — int8 activations — so it does not become a default without
   winning first.
   * Formats: Q8_0, Q4_K, Q5_K (Q6_K gated off, see 3).
   * `tools/gemm_i8k_verify.cpp` (built as `kraken-i8k`) is the format
     harness: a code probe over 96-256 positions x 128 outputs that
     reconstructs each weight the host decoder would see, plus a random GEMM
     against the host decoder. Worst relative error, all at the fp16 rounding
     bound: Q8_0 4.9e-4 / 2.1e-4, Q4_K 4.9e-4 / 1.1e-4, Q5_K 4.9e-4 /
     1.1e-4, Q6_K 4.9e-4 / 8.7e-5 (probe / GEMM).
   * Two shape gates, both measured rather than derived. **K >= 2048**
     (`kDp4aMinK`): the staging cost per call is fixed, and on the 135M
     model's K = 576/1536 matrices the flagged path measured 2827 -> 1641
     tok/s, so those shapes stay on the fp16 tile; with the floor in place
     SmolLM2 prefill runs **0 dp4a dispatches** and its GEMM time is unchanged
     (21.91 ms off vs 22.41 ms on, i.e. noise), while the 1B (K = 2048/8192)
     and the 9B (K = 4096/12288) keep theirs. **Q6_K off** (see 3).
   * Engine effect, measured as GPU time from rocprofv3 rather than wall
     clock, because this box's tok/s moves by ~1.7x with machine state (see
     the note under the benchmark table): llama-3.2-1B-Q8_0 prefill GEMM
     **36.92 -> 24.26 ms (-34%)** over the same 112 dispatches, uniformly
     ~1.5x per dispatch across all three of its weight shapes
     (11.32 vs 17.71, 9.20 vs 13.16, 3.74 vs 6.05 ms). Wall-clock prefill on
     the same build measured +21% to +84% across sessions depending on host
     load. Decode is untouched (gemv). Suite 1383/1383 with the flag on and
     off.
   * Honest limits: no Q8_0 *MoE* model exists on this box, so the MoE
     behaviour is shown through the same `gemm()` the expert loop calls; the
     path is lossy by construction; and the 9B numbers earlier in this
     campaign (**183.7 -> 205.7 tok/s** with Q6_K gated out, 245 ms of Q6_K
     cost before that) came from a tmpfs copy of the model that a reboot
     erased, so they are recorded as measured rather than re-checked.
3. **Q6_K is excluded from the gate, and its register budget is now fixed.**
   The cost was never the staging: it was that Q6_K is the only format with two
   *halves*, so the dot loop held `iacc[2][4][8]` — 64 live int32 accumulators
   against 32 for the rest. Reshaping the staging had not moved the numbers at
   all (206 VGPRs / 260 B both before and after packing its byte stores into
   dwords), which is what pointed at the accumulators.

   Counts below are gfx1031 metadata from the production flags, read with
   `hipcc --offload-arch=gfx1031 -S` and `.vgpr_count` /
   `.private_segment_fixed_size` in the kernel descriptors:

   | kernel | before | after |
   |---|---|---|
   | `gemm_dp4a_kernel<3>` Q6_K | 206 VGPR, 260 B, 1 block/CU | **128 VGPR, 28 B, 2 blocks/CU** |
   | `gemm_dp4a_kernel<0>` Q8_0 | 137 VGPR, 0 B, 1 block/CU | **124 VGPR, 0 B, 2 blocks/CU** |
   | `gemm_dp4a_kernel<1>` Q4_K | 116 VGPR, 0 B, 2 blocks/CU | 117 VGPR, 0 B, 2 blocks/CU |
   | `gemm_dp4a_kernel<2>` Q5_K | 117 VGPR, 0 B, 2 blocks/CU | 118 VGPR, 0 B, 2 blocks/CU |

   Two things moved it. The dot now runs **one half at a time** (`TM*TN` int
   accumulators live instead of `TM*TN` per half, folded per half with that
   half's scale and offset), and the kernel carries an
   `__attribute__((amdgpu_waves_per_eu(8)))` budget: `__launch_bounds__`'s
   second argument — the min-blocks slot that asks for the same thing — changed
   **not one register** on either arch, so the budget is spelled the AMD way.
   In the Q6_K body there are then 256 `v_dot4c_i32_i8` per k-tile, 70
   `ds_read2_b32` for the rotated fetches, and **no spill instructions at all**
   — the 28 B private segment is the prologue's reservation, not loop traffic.

   The per-half loop's correctness rests on the index map, and that part is now
   proved on the CPU rather than measured on a card: `dp4a_quad_index` is
   host-callable, and `kraken-i8k` checks before it touches the device that the
   `NH*RH` (half, round) pairs visit every quad group exactly once and that each
   one's rotated index lands inside its own half, for both NH values and all 16
   `tx` lanes. That is the whole invariant the fold depends on.

   **What is still missing is the measurement, not the budget**: per-dispatch
   time against the fp16 tile on gfx1031. `kDp4aEnableQ6K` stays false until
   that is taken, and the two lines that need to exist for it are the counts
   above (done) and a dispatch time (blocked: maclin was down).
4. **The int8 port broke the gfx1201 build, and nothing had noticed.**
   `KRK_HAS_DOT4` was defined for gfx10/gfx11/gfx12 alike, but clang rejects
   `__builtin_amdgcn_sdot4` on gfx1201 with "needs target feature dot1-insts" —
   so every `gfx1201` compile of the HIP backend failed from the moment the
   int8 path landed, because that path had only ever been compiled for gfx1031.
   The guard is now gfx10/gfx11, and the fallback is exact four-byte arithmetic
   that no longer takes the address of its operands (the old fallback forced
   them into scratch). `kraken-tests` on the 9070 XT: 1383/1383.
5. **`kraken-i8k` cannot validate the int8 path on gfx1201 yet.** On that box
   the code probe reports every position skipped while the host activations are
   demonstrably correct (`x[0]=1` at the one-hot index) and a Q4_K output is
   nonzero — i.e. the codes and scales it reads back are all zero while the
   kernel that consumes them evidently did not see zeros. That is a readback or
   launch problem on that platform, not a staging bug, and it is open. It now
   has a data-sanity check (an all-zero host decode fails loudly instead of
   reporting a clean pass over nothing) and `KRK_I8K_DEBUG=1` prints the first
   launch's activations, codes, scales and outputs.
6. **A guard now exists for this class of bug** — `scripts/check_lds.sh` runs
   a real workload under rocprofv3 and hands
   `SQC_LDS_BANK_CONFLICT / SQ_INSTS_LDS` to `tools/prof_agg.py --guard`,
   which exits non-zero when a kernel exceeds 6.0 conflicts per LDS
   instruction (the two buggy tiles measured 16.4 and 19.4; the fixed ones sit
   at 0.0-2.25). It takes a fresh profile, or a saved CSV/directory, or a
   model through `KRK_GUARD_MODEL`; bad usage and unmeasurable runs exit 2
   rather than passing silently. It needs ROCm and a GPU, so it is a
   release-step check, not part of `kraken-tests`.
7. **The other gfx10 kernels were audited for the same aliasing: none share
   it.** Per-kernel conflicts per LDS instruction, from the flagged 9B profile
   (`/tmp/prof_gate`) and the default path: `gemm_simt_kernel` 0.000,
   `quantize_act_kernel` 0.000, `gdn_delta_rule_kernel` 0.000,
   `gdn_l2norm_kernel` 0.000, `rmsnorm_kernel` 0.000, `attention_kernel`
   0.000, `gemv_kernel` 0.000 (its per-dispatch maximum is 0.036),
   `fused_layer_gemv_kernel` 0.000 (max 0.021),
   `attn_fused_decode_kernel` 0.079, `silu_mul_kernel` 0.000. The two int8
   tiles are the only kernels above the noise floor, at 2.25.
8. **The `rows<=16` decode shapes** remain occupancy-starved (0.4-1.2 waves);
   decode is gemv and already at 21.3 tok/s on the 9B, so this needs its own
   measurement before anyone invests.

## P1c — device kernel speed on gfx1201 (plan: [docs/PERF-PLAN.md](PERF-PLAN.md))

Ranked by measured end-to-end impact. Every claim is a number in PERF-PLAN.md;
proven causes and open hypotheses are separated there.

1. **Q6_K (and Q5_K) decode is the format that is 3-4x slow**, and Q4_K is not:
   `kraken-bench` measures Q4_K at 572-674 GB/s (89-105% of the 640 GB/s DRAM
   envelope, so at or above parity with f16) but Q6_K at **143 GB/s on a 13.8 MB
   matrix that fits the 64 MB Infinity Cache** and 195 GB/s on a cache-exceeding
   208 MB one — i.e. instruction-bound in the chunk decode, not memory-bound.
   Q5_K sits at 352. Fix = packed-f16 decode (`v_pk_mul_f16`/`v_pk_fma_f16`, the
   family `krk_hip.hpp` already exposes) with a branch-free 6-bit unpack and a
   precomputed `d * scale` in f16. Falsifier: 4096x4096 Q6_K > 400 GB/s.
2. **Prefill GEMM runs at 22% of its memory-bound ceiling because the dequant
   staging blocks it.** rows=32 Q4_K: 15.67 TFLOP/s / 137 GB/s against a ~72
   TFLOP/s memory-bound ceiling (9.44 MB per 1.07 GFLOP at 640 GB/s), while
   rocBLAS' f16 GEMM at the same shape is at 773 GB/s (memory-bound, 43.4 µs).
   Reading 3.5x fewer bytes and running 1.58x slower than that is only
explainable by time before the WMMA units see data — which the kernel's own
§2c note already measured as 55-97% of prefill GEMM time. WMMA itself is not
the suspect. Fix = take dequant off the critical path: double-buffered LDS, a
   producer/consumer split, then decode the B fragment straight into VGPRs.
   Target > 40 TFLOP/s at rows=32.
3. **Call count is the dense-decode tax.** The launch floor is 2.65 µs on a
   6.2-7.8 µs call for the k/v, g/u and down shapes: 34-43% of each call, every
   token. The engine's fused group already costs 2.9x less *per matrix* than the
   separate calls. Fix = group more projections per launch.
4. **One sync point per token.** `download_f32` = 3.4 ms/token on the 8B (21%,
   2.7 ms of it device *idle*) and ~0.6 ms on the 0.8B. Fix = softcap + top-K
   candidates on device, copy back 8-128 pairs instead of the 151k row, with a
   tail-mass guard so the host-side filters provably do not need a candidate
   they were not given.
5. **Small models are op-count-bound.** 541 device ops/token on the 0.8B against
   a 10.3 us budget each, 30.6% compute-busy; ~250 of those ops are tiny
   elementwise/norm calls. Fix = fuse the norm+elementwise chain, and add the
   per-op timings to the fused layer path.
6. **MoE is not compute at all.** 7.3% compute-busy, 69 ms/token on the 4x0.6B
   (13 GB/s). Fix = trace the expert cache first, then expert-only residency,
   device-side router top-k, one batched transfer per step.
7. **Attention at decode context.** 35-190 us/call where the KV traffic is
   ~4 us; one thread per key, each reading a whole head vector. Fix =
   warp-per-(head,q-token) with lanes across channels and a shuffle reduce.
8. **Split-K policy and `rows<=16` occupancy** need their own measurement on
   gfx1201 before anyone invests (the gfx1031 numbers do not transfer).

### Capability work the same pass owes

- **Landed**: final-logit softcapping (one place, all paths); per-stage
  coherence instrumentation; `KRK_TIME` per-op timing with a filter; the
  coherence gate with instruct templates.
- **Sliding-window attention** (per-layer patterns) — the arch table records it
  as a blocker for gemma2/gemma3; needs the per-layer window in `AttnDesc` and
  a mask in the attention kernels.
- **Per-layer input embeddings** (gemma4) — a second embedding table read per
  layer, plus the shared-KV 18-layer geometry it comes with.
- **YaRN rope scaling** — `rope_scale`/`rope_frac` exist; YaRN needs the
  frequency-dependent ramp and the attention scale that goes with it (laguna,
  long-context use generally).
- **RadixAttention / prefix cache** — shared prefixes across requests in the
  server, so a repeated system prompt is prefill-free. Needs a token trie over
  the KV cache and a rollback story for the recurrent models (see Tail-Replay
  in [docs/implementing-papers.md](implementing-papers.md)).
- **LLaDA / diffusion (DLLM)** — loader schema (fused `attn_qkv`, group-limited
  routing + gate bias) and the block-diffusion sampler; research and tensor map
  already in [docs/LLaDA.md](LLaDA.md).
- **MTP / dflash / dspark draft blocks** — currently skipped with a note; the
  verdict counts them (`mtp_blocks`), so the next step is to load them as a
  draft head and wire them into the existing speculative loop.
- **Architecture pass over every file in the model folders** — the inventory is
  at 44 runnable of 87; the refused ones need either a named dequantizer
  (quant type #100/#101/#102/#107) or a documented reason, and the runnable set
  needs a load-and-generate smoke test each.
- **Paper backlog** — all 12 summarized with what each would change here in
  [docs/implementing-papers.md](implementing-papers.md); Tail-Replay first
  because it repairs the recurrent `kv_rollback` refusal.

## P2 — robustness and diagnostics

- **CU count was under-reported (fixed).** HIP returns 20 CUs for a 40-CU
  gfx1031; every occupancy heuristic sized itself against the wrong number.
  `kfd_compute_units()` now reads the KFD topology. Reported as 40 CU. It did
  not move prefill at the time (3245 -> 3245 ms) because the constraint was
  the LDS bank conflict in P1b, not occupancy; with that fixed the same 9B
  prefill went 27.9 -> 180 tok/s.
- **Engine utilization counters are Windows-only.** `--bench` prints
  "engine counters are Windows-only" on Linux, so there is no per-engine
  GPU utilization on the machine where the profiling work happens. The
  gfx1031 and gfx1201 campaigns in [STATUS.md](STATUS.md) §8–§12 were run on
  Windows for this reason.
- **Stale processes are easy to leave behind on maclin.** This session found a
  `llama-server`, a `kraken-server` and a `kraken-oracle --gdn` that had been
  running for 3 hours and were holding 6.44 GiB of VRAM. The SIGPIPE fix stops
  the oracle from hanging that way in future; a "device busy" hint at startup
  would have surfaced it sooner.
- **`/tmp/q9b.gguf`** (5.87 GB Qwen3.5-9B Q4_K_M) was parked in tmpfs on
  maclin for the nondeterminism and MoE work; a reboot on 2026-10-03 cleared
  it, so both now need it re-fetched before they can be exercised. Nothing
  deleted it deliberately.

---

## P3 — documentation

- **`qwen35` / `qwen35moe` are missing from the supported-models list** in
  [README.md](../README.md) even though both load and run. Add them once the
  GPU path is reproducible.
- The README benchmark table has no 9B / gated-delta-net row, because there is
  no trustworthy number to put there yet.
- `--debug-topk` is documented in `--help` only; it belongs in the README's
  verification section alongside the CPU-oracle workflow.

---

## P4 — known limits carried forward from STATUS.md §6

- IQ\* formats except IQ4_NL are rejected at load by design.
- Serving is serialized: no prefix cache, no cancellation, no concurrent
  batching, no beam search.
- Speculative decoding is greedy-only; no MTP, tree attention, or n-gram
  speculation.
- Attention decode launches one block per head (no split-K); sampling is on
  the host.
- The fused decode-layer GEMM phase
  ([src/hip/kernels/fused.hpp](../../src/hip/kernels/fused.hpp)) is a measured
  prototype — `backend_hip.hip` still launches the gemvs separately, so its
  measured saving does not reach the engine.

---

## Housekeeping

- The int8 port, the guard, the K-quant work and this file are committed and
  pushed through `251d694`.
- **Uncommitted**: the Q6_K register work (`gemm_dp4a.hpp`, `krk_hip.hpp`,
  `tools/gemm_i8k_verify.cpp`) — counts and the CPU index-map proof are in, the
  gfx1031 dispatch time is not, and nothing here is pushed without being asked.
- **maclin was down** (`10.0.0.12`, unreachable from 03:30 onward) when this was
  written, which is why the Q6_K timing, the fp16 staging work and any 9B
  measurement are all stated as pending rather than done. Everything measured
  in this round was measured on the Windows box (RX 9070 XT, gfx1201) or read
  out of a gfx1031 assembly dump compiled there with
  `hipcc --offload-arch=gfx1031`.
- **`/tmp/q9b.gguf` is gone.** The Qwen3.5-9B Q4_K_M copy lived in tmpfs on
  maclin and a reboot erased it (the box came back up at 03:05). Re-fetch it
  before any further 9B or MoE work — the 9B rows below cannot be re-measured
  without it.
- **This box is shared and rebooted.** Another agent's workload (a kilo CLI
  plus a pytest run) was consuming CPU during the last measurement pass, and
  one interrupted `rocprofv3` run left a process holding the card long enough
  to make a whole three-model sweep wrong. Check `rocm-smi --showpids` and
  `ps` before trusting a number, and prefer the profiler's per-kernel times
  over wall-clock tok/s when the host is busy.

---

## Measured performance (6700 XT, gfx1031, ROCm 7.15/7.17, no WMMA)

`kraken --bench --ctx 512 --chunk 256 --greedy`, default (fp16-tile) path,
medians of 5 runs, taken on a *busy* box after the reboot:

| model | prefill | decode |
|---|---|---|
| Qwen3.5-9B Q4_K_M | not re-measurable — `/tmp/q9b.gguf` lost with the reboot | 21.3 tok/s |
| Llama-3.2-1B Q8_0 | 1119 tok/s | 182 tok/s |
| SmolLM2-135M Q4_K_M | 1608 tok/s | 343 tok/s |

The same measurements taken earlier on an idle box were 183.7 / 1120 / 2827
for the same three models, so **treat these as ±1.7x by machine state**, not as
a fixed property of the build. The decode column is stable to ~2%.

With `KRK_GEMM_INT8=1` (and the K and Q6_K gates described in P1b) the same
runs gave 1357 tok/s on the 1B and 1517 on SmolLM2 — i.e. +21% wall clock on
the 1B in this noisy session, against **-34% GEMM time** measured with
rocprofv3 on the same build, and no change at all on SmolLM2 (which takes no
int8 dispatches).

llama.cpp (Vulkan, same card, same 9B) manages 6.8 tok/s decode, so kraken is
~3.1x faster on the recurrent path.

Prefill was investigated this session and is **not** an outlier: throughput
scales with model size (the 9B is ~6.8x slower than the 1B for ~9x the
parameters) and per-token prefill cost (~35 ms) is close to decode (~47 ms).
An earlier suspicion that prefill was pathological was wrong. The remaining
decode headroom is the usual one: 5.3 GB of weights against a ~384 GB/s bus
implies a ~72 tok/s roofline, so decode sits ~3.4x off bandwidth-bound.
---

## Gap audit against the two tiering specs (2026-10-05)

Evidence: `docs/traces/decode-dissection.txt`. Measured on the RX 9070 XT;
a Laguna XS.2 Q4_K_M decode token is **75.1-75.7 ms**, of which **60% of the
generation phase is expert weight movement** and 0% is expert arithmetic.
The whole routed expert set is 19.43 GiB against a 12,425 MiB VRAM budget
(64% resident), so ~160 MiB per token comes off NVMe and is DMA'd to VRAM,
every token, forever. There is no steady state to prefetch *into*, which is
why prefetch/async/prediction cannot rescue this on their own.

Status of each spec item:

* **Spec section 22, CPU Expert Fallback (Fiddler) — NOT IMPLEMENTED.**
  The single biggest missing piece and exactly the 6-10x lever. Zero hits for
  any CPU expert path in the tree. The arithmetic that settles it: for a
  one-row activation the operand is ~8 KB of f32 and the weight is
  1.9455 MiB, so moving the weight costs ~250x more than moving the operand.
  A cold expert must be *computed*, not moved. This is also what Strata
  ("your processor works on the rest at the same time"), what
  `lna-lab/flash-next-8gb` does with `-mcl 64`, and what ExLlamaV3's
  `--moe_cpu_offload` does. Highest priority item on this list.

* **Spec sections 16/17/28, RAM tier — IMPLEMENTED BUT MIS-SIZED.**
  `ExpertCache`'s WARM tier *is* this section, but the default budget is
  `ram_total / 4` (src/engine.cpp:861): 24 GiB of 96, giving WARM 18102 MiB
  against a 19,860 MiB routed corpus. The spec table says 96 GiB -> 70 GiB
  warm; we run 18 GiB, a 3.9x shortfall, and a tier that cannot hold the
  corpus churns at the margin. Fix the default to the spec's table and make
  it a function of (RAM, corpus), not of RAM alone.

* **Spec section 20/38, expert prediction — NOT IMPLEMENTED.**
  Only `KRK_TRACE_EXPERTS` instrumentation and LFU+aging exist; there is no
  predictor. This is the part of the "revolutionary expert cache" that is
  actually missing: ExpertFlow / MoE-Infinity Level 3, cross-layer
  prediction. Note that prediction is worth much less before the CPU
  fallback exists — predicting correctly still leaves you moving the bytes.

* **Spec section 29, GPU profiles (6/12/16 GiB) — PARTIALLY IMPLEMENTED.**
  The budget does adapt to the card (`hard = total_vram - 512 MiB`, so a
  6 GiB card gets 5.5 GiB), so the "flat kCapMax regardless of card" worry
  is unfounded. What is missing is the per-class *slot* policy (8/16/32
  slots per active sparse layer). At 39 sparse layers that is 312/624/1248
  slots; our flat byte budget gives a similar count by accident for the
  current geometry, but it is not the stated policy and will not hold for
  a model with different expert size.

* **Spec section 35, logical page != physical transfer — IMPLEMENTED.**
  `ExpertCache::prefetch_layer` plus the offset-sorted contiguous-per-worker
  batch dispatch in `HipBackend::read_host_batch`. This is the one that is
  done, and it is why the NVMe read rate went 593 -> 1446 MiB/s.

* **KV tier spec — NOT IMPLEMENTED AT ALL.**
  `Engine::configure` does one flat
  `alloc(n_layer * kv_cap_ * kv_dim_)` for K and the same for V, entirely in
  VRAM (src/engine.cpp:275-278). No dtype switch, no paging, no RAM or NVMe
  tier, no radix prefix cache, no async H2D/D2H, no residency states. On a
  15.9 GiB card this is affordable (KV was 320 MiB at ctx 2048), but on the
  6 GiB class the KV cache is the difference between running and not, and
  it is the reason the 6 GiB profiles in the spec cannot be honoured today.
