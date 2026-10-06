# KRAKEN test results

Measured on the reference box, 2026-10-06.

| | |
|---|---|
| GPU | AMD Radeon RX 9070 XT, gfx1201 (RDNA4), 64 CU, 15.9 GiB VRAM |
| SDK | ROCm 10.1 / HIP 7.16.26332, clang 23.0.0 |
| Build | `build-hip`, `CMAKE_BUILD_TYPE=Release`, `-O3 -DNDEBUG`, `--offload-arch=gfx1201` |
| Host | Windows 11, 96 GB RAM |

Every number below comes from the binary in `build-hip/`, run with the staged
SDK DLLs beside it (`amdhip64_7.dll`, `amd_comgr.dll`, `rocm_kpack.dll`), so
`--info` reports `HIP 7.16.26332` with no runtime warning.

Reproduce any row with the command in its section. The GPU-utilization column
comes from the engine's own per-process sampling (`--bench` prints it), which is
why it is quoted as "avg % over the run" rather than a single reading.

---

## 1. Prefill and token generation

```
kraken.exe -m <model> --bench --greedy
```

`--bench` uses a fixed prompt (51–55 tokens here), 256 greedy tokens,
`--ctx 4096`, `--chunk 256`.

| model | prefill tok/s | decode tok/s | ms/step | load ms | GPU compute avg |
|---|---:|---:|---:|---:|---:|
| SmolLM2-135M-Instruct Q4_K_M | 2446 | 474 | 2.1 | 415 | 75.5% |
| Qwen3.5-0.8B Q4_K_M (GDN hybrid) | 2874 | 341 | 2.9 | 540 | 78.0% |
| Qwen3-MoE-4x0.6B-2.4B Q4_K_M | 199 | 108.6 | 9.2 | 417 | 43.8% |
| Qwen3-8B Q4_K_M | 876 | 88.4 | 11.3 | 1558 | 93.4% |

**Repeatability.** Three back-to-back runs of Qwen3.5-0.8B gave decode
334.4 / 335.3 / 334.4 tok/s (0.3% spread) and prefill 2725 / 2376 / 2457
(13% spread); SmolLM2-135M gave 474.5 / 455.1 decode across two runs (4%). So
decode is stable to well under 1% once warm and prefill to ~13%, and the
`--bench` prompt is short enough that prefill is a small sample. Treat anything
under ~5% on decode as noise.

**A warning about the first run after a cold start.** The very first
Qwen3.5-0.8B run in this session measured **85.9 tok/s** — 3.9× below the stable
value — and the first Qwen3-8B run measured 72.4 tok/s against 88.4 later. Both
were taken immediately after a build with cold file cache and a freshly
initialised driver. This is the same trap `AGENTS.md` records for A/B probes:
the machine has a first-run state that is not the steady state, and reporting it
as "the model's speed" understates the engine by 4×.

### Where the time goes when decode is fast, and when it is not

The GPU-utilization column is the honest summary of the CPU/GPU balance:

- **Qwen3-8B: 93.4%.** Decode is weight-bandwidth bound and the device is never
  starved. This is the engine working as intended.
- **Qwen3.5-0.8B / SmolLM2-135M: 75–78%.** Roughly a quarter of each step is
  device idle — the step is short enough (2.9 / 2.1 ms) that host-side work
  between launches is no longer negligible.
- **Qwen3-MoE-4x0.6B: 43.8%.** More than half the step is idle: the MoE path
  does router round-trips and expert promotion between launches.
- **KV-tiered (see §3): 1.8% and 0.5%.** The device is essentially unused.

`--profile` now prints a host-phase table beside the device op table, so the
CPU/device split is readable rather than inferred:

```
[stats ] host time  (wall clock per phase; KRK_PROFILE to enable)
    phase                    total ms    calls    ms/call    % wall
    decode.step                4737.7      256     18.507     99.7%
    prefill.chunk                13.5        1     13.544      0.3%
```

`decode.step` covers host sampling, the forward and the logits fetch;
`decode.forward` and `decode.fetch` break it down. When `decode.step` is large
while `decode.forward` is small, the step is host-bound and that is the thing to
fix.

---

## 2. Speculative decoding

Draft and target must share a tokenizer, so the pair is
**Qwen3-8B-Q4_K_M (target) + Qwen3-MoE-4x0.6B-2.4B-Q4_K_M (draft)**.

```
kraken.exe -m Qwen3-8B-Q4_K_M.gguf --bench --greedy
kraken.exe -m Qwen3-8B-Q4_K_M.gguf --draft Qwen3-MoE-4x0.6B.gguf --draft-tokens 4 --bench --greedy
```

| run | prefill tok/s | decode tok/s | unit cost | accept |
|---|---:|---:|---:|---:|
| target alone | 853–876 | 88.1–88.4 | 11.3 ms/step | — |
| target + draft (window 4) | 154.4 | **32.4** | 79.09 ms/round | 179/400 = 44.8% |

> **PRE-FIX NUMBERS, KEPT ON PURPOSE.** Everything in this section was measured
> *before* the draft and verify paths were routed through the device top-k (the
> `src/engine.cpp` change: `fetch_logits(true)` / `topk_argmax`, 16 bytes back
> instead of the whole row). These rows are the baseline the fix has to beat,
> not a description of the current build. Re-run the two commands above to
> compare, and treat the rate below as the *old* path's.

**Speculative decoding was a 2.7× net loss on this pair.** A round cost
79.09 ms and returned ~2.56 accepted tokens (32.4 tok/s), where two plain steps
would have cost 22.6 ms and returned 2 tokens (88.4 tok/s).

Diagnosis, from the code path rather than the number — **two compounding costs,
both of them the same mistake the plain decode path had already been cured of**:

1. **A full logits row per proposal, for an argmax.** The generic (`--draft`)
   path called `draft.fetch_logits()` with no argument, which syncs and pulls
   the *entire* row — 993 kB at a 151936 vocab — just to take the argmax on the
   host, once per proposal:

   ```cpp
   for (; d < k && res->generated + d < p.max_tokens; d++) {
       const i32 t = argmax_of(draft.logits_host_, n_vocab_);
       ...
       draft.forward(&t, 1, static_cast<i32>(pos + d - 1), true);
       draft.fetch_logits();            // 993 kB + a sync, per proposal
   }
   ```

   Plain greedy decode had already stopped doing this — `fetch_logits(true)`
   reduces the row to one integer on the device and moves 16 bytes
   (`topk_argmax`). The draft path was left on the old call. A window of 4 paid
   four of those downloads per round.

2. **The same thing on the verify side, on the target.** The verification block
   downloaded one full row per proposal (`head_compute(j)` then
   `download_f32(rows + j*n_vocab, ...)`) purely to argmax it on the host — d
   more 993 kB transfers per round, on the larger model.

Both are now routed through `topk_argmax`, which is what the plain path uses and
what the comment above `fetch_logits` already claimed every other caller did.
That comment was the tell: it asserted the speculative and draft paths passed
`false` and "get the whole row exactly as before", which was true and was the
bug.

Independent of those two, the generic path proposes with **one single-token
forward per proposal** — an autoregressive draft has no alternative, so a window
of 4 still pays four draft forwards before verifying. The batched drafter path
(`generate_speculative_dflash`) proposes all *k* tokens in **one** forward and
does not have this shape; the generic path does.

Also fixed while measuring: both speculative paths declared `timed_decode_steps`
but never incremented it, so every speculative run reported `0 timed steps =
0.0 tok/s`. It now reports rounds, tokens/s and ms/round.

**Status: speculative decoding is correct** — greedy output is bit-identical to
the non-draft run, which the test suite asserts. Its profitability is what the
device-top-k change above is for, and it is measured by re-running this section,
not assumed.

---

## 3. KV cache tiering

```
kraken.exe -m SmolLM2-135M-Instruct.Q4_K_M.gguf --bench --greedy
kraken.exe -m SmolLM2-135M-Instruct.Q4_K_M.gguf --ctx 4096 --kv-hot-mb 8  --kv-warm-mb 64 --bench --greedy
kraken.exe -m SmolLM2-135M-Instruct.Q4_K_M.gguf --ctx 4096 --kv-hot-mb 1  --kv-warm-mb 64 --bench --greedy
```

The KV-tier counters are now printed at the end of **every** run, dense models
included (they were collected from the start but the only reader was a
load-time one-liner, and the MoE-only stats block returned before reaching them).

| config | decode tok/s | ms/step | GPU compute avg | WARM→HOT | HOT→WARM | →COLD | migrated |
|---|---:|---:|---:|---:|---:|---:|---:|
| flat (fits in VRAM) | 474.5 | 2.1 | 75.5% | — | — | — | 0 |
| 2 HOT slots (3 MiB) | 10.8 | 92.3 | 1.84% | 11088 | 11130 | 4226 | 33327 MiB |
| 1 HOT slot (2 MiB) | 2.3 | 432.0 | 0.50% | 10710 | 10752 | 4606 | 32193 MiB |

Raw output for the 2-slot case:

```
[info ] kv tiered: 30 layers, 2 HOT slots of 3 MiB, layer=1.5 MiB, WARM budget=2 layers
[stats ] kv        tiered: 30 layers x 1.50 MiB, 2 HOT slots (3 MiB) | promoted WARM->HOT 11088, COLD->HOT 0 | evicted HOT->WARM 11130, ->COLD 4226 | 33327.0 MiB migrated
```

**What this says.** The tiering machinery works exactly as designed — layers
promote and demote, WARM absorbs every HOT eviction (`COLD->HOT 0` means no
layer was ever lost to disk) — but a HOT tier smaller than the working set is
catastrophic: **44× slower at 2 slots, 206× at 1 slot**, because every decode
step touches all 30 layers and each touch migrates a layer. 33 GB migrated in a
run that generated 256 tokens is pure thrash.

The rule this measures: **a partial HOT budget below the working set is worse
than no tiering at all.** Tiering only pays when the working set is genuinely
larger than VRAM and the access pattern has locality the policy can exploit.
The `--kv-hot-mb 0` default (take what the weights leave free) is the right
setting for anything that fits.

---

## 4. Coherence (device vs scalar CPU reference)

```
bash scripts/coherence_check.sh <model.gguf>
```

Runs each model twice — once on the GPU, once with `--cpu` — and diffs the text.

| model | command | result |
|---|---|---|
| Qwen3.5-0.8B Q4_K_M | n=32 ctx=512 | **PASS** |
| Qwen3.5-0.8B Q4_K_M | n=128 ctx=2048 | **PASS** |
| Spark_one.Q6_K.gguf | n=32 ctx=512 | **PASS** |
| SmolLM2-135M Q4_K_M | n=32 ctx=512 | PASS |
| Qwen3-MoE-4x0.6B Q4_K_M | n=32 ctx=512 | PASS |
| Qwen3-8B Q4_K_M | n=16 ctx=512 | PASS |

Two of these were recorded as open failures in `AGENTS.md` and
`docs/USER-GUIDE.md`:

- **Qwen3.5 / GDN GPU path** was "not yet reproducible against the scalar
  reference". It passes, at both context lengths.
- **`Spark_one.Q6_K.gguf`** "fails coherence at drift step 12". It passes.

Both notes had gone stale. The check is cheap and re-running it is the only way
to know.

---

## 5. Build honesty

```
sh scripts/build_check.sh
```

```
ok    up-to-date   ninja has no pending work for: kraken kraken-tests kraken-bench
ok    build-type   Release, -DNDEBUG in 51 command line(s), no -O0
ok    hip-backend  KRK_ENABLE_HIP=ON, HIP enabled, kraken.exe links amdhip64
ok    gpu-target   KRK_GPU_TARGETS=gfx1201
ok    device       AMD Radeon RX 9070 XT (gfx1201) is in KRK_GPU_TARGETS

0 failed, 0 warning(s)
```

Two real defects it caught in this session, both now fixed:

1. **`FAIL up-to-date` — 5 pending build steps.** Source had changed without a
   rebuild, so a run would have executed the previous `.exe`. Rebuilt; now
   clean.
2. **`WARN runtime` — wrong SDK DLLs staged.** `build-hip/amdhip64_7.dll` and
   `amd_comgr.dll` were copies from a *different* SDK revision than the one the
   binary was built against (`--info` reported `HIP 7.16.26323 (SDK
   7.16.26332)`). Re-staged from `G:/ROCM10RT-gfx1201/bin/`; `--info` now
   reports `HIP 7.16.26332` and the warning is gone.

---

## 6. Profiling coverage (`--profile`)

`--profile` (or `KRK_TIME=1`) turns on **two** tables, and a step needs both:

| table | source | answers |
|---|---|---|
| device op table | `src/hip/op_time.hpp` (HIP TU) | which kernel, and the idle gap before each launch |
| host phase table | `include/krk/host_time.hpp` (portable) | which phase of the step, in wall clock |

The host table was added because the device profiler only exists in the HIP
translation unit — portable code could not reach it, so whole phases went
unmeasured. It now covers:

| scope | what it wraps |
|---|---|
| `prefill.chunk` | one prefill chunk forward |
| `decode.step` | host sampling + forward + logits fetch, per token |
| `decode.forward` | the forward alone |
| `decode.fetch` | the logits fetch alone |
| `spec.round` | one full speculation round (propose + verify + emit) |

`decode.step` minus `decode.forward` minus `decode.fetch` is host time no device
op accounts for — which is exactly the "CPU busy, GPU idle" quantity.

The HTTP server reports too (`KRK_PROFILE=1 kraken-server ...`), one line per
request plus a cumulative host table:

```
[stats ] request   prefill 16 tok in 37.3 ms | decode 16 tok in 259.5 ms over 15 steps
[stats ] host time  (wall clock per phase; KRK_PROFILE to enable)
```

KV-tier counters print at the end of **every** run, dense and MoE alike (§3).

---

## 7. Summary of what these numbers changed

| finding | evidence | consequence |
|---|---|---|
| Cold-start runs understate decode by up to 4× | 85.9 vs 334.4 tok/s, same binary | always warm up before quoting a number |
| Speculative decoding was a 2.7× loss with a generic draft (pre-fix rows, §2) | 32.4 vs 88.4 tok/s, 44.8% accept | route the draft and verify paths through the device top-k |
| A HOT tier below the working set is 44–206× slower | 10.8 / 2.3 vs 474.5 tok/s | leave `--kv-hot-mb 0` unless the cache truly does not fit |
| GPU is idle 56–99.5% on MoE and tiered paths | 43.8% / 1.84% / 0.50% compute | host-side and migration work dominates there |
| Two documented coherence failures no longer reproduce | §4 | re-run before believing a stale failure note |
| KV tier and host-phase timings were invisible | §3, §6 | both now print on every run |
| The CPU expert fallback spec §22 predicted would win exists and LOSES 3.1× | 5.5 vs 17.2 tok/s at a 16 MiB budget; 5.392 ms/expert | opt-in and default off; it buys memory, not speed (§9) |

---

## 8. Determinism: the decode-attention split-K race

The decode-attention key-range split (`attention_decode_split_kernel`,
`src/hip/kernels/attention.hpp`) is taken when `attention_decode_splits()`
returns > 1, which starts at ~128 keys. Its per-block epilogue wrote the
partial denominator from **every** warp-leader thread, each seeded with its own
warp's term:

```cpp
if (lane == 0) lt = s_l[warp] * aw;            // eight leaders, eight values
...
for (int w = 0; w < kAttnDecWarps; w++) lt += s_l[w] * __expf(s_m[w] - mstar);
...
if (lane == 0) { p[0] = mstar; p[1] = lt; }    // ...all storing to p[1]
```

Last writer won, so one head's normalizing denominator — and therefore that
head's output — depended on warp scheduling. The single-block kernel never had
this shape: it scales `sh_l[warp] = l * aw` once, after the barrier, and then
sums `lt += sh_l[w]`.

```
kraken.exe -m Qwen3-8B-Q4_K_M.gguf \
  -p "The history of computing is a history of abstraction, from relays to vacuum tubes" \
  -n 128 --greedy --ctx 512 --chunk 256
```

md5 of stdout (banner + generated text), the identical command every time:

| build | six runs | distinct |
|---|---|---|
| before the fix | `fc30f6c8cf1d`, `d8448e90a0a3`, `7ee2e7dec3aa`, `7ee2e7dec3aa`, `fc30f6c8cf1d`, `fc30f6c8cf1d` | **3** |
| after the fix | `7ee2e7dec3aa`, 10 of 10 runs | 1 |

The fixed split path is now bit-identical to `KRK_ATTN_SPLIT=0` (the
single-block arm) at this shape, which is what that A/B knob was documented to
prove and never did. Gates after the fix: ninja rc=0 with 0 `error:` lines,
`kraken-tests` 2154/2154, `kraken-bench --gate` rc=0.

**Why it stayed hidden for so long.** Below 128 keys `attention_decode_splits`
keeps `n_split == 1`, so the kernel is never launched and a short-context probe
is structurally blind to it. Every "byte-identical every run" check in these
docs, including the coherence sweep in §4, ran in that region. The bug looked
model- and shape-specific because only the corner where the split engages could
show it.

---

## 9. Hybrid CPU + GPU expert compute

```
kraken.exe -m Qwen3-MoE-4x0.6B-2.4B-Q4_K_M.gguf \
  -p "The capital of France is" -n 32 --greedy --ctx 512 --chunk 256 \
  [--hybrid-experts 1] [--expert-cache-mb N]
```

`--hybrid-experts 1` computes on the host, out of the weight mapping, the
experts the device tier cannot hold, while the device runs the ones it can.
One decode row needs 8 KiB of activation against a ~5.9 MiB expert here, so an
expert that is not resident is cheaper to compute where its bytes already are
than to read in and promote.

**The host arm takes the OVERFLOW, not every miss.** An expert the device tier
can still afford is promoted exactly as before (`!in_vram(layer, e) &&
full_for(source)`), so a budget large enough for the routed set keeps every
expert resident and the host arm never runs at all. Taking every miss instead is
a livelock — an expert that is never promoted never becomes resident — and the
first version did exactly that: with the 610 MiB auto budget, which holds the
whole routed set, decode ran at 6.4 tok/s with the cache stuck at 55/104 slots.
Found by measuring, then fixed.

Three interleaved repetitions per arm:

| expert budget | no hybrid | `--hybrid-experts 1` | host arm |
|---|---:|---:|---|
| auto, 610 MiB (holds the routed set) | 94.7 / 104.4 / 105.7 tok/s | 109.0 / 107.0 / 105.7 tok/s | 0 experts — **inert** |
| `--expert-cache-mb 16` (2 slots) | 16.8 / 17.3 / 17.5 tok/s | 5.5 / 5.5 / 5.5 tok/s | all 922 — **3.1× loss** |

```
[stats ] hybrid    0 expert(s) on the host over 0 row(s) in 0.0 ms (0.000 ms/expert, 8 thread(s)) | device took 922
[stats ] hybrid    922 expert(s) on the host over 1008 row(s) in 4971.1 ms (5.392 ms/expert, 8 thread(s)) | device took 0
```

Generated text was **identical** to the no-hybrid baseline in every arm, so the
arithmetic is right and throughput is the whole problem. The host arm costs
5.392 ms per expert on 8 threads against roughly 2.0 ms per expert of effective
read + promote (5.9 MiB expert, 28.8 experts/token at 17.2 tok/s ≈ 58 ms/token).
The host arm is a **per-layer barrier**: the merge needs both arms finished, so
the CPU work cannot hide behind the device. That is exactly the assumption
behind the 1.84× "perfect overlap" ceiling in
`docs/traces/cpu-expert-ceiling.txt`, and it does not hold. The split only pays
where the file is slower than the CPU.

**Verdict: correct, inert when the budget fits, a ~3.1× loss when it engages on
this machine — opt-in and default off.** Its value is the memory it declines to
use: a host-computed expert needs no VRAM slot and no WARM buffer, which is
memory the KV cache can have. It is not a decode accelerator here.

Arithmetic is covered by `test_cpu_expert_pool()`, which checks the pool
against an independent reference implementing the same four f16 rounding points
(`f16(g)`, `f16(u)`, `f16(silu(g16)*u16)`, f16 accumulation) in `F32` and
`Q8_0`, across five multi-threaded claim orders — jobs are claimed from an
atomic counter, so a shared row's accumulation order varies — plus the
structural cases (an unselected row stays exactly zero, a row two experts share
carries both contributions, an empty job list writes nothing). The suite went
2154 → **2214** checks, all passing. `kraken-bench --gate` rc=0,
`scripts/build_check.sh` 0 failed / 0 warning(s), `scripts/coherence_check.sh`
3/3 PASS.

---

## 10. Expert cache: nine iterations against one model

Nine configurations, one change each, on Qwen3-MoE-4x0.6B-2.4B-Q4_K_M
(`-p "The capital of France is" -n 64 --greedy --ctx 512 --chunk 256`), after a
warm-up run. Each arm is one pass; see the batch effect below for what that
permits.

| # | change | decode tok/s | HOT hit | promotions | us/expert promoted |
|---|---|---:|---:|---:|---:|
| 1 | auto budget, 610 MiB | 100.8 | 96.8% | 117 | 1172.3 |
| 2 | `--expert-cache-slots 104` (whole routed set, explicit) | 108.9 | 96.8% | 117 | 1083.4 |
| 3 | `--expert-cache-mb 400` | 62.7 | 81.0% | 692 | 545.8 |
| 4 | `--expert-cache-mb 200` | 32.7 | 46.5% | 1947 | 507.4 |
| 5 | `--expert-cache-mb 100` | 25.1 | 19.1% | 2942 | 505.6 |
| 6 | 400 MiB, `--expert-warm-mb 0` (WARM off) | 24.6 | 75.0% | 0 | 0.0 |
| 7 | 400 MiB, `--expert-warm-prefetch 1` | 62.6 | 81.4% | 676 | 542.3 |
| 8 | 400 MiB, `--hybrid-experts 1` | 57.1 | 81.3% | 677 | 551.6 |
| 9 | 400 MiB, hybrid, `--hybrid-threads 16` | 55.9 | 81.3% | 677 | 576.4 |

**The HOT budget is the only lever with a reproducible, monotone effect.**
610 -> 400 -> 200 -> 100 MiB costs 100.8 -> 62.7 -> 32.7 -> 25.1 tok/s, and it
tracks the HOT hit rate almost exactly (96.8 -> 81.0 -> 46.5 -> 19.1%). That is
the price list for trading expert VRAM away for the KV cache.

**WARM is the tier that makes a small HOT budget survivable.** With WARM off at
400 MiB, decode is 24.6 tok/s against 62.7 with it on — and the promotion count
is *zero*, i.e. every miss became a file read. 2.5x, from one flag. "Use less
VRAM" is affordable here only if the RAM tier stays.

**Promotion cost per expert does NOT degrade as the cache shrinks** (1172 -> 546
-> 506 us). An overflowing cache is served mostly out of RAM, so each move is
*cheaper*; what hurts is how many of them line up inside one step.

**Prefetch is a measured loss, not a neutral default.** At 400 MiB it is 62.6
against 62.7 tok/s, and at the auto budget (3 interleaved reps) it is
109.5 / 103.5 / 109.7 against 111.0 / 109.9 / 111.0 — while costing ~160 ms more
load (600/586/633 ms vs 435-472 ms) and 50-70% more file read (148-165 ms vs
95-98 ms). It buys nothing and pays for the whole corpus up front.

**An oversized WARM tier is inert.** 2048 MiB vs the 610 MiB auto: 109.7 / 112.1
/ 110.0 against 111.0 / 109.9 / 111.0. The auto WARM sizing already holds this
model's set.

**Slot-vs-byte sizing is noise.** Arm 2 read as +8% in the sequential pass above
and as +1.5% in four interleaved reps (auto 108.7/104.5/108.3/111.2 vs slots
111.6/109.4/106.8/111.5). It is reported only to mark it as not real.

**The hybrid arm costs ~9%** (57.1 vs 62.7) and does not scale past 8 host
threads (55.9 at 16) — the host arm is dequant-bound and already memory-bound.
See section 9.

### The batch effect, stated because it invalidated one result

The SAME auto configuration measured **100.8 tok/s** in the first sequential pass
and **110.6 tok/s** in a later pass over the same binary — a 9% difference, with
only +-1% between reps *within* a pass. Arm 2 was the casualty: an 8% "win" in
pass 1 that is 1.5% when interleaved. Only arms measured in the same interleaved
batch may be compared; arm ordering alone is worth more than most of the effects
in the table above.

---
## 11. The ternary `Q2_0` file, the id-142 file, and where host RAM goes

Three separate things came out of one question — "why does this ternary Bonsai
file load on llama and not here" — and only one of them was a decoder bug.

### 11.1 ggml id 42 is two different formats

`Ternary-Bonsai-27B-Q2_0.gguf` declares 498 tensors as ggml type 42 (`Q2_0`).
Sorting those tensors by their own file offset and comparing
`next_offset - offset` against the computed span gives **exactly 2.125 bits per
element for all 498 of them**, with not one span mismatch and every row length a
multiple of 128. Upstream's `Q2_0` is `#define QK2_0 64` with an 18-byte block,
which is 2.25 bpw, so the file could not have been made by upstream and the old
engine's `18 bytes / 64 values` reading was the wrong variant, not a wrong file.

The llama-dx fork (`github.com/Maxritz/LLAMA-DX`, cloned on this machine at
`H:/llamadx/llama.cpp/.llama-dx-reference`) has the answer in its own
`ggml/src/ggml-common.h`:

| | block values | block bytes | bpw | where |
|---|---|---|---|---|
| `Q2_0` (fork) | 128 | 34 (f16 + 32) | 2.125 | id 42, the Bonsai ternary exports |
| `Q2_0_64` (fork) | 64 | 18 (f16 + 16) | 2.25 | id 48, byte-identical to upstream's `Q2_0` |

`QK2_0` is 128 there and the header says in so many words that the two are not
byte-compatible. The decoder forks on the *row* length, so on the code side the
change is small: id 42 becomes a 128-value / 34-byte block with four 32-value
chunks, and the 64-value variant is added as id 48 (`Q2_0_64`) rather than being
thrown away. A 42 tensor whose measured density is 2.25 bpw came from upstream
and is the one that must be read as 48; the Swift `Qwen3.8-Flash-Next` shards are
exactly that case.

The code map did not have to change: the reference's `y = ((int)q - 1) * d` is
what the fork computes too, and the fork is the source of truth for its own
containers.

### 11.2 "The file is malformed" was the wrong call

The earlier session measured 2.125 bpw for this file, compared it against
upstream's 2.25, and concluded the file was broken because upstream llama.cpp
refused it with a number identical to ours (`tensor 'output_norm.weight' has
offset 337715200, expected 357580800`). That agreement is real, but it only says
both engines were applying the same wrong block size. The file's own offsets are
self-consistent at 34 bytes per 128 values, and the fork that wrote them defines
it that way. A file can be refused by two engines and still be well-formed; the
arbitration has to be the file's own geometry plus its producer's header, not the
number of engines that agree.

### 11.3 Verification

Four independent checks, all on the real file:

1. **Offsets.** 498/498 type-42 tensors match their span at 34 bytes per 128
   values, 0 mismatches (`python3 tools/gguf_scan.py --audit`, added here).
2. **Scale field.** Over 1,830,912 block starts the leading f16 is a plausible
   scale every time — no NaN, no Inf, no zero, no denormal, median `|d|`
   0.010498, max 0.037598. Read at 18, 26 or 36-byte strides instead, roughly a
   quarter of the sampled pairs land on packed code bytes instead. For reference
   the upstream-verified sibling `Bonsai-27B-Q1_0.gguf` measures median `|d|`
   0.009949 over the same number of blocks, so the two quantizations agree on the
   model's scale structure to within 5%.
3. **Code alphabet.** Over 254,976 blocks the 2-bit codes are 0/1/2 at
   35.14 / 29.79 / 35.07%, **code 3 at 0.00%**. The container is a symmetric
   ternary `{-d, 0, +d}`; the `q == 3` branch that would read `+2d` is dead code
   in practice, though it stays because that is what the reference computes.
4. **End to end.** It loads and runs: 6.66 GiB, prefill 29.1 tok/s, decode
   35.9 tok/s, and `scripts/coherence_check.sh` reports **PASS** — GPU and
   `--cpu` produce identical text.

The equal code weights are worth noting for whoever comes next: because the three
levels are within a percentage point of each other, *no* byte statistic can tell
which code maps to zero. A wrong assignment is invisible in a histogram and only
shows up as non-words, which is why the end-to-end run is part of the evidence
rather than a formality.

### 11.4 The external oracle, and why the first "ROCm" run was not

`H:/LLAMA-bins/rocm10/llama-cli.exe` was reporting CPU speeds because
`ggml-hip.dll` cannot load without `G:\ROCM10RT-gfx1201\bin` on `PATH`; the
loader then prints `load_backend: failed to load` with an empty error and carries
on with CPU only. `--list-devices` must print `ROCm0` (16,304 MiB) before anything
from that binary counts, and `llama-completion.exe -no-cnv --no-jinja` with stdin
from `/dev/null` is the bounded, non-interactive form — without it the tool starts
a chat REPL and waits.

With that fixed, the oracle did its job on the format the two engines share.
`G:/More-models/Bonsai-27B-Q1_0.gguf` (Q1_0, id 41) on ROCm generates, greedy,
`the evolution of the fundamental building blocks of computation. The first
computers were mechanical, and the first electronic computers were vacuum tube`
— **byte identical to kraken's 24 tokens** (llama.cpp 35.8 tok/s, kraken
49.5 tok/s). On `Qwen3.8-27B-GSQ-RCO-IQ2_XS.gguf` the two agree for about twenty
tokens and then part at a near-tie (`machines were developed` against
`computers were built`), which is what a 2.06 bpw model's logits look like when
two engines sum in a different order. The rule that falls out: an oracle agrees
word for word on a ternary model and is only a tie-breaker on a 2-bit one.

### 11.5 ids 102 / 104 / 107: resolved 2026-10-07

Two files carry these: `ornith-1.0-35B-Q3_0_ROCMFPX.gguf` (101 ×2, 102 ×40,
104 ×268, `general.file_type = 112`) and
`Qwen3.8-Distill-35B-A3B-Coder-Abliterated-Q2KXL_ROCMFPX.gguf` (102 ×229,
107 ×214, `file_type = 119`). Their own offsets put 102 at 6.5 bpw, 104 at
3.5 bpw and 107 at 2.5 bpw.

These are now implemented from the ROCmFPX fork (github.com/charlie12345/ROCmFPX):
host dequantizers `deq_q6_0_rocmfpx` / `deq_q3_0_rocmfpx` / `deq_q2_0_rocmfpx`
and device kernels in `src/hip/kernels/dequant.hpp`, all three using UE4M3
half-block scales (e[2], e ≤ 0x7E valid). The layouts are
`block_rocmfp6` {u8 qs[24]; u8 e[2]} = 26 B / 32 (6.5 bpw),
`block_rocmfp3` {u8 qs[12]; u8 e[2]} = 14 B / 32 (3.5 bpw),
`block_rocmfp2` {u8 qs[8]; u8 e[2]} = 10 B / 32 (2.5 bpw). FP2 codes use the
S40 ladder {-4,-1,+1,+4}; FP3 codes use magnitudes {0,1,2,4} with sign in bit 4;
FP6 codes use magnitude = code & 31 with sign in bit 5 (mag 0 → -32).

100 and 101 (Q4_0_ROCMFP4 / Q4_0_ROCMFP4_FAST) were already implemented and
`Ornith-1.0-9b-ROCmFPX-STRIX_LEAN.gguf` (100 ×40, 101 ×209, `file_type = 106`)
loads.

### 11.6 id 142: geometry measured, and the map is not a level map (resolved 2026-10-07)

`Ternary-Bonsai-2-27B-PQ2_0.gguf` (402 tensors, `file_type = 141`,
`general.name = Hf`, `general.basename = folded`) has the same *block shape* as
the 128-value `Q2_0`: 34 bytes per 128 values, leading f16 scale, 0 of 823,296
sampled scales implausible against ~25% for 18, 26 and 36-byte strides, and a
ternary code histogram (0/1/2 at 33.6 / 32.8 / 33.6%, code 3 at 0.00%). Its F32
tensors also differ from the Bonsai file's, e.g. `blk.0.attn_norm.weight` is
1.0655 against 1.0584 — an F32 norm weight is bit-identical across two
quantizations of one model, so this is a different source model and cannot be
used as a cross-file oracle.

What the shape does *not* give is the code map, and this is the negative result
worth keeping: `(q - 1)` and three other candidate level assignments were each
implemented and run end to end, and **every one produced non-words**. A rotated
container explains that: the same fork's newer weight types are described as
WHT-rotated (`GGML_TYPE_TQ3_1S`, `TQ4_1S` — "WHT-rotated … Lloyd-Max"), and a
block that has been transformed cannot be recovered by any per-code level map,
which is also consistent with the flat three-way code histogram.

**Resolved 2026-10-07.** The negative result above was real; its conclusion was
wrong. The PrismML fork's header (`H:/llamadx/prism-llama`, `GGML_TYPE_PQ2_0 =
142`, `GGML_TYPE_PTQ1_0 = 143`) defines the blocks outright — `block_pq2_0 { f16
d; u8 qs[32] }` = 34 B per 128, `block_ptq1_0 { u8 qs[24]; u8 qh[2]; f16 d }` =
28 B per 128 — and the piece every end-to-end attempt lacked was the file's
**block-1024 Hadamard activation transform** (`prism.hadamard` metadata: per
weight-site flags, sign vectors, the inverse-folded token embedding). Correct
weight codes are not enough when the activation space is wrong, so every level
map produced non-words end to end. With the fork's `dequantize_row_pq2_0`
transcribed byte for byte (diffed against real tensor bytes of
`blk.0.ffn_down`, `ffn_up`, `output.weight` and `blk.3.attn_q`: 4 cases, 0
mismatches) and the transform implemented on host and device,
`Ternary-Bonsai-2-27B-PQ2_0.gguf` passes device-vs-CPU coherence.
`PTQ1_0` (143) landed alongside it, though no file on this machine carries
that id. The four failed maps stay on record: they are why "rotated, so no
level map can work" looked like the only explanation.

### 11.7 Where host RAM goes, and a real `--cpu` defect

Chasing a 72 GiB reading gave a measured profile instead of an explanation.
On the dense 27B (`Ternary-Bonsai-27B-Q2_0.gguf`):

| run | peak private | peak working set | virtual |
|---|---|---|---|
| GPU (`--gpu`, defaults) | 0.35 GB | 0.37 GB | 267 GB |
| `--cpu` | 7.0 GB | 13.6 GB | 17.9 GB |
| MoE, `olmoe-1b-7b` on GPU | 3.0 GB | 3.0 GB | — |

The 267 GB is virtual address space only, reserved by the HIP runtime under
`HIP_VMEM_MANAGE_SUPPORT=1`; it is not committed and is not a leak. The `--cpu`
run is large because the engine copies all 6.66 GiB of quantized weights into
backend buffers and on the CPU backend those buffers are host RAM, on top of the
mapping's own pages. The number that can genuinely grow is the expert WARM tier
on a MoE run, and the policy says so out loud:

```
[info ] ram policy: 37.6 GiB budget (95.9 GiB installed, 75.2 GiB free) -> WARM tier 3720 MiB
```

That budget is half of free RAM, and the WARM tier takes `min(expert corpus,
budget)`. On a small MoE it lands at the corpus size (3.7 GiB above); on a model
with a large expert corpus it can legitimately claim tens of gigabytes, which is
where a 72 GiB figure comes from. The auto budget is a deliberate policy, not a
bug, but it is worth an explicit `--expert-warm-mb` when RAM is needed elsewhere.

One thing that *was* a defect: on the CPU backend `device_free_bytes()` returns 0,
so the KV planner computed a 0-byte HOT budget, warned twice that the resident set
"cannot hold one full pass", and paged the entire cache through WARM once per
layer per step — the warning told the user to raise `--kv-hot-mb`, which cannot
help on a backend with no device/host split. Giving the "device" budget the whole
cache on a CPU backend makes the tier inert, which is what a run whose cache fits
does on the GPU too:

| 27B, `--cpu`, `-n 16 --ctx 128` | before | after |
|---|---|---|
| prefill | 27,532 ms (0.3 tok/s) | 17,236 ms (0.5 tok/s) |
| decode | 4,951 ms/step (0.2 tok/s) | 3,454 ms/step (0.3 tok/s) |
| KV | tiered, 495 MiB migrated, 2 warnings | flat, resident, no traffic |
| output | `people. It is a history of the people who have worked…` | identical |

The CPU path is still slow — it is a 24B-parameter model computed scalars — but
it no longer pays for paging and no longer advises a fix that does not exist.

---
