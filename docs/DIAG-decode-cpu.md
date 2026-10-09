# Decode-path diagnosis — CPU, step accounting, and the speculative round

Mode: `ponytail-diag` (expanded) for structure, `sherlock-it` for every number.
Every claim below carries the command that produced it. Claims that need hardware
or a run I did not make are marked `PROBE NEEDED`.

## 0. Reproduction

```sh
# uninstrumented CPU/thread/RSS per run (new tool)
python3 tools/probe_cpu.py -- ./build-hip/kraken.exe -m models/Qwen3-MOE-4x0.6B-2.4B-Q4_K_M.gguf \
    --greedy --seed 7 -n 64 --ctx 512 -p "The history of computing is a history of abstraction, from relays to vacuum tubes"
# instrumented step timeline (NOT for timing — see F2)
./build-hip/kraken.exe -m models/Qwen3-MOE-4x0.6B-2.4B-Q4_K_M.gguf --greedy -n 64 --ctx 512 -p "..." --profile
# speculative arm, same prompt, target used as its own draft (exercises mismatch + full-accept)
./build-hip/kraken.exe -m models/Qwen3-MOE-4x0.6B-2.4B-Q4_K_M.gguf --draft models/Qwen3-MOE-4x0.6B-2.4B-Q4_K_M.gguf \
    --draft-tokens 4 --greedy --seed 7 -n 64 --ctx 512 -p "..."
```

## 1. Inventory (decode-relevant surface)

| Subsystem | Owner | Feature |
|---|---|---|
| Token loop (plain) | `Engine::generate` | one token/step, device top-k argmax |
| Token loop (spec) | `Engine::generate_speculative`, `generate_speculative_dflash` | propose / pre-check / verify / accept / realign |
| KV cache | `Engine::forward_core`, `kv_rollback`, `kv_rollback_zero` | contiguous rows, logical shrink |
| KV tiers | `src/kv_tier.cpp` | HOT/WARM/COLD residency, markers |
| Expert cache | `src/expert_cache.cpp` | LFU+aging, pins, WARM inclusive, read-through |
| MoE step | `Engine::moe_ffn` | host plan build + device grouped GEMM |
| Recurrent block | `Engine::gdn_forward` | delta net + conv, no KV rows |
| Mamba-2 block | `Engine::mamba2_forward` (working-tree) | SSD scan |
| Emission policy | `TokenEmitter` | stop strings, holdback, sink |
| Sampling | `Sampler` | penalty → temp → top-k → top-p → min-p → CDF |
| Device timing | `src/hip/op_time.hpp` | per-op spans (`KRK_PROFILE=1`) |
| Host timing | `include/krk/host_time.hpp` | per-phase wall (`decode.step` …) |
| GPU engine split | `engine_usage.hpp` | Windows PDH Compute/3D/Copy |
| Host CPU/RSS | `tools/probe_cpu.py` (new) | CPU time, threads, RSS per run |

## 2. Flow

### 2.1 One plain decode step

```
sample/argmax ─┬─ EOS ──────────────► finish(Eos)
               └─ else
                  emit.accept(token) ─┬─ stop string ─► finish(Stop)
                                      └─ else
                                         forward(token, pos)      # q/kv/attn/ffn
                                         pos++
                                         fetch_logits(topk)       # 16 B back on device
                                         [split counters if --profile]
```

### 2.2 One speculative round (model-draft)

```
first = argmax(target logits @pos)
propose d tokens from the draft (device top-k each)
        │
        ├─ d == 0  OR  first != prop[0]  ─► emit.accept(first)      [single emission]
        │                                   forward(first); pos++
        │                                   draft re-align @ pos-1
        │                                   continue               [block never run]
        └─ match
           verify: forward_core(prop[0..d)) in ONE batch
           per-row device argmax → row_pred
           a = longest prefix where row_pred(j)==prop[j]; full = (a==d)
           emit prop[0..a)                                  [single emission]
           full?  ─► bonus = row_pred(d): emit, one plain step
           partial ─► kv_rollback(pos+emitted)
           realign draft to the accepted prefix
```

### 2.3 MoE expert acquisition

```
acquire(layer, expert)
  ├─ HOT  (VRAM resident)  ─► device table read
  ├─ WARM (host RAM copy)  ─► DMA to VRAM (promotion)
  └─ COLD (no copy)        ─► file read → WARM → DMA to VRAM
prefetch at load charges the same COLD counters (see F3)
```

## 3. Truth tables

### 3.1 Speculative round — where a token is emitted

| d==0 | first==prop[0] | full | emission site | count |
|---|---|---|---|---|
| T | — | — | mismatch branch `emit.accept(first)` | 1 |
| F | F | — | mismatch branch `emit.accept(first)` | 1 |
| F | T | F | verify loop `emit.accept(prop[j< a])` | a |
| F | T | T | verify loop | a, then `emit.accept(bonus)` → a+1 |

No row emits the same token twice; the mismatch rows precede `continue`, so the
verify loop is unreachable in them. `←` the hypothesis "first token emitted
twice" would require a fall-through that does not exist (F1).

### 3.2 Model dispatch (`generate`)

| is_recurrent | draft loaded | greedy | path |
|---|---|---|---|
| T | any | any | plain (spec refused at `load_draft`) |
| F | T | T | `generate_speculative` / `_dflash` |
| F | T | F | plain + warn once |
| F | F | any | plain |

### 3.3 Expert tier decision

| in VRAM | in WARM | action | charged as |
|---|---|---|---|
| T | — | use as-is | HOT hit |
| F | T | DMA promotion | WARM hit |
| F | F | file read → WARM → VRAM | COLD miss |

## 4. Measurements

| Arm | wall | CPU (user+sys) | mean | peak | peak threads | peak RSS | decode |
|---|---|---|---|---|---|---|---|
| MoE 4x0.6B, device | 1.403 s | 2.266 s | **1.62 cores (5.1% of 32)** | 3.56 cores | 17 | 1009 MiB | 113.6 tok/s (8.8 ms/step) |
| SmolLM2-135M, device | 0.955 s | 0.938 s | **0.98 cores (3.1%)** | 1.71 cores | 14 | 275 MiB | 543.4 tok/s (1.8 ms/step) |

MoE step accounting (uninstrumented): 8.8 ms/step; runs shown by the app itself:
prefill 67.0 tok/s, load 419 ms (0.30 GiB @1.9 GB/s), expert path read 77.4 ms +
alloc 29.6 ms + xfer 101.4 ms = **208 ms inside a 555 ms decode**.

Instrumented (`--profile`, one step): 817 ops, host wall 46.9 ms, device spans
27.2 ms (58%), **device idle 21.3 ms (45%)**, `download_f32` host 42.8 ms.

## 5. Findings

**F1 — "the first accepted token is emitted twice" — REFUTED.** The mismatch
branch ends in `continue`; the verify/accept loop cannot run in the same round.
Bit-identity with plain greedy holds on device and CPU for a run where the
mismatch path fired 36 times (40/144 accepted, 27.8%); `kraken-tests`
2675/2675. Status: confirmed refuted (table 3.1 + runs in §0).

**F2 — `--profile` inflates the host column ~5x — CONFIRMED.** Uninstrumented
step 8.8 ms vs profiled 44.1 ms. The 21.3 ms "device idle" and the 42.8 ms
`download_f32` host figure are recorder artifacts plus dependency stalls, not a
per-token CPU cost. Never quote a step time from a `--profile` run.

**F3 — the "610 MiB read from the file" during decode is the load-time
prefetch, charged deliberately — CONFIRMED by code.** `expert_cache.cpp:778-786`
charges a prefetched expert as one COLD miss so the counters sum against
acquires; `9.54 MiB/tok` is just `610/64`. No re-read bug.

**F4 — where host CPU actually goes (ranked).** (a) load/prefetch: peak 3.6
cores, 17 threads, 610 MiB; (b) expert promotion xfer+alloc: 131 ms of a 555 ms
MoE decode; (c) decode proper: well under one core. On a resident small model
this engine is **not** CPU-hungry; the "lots of CPU" impression comes from the
load/paging window.

**F5 — `PROBE NEEDED`: a paged big-model decode.** The docs record 48 x 26.6 ms
`warm_read` at 95% of a 27B MoE decode. Nothing in this report measures that
arm; it is the case where host CPU genuinely saturates.

**F6 — host parallelism is per-phase and deliberately NOT core-scaled.**
- decode: one host thread submits every op and builds the MoE plan; ~1 core by
  construction, unchanged on a 32-thread box (`src/par_pool.hpp` is the CPU
  backend's pool and is idle on a device run).
- load/prefetch: `kPullThreads` reader threads (`backend_hip.hip` ~906-947,
  tunable via `KRK_PULL_THREADS`), chosen by measurement: 4 readers x 1 MiB
  chunks = 6.64 GB/s, four handles = 15.9 GB/s, and more threads do not help
  because NVMe, not the CPU, is the limit. The measured 3.6-core peak is 4
  readers + the main thread.
- `--cpu`: `Pool` sized to hardware threads - 1 (or `--threads N`).
- `--hybrid-experts`: `--hybrid-threads` (0 = min(hardware, 8)) -- opt-in and
  default off (measured 3.1x loss, AGENTS.md).
So the answer to "does it scale to my cores?" is: no, and adding cores will not
move the MoE numbers; the limits are the single-threaded decode host path and
the storage.

**F7 — the MoE "roulette" is real but bounded, and one flag removes it.**
Interleaved A/B, same prompt, 2 reps per arm:

| expert budget | decode tok/s (rep1/rep2) | HOT | VRAM evictions | promotions |
|---|---|---|---|---|
| auto 610 MiB | 116.3 / 118.3 | 96.4% | 23 | 131 |
| `--expert-cache-mb 1024` | 124.4 / 122.0 | 96.9% | **0** | 112 |

The corpus is 112 experts (~610 MiB) and the auto budget was 104 slots, so the
wheel spun by exactly one slot: 23 evictions, 131 promotions, +5% decode once
the budget covers the corpus. An evicted expert keeps its WARM copy (WARM is
inclusive), so the cost of the spin is a ~1 ms DMA, never a file read.

**F8 — speculation is currently a 3.6x loss, and the fingerprint says why. (Reproduced 2026-10-09: still a loss, now 4.0x — 20/72 accepted = 27.8%, the same 5/18 per round, 171.4/172.7 tok/s against 669.1/707.0 plain, text identical. See docs/MTP-DRAFTER.md §3.)**
Same model as target and draft, `--draft-tokens 4`: 36 rounds, **40/144
accepted (27.8%)**, 64 tokens in 2005.9 ms = 31.9 tok/s against 113.6 tok/s
plain. A self-draft must propose its own argmax, so acceptance should be ~100%
(4 + bonus per round, exactly what the test suite asserts bit-identically).
Output still matched plain greedy byte-for-byte, so the verify/accept machinery
is correct and the loss is proposal alignment, not acceptance semantics.
`PROBE NEEDED`: `--debug-topk 1` at pos and pos-1 to see whether the draft's row
being argmaxed is one position stale. Embedded speculative heads (MTP/`nextn`,
dspark) are counted at load and never used; DFlash head sets ARE used
(`--draft`, `src/dflash.cpp`); `dspark` is refused by the arch table.

## 6. Fixes applied this session

- `src/engine.cpp` speculative loops: an interrupted edit had duplicated the
  verify/row_ids/row_pred blocks and pasted a dflash `commit` into the
  model-draft path. Repaired to the original single-copy text; build rc=0 with
  0 `error:` lines; `kraken-tests` 2675/2675; spec output byte-identical to
  plain greedy on device and CPU, and to itself across repeats.
- `tools/probe_cpu.py`: process CPU/RSS/thread sampler (the app logs device and
  host *wall* time, GPU engine split, VRAM/RAM budgets — but not host CPU%).

## 7. TODO

```
TODO src/engine.cpp:2141: [P1] MoE host plan = 28 blocking 1 KB downloads/token (dependency stalls)
      | fix: device-side router top-k feeding the existing gemm_group path
      | test: probe_cpu + --bench decode tok/s, interleaved A/B
TODO tools/probe_cpu.py: [P2] fold host CPU%/RSS into --bench/--info so the
      number is the app's own, not an external probe
TODO docs: [P2] AGENTS.md "engine.cpp is CRLF" is stale — HEAD blob is LF-only
TODO src/hip/backend_hip.hip: [P2] KRK_PULL_THREADS tops out at kPullThreads;
      probe a larger ring on a cold-cache 27B load before changing the constant
TODO src/engine.cpp:2717: [P0] spec acceptance 27.8% on a self-draft where it must
      be ~100%; output is correct so it is alignment, not acceptance semantics
      | fix: find the stale proposal row (draft head feed @ pos-1)
      | test: --debug-topk 1, expected 4+bonus tokens per round
TODO src/expert_cache.cpp: [P1] auto VRAM policy under-sizes a MoE whose corpus
      is one slot bigger than the budget; the 6.0 GiB cap sat far below the
      card's 15.3 GiB free | fix: floor the expert budget at the corpus size
      when it fits | test: 0 VRAM evictions and >=112 promotions over 64 tok
```
