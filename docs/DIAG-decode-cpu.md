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

## 8. Findings F9-F28 — the DFlash drafter and speculative cost (2026-10-10)

Continues the numbering in §5. Everything below was measured on the current
tree; the two drafter files are `laguna-xs21-dflash-q8.gguf` (head set) against
`laguna-xs2-Q4_K_M.gguf` (target), command
`-p "The history of computing" -n 8 --greedy --ctx 512 --chunk 256 --expert-warmup 1`
unless a line says otherwise.

**F9 — the DFlash round never reaches its block, so 0/64 is not a verdict on the
drafter — CONFIRMED.** With a trace inside `generate_speculative_dflash`, all
16 rounds of a `-n 16` run die at the pre-check: `first(target)=395` against
`prop[0](draft)=34386`, `330` vs `16815`, `16708` vs `15356`, and so on. The
verify block is never entered, so the counters report the proposer's *first*
token 64 times and nothing about the rows. Status: the block's own quality has
not been measured once on this pair.

**F10 — the block's position is not the cause — CONFIRMED by measurement.**
`KRK_DFLASH_BLOCK_POS` sets `block_base_`; `-1`, `0` and `+1` each accept
**0/32**, with byte-identical stdout (`2255aa8533c5`, `-n 8`).

**F11 — the code and its own comment disagree about the block's contract —
CONFIRMED by code.** `dflash.cpp` says mask row *k* is "the candidate for
pos+k-1" (which needs `block_base_ = -1`); the default is `0`, i.e. the candidate
for `pos+k`. The knob takes any integer, and **no run reported which value it
used** (fixed: the load line now prints `dflash block placement base N row(s)`).

**F12 — the drafter's rows are flat, and that is the defect — CONFIRMED.**
Same round, same position:

| pos | target row (top-3, logit) | draft row 0 (top-3, logit) |
|---|---|---|
| 5 | 395(29.297) 372(29.172) 95(28.922) | 34386(5.289) 12627(5.031) 2447(4.949) |
| 6 | 330(30.609) 11114(30.25) 10208(29.45) | 16815(5.105) 34855(4.910) 18876(4.848) |

The target separates its top-3 by 0.13–3.8 logits on logits of 29–36; the
drafter's top-3 sit within 0.3 of each other on logits of 4.6–6.2 against a row
mean near 0. Both rows go through the *same* head weights (the target's own
`out_head`), so a flat row is a hidden state near-orthogonal to the head's
directions — the block's attention is not constraining the mask rows. The rows
are also self-similar: `84601` is top-1 of rows 2 and 3 of block 6.

**F13 — the fusion is live and token-dependent, so "the features never arrived"
is REFUTED.** `KRK_DFLASH_DEBUG=1`: `enc_out` min/max move per position
(`-3.91..4.15` at prefill, `-2.87..3.62` five tokens later) and the new
`row0 moved` field — row 0 against the previous commit — reads 1.285, 0.837,
0.942. The capture pipeline is not the failure.

**F14 — `feat rms` in the commit probe is a constant by construction —
CONFIRMED.** It printed `0.139751` at every commit: each aux block is RMS-normed
*before* it is scaled by its own weight, so the assembled RMS cannot vary. The
probe could not distinguish "arrived" from "constant". Fixed (F13's field).

**F15 — the target's row was invisible in the trace — CONFIRMED.** `--debug-topk`
prints an untagged `step N pos P nan=.. inf=.. top3:` line, so a `topk` grep finds
nothing and the comparison needed a second run. The `[dflash]` line now carries
`target_top3=[395(29.3) 372(29.17) 95(28.92)]` next to `prop=[...]`.

**F16 — the realign fix works where it applies — CONFIRMED.** Self-draft on the
same file (`--draft laguna-xs2`): 8/32 = **25.0%**, and *both* rounds that pass
the pre-check verify **fully** — `accept=4 full=1`, `row=[330,16708,12906,1388]`
matching `prop=[395,330,16708,12906]` position for position — with the trace's
`accept` equal to the engine's own 8 = 2 x 4.

**F17 — the residual loss is two single-row forwards of the same weights
disagreeing at the argmax — CONFIRMED.** The six dead rounds are near-misses:
`target=81 draft[0]=83`, `83` vs `81`, `882` vs `1030`, `672` vs `2316`,
`10208` vs `34415`, `32440` vs `58754`. Same file, same tokens, same backend: a
one-or-two-token disagreement, not an alignment one. This is F8's real remaining
cause and it caps *any* drafter, so it is P0.

**F18 — speculation is a loss on this pair too, and now it says so —
CONFIRMED.** `-n 16`: plain **19.6 tok/s**, self-draft 1.8 (10.9x), DFlash 2.2
(8.9x). A run that accepts nothing now prints
`[stats ] speculation accepted no draft tokens in N round(s): the drafter is pure
cost at this rate`; before it exited 0 silently.

**F19 — a round costs what the expert tier costs, not what the drafter costs —
CONFIRMED, and it explains a 5.7x swing.** One `-n 8` arm read 74.66 ms/round;
four later interleaved readings (2 pairs, base `0` vs `-1`) read 536.8 / 518.1
against 502.4 / 479.6 ms/round — base `-1` is ~5% cheaper, not 5.7x. The
`[stats] expert-path` line from a 6-token DFlash run shows why the per-round
figure moves: `room 158.7 | alloc 1081.7 | xfer 96.7 | promote total 1337.3 ms`.
Round cost tracks promotion state; no speculation cost may be quoted without the
expert-path line beside it.

**F20 — `kraken-bench --gate` has no speculation arm — CONFIRMED by inspection.**
An 8.9x speculative regression is invisible to the gate suite. The repo's own
models make this testable without a new artifact: `--draft <the same file>` is a
self-draft, so a gate arm can assert "accepted > 0 and text identical to plain".

**F21 — the two placement knobs were invisible in every report — CONFIRMED.**
`KRK_DFLASH_BLOCK_POS` and `KRK_DFLASH_LAYER_OFFSET` are the only way to reach
these behaviours and appeared in no `--help`, no `[info]` line and no report.
Fixed for the placement (F11); the layer offset stays env-only by design.

**F22 — a stored text hash without its protocol is not a fact, measured twice —
CONFIRMED.** Device and `--cpu` stdout differ only in the banner line
(`AMD Radeon RX 9070 XT ...` against `CPU (scalar reference), cpu`), so a hash
that includes stdout can never match across backends. Today's device-arm md5 at
the recorded protocol is `6b14f62c128e`, not the recorded `dc9c47cfa1bd`.

**F23 — laguna decodes, and now matches the scalar reference exactly —
CONFIRMED.** `Laguna-XS-2.1-IQ3_XXS.gguf` (the file that decoded all-NaN),
`-p "The history of computing" -n 16 --greedy --ctx 512`: device text
`" is a story of human ingenuity overcoming physical limitations. From Charles
Babbage"` is **byte-identical** to the `--cpu` f32 arm (`cmp` rc=0), with
`silu 0, experts-sum 0, residual 0` bounds fired and `residual rows stored
scaled: max exponent 5`. The residual storage scale is doing the work and no
store is being clipped at this protocol.

**F24 — the recorded `experts-sum 1020` for that file does not reproduce at this
protocol — CONFIRMED.** The same file at the same flags reads `experts-sum 0`.
The earlier figure came from a different prompt/protocol; both numbers are real
and neither is a property of the file alone.

**F25 — the device/CPU gap on that file is 12.6x — CONFIRMED.** 5.1 tok/s
(195.7 ms/step) against 0.4 tok/s (2475 ms/step) for a 12.95 GB IQ3_XXS.

**F26 — `build_check.sh` section 6 flags any legitimate rebuild — CONFIRMED by
design.** It read `the binary on disk was NOT produced by this build` after the
session's rebuilds. Re-recorded (`KRK_REFRESH_FINGERPRINT=1`) and re-run plainly:
`ok fingerprint — all 5 artifact(s) match the recorded build (sha256)`, rc=0,
with one residual warning that is the environment's `amdhip64_*.dll`, not the
build.

**F27 — the block is computed every round and thrown away on pre-check failure —
CONFIRMED.** Six rounds of a DFlash run propose 24 tokens and verify none, and
the drafter's whole masked block is paid for each one. The counters cannot tell
"block never verified" from "block verified and rejected"; the trace can, and now
does. A `[stats]` split is the durable fix.
**F28 — `commit()` is host work inside the target's decode step — CONFIRMED by
code.** Per committed token it does one device->host download *per captured
layer* (5 here), a host-side RMS norm per block, one upload, the `fc` GEMM and an
RMS norm, then, per drafter layer, a K/V GEMM, head-norm, rope and a KV append —
all serialized against the target's next step. Its cost is inside every
speculation round, including the rounds whose block is discarded.

### Fixes applied in this pass

- `src/engine.cpp`: the `[dflash]` pre-check trace prints the target's own top-3
  (`logits_host_`) beside the proposal rows (F15); both accept traces moved after
  their accept loop, where `accept`/`full` were pre-loop values and the `REJECTED`
  line quoted the wrong pair before (F16).
- `src/engine.cpp`: the drafter load line reports the block placement in effect
  (F11, F21) — verified: `[info ] dflash block placement base 0 row(s)`.
- `include/krk/dflash.hpp`, `src/dflash.cpp`: a `block_base()` accessor, and the
  commit probe's constant `feat rms` replaced by `row0 moved` — row 0 against the
  previous commit — which reads 1.285, 0.837, 0.942 and so is able to fail (F13, F14).
- `src/main_cli.cpp`: a run whose speculation accepts nothing says so, on both the
  CLI and bench paths (F18) — verified in a 6-round DFlash run.
- `docs/DIAG-decode-cpu.md`: this section; `tools/_patch_*.py` are scratch and stay
  out of the tree.

### TODO (F9-F28 pass)

```
TODO src/engine.cpp:3687: [P0] the DFlash block's mask rows come out flat (top-3
      within 0.3 of each other on a logit scale of ~5, against the target's 29-36)
      while the fusion is live and moves | fix: test the block's per-layer
      causality/window (layer_causal_/layer_window_, decoder_laguna) and the
      injected K/V; a wrong causal flag leaves every mask row seeing only id_last
      | test: KRK_SPEC_TRACE=1, prop[0] should approach first(target)
TODO src/engine.cpp:3319: [P0] a self-draft accepts 25% where it must be ~100%: the
      pre-check compares two single-row forwards of the same weights and they
      disagree by one or two tokens (81 vs 83), and that caps every drafter
      | fix: find the op the two instances differ in (KV layout/dtype, attention
      split threshold) | test: dump both instances' logits at one position
TODO src/engine.cpp:3764: [P1] a DFlash round pays the whole drafter block even
      when the pre-check discards it, and the counters cannot say so
      | fix: count proposed/verified/never-verified separately | test: -n 16 pair,
      expect 64 proposed, 0 verified, 64 never-verified
TODO scripts/: [P1] kraken-bench --gate has no speculation arm, so an 8.9x
      speculative loss is invisible to the gate | fix: a self-draft arm on a repo
      model asserting acceptance > 0 and identical text | test: --gate rc
TODO src/dflash.cpp:631: [P1] commit() downloads one host copy per captured layer
      per token and uploads again | fix: fuse the per-aux norm on the device and
      skip the round trip for a single row | test: --profile on the commit span
TODO docs/: [P2] every stored text hash needs its protocol inline (F22): the device
      and CPU banners differ, so a stdout hash never matches across backends
```
