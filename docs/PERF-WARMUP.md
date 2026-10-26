# The ranked expert warmup: what it is worth, and where it turns into a loss

Measured on the RX 9070 XT (gfx1201, driver 71260602), greedy, one process per
arm, one variable at a time. `--expert-warmup -1` is the default.

The phase stages the ranked expert set into the WARM (pageable host) tier, then
promotes the top of that order into VRAM, before the first token. It exists
because a MoE's first pass has to read every expert it routes to, whatever the
budget is.

## Where it wins: the compulsory first touch

| model | file | corpus vs budget | arm | prefill | decode | COLD |
|---|---|---|---|---|---|---|
| LFM2.5-24B-A2B Q4_0 | 12.6 GiB | 12312 vs 11964 MiB | off | 2410 ms | 39.7 tok/s | 25.7% |
| | | | warm | **111 ms** | **83.4** | 0 |
| Qwen3-30B-A3B i1-Q2_K | 10.5 GiB | 10008 vs 10008 (exact) | off | — | 9.7 | 41.8% |
| | | | warm | — | **47.5** | 0 |
| llada-moe-7b-a1b Q4_K_M | 4.2 GiB | 3720 vs ~11.9 GiB (3x slack) | off | 541 ms | 19.2 | 36.4% |
| | | | warm | **34 ms** | **107.4** | 0 |

The third row is the one that settled the default: pool slack does not make a
first touch free, so "does the corpus fit" was never the right question, and the
warmup is now on for every MoE with a device tier (CPU keeps the old comparison,
because the suite's fixtures assert on the counters a staging pass would move).
The cost is wall time at load, not throughput: 3.87 s on the 24B (12312 MiB read
at 7.9 GB/s), so a single cold 24-token run is slower end to end (6.66 s against
4.77 s) and break-even is about 165 output tokens; a second request in the same
process pays none of it.

## Where it loses: a corpus far larger than the pool

`Tiel-Coder-35B-A3B-MTP-APEX` — 24.8 GiB file, 22.6 GiB expert corpus, 11920 MiB
pool, 256 experts x top-8 x 40 layers = 320 expert acquires per token. 6-token
prompt, 256 tokens out, interleaved pairs at every point:

| arm | staged | wall | decode |
|---|---|---|---|
| `--expert-warmup 0` | 0 | 20.5 / 20.8 s | 16.8 / 17.2 tok/s |
| `--expert-warmup-ms 500` | 2622 MiB | 20.4 / 20.7 s | 17.5 / 17.8 |
| `--expert-warmup-ms 2000` | 9278 MiB | 20.9 / 21.1 s | **19.1 / 19.2** |
| `--expert-warmup-ms 4000` | 15424 MiB | — | 9.5 (16-token run) |
| default (no cap) | 22604 MiB | 35.1 / 35.2 s | **10.4** |

At 16 output tokens the same file looks fine (off 6.24 s / 6.2 tok/s against
6.32 s / 7.3 at a 500 ms ceiling and 12.32 s / 12.6 at 8000), which is exactly
why a short arm cannot be used to judge this: the decode is fast enough for the
damage to start before it is long enough to show.

**That default has since changed** — the loss was the VRAM half of the phase,
not the staging; see "The prefill is the lever" below, where the same file reads
22.3 / 24.1 tok/s on the current binary.

### The mechanism is the promotion count, not the staged volume

`--stats` on the 256-token runs, same model, same prompt:

| arm | acquires | HOT hits | WARM hits | promotions | promote ms |
|---|---|---|---|---|---|
| off | 82,745 | 87.3% | 2,702 | 2,702 | 447 |
| ms=2000 (9278 MiB) | 82,745 | 82.4% | 3,491 | 3,491 | 465 |
| full (22604 MiB) | 111,113 | 70.3% | 28,050 | 28,050 | 4,381 |

A WARM hit is not a free read: it is a promotion (pin + commit + an event,
about 156 us per 1.7 MiB), while the file's own mapping is page-cached and feeds
a cell with a plain copy at 4-13 GB/s. Once WARM covers the corpus, every expert
that leaves the pool has somewhere to fall back to and is promoted back in
instead of being left to the mapping -- 10x the promotions, 34% more acquires,
and the whole regression.

### What did not work, and should not be re-tried blind

Bounding the staged bytes by the pool does not fix it. Measured on the same
file at 256 tokens: 2622 MiB -> 17.6 tok/s, 5210 MiB -> 12.3, 9278 MiB -> 19.2,
10507 MiB -> 12.8, 22604 MiB -> 10.4. Non-monotonic, with a cliff between 9.3
and 10.5 GiB, so `min(plan, room)`, `min(plan, room)/2` and the plan itself all
land on the wrong side of it for some round width. Both were implemented,
measured and reverted; the loop is unbounded again and carries a note pointing
here.

Nor is the time ceiling the lever. The phase is repaid at about 0 / 20 / 42 / 59
/ 61 output tokens for ceilings of 500 / 1000 / 2000 / 4000 / 8000 ms on this
file, so a `--max-tokens`-derived ceiling (`30 ms` of staging per requested
token) was also implemented, measured and reverted: it multiplied the loss at
256 tokens, where it grew the phase to 7680 ms.

## The next lever, measured and refuted: the mapping policy

The pool's admission policy treats WARM as an eviction target rather than as a
staging area, so the next attempt was to remove the copy entirely:
`KRK_EXPERT_WARM_MAP=1` keeps nothing in host RAM, serves an evicted expert from
the page-cached mapping, and answers the warmup with a page-cache touch
(`touch_layer`: the same batched reads, discarded). The reasoning was that a
promotion copies the bytes twice — RAM into the staging ring, ring into VRAM —
while a mapping read copies once.

It loses on both files, and the reason is the batching, not the bytes.

| model | arm | decode | prefill | HOT | COLD |
|---|---|---|---|---|---|
| Tiel 35B-A3B, 256 tok | mapping, no prefill | 13.6 | 1678 ms | 87.9% | 12.1% |
| | `--expert-warmup 0` | **17.2** | 1208 ms | 87.3% | 6.6% |
| LFM2.5-24B-A2B, 24 tok | mapping | 16.5 | 1467 ms | — | — |
| | default | **84.0** | **37 ms** | — | — |

Same hit *rate* on Tiel — 87.9% against 87.3% — and 21% less throughput, so the
cost is per miss: 385 us more for each of the 10,006 misses against the 5,460 the
default ran. A promotion out of WARM is one batched call whose slice N+1 stages
while slice N is in flight; `acquire()`'s mapping path is three `upload()` calls,
each with its own pool-honor wait and its own read. Four thousand five hundred
serialised round trips is the whole difference. Keeping the copy is the cheaper
read path; the copy was never the problem.

The arm stays, off by default, because it is the A/B for any future change to
that path — and because the LFM2 row is the one that shows how much of the
warmup's value is the copy's batching, not merely its existence.

## The prefill is the lever (this is what shipped)

The same runs isolate the actual regression in the 10.4 tok/s row above. The
phase has two halves — stage WARM, then promote the top of that order into
VRAM — and the second half is the one that costs when the corpus is a multiple
of the pool. With `KRK_WARM_PREFILL=0` (staging unchanged, VRAM left empty),
Tiel reads 22.3 / 24.1 tok/s and a 363 ms prompt against 10.4 and 1197 ms.

At that ratio the prefilled prefix is not a head start inside the working set,
it **is** the pool: 4757 experts, 10500 MiB of a 10498 MiB budget, and the set
the run routes to is a different one, so the prefix holds HOT at 79% against the
87.5% the LFU earns on its own — while every displaced expert is a WARM
promotion rather than a read.

The rule is now: **prefill only when the pool can hold essentially the whole
routed set**, `corpus <= pool + 5%`. The 5% is LFM2's measured shape (12312
against 11964, 2.9% over once the KV tier and workspaces exist) and it is the
only slack in the rule.

| model | corpus vs pool | prefill | decode | prefill (prompt) |
|---|---|---|---|---|
| Tiel 35B-A3B | 22604 vs 10498 MiB (215%) | skip | **22.3 / 24.1** | 363 ms |
| LFM2.5-24B-A2B | 12312 vs 11964 (103%) | yes | **84.0 / 84.8** | 37 ms |
| Qwen3-30B-A3B | 10008 vs 10008 (100%) | yes | 46.0 / 47.0 | — |

`KRK_WARM_PREFILL=1|0` overrides the rule either way, which is how each row was
measured. A file whose corpus is much larger than the pool is now left to the
read-through tier and the LFU, which is what `--expert-warmup 0` used to be the
only way to ask for.
