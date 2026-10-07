# Expert warmup and the RAM tier: trace report

Companion to `docs/test-results.md` section 12. That section is the summary a reader
browsing results sees; this file is the measurement record: what was run, how, what
the numbers were, and which conclusions the method does and does not support.

Nothing here is a projection. Every table is a command that was executed on this
machine, and where the schedule was not clean that is written next to the row.

## 1. The question

`G:/More-models/Qwen3.8-Distill-35B-A3B-Coder-Abliterated-Q2KXL_ROCMFPX.gguf` is a
35B-class MoE that does **not** fit VRAM: 40 MoE layers, 256 experts each, top-8
routing, expert corpus 9600 MiB, embedding 2048, 16 query / 2 KV heads, head_dim
256, `qwen35moe` "Student Merged", 30 of the 40 layers recurrent (gated delta net),
one MTP block (`blk.40`) skipped. Formats present: F32, F16, `Q6_0_ROCMFPX` (6.5
bpw) and `Q2_0_ROCMFPX` (2.5 bpw).

The engine already had read-through weight loading (large tensors `read()` out of
the file rather than faulted in from the mapping) and a WARM host tier, so *re-reads*
were already host reads. What it could not avoid was **compulsory** traffic: the
first time a token routes to an expert that has never been touched, that expert has
to come off disk. On a 256x40 routing space over a short prompt, most routed experts
are first-touches, so the first tokens pay the file, and on a device tier smaller
than the routed set decode keeps paying it.

`--expert-warmup` answers that directly: a bounded startup phase that fills WARM
hottest-first and then promotes the top of that order into VRAM, before the first
token.

## 2. Method (and why it is stated this explicitly)

Two rules govern every number below.

1. **Arms are interleaved run-by-run, not run in blocks.** Sequential arms warm the
   page cache and driver state between measurements; on this box that ordering alone
   is worth several GB/s and produced false conclusions earlier in the project. Every
   A/B here alternates arm on and arm off.
2. **The switch exists before the measurement.** `KRK_EXPERT_WARMUP=0/1` selects
   exactly one code path; the two arms are one environment variable apart, same
   binary, same model file, same seed.

Caveat that applies to §4 only: within a repetition the three *order* arms
(off -> even -> ranked) necessarily ran in sequence, because the order is a property
of the warmup phase and not switchable mid-process. Repetitions were interleaved, so
an ordering effect is heavily damped but not formally excluded. The auto-budget
comparison in §3 is a clean two-arm interleaved A/B.

## 3. Headline: auto budgets, `-n 256`, 3 interleaved reps

Auto budgets as chosen by the engine: VRAM 9600 MiB, WARM 9600 MiB -- the whole
9600 MiB expert corpus fits in the two tiers together.

| metric | warmup off | warmup on |
|---|---|---|
| prefill | 2049 / 2032 / 2060 ms = 26.8 tok/s | **354 / 359 / 356 ms = 155.4 tok/s (5.8x)** |
| decode | 24.3 / 25.1 / 25.1 tok/s | **31.6 / 32.5 / 32.7 tok/s (+30%)** |
| COLD misses | 6434 (7.5%) | **0** |
| bytes read during the run | 6032 MiB (23.56 MiB/tok) | **0 MiB** |
| HOT hit rate | ~92% | **100%** |
| warmup wall clock | -- | 4823 / 4886 / 5029 ms |

At this budget the entire corpus fits in the two tiers, so the warmup is a pure
preload. It moves 9600 MiB of compulsory read-through to the front, where it costs
~4.8 s once, and every expert a token routes to afterwards is resident. The 5.8x on
prefill is not a kernel improvement -- it is the same 55-token forward pass no longer
sitting behind disk.

### 3.1 It is a first-token-latency feature, not a throughput feature

`-n 512`, 2 interleaved reps:

| metric | warmup off | warmup on |
|---|---|---|
| total wall | 21956.8 / 22049.8 ms | 22251.8 / 21810.1 ms |

A wash start-to-end, which is the honest reading: the ~4.8 s warmup is approximately
the decode saving that 256 extra tokens would have collected anyway. The engine ships
warmup **on only when the routed set exceeds the device budget** and off when it fits,
because the benefit is "the first tokens do not wait on disk", not "the run is
faster".

`--expert-warmup-ms` (default 20000) bounds the phase. Because the order is
hottest-first, being cut short costs *coverage* -- the tail of the ranking is not
staged -- never the top of the list.

## 4. Where the ranking matters: a tier smaller than the routed set

`--expert-cache-mb 6144 --expert-warm-mb 6144`, 502-position full-forward index,
`-n 448`, 2 reps, three order arms:

| order | decode | COLD misses | bytes read |
|---|---|---|---|
| off (read-through) | 25.4 / 25.3 tok/s | 6434 (7.5%) | 6032 MiB (23.56 MiB/tok) |
| even (index ignored) | 22.9 / 23.5 tok/s | 3260 (3.8%) | 3056 MiB (11.94 MiB/tok) |
| **ranked** | 25.0 / 24.1 tok/s | **2862 (3.4%)** | **2683 MiB (10.48 MiB/tok)** |

The ranked warmup matches the off arm's decode while reading **2.3x less from the
file**. The even order is strictly worse than both.

The result to hold onto, and the one that is not a win: cutting COLD misses 2.2x did
**not** raise decode above the off arm. At this tier the bottleneck is not miss rate.
That is the thread to pull next, and it points at the per-layer blocking D2H of the
router in `Engine::moe_ffn` and the 4-op-per-expert chain rather than at the cache.

### 4.1 Ranking sanity (layer 0)

Top-8 experts by summed routing mass: 243, 64, 49, 161, 85, 213, 245, 112 (mass 7.93
down to 5.65). Bottom four: 113, 25, 205, 252 (0.157 down to 0.049). **The top 64 of
256 experts carry 51.4% of the layer's mass.**

Mass is comparable *within* a layer only, and every layer runs on every token, so the
warmup reads the ranking **round-major** -- every layer's r-th hottest expert -- rather
than sorting globally. A global sort would spend the whole tier on the few layers
whose experts happen to carry the most probability and leave the other ~30 layers with
nothing.

At tiers that hold the whole corpus the ranking cannot matter, and the measurement
agrees: ranked 29.6 / 27.0 vs even 31.3 / 32.5 tok/s at auto budget -- both
meaningless, both tiers full.

## 5. `--ram-tier`: the spec table applied to an intended machine

`--ram-tier 16|24|32|48|64|96` plans the host tier for that RAM class instead of the
installed 96 GiB. The clamp is `ram_total = min(tier, installed)` and the `ram policy:`
line marks the run as planned. Sweep on the 35B, `-n 8`, warmup off, installed 95.9
GiB:

| `--ram-tier` | host budget | WARM |
|---|---|---|
| 0 (installed) | 36.6 GiB | 9600 MiB |
| 16 | 8.8 GiB | 9011 MiB |
| 24 | 14.7 GiB | 9600 MiB |
| 32 | 20.6 GiB | 9600 MiB |
| 48 | 32.4 GiB | 9600 MiB |
| 64 / 96 | 36.4 GiB | 9600 MiB |

At `--ram-tier 16` the run prints `[planned for the --ram-tier class]` next to the
policy line. Only the 16 GiB class actually shrinks WARM here, because the WARM want
is `min(host_budget, total_expert_bytes())` and this model's expert corpus is only
9600 MiB. Correct per spec, a no-op for this model -- the same shape the Laguna note
records. It binds for a corpus larger than the class's budget. The VRAM side is
untouched by the tier.

## 6. A defect found while measuring

The bench branch of `src/main_cli.cpp` returned **before** `write_expert_index`, so
`--expert-scan --bench` -- the exact combination the usage text advertises, and the
one that strips tokenisation and sampling out of the scan -- wrote no
`.krakenexperts.json` at all. This is not a crash; it looks like "the scan found
nothing". Fixed by writing the index before `engine.shutdown()` in the bench branch.
Verified: the same command now writes a 441991-byte file where it previously wrote
nothing, and `kraken-inspect`-style sanity on the index shows 502 positions over 40
layers, mode `full-forward`.

## 7. Verification

Suite count 2664 -> **2675/2675** with the 11 new checks in `test_expert_warmup()`,
which pin the three properties the measurement depends on -- a matching index stages
exactly the experts it names, a stale or missing index falls back to the even order,
and `--expert-warmup 0` stages nothing -- plus that `ExpertCache::reset_counters()`
zeroes the warmup's own loads/promotions/hits before the run's report, so the
HOT/WARM/COLD split is not partly a measurement of the phase meant to improve it.

Full gate suite for this commit:

| gate | result |
|---|---|
| `ninja` six-target line | rc=0, 0 `error:` lines |
| `kraken-tests` | 2675/2675, rc=0 |
| `kraken-bench --gate` | rc=0 |
| `coherence_check.sh` (SmolLM2-135M, Qwen3.5-0.8B, Qwen3-MoE-4x0.6B) | "3 coherent, 0 not" |
| `scripts/build_check.sh` | 0 failed, 0 warnings (fingerprint record refreshed once, after the rebuild) |

## 8. What is still open

- A **decisive ranked-vs-even measurement**: needs a tier strictly below the routed
  set *and* a fully interleaved three-way schedule. §4 is suggestive, not decisive.
- The **per-layer blocking router D2H** in `Engine::moe_ffn`, the **4-op-per-expert**
  chain, and the **O(n_expert) host scan per layer** (41 router pulls per token). §4's
  "2.2x fewer misses, no speedup" is the evidence that these, not the miss rate, are
  the mid-tier bottleneck.
- Real **long-prompt prefill**: 155 tok/s at `-n 256` is a 55-token fixed prompt, not
  a 4k-token prefill number.
