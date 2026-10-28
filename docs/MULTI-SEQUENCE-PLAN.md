# Multiple sequences: what it actually takes

Requirement: serve N sequences concurrently, sharing the model, rather than one
sequence start-to-finish. This file records the code as it stands, what the
change costs, and the order to do it in. It is scoping, not a result.

## The one fact that makes this tractable

**Every kernel except attention is already row-independent.** A batched GEMM
over `rows` activations does not care whether those rows came from one sequence
at positions 100..107 or from eight different sequences at position 40 each. The
GEMM is 85% of prefill and 83% of decode time on the dense fixture, so the
expensive part of the engine needs **no change at all** for batching.

Only attention knows where a row's keys live. That is the whole job:

```
forward(toks, n, pos0)          <- one position for all rows
AttnDesc { pos0, pos_stride }   <- keys at pos0 + t, one contiguous KV
kv_cap_                         <- one KV region sized for one context
```

Three things have to become per-row. Nothing else.

## What exists today (verified by symbol)

| piece | state | where |
|---|---|---|
| Batch depth `C` | **exists** | `src/engine.cpp:1166`, `const i64 C = chunk_` |
| Batched forward | **exists** | `Engine::forward(const i32*, n, i32 pos0, ...)`, `engine.hpp:661` |
| Per-row position | **absent** | `AttnDesc::pos0 = position of query row 0` (`backend.hpp:126`) |
| Per-sequence KV base | **absent** | single `kv_cap_`, `nv_ctx`-sized (`engine.cpp:1048`) |
| KV block table | **absent** | no `block_table`/`page_table`/`kv_page` symbols |
| Sequence state reset | **exists** | `recurrent_reset()` (`engine.hpp:893`) |
| Scheduler / N in flight | **absent** | no `continuous`, `n_parallel`, or slot concept |
| Prefix reuse | **partial** | `server.cpp` reuses a matching prompt prefix in the KV |

So the batch machinery is half-built: it batches *rows* (prefill chunks) but
assumes all rows belong to one sequence.

## The change, in four stages

### S1 — per-row position and per-sequence KV base (the enabling change)

Add to `AttnDesc`:

```c
const i64 *row_pos = nullptr;   // null: pos0 + t, the current behaviour
const i64 *row_kv_base = nullptr;  // null: one contiguous region
```

Both default to null, so **every existing caller keeps its exact behaviour** —
this is an additive change with no behaviour change to gate. Attention then
indexes `kcache + row_kv_base[r] + row_pos[r] * pos_stride` and compares
`pos_pos[r]` against `pos_stride` for the causal bound.

Touchpoints, all of which currently read the scalar `pos0`:

* `src/backend_cpu.cpp:964` `CpuBackend::attention` — three bodies (f32, `kv_int8`,
  and the gather path) plus the `KRK_KTIME_MAC` accounting, which must sum over
  rows rather than assume `pos0 + n_tok`.
* `src/npu/npu_offload.cpp` `npu::AttnCall` — the device copy of the descriptor.
  It can keep refusing non-contiguous rows (`npu::attention` already refuses what
  it cannot represent) and fall through to the host.
* `src/hip/` attention kernel — same descriptor, when HIP is enabled.
* `mla_attention` / `mla_shared_attention` — same treatment, or refuse.

**Acceptance test:** N single-token rows with per-row positions and bases must be
**bit-identical** to running the N sequences one at a time. That is a strong,
cheap gate and it should be written before the change, against today's
one-sequence-per-call path.

### S2 — KV partitioning

`kv_cap_` becomes a budget divided across slots. The simple version is fixed
slots (`kv_cap_ / N` each), which wastes the difference between sequences and is
what paged KV exists to fix — but fixed slots are correct and one afternoon,
whereas a block table is a rewrite of the KV indexing everywhere.

Per-slot state that must also be partitioned: `recurrent_reset()`'s convolution
windows and delta-rule states (one set per recurrent-layer sequence), plus
`kv_tier.cpp`'s residency accounting.

### S3 — scheduler and server

A step loop that, each token boundary, assembles the active sequences into one
row batch, calls `forward` once, and scatters the sampled tokens back. Admission
and eviction policy lives here. `src/server.cpp` already has a request queue and
prefix reuse; this is where continuous batching becomes visible.

### S4 — the payoff

With S1–S3, one GEMM call covers N sequences, so the weight read is amortised N
times. That is the entire throughput argument, and it only pays where the engine
is weight-read-bound — which on a 2–3 GB VRAM / NVMe setup is always.

## What this does and does not buy

**Buys:** throughput. N sequences per weight read. On this host's CPU backend the
same effect can be measured today, without any scheduler, by batching rows: the
GEMM already does.

**Does not buy:** single-stream latency. One sequence alone decodes exactly as
fast as before. This is why it is the wrong first move if the goal is "my one
chat is slow".

### Measured: batching does not amortise the GEMM on this host

`kraken-q80-gemm-bench 768 2048 <rows>`, GMAC/s and the per-row cost derived
from it:

| rows | GMAC/s | us per row |
|---:|---:|---:|
| 64 | 18.20 | 86 |
| 128 | 21.26 | 74 |
| 256 | 15.35 | 102 |
| 512 | 16.00 | 100 |
| 1024 | 19.35 | 81 |

Per-row cost is **flat from 64 to 1024 rows** within this host's noise. The
single-row kernel (`dot1`, the `kiters` variant) and the 8-row kernel measure
about the same GMAC/s, so there is no efficiency cliff at one row either. The
reason is that this kernel is ALU-limited: every row does the same MACs, and the
weight block decode is already amortised across the 8 rows of one tile, so a
9th row from a different sequence reuses nothing the 8th did not.

**So the amortisation case for batching is about weight READS, not FLOPs.** It
pays when re-reading weights is expensive — disk-resident experts (one expert
read per batch instead of per sequence) or a GPU (kernel launch and weight
upload per step). On this host, with weights in page cache, batching N sequences
mostly buys throughput by keeping the ALU busy while *other* sequences' I/O
completes, not by making the arithmetic cheaper per row.

That is not an argument against doing it — it changes where the payoff is
claimed and how it is measured. On the target setup the number to watch is
`[stats ] expert-path read <ms>` per sequence, which should fall by the batch
factor as sequences are co-scheduled; the GEMM's GMAC/s should not be expected
to move.

## Order

1. **S1 with the bit-exact test** — self-contained, no behaviour change when
   unused, and it is the piece every other stage depends on.
2. Measure the batching curve (above) to confirm the payoff is real on the
   target host.
3. S2, then S3.
4. Paged KV (block table) only after fixed slots are working and the waste is
   measurable, because it is the expensive version of S2.
