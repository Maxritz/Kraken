# Research references, with what this repo has measured about each

Where the external plans' papers meet this code. Each entry states the paper's
idea, then **what is already true here** with the number that settles it, then what
is genuinely open. Numbers are measured on this box (RX 9070 XT, gfx1201, 15.92 GiB
VRAM) unless a file is cited.

Source documents: `NATIVE_LLM_ENGINE_RESEARCHED_MASTER_PLAN.md` (§7 primary list),
the Aug-2026 cs.LG digest, and the RDNA2/RDNA4 ISA PDFs.

---

## Expert storage, streaming and residency

### The storage question, measured (this box, re-measured 2026-10-21) — which limit binds

The three candidates -- file reads, H2D transfers, the per-layer barrier -- were
separated by varying only the VRAM expert budget on one file
(`Qwen3.8-Distill-35B-A3B-Coder-Abliterated-Q2KXL`, 40 layers x 256 experts = 10240
x 0.937 MiB = a 9600 MiB corpus, RX 9070 XT). The engine prints its own four-way
split, so no profiler is involved. **These rows are the current binary's, and two
of them no longer agree with the 2026-10-09 table this replaces** -- see the note
under it for what changed and why.

| arm | command (tail) | WARM pinned | read | room | alloc | xfer | decode |
|---|---|---|---|---|---|---|---|
| corpus resident | `--expert-cache-mb 9600 --expert-warmup 1 --ctx 512 -n 24 --greedy` | 9600/9600 | 0.0 | 4.0 | 0.9 | 4.9 | **38.7 tok/s** (25.8 ms/step) |
| corpus resident, no warm-up | same, without `--expert-warmup` | 2295 of 9600 used | 442.5 | 0.1 | 238.6 | 24.0 | 16.1 tok/s (62.0 ms/step) |
| tiny VRAM cache, WARM staged | `--expert-cache-mb 256 --expert-warmup 1 --ctx 512 -n 24 --greedy` | 9600/9600 | **0.0** | 33.2 | **3.5** | 31.3 | 14.8 tok/s (67.6 ms/step) |
| the same arm with the OLD pin policy | plus `KRK_HOST_PIN_MB=4800` | 4800/9600 | 0.0 | 29.6 | 3.4 | 2815.3 | 7.2 tok/s (139.8 ms/step) |

(The 2026-10-09 table read 35.8 / 16.5 / **7.0** tok/s with `alloc 2445.5` on the
last arm. Two things moved: the pool's hand-out learned to express its reuse
dependency as a stream wait and save the host query -- `alloc` 2445.5 -> 3.4 ms,
with `alloc 819 reuse 18231` hand-outs -- and `pinned_warm_prefix_bytes` stopped
keeping half the tier pageable, which is the 7.2 -> 14.8 row. Both changes are
recorded with their own A/Bs in `docs/AUDIT-gpu-resident-expert-cache.md`; the
old rows are kept here only as the historical reading.)

So the ranking is *conditional*, and the condition is where the working set sits:

1. **The file binds when WARM is cold.** Without the staged warm-up the same
   model reads its misses from the mapping as it goes: **read 442.5 ms** against
   24.0 ms of xfer, and 16.1 tok/s against 38.7 once WARM holds the corpus. The
   fill path is the slow one — 9600 MiB read in even order took 16.0 s (0.6 GB/s)
   on the 2026-10-09 protocol, per-transfer overhead rather than the bus.
2. **When WARM holds the corpus but VRAM does not, the COPY binds — not the
   allocator, which is what this row said before it was re-measured.** At
   `--expert-cache-mb 256` every acquire is served from RAM (read **0.0 ms**) and
   the whole promote total is `xfer` plus a 33.2 ms room row: `alloc 3.5 ms` of
   69.6 ms, i.e. 0.5 us per hand-out (`alloc 819 reuse 18231`), because the pool
   hands reused cells out with a stream-side dependency instead of a host query.
   What remains is the pageable promotion rate, and it is worth **2.14x** on this
   arm: 139.8 -> 67.6 ms/step once the whole WARM tier is pinned (7.2 -> 14.8
   tok/s), with `xfer` 2815 -> 31 ms. Shrinking HOT by 37x still costs 2.6x the
   step (38.7 -> 14.8 tok/s) — that part of the old conclusion holds — but the
   fix it pointed at was the source's pinning, not a slot plan.
3. **The per-layer barrier is not in this list.** It is the *hybrid CPU* arm's
   cost (3.1x, `docs/test-results.md` §9): the device-only path never waits on a
   host merge.

### llama.cpp's GPU-resident expert cache (#29887, #30112, #29949) -- decision and copy moved onto the card
**What it is:** three pieces of upstream work on the same problem this tier solves, and the
only one of them is about policy. #29887 (merged 2026-10-07, `--moe-cache-mib`) keeps expert
weights in host memory and caches recently routed ones in a GPU bank, decided on the HOST in
the scheduler's copy callback. #29949 (branch `lru-upstream-v3`, commit `c81901a`) moves that
decision and the copies INTO the compute graph -- `LRU_DECIDE` reads the router's top-k on the
device against a GPU i32 LRU state, `LRU_FILL` copies the misses, and the expert matmul is
handed slot ids in place of expert ids. **The number that matters is the equal-budget one:** at
the same ~6.5 GiB pinned-host budget on the same 7900 XTX, #29949 reads **31.2 tok/s at 77.9%
hit against #29887's 20.1 at 75.0%** -- so the gap is not capacity and not the bus, it is where
the work runs. Their correctness evidence is the shape to copy: PPL 1.4290 without a pool
against 1.4292 at every slot count 64..192, and 12/12 greedy runs with one text hash.
**Measured here:** the dependency they removed is the one this engine still pays --
`moe.route.fetch` **0.523 ms/layer** on Mellum2.1 (28 layers, 23.7 ms/step) of which only
0.077 ms is the copy; 14.6 ms of that step is the layer waiting for its own GPU work to retire
so the host may read the router row. Our device dispatch table exists but is PUBLISHED ONCE
and only while the whole routed set is resident (`include/krk/moe_table.hpp`), which is never
true on a model that does not fit -- so the per-expert loop with a host round trip is the
default exactly where a cache is needed.
**Open, and the order:** a fixed slot bank first -- but for STABLE ADDRESSES, not for
alloc time. The audit's `alloc 2445.5 of 3408.1 ms` row was 2026-10-09; on the current
binary the 256 MiB arm is **alloc 3.4 ms of 2849.9** (819 real allocations against 18231
reuses). The arm's real cost is the pageable copy at promotion shape, **469 us/expert at
2.0 GB/s, 7.2 tok/s**, against **15.3 tok/s when all of WARM is pinned with identical
stdout** -- and making that publish non-blocking changed nothing, so the rate is the DMA.
Then slot-addressed dispatch, then the in-graph fill, and the LFU port last. Do not take their plain
LRU (rejected here on measurement), their per-layout-group budgets, or zero-slot substitution
(every routed expert must run). Full comparison, gates per step and the REAP-288 file's 78.04
GiB / 65.4 GiB-of-experts shape: `docs/AUDIT-gpu-resident-expert-cache.md`.

### PS5LM-Plus (abhi245y, 2026-10) -- a console appliance, not an engine to borrow from
**What it is:** a jailbroken-PS5 homebrew app that embeds llama.cpp and runs GGUF on
the console's RDNA2 iGPU through a PS5-specific RADV (Vulkan) build. The inference is
upstream's; the project is the title, the downloader, the on-TV dashboard and a
**context/KV planner** that walks context lengths (131072 down to 4096) against cache
types (f16, q8_0, q4_0) until a plan fits the console's ~11.4 GiB unified heap. Its
weights budget is the whole GGUF file size plus a 0.75 GiB reserve and recurrent state.
**What it does not have:** any host/VRAM tiering, expert residency policy, pinned
transfer buffers, or read/copy overlap -- MoE is "supported because llama.cpp supports
it", and expert streaming from SSD is listed under the roadmap's *Later*. The payload
mode's `-lm mlock` is page-locking for a CPU-only path, not a promotion pipeline.
**Verdict for this engine:** nothing to port. The one idea worth naming is the
planner's shape (longest context first, best cache that fits) -- which is what
`configure_expert_cache` and the KV tier already do against measured free VRAM rather
than a static heap. Its reported numbers (20.3 tok/s on a 27B UD-Q2_K_XL, 9.2 GiB) are
console numbers on unified memory and do not compare to a discrete-card arm.
Sources: <https://github.com/abhi245y/PS5LM-Plus>,
`docs/RESEARCH.md` and `docs/ROADMAP.md` there, `ps5/app/model_plan.cpp`.

### SSD-LLaMA (2609.18110) — aligned per-(layer,expert) records, manifest, checksums
**Measured here:** the tier is already HOT/WARM/COLD with byte budgets, and on
`Qwen3.8-Distill-35B-A3B` (256 experts × 40 layers = 10240 slots, 0.937 MiB each)
the whole expert corpus is **9600 MiB, which fits the VRAM budget**: the auto
policy residented 3018 slots on a 32-token run and 10240 under forced warm-up.
So on this model the pack's premise — "a miss must become a few predictable
direct reads" — is not the binding constraint; capacity is not the problem and
there is nothing to evict.
**Measured here:** what the pack would remove is *cold-start* traffic: the WARM
fill read 2618 MiB in 438.8 ms (6.0 GB/s) and a full-corpus warm-up read 9600 MiB
in **16026 ms (0.6 GB/s)** — 6% of the 15.9 GB/s the per-thread `FileReader` path
measured with separate handles. A pack's one large aligned read per layer is the
obvious candidate for that 16-second load phase.
**Open:** the pack, the manifest, the checksums, and any io_uring path. Nothing of
that exists; reads go through the GGUF mapping plus per-thread `FileReader`.

### Mira (2609.38090) — HOT + STAGE windows, routing-aware rebalancing
**Measured here:** HOT/WARM/COLD exist with per-expert traffic counters and a
printed split (`include/krk/expert_cache.hpp:223`). There is no STAGE window and
no rebalancing policy driven by that telemetry.
**What the trace says about sizing** (`KRK_TRACE_EXPERTS=1` →
`tools/expert_trace.py`, this model, 24 tokens, 8918 acquires): HOT hits 68.7%,
WARM hits 31.3%, **COLD 0%** once warm-up has staged the set; 2793 distinct
(layer, expert) pairs touched; reuse distance **p10 319, p50 641, p75 1919**
acquires. At ~320 acquires/token (8 experts × 40 layers) those percentiles are
**≈1, 2 and 6 tokens** — so reuse is short-horizon and a lookahead of a couple of
layers is the right length, which is Mira's two-layer finding reached from this
side.
**A per-layer hot set is a real design point, and it is bimodal** (measured: 48
tokens, 26838 acquires, **all 10240 (layer,expert) pairs touched**). A per-layer
LRU of K slots serves: **K=2 -> 0.0%**, K=4 -> 0.1%, **K=8 -> 14.1%**, K=16 ->
25.6%, K=32 -> 38.3%, K=64 -> 49.3%. But the hottest **240 experts (6 per layer,
2.3% of slots, ~225 MiB) carry 42% of all acquires**, and single experts are used
on every token (L39/e89 48/48). So the "under 8 or 2" hot core is real and cheap
to hold, but it is *narrow*: it is what a small VRAM budget should pin, and the
long tail is what a STAGE window or an on-demand path has to serve.

### SeqMoE (2609.12978) — deadline-aware prefetch, forecast-driven eviction
**Measured here:** what the repo has is the *synchronous* version of this: the
expert loop acquires, promotes and computes per expert, so a miss's latency is on
the critical path. Measured on this model at the default policy: 275 µs per
promotion, ~100 promotions per step = **29.4 ms of a 60.6 ms step**, plus 4.0 ms
of allocator wait — 55% of the step in the promotion path, before any kernel runs.
With the corpus pre-warmed that becomes **0 promotions and 35.0 tok/s against
16.5** on the identical command.
**Open:** there are no in-flight DMA generations to tag because the path never has
one outstanding; a prefetch that runs ahead of the layer is what would create the
I/O window this paper is built around.

### FATE (2502.12224) / pre-attention prediction (2511.10676) — adjacent-layer prefetch
**Measured here:** rung 0 exists (`ExpertCache::prefetch_layer`,
`--expert-warm-prefetch`, and a ranked warm-up that stages the routed set before
the first token), and the tooling to judge rung 1 exists (`tools/expert_trace.py`,
`scripts/ab_prefetch.sh`, `scripts/ab_slots.sh`, `scripts/trace_ab.py`).
**The measurement that matters for the ladder:** on this model the reuse distance
is 1-6 tokens and the whole corpus fits in VRAM, so a predictor's value is
**cold-start coverage only** — after one pass over the corpus there is nothing to
predict. On a model whose expert set exceeds VRAM the same trace would say
otherwise, and that is the case to re-run before building a predictor.

### AcceptMoE (2608.02989) and EdgeXpert (2608.05303) — residency-aware offload, prompt-wise reuse
**Measured here:** the "condition on cache residency" idea is what
`--expert-warmup` already does (ranked order, `--expert-warmup-ms` ceiling,
`<model>.krakenexperts.json` index). One caution measured today: the index beside
`Qwen3.8-Distill-35B-A3B-Coder-Abliterated-Q2KXL_ROCMFPX.gguf` records
`source_size 0`, so the engine **refuses it** (`an index whose provenance cannot
be checked is not one to warm the tiers from`) and falls back to an even-order
fill — the ranked policy silently degrades to an arbitrary one unless the index is
regenerated with `--expert-scan`.

### Tied Trit-Planes (2608.08910) — folded 1.6-bit format for disk-streamed MoE
**Measured here:** the engine already runs the two ternary layouts (`Q2_0` as the
fork's 128/34 and `Q2_0_64` as 48) and the local review of the ternary storage
question is `docs/compressed-expert-spec-review.md` §3.3 (BITCOS), which decides
the *lossless*-side question this format sits next to. Nothing of the folded
9-level plane is implemented.

---

## KV cache

### AnchorKV (2608.02901), HiSparse (2608.07009), OasisKV (2608.08097), KVFetch (2610.08811)
**Measured here:** KV is flat and per-layer packed (`sum(len) * 2`; **24 MiB at
ctx 4096** on the Nemotron 30B-A3B where 208 MiB was charged before), it tiers
HOT/WARM/COLD with a cold-dir spill, and every tier knob is proven byte-identical
to the flat cache on the small models (7 decode arms, `scripts/kv_tier_geometry_check.sh`).
**Measured here, the hard way:** the failure mode to design against is silent —
two KV planes sharing one cold-file namespace spilled 1916 pages, restored 1860,
reported 0 dropped, and produced fluent text that differed from the flat cache.
That is why every compression scheme above needs the verbatim-copy gate the plan
asks for, and why `docs/AUDIT-plan-vs-code.md` calls paged KV a rewrite of the
per-layer geometry rather than an addition beside it.
**Open:** page tables, refcounts, prefix hashes, any KV quantisation.

---

## Execution core, kernels and hardware capability

### RESOLVE (2610.05683) — race/edge-case kernel gates
**Measured here:** this repo's version of it is `--selfcheck` (498 suite runs: six
arms per group including crash-at-last-check and a fuzz arm, each arm a FULL run
judged against every group's own counts, the arms of one group in flight at once), the isolation check, constructed-failure tests, and md5
determinism A/Bs. It has already paid: `attention_decode_split_kernel`'s
warp-leader race made one head's output depend on scheduling — **three distinct
outputs in six runs of one command**, now stable 10/10.

### Meganeura (2608.01563) — Vulkan/Metal portable execution
**Measured here:** no Vulkan, CUDA, Metal or WebGPU backend exists (zero hits in
`src`/`include`/`CMakeLists.txt`). What exists is the plan's *principle* applied
inside one backend: separate tuned paths per arch (`gfx1031` vs `gfx1201`,
`src/hip/kernels/gemm_dp4a.hpp` among others), which is also why RDNA2 can be
compiled and gated but never executed from this machine.

### The RDNA2/RDNA4 ISA — "so many unused extensions"
**Measured here:** the instruction classes in use are `wmma` (`src/hip/kernels/gemm.hpp`,
prefill/multi-row), `sdot4`/dp4a (`gemm_dp4a.hpp`), `sdot2` (`dequant.hpp`,
`gemm.hpp`, `attention.hpp` — the per-32-channel dot the GEMV path dispatches to),
`v_perm` (`dequant.hpp`) and wave shuffles in every reduce. There is no MFMA (CDNA
only) and no explicit `v_bfe` (the compiler emits it for the mask/shift dequant).
**Where an extension cannot help, measured:** decode visits experts at **rows=1**,
and a one-row matvec cannot use a matrix/dot instruction's cross-row capability;
that is why WMMA appears only on the prefill side. On the 256-expert model a decode
step is **2965 launches** — 320 expert visits × (upload_i32, upload, gather_rows,
pair-group, down, scatter_axpy) = 1920 of them, plus 531 `gemm(gemv)` and 40 router
readbacks. Removing 640 of those launches (the up/gate pair fold, `KRK_MOE_GROUP`)
is what one session bought; the *instruction-level* win needs the rows batched, not
a different opcode for one row.

---

## Speculative decoding and MTP drafters

### Approximate Speculative Decoding (2608.03447), Bole's tree speculation (2608.01651)
**Measured here:** the repo has a working `--draft` speculative path, and its one
recorded loss was a *call-site* defect, not the idea: the draft loop and the
target's verify block both pulled the **whole 993 kB logits row per proposal** to
argmax on the host, where plain greedy had already moved to the device top-k
(`fetch_logits(true)` / `topk_argmax`, 16 bytes back). Measured at the time: 32.4
tok/s with a generic draft against 88.4 tok/s target-only (Qwen3-8B target +
Qwen3-MoE-4x0.6B draft, 44.8% accept), i.e. a 2.7x net *loss* that was entirely
per-proposal host traffic. That is this paper family's own thesis -- a verifier is
only as good as its cheapest path -- and it is why any tree/token-count extension
must be priced per proposal, not per token.

**The artifact this repo does not have:** on the locally-installed sibling engine
the drafter is not trained, it is *shipped*. `C:\Strata-HIP-data\mtp\` holds
`mtp-inventory.md` ("MTP block in the BF16 checkpoint -- 31 tensors, 5.214 GB, in 28
shards"), `mtp-q2_0.gguf` (889,014,272 bytes, with a per-tensor format report) and
the `--mtp .../rt` directory its configs pass beside `--spec 4 --spec-min-p 0.5`.
kraken recognizes `mtp.*` shards only to refuse them (`src/arch.cpp:113`). See
`docs/STRATA-VS-KRAKEN.md` for the measured head-to-head this came out of.

---

## Quantisation

### BITCOS (2609.16338), Disaggregated Quantization (2609.26333)
**Measured here:** BITCOS is analysed in `docs/compressed-expert-spec-review.md`
§3.3 (lossless with respect to an already-ternary tensor; degenerate for anything
else), with open items named and a correctness bug blocking one of them. No custom
W4/W6/W8 format exists; the engine runs GGUF formats, including NVFP4 with
per-matrix `.scale` sidecars.

### Expert-reduction order and determinism (2607.28097)
**Measured here:** the repo's md5 discipline is exactly the guard this paper asks
for. Two examples: the MoE per-expert pair fold left stdout byte-identical
(`a56e02b731fdf93004fd10b3a591df80`, the recorded acceptance hash), and a wrong
`wt` in the group's fallback showed up as `-nan(ind)` in a `KRK_DUMP` field rather
than as a timing or a count. Any sharding of experts across devices inherits this
requirement.

---

## CPU execution and hybrid offload

### ATSInfer (2607.10183), Hybrid CPU+GPU offload generally
**Measured here:** the engine's hybrid CPU expert path is opt-in and **loses
3.1×** when it engages: `--expert-cache-mb 16` put all 922 experts on 8 host
threads at 5.392 ms/expert and decode fell from 17.2 to **5.5 tok/s**, while the
device path cost ~2.0 ms/expert of read+promote. The cause is a per-layer merge
barrier the device waits on, which is the same reason this paper's async
coordination has to come *before* the split, not with it.

---

## What this doc is not

No paper's internal numbers were reproduced here, no GPU other than gfx1201 was
run, and nothing in the notes above is a claim about a paper's implementation.
Where a number appears, it came from a command in this repo; where a capability is
called missing, the search that found nothing is stated in
`docs/AUDIT-plan-vs-code.md`.
