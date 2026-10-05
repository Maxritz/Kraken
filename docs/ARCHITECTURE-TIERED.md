Kraken target architecture — a tiered, device-resident inference engine
================================================================================
Status: design. Nothing here is implemented. Every "current" number was measured
on the RX 6700 XT (gfx1031) against llama.cpp built natively for HIP on the same
card and the same files. Evidence: docs/traces/*.txt; gap register in
docs/TODO.md from the G12 session entry onward.


1. What the measurements say the problem is
--------------------------------------------------------------------------------
Three findings, and they point at one shape of answer.

(a) Prefill is one kernel. 74.2% of the 1,784.7 ms of a 361-token Qwen3-8B
    prefill is gemm(simt); attention is 24.6%; everything else is ~1%. The GPU
    is never idle, so this is not launch overhead and not bubbles.

(b) Kernel efficiency, not instruction choice, is the gap. Measured against the
    gfx1031 roofline (12.3 TFLOP/s fp32, 24.6 packed fp16):

        gemm(simt)   504 calls   3.78 TFLOP/s   15% of packed ceiling
        gemm(dp4a)   432 calls  10.33 TFLOP/s   42% of packed ceiling

    Same shapes, same card, same binary, one environment apart. 2.7x from the
    rung choice alone. llama.cpp implies ~13.6 TFLOP/s on the same work, i.e.
    55% of the ceiling.

(c) The MoE step is not compute and not bandwidth. It is 48 blocking reads.
    warm_read is 95.2% of a 3,817-op decode window at 26.6 ms per call, with
    host_ms within 1% of dev_ms. The host is blocked and the GPU has nothing
    queued.

So: arithmetic efficiency (b), and host round-trips that idle the GPU (c).


2. The three principles this design follows
--------------------------------------------------------------------------------
P1  Keep the device busy. Anything that blocks the host while the GPU has work
    is a defect, not an implementation detail. Today's 48 x 26.6 ms warm_read
    is 95% of MoE decode and it is all host stall.

P2  Match the rung to the shape, and let measurement pick. The int8 rung is now
    default where dp4a_ok's K >= 2048 boundary says it wins (commit fc51584),
    verified on both sides of the gate. What remains excluded is a policy
    decision (Q6_K), not an architectural one.

P3  Reuse is a cache problem, and caches are sized by policy. Hot experts and
    hot KV must be resident; everything else must be one async fetch away, never
    a synchronous wait.


3. The tiers
--------------------------------------------------------------------------------
Three tiers, one policy engine, applied to both KV and expert weights.

    HOT    VRAM            fastest. Resident. What the next ~50 tokens will hit.
    WARM   host RAM        paged, pageable. One async DMA behind the compute.
    COLD   NVMe            read-through. Prefetched, never awaited inline.

The sizing rule that matters, and that the current tree gets right for experts
but not at all for KV: subtract, do not clamp. budget = total − reserve −
everything_else_resident. On the 6700 XT (12.0 GiB) with a 4.68 GiB dense
model, the expert budget lands at ~3.7 GiB — which is exactly what
configure_expert_cache already computes.

    HOT/WARM  experts    exists   src/expert_cache.cpp, 745 lines, LFU+aging
    HOT/WARM  KV         ABSENT   engine.cpp:276, one flat alloc
    COLD      experts    exists   read-through, but synchronous (see P1)
    COLD      KV         ABSENT

KV is the gap that is cheapest to describe and most expensive to ignore. Today:

    engine.cpp:276   kcache_ = alloc(n_layer * kv_cap_ * kv_dim_ * as);
    engine.cpp:278   vcache_ = alloc(n_layer * kv_cap_ * kv_dim_ * as);

One flat VRAM allocation, no tiers, no eviction, no reuse. Qwen3-8B at ctx 4096
is 576 MiB, which fits. At ctx 32768 it is 4.6 GiB, which does not, and the
only current answer is to refuse the context.


4. Split-K for the Q6_K GEMM — the largest single prefill item
--------------------------------------------------------------------------------
Measured: 72 GEMMs costing 654 ms at 1.08 TFLOP/s, the worst rate in the
profile, about half the arm's total time for one seventh of its GEMMs. These
are the Q6_K tensors in a mixed Q4_K_M model (attn_v and ffn_down per layer),
excluded by gemm_dp4a.hpp:563 `kDp4aEnableQ6K = false`.

The recorded reason is register pressure: 245 ms over 32 dispatches, 327 ms of
GEMM against the fp16 tile's 256 ms. Re-tuning the register budget fights the
symptom. Splitting K removes the pressure by construction:

    problem   one block owns the whole K axis, so its accumulators scale with K
              and the Q6_K staging (6-bit scales + int6 values) will not fit
    fix       partition K across S compute units. Each unit accumulates over its
              own slice into a private fp32 partial, and the S partials are
              reduced once, afterwards.

    grid      (n_out/BN) x (rows/BM) x S        S = split factor
    partials  S x rows x n_out fp32
    reduce    one kernel, S-way, tree or serial; run once per GEMM

Why this and not register re-tuning:
  - The register budget stops depending on K. A unit's accumulators are fixed
    by BM x TN, which is what the tile already bounds.
  - Q6_K and Q4_K then share one kernel shape. The split is orthogonal to the
    weight format, so the format-specific special case disappears rather than
    being maintained.
  - It is the standard answer for exactly this failure: llama.cpp's own path
    handles long-K low-precision weights by splitting rather than by staging
    more.

Cost, stated honestly: S partial buffers and one extra kernel launch per GEMM,
plus the reduction traffic (S x rows x n_out floats). At rows=256, n_out=4096,
S=4 that is 16 MiB of write plus the same read back — small next to 654 ms, but
it must be measured, not assumed.

Staging stays separate from splitting, and that separation is what makes this
incremental:
    step 1  split-K on the existing Q8_0/Q4_K dp4a kernel, verify no regression
    step 2  extend the weight staging to Q6_K now that registers are bounded
    step 3  re-enable kDp4aEnableQ6K and measure the 654 ms directly

Step 1 is independently valuable: it validates the split before the format work
depends on it.


5. Device-resident MoE step
--------------------------------------------------------------------------------
Today the routing plan is built on the host (engine.cpp:1252 moe_ffn), with a
blocking download_f32 per layer per token and a per-expert upload of plan_dev_
and alpha_dev_. That is why warm_read dominates: the host cannot proceed to the
next layer until the fetch lands.

Target shape:

    device-resident routing plan
      router GEMM              -> [rows, n_expert] in VRAM
      top-k + normalise         -> one kernel, selection AND weights in VRAM
      gather experts            -> grouped, by slot, never by host choice
      one grouped GEMM          -> all experts, one launch
      scatter + scale           -> weighted accumulate, one launch

Three properties this buys, each of which is currently false:
  - No host round-trip between layers, so P1 holds.
  - Expert weights are addressed by (layer, expert) rather than by host pointer,
    which is what finally makes HIP graph capture possible. Today eviction moves
    a host-chosen pointer, so a captured graph would address freed memory.
  - Prediction (G3, absent entirely) becomes expressible: a plan computed on
    device can be reused or rolled back without a round trip.

The expert cache stays the owner of residency; what changes is that the engine
hands it a plan instead of a pointer.


6. Tiered KV — the missing piece
--------------------------------------------------------------------------------
Same three tiers, same policy engine, different unit of management.

    unit        (layer, position-block)      e.g. 256 positions
    HOT         VRAM, pinned, recent         swept by a radix/prefix structure
    WARM        host RAM, pageable            evicted HOT lands here, never the file
    COLD        NVMe                          prefix-matched on session restart

The radix/prefix reuse is the part that has no analogue in the expert cache:
attention over a shared prefix is what makes long-context serving cheap, and it
needs the prefix tree to survive eviction. That is why WARM eviction must not
write to the file — a prefix already on disk is cheaper to re-read than to
rewrite, and rewriting 4.6 GiB per session is the cost the tiering exists to
avoid.

Prefetch, not await: a token whose KV is in WARM must overlap its DMA with the
current layer's compute, which is the same requirement as P1 and the same code
path as the expert WARM tier. Reuse it rather than build a second one.


7. Flowchart
--------------------------------------------------------------------------------
                              token ids
                                  |
                    +-------------+-------------+
                    |                           |
              PREFILL (rows>1)            DECODE (rows==1)
                    |                           |
        +-----------+-----------+       +-------+-------+
        |           |           |       |               |
  router GEMM  attn (qtile)  MoE    gemv per layer   MoE
        |           |           |       |               |
        |           |           |       |        plan in VRAM
        |           |           |       |        (no host hop)
        |           |           |       |               |
        |           |           |       |        grouped GEMM
        |           |           |       |               |
        +-----------+-----------+       +-------+-------+
                    |                           |
             rung ladder:                  KV touch:
             wmma > dp4a > simt             HOT  hit
             (split-K for Q6_K)             WARM prefetch+overlap
                    |                       COLD read-through
                    +-------------+-------------+
                                  |
                        logits -> sampler -> token
                                  |
                       KV write: tier by policy
                                  |
                     eviction: HOT->WARM->(reuse)

Data-path invariants:
  - no host round-trip inside the layer loop
  - every tier access is a prefetch issued >= 1 layer ahead
  - the GEMM rung is chosen per shape, by measurement, never by build config


8. Gap analysis: target vs current, and what can be fixed now
--------------------------------------------------------------------------------
Ranked by measured cost, not by how interesting the work is.

  #  item                              measured          now      target
  1  Q6_K GEMMs                        654 ms, 1.08 TF/s 1.08     ~10 TF/s
  2  SIMT tile                         3.78 TF/s (15%)   15%      45-55%
  3  MoE host read stall               95.2% of decode   blocked  overlapped
  4  KV tiering                        absent            none     3 tiers
  5  attention beyond qtile            274 ms            qtile    flash
  6  expert prediction (G3)            absent            none     on-device
  7  graph capture                     blocked            blocked  possible after 5
  8  Q6_K int8 quant for experts (T15)  absent            none     staged

Can be fixed now, in order, each with a measurement attached:

  1  split-K on dp4a, then Q6_K staging            -> attacks 654 ms, both cards
  2  SIMT tile re-tiling for occupancy              -> attacks 1,319 ms, both cards
  3  overlap the 48 blocking warm_reads             -> attacks 95% of MoE decode
  4  KV HOT/WARM split before COLD                 -> unlocks long context at all

  5-8 depend on 3: they are all "keep the device busy" work, and doing them
  before the host round-trip is gone means optimising a path that is about to
  be replaced.

Note on scope: 1 and 2 help every card, because dp4a has no arch gate (the
kernel emits v_dot4c_i32_i8, present on gfx103 and gfx1201) and the SIMT tile is
the fallback everywhere. 3 and 4 are where the two cards diverge, because the
RDNA2 box reads experts off an ntfs3 mount at 0.1-0.3 GB/s — that baseline
conflates the filesystem with the cache, and should be re-taken on a fast
filesystem before drawing policy conclusions from it (T27).


9. What is deliberately not in this design
--------------------------------------------------------------------------------
  - A fourth tier. Three is what the hardware justifies; a fourth would need a
    measurement showing the third is live.
  - Speculative decoding improvements. The entry points exist; the drafter's
    cost is not on the measured critical path.
  - Replacing the expert cache. It is 745 lines, does LFU+aging correctly, and
    the measured problem is the host read around it, not its policy.
  - Optimising the MoE host step before the plan is device-resident. It is the
    code that the design deletes.