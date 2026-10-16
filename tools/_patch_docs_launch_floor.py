#!/usr/bin/env python3
import io, sys

def load(p):
    raw = io.open(p, 'r', encoding='utf-8', newline='').read()
    return raw.replace('\r\n', '\n'), raw.count('\r\n') > 0

def save(p, t, crlf):
    io.open(p, 'w', encoding='utf-8', newline='').write(
        t.replace('\n', '\r\n') if crlf else t)

def sub(t, old, new, count=1, tag=''):
    if t.count(old) != count:
        print('ANCHOR FAIL %s: %d != %d' % (tag, t.count(old), count))
        print(old[:400]); sys.exit(1)
    return t.replace(old, new)

A, crlf = load('AGENTS.md')

# the table no longer calls the 0.32 us a queue effect
A = sub(A,
"""  | primitive | quiet (bench) | busy host enqueue | in-step (switch A/B) | factor |
  |---|---|---|---|---|
  | record on the default stream | 28.2-30.6 us | 0.32 us | **2.56 us** | quiet overstates **11-12x** |""",
"""  | primitive | quiet (bench) | host enqueue, busy queue | in-step (switch A/B) | factor |
  |---|---|---|---|---|
  | record on the default stream | 28.2-30.6 us -- **the DEVICE half** | 0.33 us, the same empty or busy | **2.56 us** | quiet overstates **11-12x** |""",
tag='table')

# the paragraph that explains the arm, rewritten to what the arm now proves
A = sub(A,
"""  The arm that makes this checkable is `kraken-bench`'s **busy-queue line**: the
  queue is fed 512 x `copy_act n=32` of real device work and 64 calls are timed
  inside it (median of 8 blocks), so the quiet and busy figures print side by
  side. It cannot use `time_calls` -- that helper's cold-call subtraction is
  bigger than a host enqueue, and the first version of the arm printed a
  **negative** marginal record cost (-12.82 us). `--depth N` sets the pre-issued
  count, `--depth 0` skips the arm, and the gate's floor line now says
  `unresolvable at iters=2` instead of printing `-24.00 us per call` (the gate
  sets iters=2, where the subtraction is the difference of two single
  measurements).""",
"""  **The arm measures a call's two halves apart, and that is what the table above
  was actually comparing.** The queue is fed 512 x `copy_act n=32` of real device
  work and 64 calls are timed inside it (median of 8 blocks); `--depth N` sets the
  pre-issued count and `--depth 0` skips the printed arm. It cannot use
  `time_calls`: that helper subtracts one-call-plus-one-sync from
  many-calls-plus-one-sync, so its quotient is whichever half dominates the queue
  -- and for a record that is the DEVICE half. Measured 2026-10-11 (plain run,
  medians of 8, and the gate's own arms): a record enqueues in **0.33-0.34 us in
  BOTH queue states** and leaves **27-32 us of device time** behind it; a launch
  enqueues in 0.75-0.97 us and leaves 2.8 us. What the queue state changes is the
  **tail** (177 -> 583 us on the launch arm), never the enqueue. So the 28.2-30.6
  us this table used to call "quiet" was never a queue-state effect: it is the
  device half, which is the right figure only for a call that cannot overlap. The
  **in-step** column stands on its own switch A/B and is the number to rank with
  (2.56 us for the record, because in a step the markers overlap hundreds of
  queued launches); the "0.32 us busy enqueue" this table once called the
  alternative was the same 0.33 us as the empty arm, mislabelled as a state.
  **The arm is in the gate now and proves itself there** -- it asserts the queue
  really is fuller (launch-arm tail, >=3x), that a record's device half is >=4x
  its host half, that the enqueue does NOT move between states, and that the
  instrument resolves the device cost per op. `KRK_TEST_QUEUE_NOPREISSUE=1` skips
  the pre-issue and fails exactly the two fullness checks (rc=1), which is the
  red construction. The gate's floor line says `unresolvable at iters=2` instead
  of printing `-24.00 us per call` (the gate sets iters=2, where the subtraction
  is the difference of two single measurements).""",
tag='paragraph')

# new bullet: the two folds this session measured
A = sub(A,
"""  **Rules:** rank a fold by running the fold (one session, interleaved,
  order-balanced) and treat any constant-times-count estimate as good to ~2x;
  never compare a host-to-device row against `--floor`; and remember that a
  *drain* is 1000x a wait.
""",
"""  **Rules:** rank a fold by running the fold (one session, interleaved,
  order-balanced) and treat any constant-times-count estimate as good to ~2x;
  never compare a host-to-device row against `--floor`; and remember that a
  *drain* is 1000x a wait.
- **A small `upload` is a blocking copy, not a launch -- the per-expert plan pair
  cost ~15 us, not ~2 launches, and folding it was worth +10%.** `Engine::moe_ffn`
  used to upload each visit's row ids and gate weights with an `upload_i32` +
  `upload` pair; both are BLOCKING `hipMemcpy` calls (`src/hip/backend_hip.hip`),
  so 56 of them per step are worth ~10% of an 8.5 ms step, not the ~1.6% a
  2.4-2.8 us launch-fold estimate predicts. The plan for the whole layer is now
  bucketed once (one O(n*k) pass with a prefix sum, replacing an O(n_expert*n*k)
  rescan per layer) and uploaded with **two** calls per layer; a visit reads
  `[off(e), off(e+1))` of it and issues no copy at all. `ws_plan_`/`ws_alpha_` are
  `C * k` entries now (`C` is the chunk; the layer's pair count is bounded by
  n*k). `KRK_MOE_PLAN=0` restores the per-visit pair. Measured on
  Qwen3-MoE-4x0.6B (28 layers, top-2): ops/step **705 -> 649** (56 = 2 x
  (visits-1) per layer, the "on" arm closing the count exactly), A/B **+10.027%
  order-balanced, 5/5 pairs, MAD 2.870%**, stdout md5 `995c6b60e7` identical --
  and on Qwen3.8-Distill-35B-A3B (256 experts, 40 layers) `upload_i32` goes from
  one per expert visit to **one per layer** (320 over 8 tokens = 40/step, down
  from ~320/step). Ranking a blocking micro-copy as a launch is the same error
  class as the record row above.
- **The per-token drain in front of the topk readback was redundant, and removing
  it is worth ~4-5%.** `Engine::topk_argmax` issued an explicit `hipDeviceSynchronize`
  between its `logits_topk` launch and the 16-byte `download` -- but `download` is
  a blocking `hipMemcpy` on the same stream, so it already orders itself behind
  that launch. That is the same argument `Engine::moe_ffn` records for the router
  readback, and it is now the same switch pattern: `KRK_TOPK_SYNC=1` restores the
  drain (default off). Measured: SmolLM2-135M **+5.170%** and Qwen3.5-0.8B
  **+4.412%** decode (5/5 interleaved order-balanced pairs each, MAD 2.076% and
  0.631%, text identical). The remaining drains in the decode path now each have
  a switch too: the router readback (`KRK_ROUTER_SYNC=1`, ~129 us/layer),
  the expert request after a promotion (`KRK_SYNC_EXPERT=1`), the logits row
  readback on the fallback path (`KRK_LOGITS_SYNC=0`) and the speculative
  single-row recompute (`KRK_SPEC_ROW_SYNC=0`). The last two are not measurable
  on a GPU run today -- the fallback is unreachable while `logits_topk` works,
  and the speculative one needs a drafter (F8: self-draft acceptance 27.8%) --
  which is why they are switches and not rows.
""",
tag='new bullets')

# the next-folds bullet: one of its two items is now done
A = sub(A,
"""  is where the next folds are: the scatter into the down-projection epilogue, and
  the per-expert `upload_i32`+`upload` pair becoming one per-layer plan upload.""",
"""  is where the next folds are: the scatter into the down-projection epilogue, and
  the per-expert `upload_i32`+`upload` pair -- **which is now folded into one plan
  per layer (2026-10-11): `upload_i32` is 40/step here, one per layer, and the
  per-visit pair it replaced was two BLOCKING copies, not two launches, so the
  fold was worth +10.0% on the 28-layer MoE rather than the ~1.6% a launch
  estimate gives.** What is left of this bullet is the launch-count question
  itself: the **launch floor on this 2965-op step is still measured only by
  construction.** The A/B (`KRK_MOE_GROUP=0` against the default, which adds 2
  launches per expert visit) needs ~10 runs of a 13 s load + 16 s warm-up, and the
  unprofiled baseline run of it was stopped because the model thrashes the GPU:
  at 11 GiB of expert corpus against a 9.4 GiB budget the step is
  allocation-bound (`alloc` dominates, HOT 96.4% still promoting 136 slots in 8
  tokens), which is a different measurement than the one the fold A/Bs make on
  the dense models. Run it when the card is otherwise idle.""",
tag='next folds')

save('AGENTS.md', A, crlf)
print('patched AGENTS.md (busy-queue table + paragraph, two new bullets, next-folds)')
