"""Record the GDN four-projection group: report section, session log, AGENTS note.

Every number below is copied from the runs in this session's logs (/tmp), not
retyped: the op tables, the md5s and the harness output.
"""
import io

REPORT_SECTION = """

---

## 9. Qwen3.5-0.8B: what is left, and the next fold (this session's third pass)

The two folds of section 8 are active on this model too (its `residual` row is
already gone), so the measurement starts from **451 ops per decode step** and
24 layers: **18 recurrent (gated-delta-net) layers at 20 ops each, 6 attention
layers, plus the head**.

Per-step components, `KRK_PROFILE=1`, ctx 512, n 32, greedy:

| component | ops | note |
|---|---|---|
| projections | 159 | `gemm(gemv)` 91, `gemm_acc(gemv)` 42, `gemm_group` 26 |
| recurrent | 90 | `l2norm` 36, `conv1d_silu`/`delta_rule`/`softplus_act` 18 each |
| attention-mix | 72 | `sigmoid_act` 24, `scale_act`/`scale_cols` 18 each |
| norms | 61 | `rmsnorm` (3 per recurrent layer) |
| attention | 24 | `rope`/`qk_norm`/`kv_append`/`attention` 6 each |
| ffn-activate, bias, other, transfer, embed, head+sample | 45 | |

One recurrent layer, in order (indices 1..20 of the step):

```
rmsnorm, gemm, gemm, conv1d_silu, l2norm, l2norm, scale_act, gemm,
add_bias_cols, softplus_act, scale_cols, gemm, sigmoid_act, delta_rule,
rmsnorm, silu_mul, gemm_acc, rmsnorm, gemm_group, gemm_acc
```

The first four `gemm` calls are `[q | k | v]`, the z gate, `ssm_alpha` and
`ssm_beta` -- **all four read the attention-norm output and none depends on
another**, which is the exact condition `gemm_group` exists for. They were
issued from two different places in `gdn_forward` (steps 1 and 4) only because
of the order the code narrates them in. Grouping them is **4 launches -> 1 per
recurrent layer, 54 of the step's 451 ops**.

### The wall: a group carried one weight type

The first implementation was inert (451 ops in both arms, md5 unchanged). The
guard that fired was the dtype check: `blk.*.attn_qkv.weight` is **Q5_K** while
`ssm_alpha`/`ssm_beta` are **Q4_K** on this file, and `gemm_group` passed a
single `DType wt` to the fused kernel. Real checkpoints mix formats across the
projections of a group routinely, so the fix is the useful one: `FusedLayer`
now carries `wt[8]`, one per matrix, with a nullable `wts[]` on the launcher --
the weight addressing (`w_row_bytes[m]`) was always per matrix, so the dequant
dispatch tag was the only shared thing. Null means "every matrix uses `wt`", so
no existing caller changes.

### The bug the md5 falsifier caught

With the group engaged the op count fell exactly as designed (451 -> 397, gemv
calls 2822 -> 590) -- **and the stdout md5 changed**. Not a timing artifact: a
different greedy text. `KRK_DUMP` localized it in one run:

| stage | group on | group off |
|---|---|---|
| `gdn.conv L00` | `0.01457977 0.01085663 0.004024506 -0.2780` | identical |
| `gdn.delta L00` | **`-nan(ind) -nan(ind) -nan(ind)`** | `-1.66893e-06 -6.556511e-07 8.285046e-06` |

The cause was in the group's **fallback**, not its fused path: both the HIP
override and the base-class default issued `gemm(out[m], x, w[m], wt, ...)` --
the group's single `wt` -- so during **prefill** (rows > 1, where the group
falls back to per-matrix gemms) the two Q4_K matrices were decoded as Q5_K.
Every block read from the wrong place, NaN out, fluent-but-wrong text after.
Fixed by using `wts ? wts[m] : wt` in both fallbacks.

The general lesson, worth more than the win: **the fused path is decode-only, so
on this model the group's fallback is what prefill runs** -- and a fallback that
ignores per-matrix metadata is a silent wrong-answer bug, not a performance bug.
The md5 check across arms is what caught it; the op count and the timing both
looked right.

### Verified

| check | result |
|---|---|
| stdout md5, group on vs off vs pre-change baseline | `08afb598d17b0c7ac6466d824163e1a6` in all three (**bit-identical**) |
| `KRK_DUMP` `gdn.delta`, group on | `-1.66893e-06 -6.556511e-07 8.285046e-06` -- matches group off |
| op count per step | **451 -> 397** (-12%); `gemm(gemv)` 2822 -> 590 calls |
| interleaved A/B, 5 pairs, ctx 512, n 64, greedy | paired deltas `3.154 4.899 6.084 6.131 6.284`, **median +6.084%, MAD 0.200%, 5/5 pairs**, all ten runs md5 `08afb598d1` -- verdict **WIN** |
| SmolLM2-135M (no GDN layer: the group must be inert) | `d6e2b2b3b00b397dd3fa89722790218e`, both switch values |
| Qwen3-MoE-4x0.6B (documented protocol) | `a56e02b731fdf93004fd10b3a591df80`, both switch values -- reproduces the hash in AGENTS.md |

Gate on the final tree: `ninja` six targets rc=0 with 0 `error:` lines and no
pending work; `build_check.sh` 0 failed (fingerprints re-recorded, then verified
`all 5 artifact(s) match`); `kraken-tests` **2751/2751** rc=0; the baseline gate
rc=0 with `18/18 selftest` and `proof 17/17`; `kraken-bench --gate` rc=0;
coherence on the three small models rc=0, **3 coherent, 0 not**.

`KRK_GDN_GROUP=0` restores the four separate launches.
"""

AGENTS_BULLET = """- **A fused group's fallback runs on PREFILL, and it must use each matrix's own
  metadata.** `gemm_group`'s fused path is decode-only (rows == 1), so a prefill
  pass takes the per-matrix fallback -- and a fallback that decodes a Q4_K
  matrix with the group's single `wt` (Q5_K here) reads every block from the
  wrong place. Measured on Qwen3.5-0.8B: `KRK_DUMP` `gdn.delta` came out
  `-nan(ind)` against `-1.66893e-06` from the separate-launch arm, i.e. a
  wrong-answer bug that leaves the op count and the timing looking healthy. The
  md5 comparison across arms found it in one run; `FusedLayer` now carries
  `wt[8]` (per-matrix types, nullable `wts[]` on the launcher and `gemm_group`)
  and both fallbacks use `wts[m]`.
- **Four projections that share an activation are one launch where the backend
  can express it.** `Engine::gdn_forward`'s `[q | k | v]`, z gate, `ssm_alpha`
  and `ssm_beta` all read the attention-norm output and have no dependency
  between them, but were issued 4 launches per recurrent layer -- `KRK_GDN_GROUP=0`
  restores that. Measured on Qwen3.5-0.8B (18 recurrent layers of 24), ctx 512,
  n 64, greedy, 5 interleaved pairs: **451 -> 397 ops/step** (-12%, `gemm(gemv)`
  2822 -> 590 calls) and **+6.084% decode** (MAD 0.200%, 5/5 pairs), with stdout
  md5 `08afb598d17b0c7ac6466d824163e1a6` unchanged on all ten runs and identical
  to the pre-change baseline -- the fold is bit-identical because the group runs
  `row_parallel_only` (splits pinned at 1), which is also what keeps its tiny
  `dt_rank` matrices from paying a reduce launch each.
"""


def append(path, text):
    s = io.open(path, encoding="utf-8", newline="").read()
    if not s.endswith("\n"):
        s += "\n"
    io.open(path, "w", encoding="utf-8", newline="").write(s + text.lstrip("\n"))


append("sherlock_report.md", REPORT_SECTION)

s = io.open("AGENTS.md", encoding="utf-8", newline="").read()
anchor = "- **`env A=0,B=0 cmd` is ONE variable, not two.**"
assert s.count(anchor) == 1
s = s.replace(anchor, AGENTS_BULLET + anchor)
io.open("AGENTS.md", "w", encoding="utf-8", newline="").write(s)
print("report section 9 appended; AGENTS.md bullets inserted")

try:
    s = io.open("sherlock-kraken-session2.log", encoding="utf-8", newline="").read()
except FileNotFoundError:
    print("no session log")
else:
    append("sherlock-kraken-session2.log", """
================================================================================
G. Qwen3.5-0.8B: four shared-activation projections -> one launch (+6.08%)
================================================================================

G1. What a decode step costs on this model (KRK_PROFILE=1, ctx 512, n 32):

      451 ops/step  (18 recurrent layers x 20 + 6 attention layers + head)
      projections 159, recurrent 90, attention-mix 72, norms 61, attention 24,
      ffn-activate 18, bias 18, other 6, transfer 1, embed 1, head+sample 1

    One recurrent layer, in order: rmsnorm, gemm, gemm, conv1d_silu, l2norm,
    l2norm, scale_act, gemm, add_bias_cols, softplus_act, scale_cols, gemm,
    sigmoid_act, delta_rule, rmsnorm, silu_mul, gemm_acc, rmsnorm, gemm_group,
    gemm_acc. The four `gemm` calls are [q|k|v], z, ssm_alpha, ssm_beta -- all
    four read the attention-norm output, none depends on another.

G2. First attempt was INERT: 451 ops in both arms, md5 unchanged. The guard that
    fired was the dtype check -- blk.*.attn_qkv.weight is Q5_K while
    ssm_alpha/ssm_beta are Q4_K, and gemm_group carried ONE wt for the group.
    Fix: FusedLayer gains wt[8] (per-matrix dequant tag) and the launcher and
    gemm_group take a nullable wts[]; null keeps every existing caller's meaning.

G3. With the group engaged the op count fell as designed and the MD5 CHANGED:

      group on : 397 ops/step, gemv calls 2822 -> 590, md5 30bffca233e0a3b9...
      group off: 451 ops/step,                            md5 08afb598d17b0c7a...

    KRK_DUMP localized it in one run:
      gdn.conv  L00  0.01457977 0.01085663 0.004024506 -0.2780      (both arms)
      gdn.delta L00  -nan(ind) -nan(ind) -nan(ind)   [group on]
      gdn.delta L00  -1.66893e-06 -6.556511e-07 8.285046e-06 [group off]

    Cause: the group's FALLBACK (used by prefill, rows > 1) issued
    gemm(out[m], x, w[m], wt, ...) with the group's single wt, so two Q4_K
    matrices were decoded as Q5_K during the prompt. Fixed in both fallbacks
    (HIP override and the base-class default) with `wts ? wts[m] : wt`.

G4. After the fix, bit-identical and verified:

      md5 group on == group off == pre-change baseline
          08afb598d17b0c7ac6466d824163e1a6
      KRK_DUMP gdn.delta group on == group off
          -1.66893e-06 -6.556511e-07 8.285046e-06
      397 ops/step with the group on (54 = 3 per recurrent layer x 18)

    $ KRK_AB_A_SWITCH=KRK_GDN_GROUP=0 KRK_AB_SWITCH=KRK_GDN_GROUP=1 \\
      KRK_AB_REPS=5 bash scripts/kraken_sherlock_kernel_ab.sh \\
      models/Qwen3.5-0.8B.Q4_K_M.gguf
      paired deltas (B vs A, %): 3.154 4.899 6.084 6.131 6.284
      median 6.084%   MAD 0.200%   B faster in 5/5 pairs
      verdict: WIN -- Text identical. (all ten runs md5 08afb598d1)

    Non-GDN models unaffected (the group is in gdn_forward only):
      SmolLM2-135M  d6e2b2b3b00b397dd3fa89722790218e  (both switch values)
      Qwen3-MoE     a56e02b731fdf93004fd10b3a591df80  (both switch values;
                    reproduces the documented hash)

G5. Gate on the final tree: ninja rc=0 (0 error lines, no pending work);
    build_check 0 failed with fingerprints re-recorded then verified;
    kraken-tests 2751/2751 rc=0; baseline gate rc=0 (18/18 selftest, 17/17
    proof); kraken-bench --gate rc=0; coherence rc=0 "3 coherent, 0 not".
""")
    print("session log section G appended")
