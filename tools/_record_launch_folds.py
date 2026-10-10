"""Record the two decode launch folds: report section, session log, AGENTS note.

The numbers here are copied verbatim from the runs in /tmp (harness output, op
tables) -- nothing is retyped from memory, and the switch-off arm's figures come
from the same interleaved runs as the switch-on arm's.
"""
import io

REPORT_SECTION = """

---

## 8. Cutting the launch count (this session's second pass)

The op table says a decode step is structure-bound: 303 launches on SmolLM2-135M
at a constant ~5.3 us per op across two models whose weight traffic differs
6.7x, against a `kraken-bench` launch floor of 3.64 us per call. Two folds, both
**bit-identical by construction** and both default-on behind a switch, remove
two launches per layer:

| fold | what it merges | switch that restores the pair |
|---|---|---|
| residual into the projection | `gemm(scratch)` + `add_inplace(dst, scratch)` -> `dst += W*x` in the GEMV epilogue (`gemv_kernel<256, true>`) | `KRK_ACC_GEMM=0` |
| activation into the gate/up group | gate + up + `silu_mul` -> one warp computes the same row of both and applies silu in the epilogue (`fused_gate_silu_kernel`) | `KRK_FUSED_SILU=0` |

Why they are bit-identical, not merely close: the residual form rounds the
projection to the activation type *before* summing (which is what the two-launch
form stores), and a sum of two f16 values is exact in f32, so the rounded result
is the same number; the silu form rounds both operands to f16 before the
activation, exactly as `silu_mul_kernel` reads them back from memory.

Measured SmolLM2-135M, ctx 512, n 64, greedy, interleaved 5 pairs per run, arms
one switch apart, stdout md5 `d6e2b2b3b0...` identical in every arm:

| comparison (B vs A) | median paired delta | MAD | pairs B faster | verdict |
|---|---|---|---|---|
| residual fold only | **+2.867%** | 1.221% | 5/5 | WIN |
| silu fold only (residual fold on in both arms) | **+6.135%** | 1.067% | 5/5 | WIN |
| both folds | **+8.347%** | 1.802% | 5/5 | WIN |

And the structure itself, from the profiler's *counts* (which are exact even
though its milliseconds are not):

| arm | ops/step | components |
|---|---|---|
| both folds off | **303** | projections 149, attention 30, norms 31, other 30, ffn-activate 30, residual 30, transfer 1, embed 1, head+sample 1 |
| both folds on | **243** | projections 149, other 30, attention 30, norms 31, transfer 1, head+sample 1, embed 1 |

The two rows that disappear are exactly `ffn-activate` (30) and `residual` (30):
60 launches = 2 per layer x 30 layers. 303 -> 243 is -19.8% of the step's ops
for +8.35% of its wall time.

**The marginal launch is ~1.5 us, not the 3.64 us bench floor.** The residual
fold removed 30 launches and bought 46 us (2.867% of a 1.6 ms step), so a launch
in this already-saturated stream costs about half of what an empty 64x256 launch
measures in `kraken-bench`. Estimating a fold from the bench floor overstates it
by ~2.4x -- the floor is the cost of *adding* a launch to a quiet queue, not the
cost of *holding* one in a busy step.

**Instrument bug found on the way:** `env A=0,B=0 kraken.exe` sets ONE variable
named `A` to `"0,B=0"`, which every switch here reads as truthy, so a
comma-separated arm comparison silently ran the same configuration twice and
reported NO DIFFERENCE (median -0.52%, MAD 0.70%) on an experiment with no arms.
`scripts/kraken_sherlock_kernel_ab.sh` now splits the list into words before it
reaches `env`; the same trap is one `env` away in any harness in this repo.

**Gate on this tree:** `ninja` six targets rc=0 with 0 `error:` lines;
`build_check.sh` 0 failed (fingerprints re-recorded per procedure, then verified);
`kraken-tests` 2751/2751 rc=0; the baseline gate rc=0 with `proof 17/17` and
`18/18 selftest`; `kraken-bench --gate` rc=0; coherence on the three small models
rc=0, **3 coherent, 0 not** -- so the fused kernels match the CPU scalar
reference on a dense, a gated-attention and an MoE model.
"""

AGENTS_BULLET = """- **A decode step's cost is its op COUNT, and two folds that remove two launches
  per layer are default-on now.** `dst += W*x` in the GEMV epilogue
  (`gemv_kernel<256, true>`, `KRK_ACC_GEMM=0` restores `gemm()` +
  `add_inplace()`) and gate/up + silu in one launch (`fused_gate_silu_kernel`,
  `KRK_FUSED_SILU=0` restores the separate `silu_mul()`). Both are
  bit-identical by construction, not approximately equal -- the residual form
  rounds the projection to the activation type before summing, as the pair
  stores it, and the silu form rounds both operands before the activation --
  and the proof is that stdout md5 is unchanged. Measured on SmolLM2-135M
  (ctx 512, n 64, greedy, 5 interleaved pairs per run, text identical): residual
  fold **+2.867%** (5/5), silu fold **+6.135%** (5/5), both **+8.347%** (5/5);
  the op table goes **303 -> 243 ops/step**, the `residual` and `ffn-activate`
  component rows disappearing (60 = 2 per layer x 30 layers). **The marginal
  launch costs ~1.5 us, not the 3.64 us `kraken-bench` floor** -- 46 us for 30
  launches -- so sizing a fold from the bench floor overstates it ~2.4x; that
  floor is the price of adding a call to a quiet queue, not of holding one in a
  busy step.
- **`env A=0,B=0 cmd` is ONE variable, not two.** The value is the literal
  string `"0,B=0"`, which any `strcmp(v, "0") != 0` test reads as truthy, so a
  comma-separated arm list silently runs the same configuration in both arms.
  `scripts/kraken_sherlock_kernel_ab.sh` hit exactly this and reported NO
  DIFFERENCE (median -0.52%, MAD 0.70%) on an experiment with no arms; it now
  splits the list into words first. Every harness here that takes switches from
  the environment needs the same check.
"""


def append(path, text):
    s = io.open(path, encoding="utf-8", newline="").read()
    if not s.endswith("\n"):
        s += "\n"
    io.open(path, "w", encoding="utf-8", newline="").write(s + text.lstrip("\n"))


append("sherlock_report.md", REPORT_SECTION)

# The AGENTS.md note goes with the other measurement lessons, after the
# `kraken-bench --sweep` bullet.
s = io.open("AGENTS.md", encoding="utf-8", newline="").read()
anchor = "- A/B switches must exist *before* the measurement, not after:"
assert s.count(anchor) == 1
s = s.replace(anchor, AGENTS_BULLET + anchor)
io.open("AGENTS.md", "w", encoding="utf-8", newline="").write(s)
print("sherlock_report.md section 8 appended; AGENTS.md bullet inserted")

# Session log, if it is where the last session's evidence lives.
try:
    s = io.open("sherlock-kraken-session2.log", encoding="utf-8", newline="").read()
except FileNotFoundError:
    print("no session log at sherlock-kraken-session2.log")
else:
    log = """
================================================================================
F. Decode launch folds: 303 -> 243 ops/step, +8.35% decode (bit-identical)
================================================================================

F1. Residual add folded into the out-projection's GEMV epilogue.
    New: Backend::gemm_accumulate() (include/krk/backend.hpp), gemv_kernel<256,
    true> (src/hip/kernels/gemm.hpp), Engine::gemm_residual() (src/engine.cpp,
    used by dense_ffn + both ssm_out sites), KRK_ACC_GEMM switch
    (src/hip/backend_hip.hip).

    $ KRK_AB_A_SWITCH=KRK_ACC_GEMM=0 KRK_AB_SWITCH=KRK_ACC_GEMM=1 \\
      KRK_AB_REPS=5 bash scripts/kraken_sherlock_kernel_ab.sh
      paired deltas (B vs A, %): 2.867 4.414 3.399 1.646 1.307
      median 2.867%   MAD 1.221%   B faster in 5/5 pairs
      verdict: WIN -- Text identical.

F2. silu_mul folded into the gate/up group's epilogue.
    New: fused_gate_silu_kernel + the launcher's silu branch
    (src/hip/kernels/fused.hpp), the silu request/answer on
    Backend::gemm_group() (include/krk/backend.hpp), KRK_FUSED_SILU switch.

    $ KRK_AB_A_SWITCH=KRK_FUSED_SILU=0 KRK_AB_SWITCH=KRK_FUSED_SILU=1 \\
      KRK_AB_REPS=5 bash scripts/kraken_sherlock_kernel_ab.sh
      paired deltas (B vs A, %): 5.068 6.247 7.576 3.818 6.135
      median 6.135%   MAD 1.067%   B faster in 5/5 pairs
      verdict: WIN -- Text identical.

F3. Both folds together, and the op count that says why.

    $ KRK_AB_A_SWITCH=KRK_ACC_GEMM=0,KRK_FUSED_SILU=0 \\
      KRK_AB_SWITCH=KRK_ACC_GEMM=1,KRK_FUSED_SILU=1 \\
      KRK_AB_REPS=5 bash scripts/kraken_sherlock_kernel_ab.sh
      paired deltas (B vs A, %): 5.838 8.454 5.480 8.347 10.149
      median 8.347%   MAD 1.802%   B faster in 5/5 pairs
      verdict: WIN -- Text identical.

    Every arm's stdout md5: d6e2b2b3b00b397dd3fa89722790218e (unchanged from
    the pre-fold runs, i.e. the folds moved no numbers).

    Op table, both folds off (KRK_PROFILE=1, n 32, ctx 512, greedy):
        303 ops | host wall  13.827 ms | device spans 7.986 ms | device idle 6.654 ms
        component: projections 149, attention 30, norms 31, other 30,
                   ffn-activate 30, residual 30, transfer 1, embed 1, head+sample 1
    Both folds on:
        243 ops | host wall  12.114 ms | device spans 6.194 ms | device idle 6.520 ms
        component: projections 149, other 30, attention 30, norms 31,
                   transfer 1, head+sample 1, embed 1
    The rows that vanish are residual (30) and ffn-activate (30) = 2 per layer x
    30 layers. 46 us bought by 30 removed launches = ~1.5 us per launch, against
    the 3.64 us empty-launch floor `kraken-bench` measures -- the floor is the
    cost of adding a call to a quiet queue, not of holding one in a busy step.

F4. The harness had a bug, and the first "both folds" run was its symptom.
    KRK_AB_A_SWITCH=KRK_ACC_GEMM=0,KRK_FUSED_SILU=0 reached `env` as one
    assignment, so arm A was "KRK_ACC_GEMM=0,KRK_FUSED_SILU=0" -- a truthy
    string for ACC, and the silu switch untouched -- i.e. both arms ran the same
    configuration and the comparison said NO DIFFERENCE (median -0.52%, MAD
    0.70%). Fixed by splitting the list into words; the re-run above is the
    fixed harness's answer. An instrument that can silently compare an arm with
    itself is the reason the arms' text is checked before their speed.

F5. Gate on this tree (all after the last edit):
      ninja six targets          rc=0, 0 "error:" lines, "ninja: no work to do"
      build_check.sh             0 failed, 1 warning (section 6 as designed);
                                 fingerprints re-recorded (KRK_REFRESH_FINGERPRINT=1)
                                 then verified: "all 5 artifact(s) match"
      kraken-tests               2751/2751 checks passed, rc=0
      baseline gate              rc=0, "tool 18/18 selftest cases passed",
                                 "proof 17/17 selfcheck cases passed",
                                 "verdict: no regression"
      kraken-bench --gate        rc=0
      coherence_check.sh         rc=0, "3 coherent, 0 not"
                                 (SmolLM2-135M, Qwen3.5-0.8B, Qwen3-MoE-4x0.6B)
"""
    append("sherlock-kraken-session2.log", log)
    print("session log section F appended")
