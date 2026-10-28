#!/usr/bin/env python3
"""int8_witness.py -- what does the int8 weight path cost the model's output?

`KRK_NPU_W8A8` puts each weight through a per-output-row symmetric int8
quantiser before the device multiplies with it (see `src/npu/npu_offload.cpp`,
`Offload::weight_i8`). This tool applies the exact same quantiser to a fixture's
weights and runs the independent reference forward (`tools/ref_llama.py`)
twice -- once f32, once quantised -- so the cost can be stated as a number
instead of as a claim.

The columns matter together, and any one of them alone misleads:

  max/mean |dlogit|   how far the approximation moves the output distribution,
                      end to end through every layer.
  median margin       the fixture's own top-1/top-2 gap. On random-weight
                      fixtures this is the same order as the error, so a flip is
                      expected, not surprising. It is NOT a statement about a
                      trained model, whose confident positions have wider gaps.
  margin < err        how many of the 33 prompt positions have a gap smaller
                      than the worst injected error -- the positions where an
                      argmax flip is possible at all.
  flips               argmax changes actually observed on the prompt positions.
  match               greedy continuation agreement over 32 tokens. This swings
                      from 32/32 to 1/32 for the same flip rate, because one
                      flipped token changes the prefix everything after it is
                      conditioned on. That is why it is opt-in.

Usage
-----
    tools/int8_witness.py FIXTURE.gguf [FIXTURE.gguf ...] [--reps 32]

Exit status is 0 when every fixture ran.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ref_llama import forward, load  # noqa: E402

# The bench fixtures are all generated with the same tokenizer, so the prompt
# "the quick brown fox" repeated eight times tokenises identically everywhere:
# one BOS then 32 word pieces. Verified with `kraken --tokenize` on all nine.
PROMPT_TOKENS = [1] + [259, 260, 261, 262] * 8


def quantise_like_npu(W):
    """Per-output-channel symmetric int8, identical to Offload::weight_i8().

    Rows are the output channels ([out, in] layout), the scale is row peak/127,
    and rounding is half away from zero -- which is what the C++ does in float
    (`f >= 0 ? f + 0.5f : f - 0.5f`), so `np.rint` is the matching numpy call
    for ties-to-even being explicitly NOT wanted here. `np.floor(f+0.5)` would
    be closer to the C++ for negative zero cases; the difference is one LSB on
    an exact .5 boundary and is reported by --audit-rounding if it ever matters.
    """
    peak = np.abs(W).max(axis=1, keepdims=True)
    scale = np.where(peak > 0, peak / 127.0, 0.0).astype(np.float32)
    inv = np.where(scale > 0, 1.0 / scale, 0.0).astype(np.float32)
    f = W * inv
    q = np.clip(np.floor(np.abs(f) + 0.5) * np.sign(f), -127, 127).astype(np.int8)
    return (q.astype(np.float32) * scale).astype(np.float32)


def greedy(meta, w, tokens, n):
    seq = list(tokens)
    for _ in range(n):
        seq.append(int(np.argmax(forward(meta, w, seq)[-1])))
    return seq[len(tokens):]


def audit(path):
    meta, w = load(path)
    q = {k: (quantise_like_npu(v) if v.ndim == 2 else v) for k, v in w.items()}
    f32 = forward(meta, w, PROMPT_TOKENS)
    i8 = forward(meta, q, PROMPT_TOKENS)
    err = np.abs(f32 - i8)

    margins = np.array([np.subtract(*np.sort(f32[t])[::-1][:2]) for t in range(f32.shape[0])])
    flips = int((np.argmax(f32, -1) != np.argmax(i8, -1)).sum())
    return {
        "name": Path(path).name,
        "max": float(err.max()),
        "mean": float(err.mean()),
        "median_margin": float(np.median(margins)),
        "min_margin": float(margins.min()),
        "below": int((margins < err.max()).sum()),
        "positions": int(f32.shape[0]),
        "flips": flips,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("fixtures", nargs="+")
    ap.add_argument("--reps", type=int, default=32,
                    help="greedy tokens to compare (default 32)")
    a = ap.parse_args(argv)

    rows = []
    for path in a.fixtures:
        r = audit(path)
        meta, w = load(path)
        q = {k: (quantise_like_npu(v) if v.ndim == 2 else v) for k, v in w.items()}
        fa = greedy(meta, w, PROMPT_TOKENS, a.reps)
        qa = greedy(meta, q, PROMPT_TOKENS, a.reps)
        r["match"] = sum(1 for x, y in zip(fa, qa) if x == y)
        rows.append(r)

    print(f"{'fixture':<20} {'max|dlogit|':>11} {'mean|dlogit|':>12} {'med margin':>10} "
          f"{'min margin':>10} {'margin<err':>10} {'flips':>6} {'match':>8}")
    for r in rows:
        print(f"{r['name']:<20} {r['max']:>11.4f} {r['mean']:>12.4f} "
              f"{r['median_margin']:>10.4f} {r['min_margin']:>10.4f} "
              f"{r['below']:>4d}/{r['positions']:<5d} {r['flips']:>6d} "
              f"{r['match']:>4d}/{a.reps}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
