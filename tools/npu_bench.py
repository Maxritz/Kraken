#!/usr/bin/env python3
"""Compare the reference, the virtual-NPU GEMM arm and the whole-graph arm.

Runs one binary per arm over a matrix of (model, prompt length) and reports what
each produced, so a speed claim and its correctness witness come from the same
run rather than from two.

    python3 tools/npu_bench.py --bin-npu build-npu/kraken --bin-cpu build-cpu/kraken \
        --models '/tmp/fix/-*.gguf' --arms cpu,npu,npu-cpu --out bench.md

What is measured, per (model, arm, prompt length):

  prefill   tokens/s over the prompt   (the [stats ] prefill line)
  decode    tokens/s over the samples  (the [stats ] decode line, which excludes
                                       the first step because it carries the
                                       prefill's logits fetch)
  load      ms to map and upload
  device    the counts the NPU run reports: calls per op, declined, batches,
            GFLOP handed over and weight bytes staged

Every arm is run `--reps` times and the BEST result is kept, because the noise
here is one-sided: a run cannot come out faster than the work it did, but it can
come out slower because the OS scheduled something else on two cores. The spread
is reported alongside so a reader can see when the difference is inside it.

Correctness is checked rather than assumed: the same greedy decode is compared
against the reference arm's text. An arm that changed the model is not a faster
arm, and the report says so in a column instead of leaving it to the prose.

Traces: --trace DIR writes one Chrome trace per (arm, shape) through
KRK_NPU_TRACE (the device's own timeline) and --trace (the engine's), which
tools/trace_stats.py then summarises.
"""
import argparse
import glob
import json
import os
import re
import statistics
import subprocess
import sys


def prompt_of(n_tokens, unit="the quick brown fox jumps over the lazy dog"):
    """A prompt whose tokenizer cost is a round number of words.

    The fixtures use an SPM vocab with a handful of word pieces, so each word is
    one or two tokens; the exact count comes back in the stats line and is what
    the report uses.
    """
    words = unit.split()
    out = []
    i = 0
    while len(out) < n_tokens:
        out.append(words[i % len(words)])
        i += 1
    return " ".join(out)


RE_LOAD = re.compile(r"\[stats \] load\s+(\d+) ms")
RE_PREFILL = re.compile(r"\[stats \] prefill\s+(\d+) tok in ([\d.]+) ms = ([\d.]+) tok/s")
RE_DECODE = re.compile(
    r"\[stats \] decode\s+(\d+) tok in ([\d.]+) ms over (\d+) timed steps = ([\d.]+) tok/s")
RE_NPU = re.compile(r"\[stats \] npu\s+(.*)")
RE_ARM = re.compile(r"^(KRAKEN .*)$", re.M)


def parse_stats(stdout):
    r = {}
    m = RE_LOAD.search(stdout)
    if m:
        r["load_ms"] = float(m.group(1))
    m = RE_PREFILL.search(stdout)
    if m:
        r["prefill_tok"] = int(m.group(1))
        r["prefill_ms"] = float(m.group(2))
        r["prefill_tps"] = float(m.group(3))
    m = RE_DECODE.search(stdout)
    if m:
        r["decode_tok"] = int(m.group(1))
        r["decode_ms"] = float(m.group(2))
        r["decode_steps"] = int(m.group(3))
        r["decode_tps"] = float(m.group(4))
        r["step_ms"] = r["decode_ms"] / r["decode_steps"] if r["decode_steps"] else 0.0
    m = RE_NPU.search(stdout)
    if m:
        r["device"] = m.group(1).strip()
        for key, pat in (("gemm", r"gemm (\d+)"), ("rmsnorm", r"rmsnorm (\d+)"),
                         ("swiglu", r"swiglu (\d+)"), ("add", r"add (\d+)"),
                         ("mul", r"mul (\d+)"), ("declined", r"(\d+) declined"),
                         ("batches", r"(\d+) batch")):
            mm = re.search(pat, m.group(1))
            if mm:
                r[key] = int(mm.group(1))
        mm = re.search(r"([\d.]+) GFLOP", m.group(1))
        if mm:
            r["gflop"] = float(mm.group(1))
        mm = re.search(r"([\d.]+) MiB staged", m.group(1))
        if mm:
            r["staged_mib"] = float(mm.group(1))
    m = RE_ARM.search(stdout)
    if m:
        r["arm"] = m.group(1).strip()
    return r


def run_once(binary, model, arm, prompt, n_decode, ctx, extra_env=None, extra_args=None):
    cmd = [binary, "-m", model, "--prompt", prompt, "--max-tokens", str(n_decode),
           "--ctx", str(ctx), "--greedy", "--seed", "1"]
    if arm != "auto":
        cmd.append("--" + arm)
    if extra_args:
        cmd.extend(extra_args)
    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)
    # Binary capture: the fixture's SPM vocab falls back to raw bytes, so the
    # generated text is not necessarily valid UTF-8 and a text=True pipe would
    # raise on a byte that is a perfectly good token.
    p = subprocess.run(cmd, capture_output=True, env=env, timeout=1800)
    out = p.stdout.decode("utf-8", "replace")
    err = p.stderr.decode("utf-8", "replace")
    st = parse_stats(out + "\n" + err)
    # The generated text is everything after the banner line, minus the stats.
    text = []
    seen_banner = False
    for line in out.splitlines():
        if line.startswith("KRAKEN "):
            seen_banner = True
            continue
        if seen_banner and not line.startswith("[stats ]") and not line.startswith("[info"):
            text.append(line)
    st["text"] = "\n".join(text).strip()
    st["rc"] = p.returncode
    st["stderr"] = err[-2000:]
    return st


def best_of(runs, key):
    vals = [r[key] for r in runs if key in r]
    if not vals:
        return None, None
    return max(vals), (max(vals) - min(vals)) if len(vals) > 1 else 0.0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True,
                    help="GGUF paths or globs")
    ap.add_argument("--bin-cpu", default="build-cpu/kraken")
    ap.add_argument("--bin-npu", default="build-npu/kraken")
    ap.add_argument("--arms", default="cpu,npu,npu-cpu")
    ap.add_argument("--prefill", default="16,128,512",
                    help="prompt lengths in words (comma separated)")
    ap.add_argument("--decode", type=int, default=64)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default="")
    ap.add_argument("--json", default="",
                    help="write the raw records here")
    ap.add_argument("--trace", default="",
                    help="directory for one Chrome trace per (arm, model)")
    ap.add_argument("--env", action="append", default=[],
                    help="extra ENV=VALUE for every run (repeatable)")
    a = ap.parse_args(argv)

    models = []
    for pat in a.models:
        hit = sorted(glob.glob(pat))
        if not hit:
            ap.error("no model matches %r" % pat)
        models.extend(hit)
    arms = [s for s in a.arms.split(",") if s]
    env_extra = {}
    for kv in a.env:
        k, _, v = kv.partition("=")
        env_extra[k] = v
    if a.trace:
        os.makedirs(a.trace, exist_ok=True)

    records = []
    for model in models:
        name = os.path.basename(model)
        # The reference arm's text is the witness every other arm is compared
        # against, so it is produced once per (model, prompt) at a fixed decode
        # length and reused.
        for words in [int(x) for x in a.prefill.split(",")]:
            prompt = prompt_of(words)
            ref_text = None
            for arm in arms:
                binary = a.bin_npu if arm.startswith("npu") else a.bin_cpu
                if arm == "cpu":
                    binary = a.bin_cpu
                runs = []
                for rep in range(a.reps):
                    trace_env = {}
                    extra = []
                    if a.trace and rep == 0:
                        tag = "%s-%s-%d" % (re.sub(r"\.gguf$", "", name), arm, words)
                        trace_env["KRK_NPU_TRACE"] = os.path.join(a.trace, tag + ".vnpu.json")
                        extra = ["--trace", os.path.join(a.trace, tag + ".engine.json")]
                    st = run_once(binary, model, arm, prompt, a.decode, a.ctx,
                                  extra_env={**env_extra, **trace_env}, extra_args=extra)
                    st["arm"], st["model"], st["words"] = arm, name, words
                    runs.append(st)
                    if st["rc"] != 0:
                        break
                ok = all(r["rc"] == 0 for r in runs)
                if not ok:
                    sys.stderr.write("  FAILED %s %s words=%d: %s\n"
                                     % (name, arm, words, runs[-1]["stderr"][-400:]))
                pre_tps, pre_spread = best_of(runs, "prefill_tps")
                dec_tps, dec_spread = best_of(runs, "decode_tps")
                step_ms, _ = best_of([{"s": -r["step_ms"]} if "step_ms" in r else {} for r in runs], "s")
                text = runs[0].get("text", "")
                if arm == "cpu" and text:
                    ref_text = text
                rec = {
                    "model": name, "arm": arm, "words": words,
                    "prompt_tokens": runs[0].get("prefill_tok"),
                    "prefill_tps": pre_tps, "prefill_spread": pre_spread,
                    "decode_tps": dec_tps, "decode_spread": dec_spread,
                    "step_ms": -step_ms if step_ms is not None else None,
                    "load_ms": runs[0].get("load_ms"),
                    "device": runs[0].get("device", ""),
                    "gemm": runs[0].get("gemm"), "batches": runs[0].get("batches"),
                    "declined": runs[0].get("declined"),
                    "gflop": runs[0].get("gflop"), "staged_mib": runs[0].get("staged_mib"),
                    "arm_banner": runs[0].get("arm", ""),
                    "ok": ok,
                    "text_matches_reference": (ref_text is not None and text == ref_text)
                    if arm != "cpu" else True,
                    "text_len": len(text),
                }
                records.append(rec)
                print("%-24s %-8s words=%-4d prefill %8.1f tok/s  decode %8.1f tok/s  %s"
                      % (name, arm, words,
                         pre_tps or 0.0, dec_tps or 0.0,
                         "" if rec["text_matches_reference"] else "TEXT DIFFERS"),
                      flush=True)

    if a.json:
        with open(a.json, "w") as f:
            json.dump(records, f, indent=1)
        print("records: %s" % a.json)

    md = render(records, a)
    if a.out:
        with open(a.out, "w") as f:
            f.write(md)
        print("report: %s" % a.out)
    else:
        print(md)
    return 0 if all(r["ok"] for r in records) else 1


def render(records, a):
    """A markdown report: one table per model, speedup against the reference."""
    lines = []
    lines.append("# Reference vs virtual NPU\n")
    lines.append("Generated by `tools/npu_bench.py`. Decode is %d greedy tokens, "
                 "context %d, best of %d runs (the fastest run is the one that "
                 "was not descheduled on %d core(s)).\n"
                 % (a.decode, a.ctx, a.reps, os.cpu_count() or 1))
    lines.append("`same text` compares the arm's greedy decode against the "
                 "reference arm's on the same prompt: an arm that changed the "
                 "model is not a faster arm.\n")

    models = []
    for r in records:
        if r["model"] not in models:
            models.append(r["model"])
    for model in models:
        rows = [r for r in records if r["model"] == model]
        lines.append("\n## %s\n" % model)
        lines.append("| prompt | arm | prefill tok/s | decode tok/s | ms/step | "
                     "vs reference | same text |")
        lines.append("| ---: | --- | ---: | ---: | ---: | ---: | :---: |")
        for words in [int(x) for x in a.prefill.split(",")]:
            ref = None
            for r in rows:
                if r["words"] == words and r["arm"] == "cpu":
                    ref = r
            for r in [x for x in rows if x["words"] == words]:
                speed = ""
                if ref and ref["decode_tps"] and r["decode_tps"] and r["arm"] != "cpu":
                    speed = "%.2fx" % (r["decode_tps"] / ref["decode_tps"])
                dec = "%.1f" % r["decode_tps"] if r["decode_tps"] else "-"
                pre = "%.1f" % r["prefill_tps"] if r["prefill_tps"] else "-"
                step = "%.2f" % r["step_ms"] if r["step_ms"] else "-"
                lines.append("| %d | `--%s` | %s | %s | %s | %s | %s |"
                             % (words, r["arm"], pre, dec, step, speed,
                                "yes" if r["text_matches_reference"] else "**NO**"))
        dev = [r for r in rows if r["device"]]
        if dev:
            d = dev[0]
            lines.append("\nDevice work (`--%s`, first prompt): %s"
                         % (d["arm"], d["device"]))
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
