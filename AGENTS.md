# AGENTS.md

Operational knowledge for working in this repo. Everything here is something a
fresh session would otherwise have to rediscover: it is not visible from the
code, the README, or `docs/`.

## Build and gates

- Build: `ninja -C build-hip kraken kraken-tests kraken-bench kraken-oracle`.
  **Always capture the exit status explicitly** (`> /tmp/b.log 2>&1; echo rc=$?`).
  Piping into `grep`/`head` masks a ninja failure, and `grep -c` exits 1 on zero
  matches — that is not a build failure.
- The gate suite is four things, not one: ninja rc=0 with 0 `error:` lines,
  `kraken-tests` 1847/1847, `kraken-bench --gate` rc=0, and a coherence spot
  check on three models (SmolLM2-135M, Qwen3.5-0.8B, Qwen3-MoE-4x0.6B).
- **A build failure leaves the old `.exe` in place.** A "run" after `ninja rc=1`
  silently executes the previous binary and can look like a pass or a new bug.
  Check rc before trusting any run.

## Line endings (this bites constantly)

- `src/hip/backend_hip.hip`, `src/common.cpp`, `src/model.cpp`,
  `include/krk/*.hpp` are **LF in git**. `src/engine.cpp`,
  `tests/test_kraken.cpp`, `src/sampler.cpp` are CRLF.
- Editing tools and Python round-trips can silently convert a file to CRLF.
  That is invisible in a diff of the code but makes every multi-line
  string-replace anchor fail with a confusing assertion. Check with
  `s.count('\r\n')` before blaming the anchor, and normalize with
  `raw.replace('\r\n','\n')`.

## Patching files safely

- **Multi-line Python patching: use a quoted heredoc (`python3 <<'PYEOF'`), not
  `python -c "..."`** — backticks and `\n` in the payload get eaten by bash.
- **A `\n` inside a C++ string literal survives exactly one level of escaping.**
  Through JSON→bash→Python it becomes a real newline and breaks the build with
  `expected ')'`. When writing a format string, build the backslash explicitly
  (`BS = chr(92)`) or write the patch to a `.py` file and run that.
- Prefer line-indexed edits for anything spanning lines; assert on the exact
  text first so a mismatch fails loudly instead of silently no-op'ing.

## Environment quirks

- Bash arithmetic: `$((n//3))` is invalid — use `$(( n / 3 ))`.
- `/usr/bin/time` does not exist here; measure wall time in the shell or in-process.
- HIP launches use the **default stream** (`<<<g, t>>>`); there is no `st_`
  member. `__shfl_xor(v, off)` (not `__shfl_down_xor`); `kWaveSize = 32`.
- Anything touching HIP must go through `hipcc` — keep HIP headers out of
  plain-C++ TUs. `src/main_cli.cpp` bridges via environment variables instead.
- Machine: 96 GB host RAM, 15.9 GiB VRAM, models also under `G:/More-models/`.

## Measuring performance (the expensive lessons)

- **Never trust a probe whose arms ran in sequence.** Page cache and
  driver state warm between arms, and the ordering alone is worth several GB/s.
  Every A/B must be interleaved run-by-run, and a win must hold across
  repetitions. This session produced three separate false conclusions from
  sequential probes before interleaving was introduced.
- A/B switches must exist *before* the measurement, not after: `KRK_WEIGHT_PULL=0`
  and `KRK_ATTN_QTILE=0` both restore the previous path so the comparison is one
  process apart.
- Instrument the wall time of the thing you are changing. Thread-summed
  per-stage timers hid a pipeline that was not actually overlapping — the sums
  looked healthy while throughput was 2.6× below the probe.
- `--profile` self-calibrates its own floor at startup (~0.7 µs host / ~20 µs
  device per record). Do not compare op timings against a nominal constant.

## Weight loading (`Backend::set_weight_pull`)

- Large tensors are read() out of the model file rather than DMA'd from the
  mapping, because a cold mapping costs one soft fault per 4 KiB page (~0.4 µs
  each) and that — not the transfer — is the bottleneck.
- **`GgufTensor::data` already includes the data-section offset.** A pointer
  into the mapping converts to a file offset by `src - map_base` alone; adding
  `data_section_off()` again reads every tensor from the wrong place.
- **`hipStreamNonBlocking` streams have no implicit ordering against the legacy
  default stream** that every kernel in this engine runs on. Draining them is
  not sufficient for the caller to observe the data — a device barrier (or an
  event the default stream waits on) is required. The failure is silent and
  model-dependent: bytes are in flight, not missing.
- On Windows a file `HANDLE` serialises concurrent `ReadFile`, so each reader
  thread opens its own `FileReader`: 7.4 GB/s shared vs 15.9 separate.
- Every buffer in the pull ring must be drained. A drain loop left over from a
  smaller ring silently leaves the last chunks unwritten.

## Coherence and correctness

- `scripts/coherence_check.sh` runs each model **twice** — on the GPU and again
  with `--cpu` — and diffs the text. A FAIL means device and scalar reference
  disagree; it does not mean the run was CPU-only.
- A greedy output that is fluent but wrong is the signature of a
  plausible-but-incorrect weight or attention path. Compare against
  `KRK_WEIGHT_PULL=0` / `KRK_ATTN_QTILE=0` before assuming the model is at fault.
- `KRK_DUMP` + `tools/dump_diff.py` localizes a divergence to the first
  differing stage; that is the fastest way to tell prefill from decode.
- **Verify a data path against the real source pointer, not against the mapping
  at the same offset.** Comparing `read_at(off)` to `mapping[off]` passes for any
  wrong `off` — it is circular.

## Known open items

- `Spark_one.Q6_K.gguf` fails coherence at drift step 12; five hypotheses were
  tested and eliminated (recorded in `docs/TODO.md`). It is fp16-vs-f32
  sensitivity, not a demonstrated bug.
- The query-tiled prefill kernel (`KRK_ATTN_QTILE=1`) is fast (~5.4× on a 2720-token
  prompt) but numerically wrong; it is opt-in and must stay that way until fixed.
- `maclin` (gfx1031 / RDNA2) is unreachable, so nothing is confirmed on RDNA2.
