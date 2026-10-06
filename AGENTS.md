# AGENTS.md

Operational knowledge for working in this repo. Everything here is something a
fresh session would otherwise have to rediscover: it is not visible from the
code, the README, or `docs/`.

## Build and gates

- Build: `ninja -C build-hip kraken kraken-tests kraken-bench kraken-inspect`
  and `kraken-oracle`.
  **Always capture the exit status explicitly** (`> /tmp/b.log 2>&1; echo rc=$?`).
  Piping into `grep`/`head` masks a ninja failure, and `grep -c` exits 1 on zero
  matches — that is not a build failure.
- **`kraken-inspect` is in the build line deliberately.** It answers loader
  verdicts from the same `krk_core` tables the loader uses, and one built before
  a new dequantizer landed reports `refused` for formats the engine already
  runs -- seen live: it refused #41/#21/#22/#18 the day after they were fixed,
  contradicting the README. `build_check.sh` requires it current for the same
  reason.
- The gate suite is five things, not one: `sh scripts/build_check.sh`, ninja
  rc=0 with 0 `error:` lines, `kraken-tests` 2329/2329, `kraken-bench --gate`
  rc=0, and a coherence spot check on three models (SmolLM2-135M,
  Qwen3.5-0.8B, Qwen3-MoE-4x0.6B).
- That 2329 is a watermark, not a constant: the suite grew 2214 -> 2316 when the
  missing dequantizers landed, and 2316 -> 2329 when Q2_0's geometry changed.
  Read the count off the run (`2329/2329 checks passed`) rather than trusting the
  number written here, which is only the last one seen.
- **A build failure leaves the old `.exe` in place.** A "run" after `ninja rc=1`
  silently executes the previous binary and can look like a pass or a new bug.
  `scripts/build_check.sh` is that check, automated: it asks `ninja -n` whether
  work is still pending instead of reading the log, and it also fails on a Debug
  build type, a CPU-only binary in a HIP configuration, and an offload arch that
  does not match the card in the machine.

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
- **Host memory: read the number, do not infer a leak.** A GPU run of a 27B model
  peaks at ~0.35 GB private / ~0.37 GB working set; the ~267 GB "virtual" figure
  Task Manager shows for it is the HIP runtime's address-space reservation under
  `HIP_VMEM_MANAGE_SUPPORT=1` and is **not committed memory**. A `--cpu` run is
  ~7.0 GB private / 13.6 GB working set, because the engine copies the quantized
  weights into backend buffers and on the CPU backend that memory is host RAM.
  The real host spend on a MoE is the expert WARM tier: the auto RAM policy hands
  it about half of free RAM as a budget (measured `ram policy: 37.6 GiB budget
  (95.9 GiB installed, 75.2 GiB free)` on this box) and the tier then takes
  min(corpus, budget), so a large-corpus model can legitimately commit tens of GB.
- **The ROCm oracle needs the ROCm runtime bin on PATH.**
  `H:/LLAMA-bins/rocm10/llama-cli.exe` loads `ggml-hip.dll`, which fails to load
  without `G:\ROCM10RT-gfx1201\bin` on `PATH`; the loader then reports an *empty*
  error and continues **CPU-only**, so a "ROCm" run silently becomes a CPU run
  (measured 2.7 tok/s against 35.8 for the same model and prompt). Export the bin
  dir first and confirm `--list-devices` prints `ROCm0` before believing any
  number from it. Run it bounded: `llama-completion.exe -no-cnv --no-jinja` with
  stdin from `/dev/null`; without `-no-cnv` it enables conversation mode and sits
  at a `> ` prompt, and an unbounded `llama-cli.exe` invocation once spun into its
  REPL and wrote a ~100 MB log.

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
- **There are two profilers and a step needs both.** The device table
  (`src/hip/op_time.hpp`) times kernel spans and lives in the HIP TU; the host
  table (`include/krk/host_time.hpp`, `KRK_TIME_HOST`) times whole phases from
  portable code and prints underneath it as `[stats ] host time`. A step whose
  `decode.step` is large while `decode.forward` is small is host-bound -- which
  is what "the CPU is busy and the GPU is idle" looks like in numbers. Both are
  enabled by the same switch, so `--profile` never reports half the story.
- The device table CANNOT nest (one `cur_` slot), so never open a host span
  around backend ops expecting a device breakdown inside it. The host scopes are
  `spec.round`, `prefill.chunk`, `decode.step`, `decode.forward`, `decode.fetch`.
- **KV tier traffic is reported at the end of every run**, dense models
  included: `[stats ] kv tiered: ... promoted WARM->HOT n, COLD->HOT n, evicted
  ... n MiB migrated`. It printed nothing before because the only reader was the
  load-time one-liner and the MoE block returned first. One HOT slot
  (`--kv-hot-mb 1`) takes SmolLM2-135M from ~54 to ~2.4 tok/s -- that is the
  shape of a tier budget smaller than the working set, and it is now visible
  instead of inferred.

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

## GGUF id 42 is two different formats

- Upstream llama.cpp's Q2_0 is a **64-value block of 18 bytes** (2.25 bpw):
  `#define QK2_0 64` and `block_q2_0` = `ggml_half d` + `uint8_t qs[QK2_0/4]`.
- The llama-dx fork's Q2_0 (`Maxritz/LLAMA-DX`, `ggml/src/ggml-common.h`) is a
  **128-value block of 34 bytes** (2.125 bpw), because that header has
  `#define QK2_0 128` and therefore a 32-byte code plane. The same header adds
  `GGML_TYPE_Q2_0_64 = 48` for the 64-value layout and states that the two are
  *not* byte-compatible.
- **The id alone does not say which one a file carries.** Kraken decodes 42 as
  the fork's 128/34 layout and exposes the 64/18 layout as `Q2_0_64` (id 48).
  Getting this backwards is not a crash: every block is read from the wrong
  place and the model decodes fluent garbage.
- **The file's own tensor offsets are the authority.** Sort the tensors by their
  own offset and compare `next_offset - offset` against the computed span; the
  ratio is the density in bpw. `python3 tools/gguf_scan.py --audit FILE` does
  exactly that and names the variant: 2.125 bpw is the fork layout (34/128),
  2.25 bpw is upstream's (18/64). Measured: `Ternary-Bonsai-27B-Q2_0.gguf` is
  498 tensors, every one exactly 2.125 bpw; the second shard of
  `Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS` is 37 tensors at 2.25 bpw.
- The ternary files use **codes 0/1/2 only**: code 3 is 0.00% of 32.6M codes
  over 254,976 blocks, so the `q == 3` branch that reads +2d is dead in practice
  even though the reference computes it. The three live codes carry nearly equal
  weight (35.1 / 29.8 / 35.1%), which is also why no byte statistic can tell one
  code-to-level assignment from another: that has to come from the tool that
  wrote the file, or from a run, never from the file.

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

- **Decode attention was nondeterministic at ≥128 keys and is now fixed.**
  `attention_decode_split_kernel` (`src/hip/kernels/attention.hpp`) had every
  warp-leader thread store the block's partial denominator to the same address,
  each seeded with its own warp's term, so the last writer won and one head's
  output depended on warp scheduling. This is a *second*, model-blind source of
  run-to-run variation, distinct from the qwen35 packed-query race, and it is why
  several "not reproducible" notes in this repo could not be reproduced. Measured
  on Qwen3-8B-Q4_K_M, prompt "The history of computing is a history of
  abstraction, from relays to vacuum tubes", `-n 128 --greedy --ctx 512 --chunk
  256`, md5 of stdout: the old kernel gave **three** distinct outputs in six runs
  of the identical command (`fc30f6c8cf1d`, `d8448e90a0a3`, `7ee2e7dec3aa`); the
  fixed kernel gives `7ee2e7dec3aa` 10 times out of 10 and now matches
  `KRK_ATTN_SPLIT=0` bit-for-bit. Below 128 keys `attention_decode_splits` keeps
  `n_split == 1`, so a short-context probe **cannot** see it — the md5 A/B and
  `--profile` are the tools that catch this class of bug, and a green
  short-context run is not evidence of determinism.
- **The hybrid CPU+GPU expert path exists and LOSES ~3.1× when it engages — do
  not turn it on by default.** `--hybrid-experts 1` (`CpuExpertPool`,
  `include/krk/expert_cpu.hpp` / `src/expert_cpu.cpp`, wired into
  `Engine::moe_ffn`) computes the experts the device tier cannot hold on the
  host, out of the weight mapping, while the device runs the ones it can. It is
  opt-in and default off, and it is **inert rather than harmful** when the
  budget holds the routed set: measured on Qwen3-MoE-4x0.6B-Q4_K_M at the auto
  610 MiB budget it took **0** experts on the host and ran 105.7 vs 104.4 tok/s
  without it. At `--expert-cache-mb 16` it took all 922 experts at 5.392 ms per
  expert on 8 threads and decode fell from 17.2 to **5.5 tok/s**. Generated text
  was identical in every arm, so the arithmetic is fine — the host arm is a
  **per-layer merge barrier** the device waits on, and ~2.0 ms/expert of
  read+promote beats 5.392 ms/expert of CPU. Do not re-derive the 1.84×
  "perfect overlap" ceiling in `docs/traces/cpu-expert-ceiling.txt`: it assumes
  the CPU work hides behind the device, which a per-layer barrier forbids. Its
  value is memory (a host-computed expert needs no VRAM slot and no WARM
  buffer), not speed — see docs/test-results.md §9. If it is ever revisited, the
  policy detail that matters is that the host arm must take only the OVERFLOW
  (`!in_vram(layer, e) && full_for(source)`): taking every miss livelocks the
  cache, because an expert that is never promoted never becomes resident.
- `Spark_one.Q6_K.gguf` **passes** `coherence_check.sh` on the current build. It
  was recorded as failing at drift step 12; that no longer reproduces. Treat any
  "known to fail coherence" note as a claim to re-run, not a fact — the check is
  cheap and this one had gone stale.
- **Speculative decoding with a generic `--draft` was a 2.7× net loss** (32.4
  tok/s against 88.4 target-only, Qwen3-8B target + Qwen3-MoE-4x0.6B draft,
  44.8% accept) and the cause was a call-site regression, not the idea. The
  draft loop and the target's verify block both pulled the **whole 993 kB logits
  row per proposal** just to take an argmax on the host, where plain greedy
  decode had already been switched to the device top-k (`fetch_logits(true)` /
  `topk_argmax`, 16 bytes back). Both now use it. The prose above `fetch_logits`
  asserted that "the speculative and draft paths pass false and get the whole
  row exactly as before" — that was true, and it was the bug.
- The query-tiled prefill kernel is the prefill default; `KRK_ATTN_QTILE=0` opts
  out. It was fixed: the old default kernel had a shared-memory race (one f32
  score row per query overwritten with exp(x) before neighbours had read it),
  now corrected — the probe sweep reports 0 bad elements everywhere, worst
  residual 1.2e-4 from fp16 summation order. Byte-identical output (md5) to the
  one-query path at 801- and 900-token prompts, and +38% prefill tok/s
  interleaved (581.5 vs 420.9, 5 reps each, arms that do not overlap) is the
  real, reproducible number; the old 5.4x was a sequential cold-cache probe and
  does not reproduce.
- `maclin` (gfx1031 / RDNA2) is unreachable from here, so nothing is *executed*
  on RDNA2. The target still builds: `-DKRK_GPU_TARGETS="gfx1031;gfx1201"`
  compiles the gfx1031 kernels and the same `kraken-bench --gate` runs over
  them. Missing hardware, not missing verification.
- **The `*-Q2_0.gguf` ternary Bonsai files are not malformed** -- see "GGUF id
  42 is two different formats" above. Their own tensor offset table implies 17
  bytes per 64 values because the fork that wrote them uses a 128-value block
  (`#define QK2_0 128`, 34 bytes) where upstream uses 64 (18 bytes); that is also
  why upstream llama.cpp refuses `Ternary-Bonsai-27B-Q2_0.gguf` with `tensor
  'output_norm.weight' has offset 337715200, expected 357580800` -- the number
  this engine used to compute, before 42 became the 128-value layout the file
  actually carries. The files that carry 42 are
  `Ternary-Bonsai-27B-Q2_0.gguf` (498 tensors), the second shard of
  `Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS` (37) and
  `Ternary-Bonsai-27B-dspark-Q4_1.gguf` (one token_embd row). **The diagnostic
  worth keeping:** read the file's own tensor infos, sort them by offset, and
  compare `next_offset - offset` against the computed span of each tensor. That
  pinpoints a wrong block size in one run -- it is how the 17-vs-18 discrepancy
  was found -- where comparing a decode against the file's mapping cannot see it,
  because only the file's own offsets are authoritative.
- **`n_embd % n_head != 0` is not a defect when the file declares
  `attention.key_length`.** `Model::load` enforces that identity only when
  head_dim is *inferred* from the embedding (`head_dim_declared`,
  `src/model.cpp`). Qwen3.5's gated attention runs 24 heads of 256 dims over a
  5120-wide embedding because `q_proj` is `2 * n_head * head_dim` = 12288 and
  emits the query gate alongside the query, so the identity genuinely does not
  hold. Before this was fixed both `Bonsai-27B-Q1_0.gguf` and
  `Ternary-Bonsai-27B-Q2_0.gguf` were refused at load with `n_embd is not
  divisible by head count` while their dtype was already fine. Same shape as the
  per-layer-head-count exemption laguna already needed in that function.
