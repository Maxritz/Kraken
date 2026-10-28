# Cleanliness audit — 2026-10-27

A pass over the whole tree for stubs, placeholders, half-written code and dead
weight, with the compiler's warning set treated as the defect list it is. What
follows is what was found, what was changed, and what could not be checked from
a machine with no GPU.

## What the sweep covered

`TODO`, `FIXME`, `XXX`, `HACK`, `TBD`, `WIP`, `placeholder`, `stub`,
`unimplemented`, `for now`, `temporary`, `dummy`, `fake`, `cheat`, `workaround`,
`#if 0`, `abort()`, `assert(false)`, `unreachable`, over `src/`, `include/` and
`tests/`; every virtual with an empty body in `Backend`; every `(void)param`
discard in `src/`; every `.cpp`/`.hip`/`.hpp` compiled with
`-Wall -Wextra -Wshadow`; and the build targets' own gate script.

## Findings that were not defects

- **`--expert-stub`** is a documented diagnostic: it requires `--expert-scan`,
  is refused on its own (`main_cli.cpp`), and warns that the ranking it produces
  is an approximation. A router-only measurement arm, not an unfinished path.
- **`src/hip/hip_stub.cpp` and the `#else` half of `src/npu/npu_offload.cpp`**
  are build-configuration shims for a tree configured without that backend.
  Both fail loudly and honestly: `make_hip_backend` returns nullptr with the
  configure line that would fix it, `npu::available()` returns false, and every
  NPU op returns false so the caller runs its own body.
- **`hadamard_act`'s base-class default** logs and then `std::abort()`s rather
  than returning with the output untouched. That is the correct shape for
  "this backend does not implement it": a lost transform is fluent garbage.
- **The `_patch_*.py` scripts under `tools/`** are applied one-shot codemods
  kept as provenance under the convention AGENTS.md describes. Nothing builds
  or runs them.
- **`KRK_ROUTE_FAKE`** hardcodes the routing to a fixed expert per row. It is
  environment-gated, off by default, and labelled at its definition and at its
  call site as a MEASUREMENT arm that changes the answer on purpose — it exists
  to price the router round trip by removing it. Left in place deliberately.

## Defects fixed

Real bugs the warning set was pointing at:

| where | what |
| --- | --- |
| `engine.cpp` draft-head load | two error paths did `delete h;` then read `h->cfg()`, i.e. a use-after-free behind a diagnostic |
| `engine.cpp` row-fidelity buckets | `1 << 40` on an `int`: undefined behaviour for the unbounded upper bound |
| `host_time.hpp` components report | `%run` inside a printf FORMAT string — `%r` is not a conversion, so the header was undefined behaviour |
| `dflash.cpp` capture probe | `snprintf(stage, 16, "post-l%d")` truncates a 32-bit layer index |

Dead code removed (each verified against its use sites): `dplan` (the leftover
of the device dispatch plan), `src` in the dequant probe, the activation size in
`kv_rollback_zero` and `mamba2_forward`, `q_dim`, `has_kv_a`, `gout`, `half`,
an unused lambda in the tests, and an unused `mb` in `bench_ops`.

Shadowing that hid meaning rather than merely tripping `-Wshadow`: a nested
`KRK_TIME_HOST` that made the sample span close the decode step's, three
capture-list keys all named `tl`, a flat index named `k` where `k` is also the
top-k count, a `u16` local shadowing the fp16 typedef, and a `host_reserve`
measured against installed RAM shadowing one measured against `--ram-tier`.

Hardening in the NPU layer: a failed submit or sync now stops the run (the
commands in that batch already returned true to their callers, so continuing
decodes garbage), and the staging allocation guard catches `std::exception`
rather than only `std::bad_alloc`.

## Gate verdicts

Run from a CPU-only sandbox (2 cores, no ROCm), so the device half of the
suite's own gate list could not run.

| gate | verdict |
| --- | --- |
| `scripts/build_check.sh` | **ok** — 0 failed, 0 warnings, Release confirmed |
| ninja, 0 error lines | **ok** — and 0 `warning:` lines in both trees |
| `kraken-tests`, CPU build | **4576/4576** |
| `kraken-tests`, NPU build | **4599/4599** |
| clean-room build from the release archive | **ok** — configure, build, both suites green |
| `scripts/kraken_tests_baseline_check.sh` | **0 regressions**, 3 new groups (`test_attention_softcap`, `test_gemma_family_chunk_gate`, `test_npu_backend`), 6 group-size changes that are CPU-vs-HIP (that baseline was recorded on a device build: its prefill gate carries 34 checks, this build's 27). Not re-recorded here on purpose — recording a CPU/NPU shape as the baseline would delete the device checks it exists to protect. |
| `kraken-bench --gate`, coherence spot check | **not run** — both need the HIP backend and real model files, neither of which exists in this sandbox. No GPU result is claimed anywhere in this tree's changes. |

Behaviour after the audit is unchanged where it could be observed: all three
arms (`--cpu`, `--npu`, `--npu-cpu`) are still chunk-invariant at
`--chunk 1/8/64/512`, each keeps its own trace hash, all three agree on the
sampled token, and `KRK_NPU_STRICT_QUEUE=1` still produces byte-identical text.
