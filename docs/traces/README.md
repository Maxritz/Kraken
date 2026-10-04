# Trace archive

Shareable performance evidence. Every file here was produced by a command
recorded below, on the machine described below. Nothing in this directory is
edited by hand.

## Host

| | |
|---|---|
| GPU | AMD Radeon RX 9070 XT, gfx1201, RDNA4, 64 CU, wave32, 15.9 GiB VRAM |
| HIP | 7.16.26323 |
| Host RAM | 95.9 GiB (96 GB) |
| OS | Windows, Git Bash |

## Reference binary

`H:/LLAMA-bins/rocm10/llama-cli.exe`, build `b11146-7fe450e19`.

**This build defaults to `--repack` ON** (env `LLAMA_ARG_REPACK`). Repacking
converts every tensor to `q4_K_8x8`, which needs a full extra host-RAM copy of
the model: peak resident set on laguna-xs2 was ~64 GiB with repack enabled
versus **12.5 GiB with `--no-repack`**. On a 24-32 GiB host the default setting
cannot run this model at all. `llama-rocm-repack-verbose.log` is the `-v`
transcript that shows the repack pass; it is kept as the evidence for that
memory cost.

## Baseline: laguna-xs2-Q4_K_M (19.3 GiB)

Prompt `"what is the capital of france?"`, `-n 24 --temp 0`, ChatML template.

| Run | Command tail | Prompt tok/s | Gen tok/s | Peak host RAM | Answer |
|---|---|---:|---:|---:|---|
| `llama-rocm-repack-default.log` | `-st` | 47.8 | 15.5 | ~64 GiB | correct |
| `llama-rocm-no-repack.log` | `-st --no-repack` | 25.2 | 14.4 | 12.5 GiB | correct |
| `kraken-laguna-xs2.log` | `--greedy --chat` | 7.9 | 10.1 | — | **wrong** |

Repack buys **1.9x on prefill** and **1.08x on decode**. That asymmetry is the
useful signal: repack is a matmul tile-layout optimization, so it helps the
compute-bound prefill phase and does essentially nothing for decode, which is
bound by streaming weights. Any decode comparison must be made against
`--no-repack`, which is also the only config that fits a normal host.

Reproduce:

```bash
bash scripts/run_ref.sh "G:/More-models/laguna-xs2-Q4_K_M.gguf" /tmp/out.log \
  -p "what is the capital of france?" -n 24 --temp 0 \
  --no-jinja --chat-template chatml -st --no-repack
```

```bash
./build-hip/kraken.exe -m G:/More-models/laguna-xs2-Q4_K_M.gguf \
  --prompt "what is the capital of france?" --max-tokens 24 --temp 0 --greedy --chat
```

`run_ref.sh` records peak working set and always kills `llama-cli.exe` when it
finishes, so a reference run cannot outlive its own measurement.

## Memory budgets and the DFlash head

| file | what it shows |
|---|---|
| `laguna-perf-ab-arena.txt` | 3 interleaved repetitions of the expert VRAM arena against one hipMalloc per slice: **90.00 -> 63.80 ms/token (11.11 -> 15.67 tok/s)**, and device memory after the run falls 15.04 GiB -> 10.23 GiB |
| `laguna-determinism-promote-wait.txt` | 8+8 interleaved runs of the promotion ordering change: 16/16 identical |
| `dflash-head-inspect.txt` | `kraken-inspect` on both Poolside DFlash heads: 6 dense blocks, `block_size=16`, `decoder_arch=laguna`, `target_layers=6`, no experts |

## Decode-cost localisation and the two fixes it produced

The files below are the evidence behind section 10 of
`docs/PERF-ANALYSIS.md`. All of them are on `laguna-xs2-Q4_K_M`, prompt
`"what is the capital of france?"`, `-n 24 --temp 0 --chat`.

| file | what it shows |
|---|---|
| `laguna-perf-ab-router-and-promote.txt` | 4 interleaved repetitions of the previous behaviour against the new default: **110.70 -> 96.25 ms/token (9.00 -> 10.40 tok/s)**, one identical text hash in all 8 runs, with the `expert-path` split per run |
| `laguna-determinism-promote-wait.txt` | 8+8 interleaved runs of `KRK_PROMOTE_WAIT=host` vs `event`: **16/16 identical**, and identical to the host-wait arm |
| `laguna-expert-path-split.txt` | the run the four-way split came from: read / xfer / alloc / room = 84% of the wall |
| `probe-sync-host-roundtrip.txt` | `probe_sync`: a host round trip on this machine costs ~85 us of fixed cost, independent of kind and of queue depth |
| `probe-promote-dma-depths.txt` | `probe_promote`: 2.06 MiB pageable-to-VRAM copies at 6-9.5 GB/s; in-flight depth changes almost nothing, so the shipped host wait, not the copy, was the cost |
| `probe-read-file-bandwidth.txt` | `probe_read.py`: the model file saturates at ~3.8-4.1 GB/s at every thread count from 1 to 32 |

Reproduce:

```bash
bash scripts/ab_perf.sh G:/More-models/laguna-xs2-Q4_K_M.gguf 4 \
  "KRK_ROUTER_SYNC=1 KRK_PROMOTE_WAIT=host :: " "X=0 :: "
bash scripts/determinism.sh G:/More-models/laguna-xs2-Q4_K_M.gguf 8 \
  "KRK_PROMOTE_WAIT=host" "KRK_PROMOTE_WAIT=event"
./build-hip/probe_sync.exe 300
./build-hip/probe_promote.exe 4096
python3 tools/probe_read.py G:/More-models/laguna-xs2-Q4_K_M.gguf 4 16
```

## Kraken per-op traces

`laguna-krk_time.err` is the full `KRK_TIME=1 KRK_PHASE=1` per-op table for
laguna-xs2, 137011 ops over 24 decode tokens. The `.err` files are the stderr
streams of interleaved A/B determinism runs (`none`, `vram3`, `pf` arms).
`results.tsv` is the 26-model device benchmark sweep.

These are the raw material for `docs/PERF-ANALYSIS.md`.
## Coherence sweep

`coherence-24-of-25-samastam-degenerate.log` is a device-versus-CPU-reference
check across 25 models, each asked the same question and compared on generated
text. **24 coherent, 1 not.**

The one failure is `Samastam-2.5B-Q8_0.gguf` with the reason `both produced no
text`. That is degenerate rather than a disagreement: Samastam is a base model
and emits EOS immediately under a ChatML wrapper on *both* backends, so there is
nothing to compare. Run it with `KRK_CHAT=0` (no chat wrapper) and it passes.
The count is 24/25, not 25/25, and the filename says so.

## Headline scores

Measured on the reference build `b11146-7fe450e19`, `--no-repack` unless noted.
`--repack` is that build's default and costs ~64 GiB of host RAM.

| model | metric | kraken | llama.cpp ROCm |
|---|---|---:|---:|
| laguna-xs2-Q4_K_M | decode tok/s | 10.6 | 14.4 |
| laguna-xs2-Q4_K_M | prompt tok/s | 7.9 | 25.2 |
| Laguna-S-2.1 (68 GiB, 3 shards) | decode tok/s | 0.9 | *not comparable* |

Laguna-S-2.1 has no valid llama.cpp baseline: at most 23.4% of a 68.09 GiB model
fits in 15.9 GiB of VRAM, so 76.6% of it was CPU-computed. Its kraken figure is
real and its output is garbage (`|UNK|`), so treat the row as "runs, wrong
answer, slow" rather than a speed comparison.

Per-model speed and coherence for the 26-model sweep is in
`results.tsv` and rendered in `docs/MODEL-STATUS.md`.
