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

## Kraken per-op traces

`laguna-krk_time.err` is the full `KRK_TIME=1 KRK_PHASE=1` per-op table for
laguna-xs2, 137011 ops over 24 decode tokens. The `.err` files are the stderr
streams of interleaved A/B determinism runs (`none`, `vram3`, `pf` arms).
`results.tsv` is the 26-model device benchmark sweep.

These are the raw material for `docs/PERF-ANALYSIS.md`.