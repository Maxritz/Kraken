# Sherlock trace bundle — 2026-10-22

Chrome-trace JSON for Perfetto (`https://ui.perfetto.dev` → Open trace file).
`*-stats.txt` files hold the matching run's stderr (stats lines, expert-path
split, op tables). All runs: RX 9070 XT (gfx1201), Release, greedy, short
prompt `"The capital of France is"`.

| File | Model | Command |
|---|---|---|
| `reap288-trace.json` | `qwen3.8-flash-next-reap-288-Q4_K_M.gguf` (78 GiB, qwen4exp, experimental) | `KRK_ARCH_ALLOW_EXPERIMENTAL=1 kraken -m <file> -p "The capital of France is" -n 4 --ctx 128 --chunk 8 --greedy --expert-cache-mb 256 --trace reap288-trace.json` → prefill 1.7 tok/s, decode 1.4 tok/s |
| `reap320-trace.json` | `Qwen3.8-Flash-Next-REAP-320-YMQ-M-TI.gguf` (63 GiB, qwen4exp, experimental) | same, REAP-320 file → prefill 2.6 tok/s, decode 1.7 tok/s |
| `qwen35flash-trace.json` | `qwen3.5-flash-Q4.gguf` (dense qwen35 27B, 16 GiB) | `kraken -m <file> -p "The capital of France is" -n 8 --ctx 256 --chunk 8 --greedy --trace qwen35flash-trace.json` → prefill 1.8 tok/s, decode 2.3 tok/s |
| `distill-trace.json` | `Qwen3.8-Distill-35B-A3B-...Q2KXL_ROCMFPX.gguf` (qwen35moe, 12 GiB) | `kraken -m <file> -p "The capital of France is" -n 8 --ctx 256 --chunk 8 --greedy --expert-cache-mb 512 --trace distill-trace.json` → prefill 9.3 tok/s, decode 5.9 tok/s (traced; untraced baseline in `distill-baseline-stats.txt`: 15.0/15.3 tok/s) |
| `smol-trace2.json` | `models/SmolLM2-135M-Instruct.Q4_K_M.gguf` (tracing verification) | `kraken -m models/SmolLM2-135M-Instruct.Q4_K_M.gguf -p "..." -n 8 --ctx 128 --greedy --trace smol-trace2.json` → 6069 events, all 7 lanes populated |
| `distill-xtrace.tsv` | expert access trace (`KRK_TRACE_EXPERTS=1`) for the Distill run, 89K | `tools/expert_trace.py distill-xtrace.tsv --hot 64 --warm 256` |

Headline findings (full analysis in `docs/DEEP-DIVE-MOE-PREFILL-DECODE-2026-10-22.md`):
- REAP: routed set 32–42 GiB vs 256 MiB HOT → HOT hits 0.8–2.6%, device lane topped by `warm_read(pool)` + `upload_paged`/`upload`, `moe-ffn` ≈ 95% of generate.
- Dense flash: weight streaming (56 stagings, 13 GiB, drain 1036 ms >> read 389 ms >> copy 11 ms).
- Router readback host-bound (`download_f32` backend 636 ms vs device 35 ms on REAP-288).
- Distill control proves the engine path: WARM-resident set → promote 38 ms at 60 GB/s.
