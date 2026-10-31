# Scores

Every number here is from this host: 2 cores, ~2.6 GHz nominal, AVX-512F/BW/VNNI
+ GFNI, no AMX. Model: synthetic `dense-768x12-q8.gguf` (12 layers, 768 wide,
q8_0, 92.6 MiB) unless stated. **This host varies by up to 1.8x between
sessions and between reps**, so only numbers taken in the same interleaved run
are comparable to each other; the historical column is context, not a control.

Regenerate the fixture:

```
python3 tools/make_bench_gguf.py /home/user/fix/dense-768x12-q8.gguf \
    --layers 12 --embd 768 --heads 12 --ff 2048 --vocab 4096 --format q8_0
```

Reproduce any row with `KRK_CPU_Q80_KERNEL=avx2|bw|vnni`.

## Prefill — 512-token prompt, ctx 4096, 3 interleaved reps

| rung | rep 1 | rep 2 | rep 3 | median | vs avx2 |
|---|---:|---:|---:|---:|---:|
| `avx2` (256-bit, pre-signflip-era kernel) | 343.3 | 370.1 | 345.4 | 345.4 | 1.00x |
| `bw` (512-bit + sign flip + f16c decode) | 391.5 | 410.4 | 395.2 | 395.2 | **1.14x** |
| `vnni` (shipped default) | 429.0 | 439.5 | 433.5 | **433.5** | **1.26x** |

Historical context: 384.7 tok/s was the best recorded before this round's kernel
work, on the 8-row AVX2 path. The `avx2` row above measures 345.4 today, which
is the same kernel — the difference is this host, and that is exactly why the
comparison above is the interleaved one and not "today vs the old number".

## Decode — 32 tokens, ctx 512, 3 interleaved reps

| rung | rep 1 | rep 2 | rep 3 | median | vs avx2 |
|---|---:|---:|---:|---:|---:|
| `avx2` (old single-row kernel) | 68.4 | 63.1 | 66.6 | 66.6 | 1.00x |
| `bw` | 81.4 | 75.7 | 74.0 | 75.7 | **1.14x** |
| `vnni` (shipped default) | 78.2 | 78.0 | 79.7 | **78.2** | **1.17x** |

Decode gained from the same three ideas the prefill kernel already had: 512-bit
width, the sign flip (one `maddubs` instead of two), and one F16C convert for
both block scales. The old decode kernel had **none** of them — every one of the
previous rounds' kernel wins had been applied only to the eight-row prefill
kernel, and decode is the path a user watches.

**VNNI buys nothing at one row** (78.2 vs 75.7, inside this host's noise): with a
single accumulator chain the loop is latency-bound, so replacing `maddubs+madd`
with `dpbusd` removes an instruction that was already free. It is kept as the
default because it is not worse and it is the faster rung at every width ≥ 8
rows.

## Components — prefill, shipped `vnni` rung, 512-token prompt

| component | ops | dev_us | us/op | %run | GMAC/s | %peak |
|---|---:|---:|---:|---:|---:|---:|
| `gemm_q8_0_i8_rN` | 84 | 952897.0 | 11344.0 | **81.8%** | **45.6** | 15.2% |
| `attention` | 12 | 141534.0 | 11794.5 | 12.2% | 34.1 | 33.4% |
| `silu_mul` | 12 | 30508.0 | 2542.3 | 2.6% | — | — |
| `rope` | 12 | 19968.0 | 1664.0 | 1.7% | — | — |
| `kv_append` | 12 | 10551.0 | 879.2 | 0.9% | — | — |
| `rmsnorm` | 25 | 8324.0 | 333.0 | 0.7% | — | — |
| `gemm_q8_0_i8_r1` | 1 | 329.0 | 329.0 | 0.0% | 9.6 | 3.2% |

Totals: `prefill` 1210.6 ms for 512 tokens = 422.9 tok/s in the profiled run
(profiling adds ~2%). Table sums to 99.9% of run.

## GEMM throughput over time

| stage | GMAC/s | %peak | source |
|---|---:|---:|---|
| int8 kernels, before | 18.0 | — | dequant path, `docs/GEMM-PERF-WALL.md` |
| i8 512-bit | 34.5 | 11.5% | #32 |
| + column blocking + sign flip | 39.1 | 13.0% | #33 |
| + F16C scale decode (1.17x) | ~45 | ~15% | #39 |
| + VNNI (1.21x on the isolated kernel) | **45.6** | **15.2%** | #42 + this round |

The last row is the interesting one: the isolated kernel measured 1.21x for
VNNI, but the end-to-end GMAC/s moves less, because at 81.8% of the run the GEMM
is now bounded by the memory it streams and by Amdahl on the remaining 18%.

## What is not a win

* **VNNI on decode** — see above, latency-bound, no measurable gain.
* **Row depth 16** — 96% of 8. The kernel is not short of independent work.
* **Reordering the activation scale chain** — four attempts, all neutral or
  worse; the ablation prices it at 0.5% of the kernel.
