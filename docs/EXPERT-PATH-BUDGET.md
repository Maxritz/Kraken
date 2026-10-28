# What expert management is worth, measured

The question this file answers: at a small VRAM budget, is the remaining
performance in expert prediction / cache management, or somewhere else?

Measured on this host (2 cores, CPU backend, page-cached weights) with a
synthetic `qwen3moe` fixture: 8 layers, 512-wide, 16 experts, top-4,
**111.8 MiB** of expert stack, 24 decode steps, ctx 2048.

```
python3 tools/make_bench_gguf.py /home/user/fix/moe-512x8.gguf --embd 512 \
    --layers 8 --heads 8 --kv-heads 2 --ff 512 --expert-ff 512 \
    --experts 16 --experts-used 4 --vocab 4096 --ctx 2048 --format q8_0
```

| expert cache | decode tok/s | HOT hit | COLD miss | read ms | read % of decode |
|---|---:|---:|---:|---:|---:|
| 8 MB | 110.1 | 0.0% | 832 | 82.4 | 39.5% |
| 32 MB | 127.8 | 70.0% | 260 | 52.6 | 29.2% |
| 512 MB | **140.3** | **88.2%** | 98 | 45.4 | 27.7% |

## Reading it

**The 512 MB row is not an 88% hit rate, it is a 100% one.** The fixture has
8 x 16 = 128 experts and the run only ever routed to 98 of them; the 98 misses
are one compulsory cold touch each. So with room to spare the cache is already
optimal: no policy can do better than 98 misses, and no predictor can either.
That is the first thing to check before buying a predictor — is the miss count
already equal to the number of distinct experts touched?

**The whole expert-cache lever is worth 1.27x here.** 110.1 -> 140.3 tok/s from
a completely thrashing cache to a perfect one. The reason is in the last column:
at 0% hit the expert read path is 39.5% of decode, so the theoretical floor
(a cache that never misses and never reads) is ~182 tok/s, or 1.65x over the
worst case.

**Why that number is host-specific and not a verdict.** These reads come from
the page cache at ~1.9 MB per expert in ~99 us each, i.e. memory speed. The
engine's own note in `expert_cpu.hpp` records 593 MiB/s for one outstanding
NVMe read on this class of host. At that rate the same 832 cold reads cost
~2.7 s instead of 82 ms, and the hit rate becomes the entire decode time. So:

* **weights in page cache / RAM** -> expert management is a ~1.3x lever;
* **weights on NVMe and not fitting VRAM** -> it is the dominant lever, and the
  gap between 70% and 100% hit is worth far more than any kernel change.

The engine already reports the numbers needed to tell which regime you are in,
on the `[stats ]` lines: `experts ... N/M slots resident`, `cache ... HOT hits
(P%)` next to `COLD misses`, and `expert-path read <ms>`. The decision rule is
`read ms / decode ms`: below ~20% the cache is not the problem, above ~50% it is
the only problem.

## What this changes about the roadmap

The prediction and prefetch machinery the master spec asks for already exists
here (`MoeRoutePredictor`, lookahead, read-ahead, LFU-with-aging residency), and
at sufficient budget it already reaches the compulsory-miss floor. What is
missing is not another predictor:

1. **The per-expert read cost.** 99 us per 1.9 MB is memory-bound, not
   policy-bound. More outstanding reads (the header's own 593 -> 1446 MiB/s
   scaling with four in flight) is worth more than better prediction when the
   miss count cannot go below the number of distinct experts.
2. **Paged KV + continuous batching** stays the structural gap, and this
   measurement does not touch it: it is a throughput limit (one sequence at a
   time), not a per-token latency limit.
3. **The GEMM** is still 85% of prefill. Expert management does nothing for
   prefill, which is where the 85% sits.

## Caveats

* One synthetic fixture, one host, page-cached weights, CPU backend. The hit
  rate curve depends on the router's skew, which a synthetic model does not
  reproduce faithfully — a real model's expert popularity is more concentrated,
  which flatters both LFU and any predictor.
* These are single runs. This host varies by up to 1.8x between runs, so
  differences under ~10% here are not meaningful; the 1.27x span is well outside
  that, the ordering of the three rows is not in question.
* `read ms` is the engine's own accounting, not an external measurement.
