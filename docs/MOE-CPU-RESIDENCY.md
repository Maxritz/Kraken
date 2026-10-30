# MoE on the CPU backend: where the time actually goes

Measured 2026-10-30 on the 32-expert q8_0 fixture
(`--embd 1024 --layers 6 --ff 512 --expert-ff 512 --experts 32
--experts-used 4 --vocab 16000 --format q8_0`, **355.8 MiB**, 373 MB on disk),
on a 2-vCPU AVX-512 host with 3.9 GB RAM. Every number below is from that host;
none of it is extrapolated from a dense model.

## The fixture had to exist first

Before this, every CPU performance number in the repo came from a DENSE model.
An MoE model does not behave like one -- it issues ~16 GEMMs per layer instead
of 4, the expert matrices are small and wide, and the weight traffic per token is
set by the active expert set rather than by the model. So the first deliverable
was the fixture, and building it turned up a real defect:

**`tools/make_bench_gguf.py` could not produce a loadable MoE file.** With
`--experts` and no `--schema` it defaults `arch` to `qwen3moe` while leaving
`schema` unset, and the q/k-norm switch was keyed on `schema`. So the tool wrote
a qwen3moe file with no `attn_q_norm.weight`, and the engine refused its own
default MoE fixture: *"layer 0 is missing attn_q_norm.weight, which arch
'qwen3moe' declares"*. Qwen3's real MoE checkpoints carry both norms, so the arch
row is right and the generator was wrong. Fixed by keying the switch on `arch`.

## Decode is one kernel, and it is not the expert read

Decode-only profile (61-token prompt is not included; 40 timed steps):

| component | ops | us/op | %run | GMAC/s |
|---|---:|---:|---:|---:|
| `gemm_q8_0_i8_r1` | 4072 | 34.4 | **86.4%** | 19.8 |
| `gemm_q8_0_i8_rN` | 114 | 131.1 | 9.2% | 12.5 |
| attention | 240 | 11.7 | 1.7% | 4.1 |
| silu_mul | 982 | 2.1 | 1.3% | – |

So on CPU the MoE decode is the same GEMV kernel the dense path uses, at 86% of
the run. The expert read is **not** in the decode path at all:

```
expert-path split  prefill: read    97.6 ms | alloc 1.2 ms
                   decode:  read     0.0 ms | alloc 0.0 ms
```

That split line did not exist for CPU runs before this change: it was printed
only when a device *promotion* happened, and a pure-CPU run has none, so the one
run that needed the breakdown never printed it.

## The real cost is populating the expert set, and it lands on prefill

`read 97.6 ms` for 169 MiB = **1.73 GB/s**, which is far below this host's
measurable rates for the same bytes. Ablated in isolation, same file, same
bytes, warm page cache:

| destination pattern | GB/s |
|---|---:|
| reused buffer | 5.25 |
| `malloc`+`free` per chunk | 9.78 |
| **every slot live at once** | **2.05** |
| huge pages, all live | 1.97 |

The engine was in the third row, and 2.05 vs the 1.73 it reported is the whole
gap. The cost is not I/O and not the copy: it is the kernel zeroing every freshly
allocated page before handing it over. Huge pages do not help (1.66 -> 1.97), so
it is per-page zeroing, not per-fault overhead.

## The fix: recycle expert slots on this backend

`CpuBackend` used the base class's `alloc_pooled`/`release_pooled`, which are
`alloc`/`free`. Every cold expert asked for memory that had never been touched,
and an evicted slot's memory went straight back to the allocator instead of
being reused. A bounded free list keyed on the slice size fixes it
(`KRK_CPU_POOL_MB`, default 512 MiB; `0` disables it and reproduces the old
behaviour exactly, which is the control below).

With a **constrained budget** -- `--expert-cache-slots 8`, i.e. the case this
project exists for, a model far larger than RAM -- the pool is used hard:
**1329 reuses against 24 allocations**. Interleaved, three runs each:

| | expert read | decode |
|---|---:|---:|
| pool off | 104.4 / 104.4 / 106.2 ms | 9.2 / 9.2 / 8.9 ms/step |
| pool on | 77.0 / 79.1 / 83.6 ms | 7.8 / 8.2 / 8.6 ms/step |

**read -24%, decode +12%**, and `room` fell 4.6 -> 0.4 ms. Unconstrained
(everything fits) it is neutral-to-better, 100.3 -> 93.5 ms, because nothing is
evicted and therefore nothing is reused.

## What is still open here

* The read is single-threaded. The same 169 MiB on two threads should be close
  to half; it is not measured yet.
* On CPU the slot is a redundant copy of bytes already in the mapping. Pointing
  the slot at the mapping instead would remove the copy and the first-touch
  entirely -- the `in place` counter exists and reads 0.0% on every CPU run.
* `--expert-cache-mb` has no effect on the CPU backend: the residency ledger it
  sizes is the simulated device one, so the run reports the same 106 COLD misses
  at 0 MiB and at 4096 MiB. Only `--expert-cache-slots` changes behaviour.
