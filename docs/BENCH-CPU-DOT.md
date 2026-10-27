# The CPU dot product: why prefill was 6x slower than it needed to be

**Measured 2026-10-27 on the reference box** (1 core / 2 SMT threads @ 2.60 GHz,
L1d 48 KiB, L2 1.3 MiB, L3 54 MiB, 3.8 GiB RAM, `avx2 f16c fma avx512f`).
Every number here is from that machine; this box has no GPU, so everything in
this document is the CPU backend.

This is the investigation that started from a 2 GiB routed-expert model, and the
finding was not in the MoE routing at all. It was in the dot product every
projection in the engine uses.

## 1. The report, and what it looked like

The complaint was that prefill is bad for a lot of models, MoE ones
especially — and that keeping experts off RAM should cost prefill "slightly",
not a factor of many. Two fixtures were built to look at it directly, both
synthetic and both regenerable from a seed (see §7):

| fixture | shape | size |
|---|---|---|
| `dense-1g-f16.gguf` | 32 layers, 1024 embd, 16/4 heads, hd 64, ff 4096, vocab 32000 | 1054.0 MiB |
| `moe-2g-f16.gguf` | 20 layers, 768 embd, 12/4 heads, hd 64, **32 experts top-4**, expert ff 768, vocab 32000 | 2315.5 MiB |
| `moe-2g-q8_0.gguf` | same stack, q8_0 | 1230.5 MiB |
| `moe-2g-q4_0.gguf` | same stack, q4_0 | 651.9 MiB |

Baseline, before any change:

| model | T=8 | T=16 | T=32 | T=64 | T=128 |
|---|---:|---:|---:|---:|---:|
| dense 1 GiB, prefill tok/s | — | 1.7 | 1.7 | 1.7 | — |
| MoE 2 GiB, prefill tok/s | 3.6 | 4.2 | 4.3 | 4.6 | 4.7 |

Dense prefill is **exactly linear**: 583 ms per token, unchanged from 16 to 64
tokens. A model that is linear in tokens is not amortising anything — each
token re-reads the whole 1 GiB. The MoE curve is worse than linear-per-byte:
4.3 tok/s at 36 tokens is 215 ms/token for a model whose per-token arithmetic
is 141.6 M MAC (20 layers x 4 experts x 3 matrices x 768 x 768), i.e. 0.66
GMAC/s.

Two checks ruled out the obvious suspects:

- **Not threading.** `taskset -c 0` against both threads: 16017.7 ms vs
  8061.7 ms for the same prefill — 2.0x, so both SMT threads were already
  busy. The kernel was parallel; it was slow per element.
- **Not the expert cache** in the first instance. At T=32 the run reported
  `608 MiB read from the file (151.88 MiB/tok)` out of ~283 MB of expert
  weights touched per token, so most expert traffic was already being served
  from the resident set, not from disk.

## 2. The cause

`vec_dot()` in `src/quant.cpp` is the only reduction the CPU backend has —
every projection, every expert matrix, the head — and its f16 branch read:

```c++
acc0 += (bf ? bf16_to_fp32(h[i]) : fp16_to_fp32(h[i])) * x[i];
```

`fp16_to_fp32` is **declared in the header and defined in `common.cpp`**: an
out-of-line call, with a branch chain and a normalisation loop for subnormals,
executed once per weight element. Nothing about that survives inlining, and
nothing about it vectorises. The result was 0.8 GMAC/s — roughly 3 cycles per
weight element on a core that can retire a 4-wide FMA in one.

This also explains the odd shape of the report. The cost is *per weight
element*, not per row, so:

- prefill and decode pay the same price per element, which is why prefill
  looked like "decode, but more tokens";
- the MoE experts — 3 matrices of 768x768 per (token, expert) pair, with an
  arithmetic intensity of one MAC per weight byte — were the worst possible
  shape for it, so MoE models were hit hardest;
- it was invisible on small fixtures, whose weights sit in the 54 MiB L3, and
  dominant on anything larger.

## 3. The fix, and why it is allowed in an oracle

The CPU backend is this project's **scalar reference** — the oracle every other
backend is tested against — so its arithmetic is not a performance decision.
The fix is therefore not a cheaper dot product; it is the *same* dot product
computed four elements at a time:

```c++
acc = _mm_fmadd_ps(_mm_cvtph_ps(h4), _mm_loadu_ps(x + i), acc);
```

Lane *k* of the vector is accumulator *k* of the scalar loop, fed the same
elements in the same order, and the final combination is still
`(a0 + a1) + (a2 + a3)`, and the scalar tail is untouched. `_mm_cvtph_ps` is
an exact IEEE half-to-single conversion, so the widened weights are bit-for-bit
the values `fp16_to_fp32` returned for every non-NaN input. The F32 branch gets
the same treatment.

Four lanes and no more: eight lanes would need eight accumulators, which is a
different summation order and therefore a different answer. The lane count is
fixed by exactness, not by the ISA.

**Proof, not assertion:** `build-cpu/kraken-tests` reports **4576/4576** before
and after the change, with every golden hash in the suite unchanged. A change
that altered one rounding in one dot product would fail dozens of them. The
change is bit-identical by construction and verified by the suite that exists
to catch exactly that.

## 4. Result

| measurement | before | after | |
|---|---:|---:|---:|
| dense 1 GiB prefill, T=16 | 1.7 tok/s | 10.0 tok/s | **5.9x** |
| dense 1 GiB prefill, T=64 | 1.7 tok/s | 10.0 tok/s | **5.9x** |
| MoE 2 GiB prefill, T=8 | 3.6 tok/s | 13.9 tok/s | **3.9x** |
| MoE 2 GiB prefill, T=32 | 4.3 tok/s | 22.9 tok/s | **5.3x** |
| MoE 2 GiB prefill, T=128 | 4.7 tok/s | 28.0 tok/s | **6.0x** |
| MoE 2 GiB decode | 254.9-264.7 ms/step | 58.9 ms/step | **4.3x** |

Decode moves for the same reason prefill does: the head and every projection
go through the same reduction.

The MoE prefill curve also changes shape. It used to be flat (4.3 to 4.7 tok/s
from 36 to 144 tokens, i.e. no amortisation); it now rises 13.9 -> 22.9 -> 28.0
tok/s, because with the conversion cost gone the expert-union reads amortise
over the chunk the way they always should have.

## 5. The block formats: why they were left behind, and the opt-in

| format | prefill T=32 | decode |
|---|---:|---:|
| f16 | 17.7 tok/s | 58.9 ms/step |
| q8_0 | 11.2 tok/s | 82.9 ms/step |
| q4_0 | 8.5 tok/s | 134.8 ms/step |

q4_0 is **half the bytes of q8_0 and 30% slower.** That is the signature of a
latency-bound loop, not a bandwidth-bound one, and it is structural: the block
formats accumulate into a *single* chain,

```c++
while (i < n) { dequant_block(t, p, buf); for (k) sum += buf[k] * x[i + k]; }
```

so the loop is one dependent FMA per element — a 4-cycle latency chain at ~0.65
G element/s per thread, which is what the two SMT threads deliver together.
Unlike the f16 path there is no legal vectorisation *that stays exact*: four
accumulators are a different summation order, and that changes every q4/q8
result against the oracle.

### 5.1 The opt-in

`KRK_CPU_DOT_FAST=1` (or `set_vec_dot_fast(true)`) switches the block formats to
four accumulators combined as `(a0+a1)+(a2+a3)` — the same shape the f16 path
already uses. It is off by default and the default path is untouched.

| fixture (2 GiB MoE, 121-token prompt) | exact | fast | |
|---|---:|---:|---:|
| q8_0 prefill | 13656.6 ms | 4443.2 ms | **3.07x** |
| q8_0 decode | 399.3 ms | 147.6 ms | **2.71x** |
| q4_0 prefill | 18685.6 ms | 9458.6 ms | **1.98x** |
| q4_0 decode | 540.6 ms | 267.9 ms | **2.02x** |

Best-of-3, interleaved arms, same binary. Less than the ~3-4x the chain analysis
suggested, because `dequant_block` itself is still in the loop: what the flag
removes is the dependency between the multiplies, not the widening. q4_0 gains
least for the same reason — unpacking nibbles costs more than the sum saves.

### 5.2 The accuracy witness

`tools/dot_fast_witness.py` reads real expert rows out of a GGUF and runs both
orders in strict f32:

```
moe-2g-q8_0.gguf: blk.0.ffn_gate_exps.weight q8_0  n_in=768  rows=512
  err / |dot|       median 4.738e-07   p99 1.533e-05   max 4.060e-04
  err / sum|w.x|    median 1.644e-08   p99 9.564e-08   max 1.399e-07
moe-2g-q4_0.gguf: blk.0.ffn_gate_exps.weight q4_0  n_in=768  rows=512
  err / |dot|       median 4.539e-07   p99 3.008e-05   max 5.786e-05
  err / sum|w.x|    median 1.530e-08   p99 7.686e-08   max 9.676e-08
```

Two denominators on purpose, because they answer different questions. `|dot|`
is what a user would divide by, and it explodes near a cancellation — a dot
that should be 0.001 and lands on 0.0013 is "30% wrong" and entirely fine.
`sum|w.x|` is the magnitude the summation order can actually perturb, and
against it the reordering costs **≤ 1.4e-07, about one ulp.** That is the claim
the flag makes, and `test_vec_dot_fast_opt_in` pins it as a bound rather than a
number so it cannot rot into a tautology.

What this is **not** is a quality claim on a real checkpoint: the fixtures are
seeded random weights, so their greedy text is degenerate and identical between
the arms for reasons that have nothing to do with accuracy. The witness is the
numeric deviation, measured on real quantized rows.

### 5.3 What it costs elsewhere

- **The CPU arm stops being a bit-exact oracle while it is on.** Every
  comparison against `--device` or `--npu` must be run with it off, which is why
  it is an environment variable rather than a default.
- It does **not** break `--prefill-hash` chunk invariance: the four lanes are a
  property of one row's dot product, so the same row produces the same value
  whatever chunk it arrives in. Verified at `--chunk` 8/128/512.
- f16/f32 weights ignore the flag entirely (they already use four lanes, in the
  same order, so they are exact either way).
- 4621/4621 with the flag off, which is the default path being unchanged.

### 5.4 The flag in context: all six arms

The flag is not only a CPU-backend switch. The NPU path takes the transformer
GEMMs the device has opcodes for and leaves the rest on the host, so a third of
its prefill is still `vec_dot`, and the block formats there are the same code.
Measured on the same fixture and prompt, best-of-2 interleaved, one binary:

| arm | prefill | tok/s | decode | vs `--cpu` prefill |
|---|---:|---:|---:|---:|
| `--cpu` | 13657.8 ms | 8.93 | 132.2 ms | 1.00x |
| `--cpu` + fast dot | 4570.2 ms | 26.69 | 48.7 ms | 2.99x |
| `--npu-cpu` | 5916.9 ms | 20.62 | 108.1 ms | 2.31x |
| `--npu-cpu` + fast dot | **3021.1 ms** | **40.38** | 74.6 ms | **4.52x** |
| `--npu-cpu` + `KRK_NPU_W8A8=1` | 3258.2 ms | 37.44 | **31.2 ms** | 4.19x |
| `--npu-cpu` + both | 3363.0 ms | 36.28 | 32.0 ms | 4.06x |

Two things this table settles:

- **The best prefill arm is `--npu-cpu` with the fast dot** (4.52x over the
  default CPU path), and **the best decode arm is w8a8** (31.2 ms/step, 4.2x).
  They are different arms: w8a8 pays a re-quantise of every activation row, which
  is cheap at one row per step and not cheap at 122 rows, so it wins decode and
  loses prefill to the fast dot.
- **w8a8 is not a prefill regression.** It is 1.82x faster than the f32 NPU arm
  on prefill (5916.9 -> 3258.2 ms) despite the extra `QUANTIZE_I8` pass. An
  earlier draft of this work claimed the opposite; the arm had been measured
  against a baseline that inherited `KRK_CPU_DOT_FAST=1` from the shell, so the
  comparison silently gave the fast dot to one side only.

That last point is worth stating as a rule: **an A/B harness must set the
environment of every arm explicitly, including the arms that want a variable
absent.** `env -u` in the harness, not a variable left exported in a terminal.

## 6. What the 2 GiB MoE actually showed about experts and disk

The engine keeps experts in three tiers: HOT (resident, 512 MiB budgeted here),
WARM (host RAM), COLD (the GGUF mapping, i.e. the file). "Only the active
expert in RAM, the rest on disk" is the COLD path, and it works — the run above
is a 2.3 GiB model on a 3.8 GiB box with a 512 MiB expert budget and a
`ram policy: 2.3 GiB of the host room is the file mapping itself` line saying so.

What it costs is **union amplification**, and that is the real answer to "why is
prefill hurt so much more than decode":

| | distinct (layer, expert) reads | share of the 640-pair pool |
|---|---:|---:|
| decode, one step | 80 (4 per layer x 20) | 12.5% |
| prefill, 9 tokens | 225 | 35% |
| prefill, 144 tokens | 278 | 43% |

Decode reads the same 4-of-32 experts per layer every step, so a warm cache
serves it almost for free. Prefill reads a *different union* per layer, and the
union saturates fast: 9 tokens already touch 35% of the pool, because 4 choices
out of 32 per token collide quickly. At a budget that cannot hold the union,
prefill re-reads experts it evicted earlier in the same chunk
(`756 MiB read`, `151/151 slots`, `19.4%` HOT hits at T=128).

So the asymmetry is not a bug in the offload: **decode is a working-set
problem and prefill is a union problem**, and they need different budgets. The
engine's `--expert-cache-mb` is a single knob for both. Sizing it for prefill
means holding ~43% of the pool; sizing it for decode means ~13%.

## 7. Reproducing

```bash
# fixtures (synthetic, seeded; the generator streams so a 2 GiB file does not
# need 2 GiB of Python objects -- see the tool's docstring for the format notes)
python3 tools/make_bench_gguf.py /tmp/fix/dense-1g-f16.gguf --embd 1024 \
    --layers 32 --heads 16 --kv-heads 4 --ff 4096 --vocab 32000 --ctx 2048 \
    --format f16 --scale 0.02
python3 tools/make_bench_gguf.py /tmp/fix/moe-2g-f16.gguf --embd 768 \
    --layers 20 --heads 12 --kv-heads 4 --ff 768 --expert-ff 768 \
    --experts 32 --experts-used 4 --vocab 32000 --ctx 2048 --format f16 \
    --scale 0.02

# the two arms (swap the binary to compare; beat the noise with 5 runs)
P="the quick brown fox jumps over the lazy dog the quick brown fox"
./build-cpu/kraken -m /tmp/fix/moe-2g-f16.gguf --cpu -p "$P" -n 4 --greedy \
    --ctx 2048 --expert-cache-mb 512
```

The correctness witness is not a hash table in this document: it is
`kraken-tests` -- 4576 checks at the time of that change, 4621 after section 5's
opt-in test was added -- of which the numeric golden hashes are the ones that
would move if the dot product had changed.

## 8. Remaining headroom, honestly

- **Compute-bound now.** After the change, dense prefill is 10 tok/s at both
  T=16 and T=64 — 4.8 GMAC/s, close to what four-lane exact-order FMA can
  sustain on one core with two SMT threads (10.4 GMAC/s theoretical, ~0.46 IPC
  achieved). The next step is scheduling, not arithmetic: unrolling the inner
  loop so four FMAs issue per iteration instead of one.
- **Weights are re-read per row.** `gemm()` iterates a flat (row, output) space,
  so a weight row is touched once per row of `x`. Loop inversion alone does not
  help — `vec_dot` re-widens the row on every call — so the win needs the row
  widened once per (output, row-block) into f32 and reused, *with* output
  blocking so the activation block stays in L2. Measured as next, not done here.
- **Block formats** are at their exact-order floor (§5).
- **Two knobs, one budget.** `--expert-cache-mb` serves a decode working set and
  a prefill union with one number. A prefill-shaped budget is a design change,
  not a tuning change.
