# Adding a model: where it can go wrong

Every failure below was hit on a real model in `docs/model-inventory.md`, and
each one is written as a *symptom* first, because that is what you see. The
point of this file is to turn "the output is garbage" into a short list of
places to look, in the order the loader and the forward pass touch them.

Companion files: `docs/PERF-PLAN.md` (perf), `docs/TODO.md` (open bugs),
`docs/ARCHITECTURE.md` (how the pieces fit).

---

## 0. The one-command triage

Before anything else:

```
kraken --model M.gguf --info          # arch, geometry, verdict, tied embd, qk norm
KRK_N=16 KRK_CHAT=1 bash scripts/coherence_check.sh M.gguf   # device vs CPU reference
```

`--info` answers "is this even supported, and with what geometry" without
running a forward pass. `coherence_check.sh` runs the model twice — once on the
9070 XT, once with `--cpu` — and diffs the generated text. When they disagree it
reports `drift step N (idA vs idB)`, which is the first step where the two
backends picked different tokens. That step index is the single most useful
number in this whole document.

**A FAIL is not automatically a bug in kraken.** See §7.

---

## 1. Loader and geometry

| Symptom | Cause | Where |
|---|---|---|
| `cannot run this file` + blockers | Arch not implemented, or a tensor names an unimplemented piece | `src/verdict.hpp` |
| `architecture 'X': ... (known: ...)` warning, model still runs | Unknown llama-schema variant | `verdict.hpp` notes |
| Loads, then decodes nonsense | Geometry misread (head count, head dim, rope base) | `Model::load`, `src/model.cpp` |
| Missing `blk.N.ssm_*` | MTP blocks counted as layers | `cfg_.n_layer_nextn` |

Two geometry traps that have already cost time:

- **`head_dim` is not always `n_embd / n_head`.** Qwen3.5-0.8B is `n_embd=1024,
  n_head=8` → 128 by division, but the real head width is **256**. It comes from
  `.attention.key_length`. Always prefer that key; the division is only a
  fallback.
- **MTP / next-token-prediction blocks** (`nextn_predict_layers`) are appended to
  `block_count` but carry no SSM tensors. Loading them as layers fails on the
  missing delta-net weights. `verdict.mtp_blocks` is the authority.

---

## 2. Quantisation and the two backends

The **CPU reference dequantizes every format it accepts, in f32, on the host**
(`dequant_row`). The **device keeps weights packed** and dequantizes inside the
kernels, and its activations are fp16. That asymmetry is the source of most
"device disagrees with reference" reports, and it is worth stating plainly:

- Both backends read the *same* file bytes. A quantisation bug would break both
  identically, so **a device-vs-CPU difference is never a quantisation-format
  bug** — it is arithmetic (fp16 vs f32) or a kernel indexing bug.
- `dtype_supported` / `dtype_row_aligned` are checked at load; a format the
  build cannot handle is reported by name, not run.
- `tests/test_kraken.cpp` validates each dequantizer **against a synthetic
  block**. Whole-tensor device-vs-host agreement is *not* covered; see §7.

**DP4A is off for Q6_K** (`kDp4aEnableQ6K`, `src/hip/kernels/gemm_dp4a.hpp`), and
`dp4a_ok()` also rejects `rows < 16`, so decode (rows=1) never takes it. A new
format needs a `dp4a_kind()` entry and a decision on that flag before it can use
the int8 path.

---

## 3. Attention

Head width is the thing that decides which kernel runs.

| Path | Accepts | Notes |
|---|---|---|
| `attention_decode` (split-K) | hd 64, 128, 256 | `attention_decode_splits()` scales splits with `cu_count` |
| `attention_qtile` | **BROKEN — opt-in only** | `KRK_ATTN_QTILE=1`; wrong output, see PERF-PLAN B6 |
| `attention_kernel` (tiled) | anything | the safe fallback, and the current prefill default |
| `attn_fused_chain` | hd 64, 128 | hd=256 models cannot use the fused decode chain |

Adding a model with a **new head width** is the most likely way to land on a
path that has never been exercised. Check `attention_decode_supported()` and
`attn_fused_decode_supported()` before assuming the fast path is taken — a
silent fallback to the tiled kernel is *correct but slow*, and shows up as
"decode collapses as context grows" (see PERF-PLAN B5).

**GQA ratio**: `hkv = h / n_rep` with `n_rep = n_head / n_head_kv`. Spark_one is
16/2 = 8:1. The tiled kernel's `n_rep > 0` guard exists because a malformed
`n_head_kv` of 0 would otherwise divide by zero.

**qk_norm**: Qwen3.5 sets it, which disqualifies the fused chain. It is read
from the presence of `attn_q_norm.weight` / `attn_k_norm.weight`, not from a
metadata flag — a model that ships those tensors gets the norm applied whether
or not the arch string says so.

---

## 4. RoPE

- `rope.freq_base` matters. Qwen2.5-Coder-3B and Spark_one both use **1e6**,
  not the 1e4 default.
- `rope.scaling.type = none` must reset `rope_scale` to 1.0; YaRN is only
  approximated by the linear factor (`src/rope.cpp`), which is a known gap.
- **Partial rotary** (`rope_dim < head_dim`) rotates only the first `rope_dim`
  dims and leaves the tail. The table is `rope_dim/2` long and the fraction is
  passed to the rotation op. Getting this wrong produces text that is fluent for
  a few tokens and then degrades.

---

## 5. MoE

| Symptom | Cause |
|---|---|
| Very slow decode, hit rate ~0% | Expert residency budget too small — see PERF-PLAN B3 |
| `total_expert_bytes()` understating | It must multiply by `n_expert`, not count one |
| Routers pick nonsense | Running the routers with expert FFNs stubbed changes the routing decision (refuted: mean TV 0.83, worse than random) |

`is_moe` is confirmed only when a layer actually exposes expert tensors, so a
checkpoint that merely declares `expert_count` cannot misroute the FFN.

---

## 6. Perf traps specific to new models

| Trap | Why |
|---|---|
| **First large `hipMalloc` costs ~165 ms** | One-time GPU/driver init, paid by whichever allocation is first. It lands on `kcache_` in `Engine::init`. Do not "fix" it by allocating early — measured a wash. |
| **Page faults on the file mapping** | Reading a mmap'd model pays a soft fault per 4 KiB page: 1.2M faults / 476 ms on the 8B. Touching the range on a helper thread while the DMA runs: **1042 → 642 ms**. |
| **Copy count, not copy size, sets the rate** | 4 KiB copies run at 0.04 GB/s, 16 MiB at 10.6 GB/s. Many small tensors (SmolLM2 has ~270) load at 1.58 GB/s even though the model is tiny. |
| **Per-layer loop is 66% of start-to-end** | If a model has unusually many layers, load dominates everything. |
| `--ctx` drives KV allocation | KV = `n_layer × ctx × kv_dim × 2 bytes`. On the 8B at ctx 4096 that is 576 MiB, allocated up front. Use `--ctx 256/512` on small machines. |

---

## 7. When device and CPU disagree: is it a bug?

Work through this before filing anything. All of these were measured on
Spark_one (Qwen2.5-3B, qwen2, Q6_K), the one model in the corpus that fails the
coherence gate.

1. **Deterministic?** Run the GPU 3× and the CPU 2× and hash the text. If either
   varies, it is a race — stop and look for one.
2. **Path-dependent?** Force each rung: `KRK_GEMM_PATH=wmma|simt|dp4a`,
   `KRK_ATTN_SPLIT=0`, `KRK_ATTN_QTILE=0`. If the failure is bit-identical across
   three independent GEMM implementations, it is not the GEMM.
3. **Model-blind?** Try 3–5 different prompts. A kernel or indexing bug fails on
   all of them; fp16-vs-f32 rounding fails on some.
4. **Where does it first diverge?** `KRK_DUMP=a.txt` / `b.txt` then
   `python tools/dump_diff.py a.txt b.txt`. The **first** flagged stage is the
   real one; everything after is cascade.
5. **Is the error in the normal band?** Per-forward end-of-model relative error
   in the 2e-3 – 1.3e-2 band is normal fp16 activation rounding. An *abrupt*
   10× jump between two consecutive forwards is not.

Spark_one: deterministic, bit-identical across all three GEMM paths, 3 of 5
prompts pass, flat ~3e-3 for prefill + 11 decode steps then a jump. That is the
signature of fp16-vs-f32 sensitivity at a close call, not a defect — which is
why the gate was left strict rather than loosened.

**The open hole:** whole-tensor device-vs-host Q6_K dequant agreement is
unverified (the suite only checks a synthetic block). That is the next
discriminating experiment.

---

## 8. Checklist for a new model

1. `kraken --model M --info` — arch supported, geometry sane, tied embd, qk norm.
2. `coherence_check.sh M` — device vs CPU reference.
3. If it fails: §7 before §3.
4. `kraken --model M --bench` — record prefill/decode against the table in
   README so a regression has a baseline.
5. `kraken --model M --profile` on a decode step — confirm the fast kernels are
   actually selected, not silently falling back.
6. MoE only: check the expert hit rate in `--info` and size `--expert-cache-mb`.
