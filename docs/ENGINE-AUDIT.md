# Engine deep dive: workspaces, and the null-pointer class

A feature-gated workspace that another family reaches is a **crash**, not a
wrong answer — and it only shows up on the family that arrives later. One of
those arrived with `qwen2moe` (below). This page records the audit that finds
them, so the next one is caught before it ships.

## The audit

`include/krk/engine.hpp` declares every scratch buffer; `src/engine.cpp`
allocates them inside `init()`. A buffer is safe only if **every** use is
reachable only when its allocation ran. The script below prints, per buffer,
the guard around its allocation and the functions that use it:

```python
members = set(re.findall(r'void \*\s*(ws_[a-z0-9_]+)\s*=', engine_hpp))
for each `ws_X_ = alloc(...)` (or `halloc(ws_X_, ...)`) in engine_cpp:
    walk up to the enclosing `if` -> guard[ws_X_]
for each member:
    print(member, guard, sorted(functions that mention it))
```

73 workspaces, 44 of them behind a feature guard. The table it produces is the
checklist; the review rule is one line each:

> **A buffer's uses must be inside its own guard, or inside a caller whose
> precondition implies it.**

## What it found

### 1. `ws_ssm_` was the shared expert's gate — and only exists on a hybrid (fixed)

```cpp
// before
if (L.shexp_inp_gate.present()) {
    gemm_w(ws_ssm_, ws_xn_, L.shexp_inp_gate, 1, n_embd_, n);   // NULL on qwen2moe
    be_->sigmoid_act(ws_ssm_, n);
    be_->scale_rows(ws_x2_, ws_ssm_, n, n_embd_);
}
```

`ws_ssm_` is allocated under `if (model_.is_recurrent())`. The family that wrote
that path — Qwen3.5 — *is* recurrent, so the gate worked there. `qwen2moe` is a
dense-MoE stack with the same 1-D `ffn_gate_inp_shexp` tensor, so its **first
forward dereferenced null** (reproduced: SIGSEGV right after `workspaces ready`,
before any token was written).

Fix: a dedicated `ws_shexp_g_` (`[chunk]`, f32) allocated beside `ws_ffn_`
whenever the model has routed experts, released in the same list as the other
MoE buffers. `test_qwen2moe` now drives the gate to both extremes, so the arm
fails if the gate is dropped rather than crashing quietly.

### 2. `mtp_ffn` on a dense model (already handled — the shape to copy)

The same class, caught earlier: an MTP block on a model with no routed experts
has no `ws_ffn_`, and the block used to `memset(NULL)` — `hipErrorInvalidValue`
before the first token. The code now says so and degrades to the residual:

```cpp
void *acc = ws_ffn_ ? ws_ffn_ : ws_x_;
i32  *exp = static_cast<i32 *>(ws_ffn_ ? ws_fexp_ : mtp_rep_);
```

### 3. Workspaces with no allocation site in `engine.cpp` — checked, they use `halloc`

Eleven buffers (`ws_hc_*`, `ws_ple_*`) have no `= alloc(` line: they go through
the `halloc(member, bytes, name)` lambda, which allocates and names in one call.
The audit script matches both spellings; a grep for `= alloc(` alone reports
them as "never allocated" and is the reason the first pass of this audit was
wrong.

## The rest of the table, reviewed

| buffer | guard | uses | verdict |
|---|---|---|---|
| `ws_had_` | `model_.had().active` | `dense_ffn`, `gdn_forward`, `head_compute`, `mamba2_forward`, `mtp_probe_step`, `prefill_trace_hash` | every use sits behind an `L.h_*` flag, and the loader sets those only when `had().active` |
| `ws_xn2_` | `mc.parallel_two_norm` | `plain_forward` | use is inside the same flag |
| `ws_gs_` | `model_.max_scaled_in() > 0` | `gemm_w`, `gemm_residual`, `mamba2_forward` | the loader records `max_scaled_in_` from the same `gs` it later tests |
| `ws_mla_`, `ws_mlaqn_`, `ws_mlaa_`, `ws_mlay_` | `mc.mla`, `mc.mla_q_lora > 0`, `mc.mla_layout == Shared` | `mla_forward`, `mla_shared_forward` | both functions are only called for an MLA layer, and the layout flag picks the function |
| `ws_qsa_*` | `qsa_active_` | `qsa_select_block`, `qsa_attention` | both called only when `qsa_active_` |
| `ws_ppl_*` (gemma4) | `n_embd_per_layer > 0 && ple_tok_embd.present()` | `forward_core` | use is inside `if (ppl_width_ > 0)`, and `ppl_width_` is set in the same block that allocates |
| `ws_agate_` | `is_recurrent() \|\| mc.attn_gate` | `forward_core` | use is inside `if (mc.attn_gate && …)` |
| `ws_qkv_`, `ws_z_`, `ws_h_`, `ws_beta_`, `ws_ssm_` | recurrent branches | `gdn_forward`, `mamba2_forward`, `lfm_conv_forward`, `recurrent_reset` | all four functions are dispatched from `model_.is_recurrent()` |
| `ws_ebatch_`, `ws_edown_`, `ws_ebscale_`, `ws_plan_`, `ws_alpha_`, `ws_scale_`, `ws_xg_`, `ws_upg_`, `ws_gateg_`, `ws_router_`, `ws_ids_`, `ws_wts_`, `ws_off_`, `ws_ffn_`, `ws_shexp_g_` | `mc.is_moe && mc.n_expert > 0` | MoE paths only | `moe_ffn`/`moe_decode_batch` are called for `L.moe` layers, and `ws_ffn_`'s one non-MoE reader (`mtp_ffn`) is null-checked |
| `ws_xlat_`, `ws_xlat_out_`, `ws_ffn_lat_` | `lat_w > 0` | `moe_ffn` | `latent` is computed from the same per-layer latency test |
| `ws_m2in_` | `mc.nemotron_moe` | `init` only | no forward use |
| `ws_qpack_` | `q_proj_ != q_dim_` | `forward_core`, `mtp_block_forward` | both go through the same `q_proj_` handling |

## Method for the next family

1. Add the arch entry, then a fixture, then run it — the crash, if any, is
   immediate and reproducible (`--run-group`, then the group's scratch dir keeps
   the fixture for the CLI).
2. If the family reaches a new piece of shared state (a gate, a scalar, a
   norm), check whether the buffer it lands in is allocated for **this** shape
   of model, not for the family the code was written for.
3. Give the new piece its own A/B arm. "It loads and generates" cannot see a
   dropped gate, a missing norm or a zeroed branch — the arms in
   `test_qwen2moe`, `test_lfm2_dense` and `test_gemma4_forward` each move a
   number in one direction only.
