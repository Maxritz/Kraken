// ============================================================================
//  mla.hpp — multi-head latent attention (DeepSeek-V2/V3/V4): host oracle.
//
//  The K/V a layer attends is not per-head rows but one compressed latent
//  (cKV, kv_lora_rank wide) plus a decoupled RoPE key (qk_rope_head_dim),
//  with the per-head V projection absorbed into the output at load. This
//  header is the arithmetic both the CPU backend and the HIP kernels will be
//  checked against before they are trusted -- the same role src/hc.cpp plays
//  for the hyper-connection stack.
//
//  Two orders, one answer:
//    unfolded  the reference clarity path: explicit per-head K/V built from
//              the cache (c @ Wuk, c @ Wuv), then scores and weighted sums.
//    absorbed  the oracle order: the query is folded through Wuk at load
//              (Wq_abs = Wqb_nope @ Wuk^T per head) and scores the cache
//              directly; only the value side still touches Wuv at runtime.
//  They must agree within f32-association tolerance. A wrong absorb looks
//  like fluent text (same failure shape as the PLE lane scatter), so the
//  suite pins this equivalence, not a text sample.
//
//  The cache is the test's business: c_cache holds [n_keys, kv_lora] latent
//  rows and kr_cache [n_keys, qk_rope] decoupled keys (RoPE already applied --
//  the existing rope_apply covers the rope slice when the engine gets here).
//  No causal mask in v1: the oracle scores every cached key, and masking is
//  the kernel's business to prove separately.
// ============================================================================
#pragma once

#include "krk/common.hpp"

namespace krk {

// Latent geometry of one MLA layer. V2-Lite: 2048 embd, 27 heads, q_lora
// 1536, kv_lora 512, nope 128, rope 64, v 128.
struct MlaGeometry {
    i32 n_embd = 0;
    i32 n_head = 0;
    i32 q_lora = 0;
    i32 kv_lora = 0;
    i32 qk_nope = 0;
    i32 qk_rope = 0;
    i32 v_dim = 0;

    bool valid() const {
        return n_embd > 0 && n_head > 0 && q_lora > 0 && kv_lora > 0 &&
               qk_nope > 0 && qk_rope > 0 && v_dim > 0;
    }
    // One cache row: the latent plus its decoupled RoPE key.
    i64 cache_row_width() const { return static_cast<i64>(kv_lora) + qk_rope; }
};

// Layer weights, row-major [rows, cols], host f32. Wqb packs all heads:
// head h owns columns [h*(nope+rope), h*(nope+rope)+nope+rope), nope first.
// Wuk/Wuv are per-head [kv_lora, nope/v_dim].
struct MlaWeights {
    const f32 *wq_a = nullptr;   // [n_embd, q_lora]
    const f32 *wqa_norm = nullptr; // [q_lora] rms gamma
    const f32 *wq_b = nullptr;   // [q_lora, n_head*(nope+rope)]
    const f32 *wkv_a = nullptr;  // [n_embd, kv_lora]
    const f32 *wkv_norm = nullptr; // [kv_lora] rms gamma
    const f32 *wkr = nullptr;    // [n_embd, qk_rope] decoupled rope key
    const f32 *wuk = nullptr;    // [n_head, kv_lora, qk_nope]
    const f32 *wuv = nullptr;    // [n_head, kv_lora, v_dim]
    const f32 *wo = nullptr;     // [n_head*v_dim, n_embd]
};

// The load-time absorb: Wq_abs[h] = Wqb_nope[h] @ Wuk[h]^T, one
// [q_lora, kv_lora] matrix per head. This fold is what makes the runtime
// order expressible; a wrong fold is the silent-wrong-text defect, so the
// suite checks the folded graph against the unfolded one, not the fold.
void mla_absorb_q(f32 *wq_abs, const f32 *wq_b, const f32 *wuk,
                  const MlaGeometry &g);

// Reference clarity path: explicit per-head K/V, then scores, softmax,
// weighted sums, output projection. out is [n_embd].
void mla_forward_unfolded(f32 *out, const f32 *x, const f32 *c_cache,
                          const f32 *kr_cache, i64 n_keys, const MlaWeights &w,
                          const MlaGeometry &g, f32 scale, f32 eps);

// Oracle order: queries folded at load score the cache directly; only the
// value side reads Wuv at runtime. wq_abs is [n_head, q_lora, kv_lora] as
// built by mla_absorb_q. Must agree with the unfolded path within f32
// tolerance -- and run deterministically (two runs, same bytes).
void mla_forward_absorbed(f32 *out, const f32 *x, const f32 *c_cache,
                          const f32 *kr_cache, i64 n_keys, const MlaWeights &w,
                          const f32 *wq_abs, const MlaGeometry &g, f32 scale,
                          f32 eps);

} // namespace krk
