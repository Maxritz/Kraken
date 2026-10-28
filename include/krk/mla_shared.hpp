// ============================================================================
//  mla_shared.hpp — DeepSeek-V4 shared-latent MLA (MlaLayout::Shared): oracle.
//
//  This is NOT the V2/V3 layout that already runs (mla.hpp). V4 keeps ONE
//  latent row per token and uses it as both K and V for every head:
//
//      q_a = rms_norm(x @ Wq_a, q_a_norm)          [q_lora]
//      q_h = Wq_b row-slice for head h             [kv_lora]  (nope | rope)
//      c   = rms_norm(x @ Wkv,  kv_a_norm)         [kv_lora]  <- the cache row
//      K_h = rope_tail(copy of c, pos, 64)         (the tail is rotary)
//      V   = c                                     (unrotated: see the note)
//      s_h = q_h . K_h * scale                     (+ sink_h, denominator only)
//      a_h = sum_j softmax(s_h)_j * c_j            [kv_lora]
//      y   = concat_g Woa_g @ concat_{h in g}(a_h) [o_groups * o_lora]
//      out = Wob @ y                               [n_embd]
//
//  Geometry is the file's own: kv_lora == key_length == value_length (512 on
//  DeepSeek-V4-Flash), qk_rope == rope.dimension_count (64), q_lora ==
//  attention.q_lora_rank (1024), output_group_count 8, output_lora_rank 1024.
//  Model::load validates exactly these (`attn_kv` [n_embd, key_length],
//  `attn_output_a` [n_head*key_length/o_groups, o_lora*o_groups],
//  `attn_output_b` [o_lora*o_groups, n_embd]) -- see src/model.cpp, the
//  "V4: ONE shared latent, a GROUPED output projection" branch, so this
//  header cannot invent a shape the loader would have refused.
//
//  **The one assumption this oracle makes that no local file can confirm:**
//  V is the UNROTATED latent. K is a rope-rotated copy of the same row, so the
//  cache stores the normed latent and rotation happens on read -- the same
//  split the lite layout uses for its decoupled key, and the reason the audit
//  calls this "kraken's existing GQA path with n_kv=1, hd=512". The file
//  cannot tell us whether the reference rotates the value as well (one tensor
//  feeds both), and the difference is exactly the class of bug that reads as
//  fluent text. It is pinned here, in one named function, so a reference
//  comparison has one place to look and a wrong answer is one edit.
//
//  Two orders, one answer -- the same discipline as mla.hpp:
//    accum   per-key: softmax per head, then an axpy of that key's latent
//    gather  weights first, then one pass over the cache per head
//  They differ only in f32 association, so the suite can pin the graph
//  without a reference model. Determinism is required of both.
// ============================================================================
#pragma once

#include "krk/common.hpp"

namespace krk {

// One V4 shared-latent layer. `kv_lora` is the whole latent (nope + rope):
// unlike mla.hpp there is no separate decoupled-key width, because the rope
// lives in the TAIL of the same row.
struct MlaSharedGeometry {
    i32 n_embd = 0;
    i32 n_head = 0;
    i32 q_lora = 0;
    i32 kv_lora = 0;   // key_length == value_length == head_dim
    i32 qk_rope = 0;   // rotary tail of the latent
    i32 o_groups = 0;
    i32 o_lora = 0;

    i32 qk_nope() const { return kv_lora - qk_rope; }
    i32 heads_per_group() const { return o_groups > 0 ? n_head / o_groups : 0; }
    // One cache row: K and V are the same latent, so there is no second width.
    i64 cache_row_width() const { return kv_lora; }
    bool valid() const {
        return n_embd > 0 && n_head > 0 && q_lora > 0 && kv_lora > 0 &&
               qk_rope > 0 && qk_rope < kv_lora && o_groups > 0 &&
               o_lora > 0 && n_head % o_groups == 0;
    }
};

// Layer weights, row-major [rows=in, cols=out] -- the same orientation
// mla.hpp uses, so `proj` means the same thing in both files.
// Woa is the grouped output: output column block g of width o_lora reads the
// concatenated heads of group g (heads_per_group * kv_lora rows). It is one
// [in, o_groups*o_lora] matrix in the file, which is why a wrong group stride
// is invisible in the shapes and must be pinned by the suite.
struct MlaSharedWeights {
    const f32 *wq_a = nullptr;       // [n_embd, q_lora]
    const f32 *wq_a_norm = nullptr;  // [q_lora]
    const f32 *wq_b = nullptr;       // [q_lora, n_head * kv_lora]
    const f32 *wkv = nullptr;        // [n_embd, kv_lora]
    const f32 *wkv_norm = nullptr;   // [kv_lora]
    const f32 *wo_a = nullptr;       // [heads_per_group*kv_lora, o_groups*o_lora]
    const f32 *wo_b = nullptr;       // [o_groups*o_lora, n_embd]
    // Per-head attention-sink logit, added to the softmax DENOMINATOR only
    // (ggml_soft_max_add_sinks): it can take probability mass away from every
    // key and is never attended. Null means no sink, which is not the same as
    // a zero sink on a file that has one.
    const f32 *sinks = nullptr;
};

// The cache row itself: c = rms_norm(x @ Wkv, kv_a_norm). Unrotated on
// purpose -- K is produced by rotating a copy on read, V is the row as it is.
// The write path and the suite both go through here so there is one definition
// of "the latent this layer caches".
void mla_shared_latent(f32 *c, const f32 *x, const MlaSharedWeights &w,
                       const MlaSharedGeometry &g, f32 eps);

// Consecutive-pair (NORM) rotation of the tail `n_rot` of a `d`-wide row, in
// place. V4's rope is the TAIL of the latent (offset d - n_rot), not a
// leading slice, and the offset is the one thing a wrong implementation gets
// silently wrong: it rotates real content and leaves the rotary dims alone.
void mla_shared_rope_tail(f32 *v, i64 d, i64 n_rot, i64 pos,
                          const f32 *inv_freq);

// Order A: per-key accumulation. `c_cache` is [n_keys, kv_lora] of NORMED,
// UNROTATED latent rows; `kpos[t]` is the position key t was written at (V4
// layers are windowed, so position is not the array index). `window` <= 0
// means every cached key; otherwise only keys with
// kpos[t] > q_pos - window take part, which is the SWA rule the file's
// `attention.sliding_window` asks for. `sinks` (when present) is per head.
void mla_shared_forward_accum(f32 *out, const f32 *x, const f32 *c_cache,
                              const i64 *kpos, i64 n_keys, i64 q_pos,
                              const MlaSharedWeights &w,
                              const MlaSharedGeometry &g, const f32 *inv_freq,
                              f32 scale, f32 eps, i32 window);

// A COMPRESSED layer (DeepSeek-V4 CSA/HCA): the cache holds two kinds of row.
// Rows [0, n_raw) are raw latents, exactly what the calls above take -- windowed,
// roped on read with the layer's attention table. Rows [n_raw, n_keys) are
// pooled keys from dsv4_comp_pool(): normed, unroped, ALWAYS visible (a block
// covers positions the window has already passed), and roped on read with
// `compress_inv_freq` -- the file's `attention.compress_rope_base`, a different
// table from the attention one. `kpos[t]` carries each row's position: the
// token position for a raw row, the block's first token for a pooled one.
//
// For CSA the caller passes only the pooled rows the indexer selected; there is
// no block mask here, because a compacted array is the same answer without a
// second definition of "visible".
void mla_shared_forward_compressed(f32 *out, const f32 *x, const f32 *c_cache,
                                  const i64 *kpos, i64 n_raw, i64 n_keys,
                                  i64 q_pos, const MlaSharedWeights &w,
                                  const MlaSharedGeometry &g,
                                  const f32 *inv_freq,
                                  const f32 *compress_inv_freq, f32 scale,
                                  f32 eps, i32 window);

// Order B: weights first, then one pass over the cache per head. Same inputs,
// same answer within f32 association. This is the order a kernel wants (the
// weights are a [n_head, n_keys] tile it can keep in LDS), so it is checked
// against order A rather than assumed equal to it.
void mla_shared_forward_gather(f32 *out, const f32 *x, const f32 *c_cache,
                               const i64 *kpos, i64 n_keys, i64 q_pos,
                               const MlaSharedWeights &w,
                               const MlaSharedGeometry &g, const f32 *inv_freq,
                               f32 scale, f32 eps, i32 window);

} // namespace krk
