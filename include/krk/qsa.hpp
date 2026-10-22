// ============================================================================
//  qsa.hpp — qwen4exp's QSA block indexer, host side.
//
//  A full-attention layer of this family does not attend to every key. Every
//  `ratio` cells (the layer's own `compress_ratios` entry) form one BLOCK; the
//  indexer scores each COMPLETE block once, and the layer then attends densely
//  to the cells of the best few blocks -- plus the tail block, which is still
//  filling and is therefore always visible.
//
//      pooled[b]  = mean over the block's `ratio` raw indexer keys
//                   (pooling precedes norm and rotation, so the keys pooled
//                   here are the projection output, untouched)
//      pooled[b]  = rope(rmsnorm(pooled[b]), position b*ratio)
//      q[t]       = rope(rmsnorm(index_q(x[t])), position pos0+t)    per head
//      score[t][b]= sum_h relu(q[t][h] . pooled[b])
//      cells[t]   = every cell of the top blocks, by score, until
//                   min(n_kv, top_k + ratio - 1) cells are chosen,
//                   with the block that is still filling pinned visible
//
//  The relu is per head and before the sum (DeepSeek's lightning indexer; the
//  reference this was read from is llama.cpp's src/models/qwen4exp.cpp —
//  build_qsa_top_k — and llama-memory-hybrid-idx.cpp's set_input_qsa, which
//  builds the block list, the block positions and the visibility bias).
//
//  Two properties are easy to get wrong and are pinned by the tests:
//  * only COMPLETE blocks are pooled. A block with fewer than `ratio` cells
//    has no mean yet; its cells are the tail and are selected by the
//    visibility rule, never by a score.
//  * the block containing the query is pinned visible only while it is
//    INCOMPLETE (its last cell position is > q). A complete block, including
//    the one the query sits in, competes on its score like every other.
// ============================================================================
#pragma once

#include "krk/common.hpp"

#include <vector>

namespace krk {

// One layer's indexer geometry.
struct QsaConfig {
    i32 n_head = 0;    // indexer.head_count, 4
    i32 head_dim = 0;  // indexer.key_length, 128
    i32 top_k = 0;     // indexer.top_k, 2048 cells
    i32 ratio = 0;     // this layer's compress ratio, >= 1
    f32 eps = 1e-6f;   // attention.layer_norm_rms_epsilon
    // rotation applied to both the pooled keys and the queries: the model's
    // own rotary width, its table and its pairing convention. `frac` is
    // n_rot / head_dim, so the indexer rotates the same leading channels its
    // attention layer does.
    i32 n_rot = 0;
    const f32 *inv_freq = nullptr;
    f32 rope_scale = 1.0f;
    bool rope_neox = false;

    // Cell budget: the reference keeps `top_k + ratio - 1` cells, i.e. whole
    // blocks plus the tail. Below this many cells nothing is dropped and the
    // layer runs plain dense causal attention.
    i64 budget() const { return static_cast<i64>(top_k) + ratio - 1; }
    bool valid() const {
        return n_head > 0 && head_dim > 0 && top_k > 0 && ratio > 0;
    }
};

// Mean-pools the raw indexer keys into one vector per COMPLETE block.
//
// `keys` is [n_kv, head_dim] row-major in position order; `out` is
// [n_blocks, head_dim] with `n_blocks = n_kv / ratio` (a trailing partial block
// is not pooled). Returns the number of pooled blocks.
i32 qsa_pool(f32 *out, const f32 *keys, i32 head_dim, i32 n_kv, i32 ratio);

// In-place rmsnorm (per row, `head_dim` wide, weight `w`) followed by rope at
// the row's own position: `pos[r]` for r in [0, n_rows). One row per head, so
// a caller norming n_tok rows of n_head heads passes n_tok*n_head rows and a
// per-row position table.
void qsa_norm_rope(f32 *x, const f32 *w, i32 n_rows, i32 head_dim,
                   const i64 *pos, const QsaConfig &cfg);

// score[t*n_blocks + b] = sum_h relu(q[t][h] . pooled[b]), q is
// [n_tok, n_head, head_dim]. Pooled and q are both already normed and rotated.
void qsa_scores(f32 *score, const f32 *q, const f32 *pooled, i32 n_tok,
                i32 n_head, i32 n_blocks, i32 head_dim);

// One query row's cell selection, ascending.
//
// `n_kv` cells exist (positions 0..n_kv-1) and row t's query sits at
// `pos0 + t`; only cells at or before it can be selected (causal).
// `score` is that row's [n_blocks] scores. Writes the chosen cell indices into
// `cells` (cleared first) and returns how many.
i32 qsa_select(std::vector<i64> *cells, const f32 *score, i32 n_blocks,
               i32 n_kv, i64 qpos, const QsaConfig &cfg);

} // namespace krk
