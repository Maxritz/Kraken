// ============================================================================
//  msa.hpp -- MiniMax Sparse Attention: which KV blocks a query block sees.
//
//  MSA's defining property is what it does NOT keep. An indexer that always
//  attends to the first N keys plus the local window is dense-with-a-hole; MSA
//  is block-sparse with **no dense prefix** -- a block that is not selected is
//  not read, including the blocks at the very start of the context. That is why
//  this lives beside the attention kernel rather than inside a router: the
//  selection is a property of the attention, and a wrong selection silently
//  changes what the model can see.
//
//  The selection, per query block:
//
//    1. Pool every KV block to one representative vector (mean of its keys, the
//       same pooling the QSA indexer uses -- `qsa_pool`).
//    2. Score the query block's pooled query against every KV block.
//    3. Keep the `budget` highest-scoring blocks, plus the local block, minus
//       any block that is entirely in the future (causal) or outside the
//       window.
//
//  Two facts are checked rather than assumed, and both are in the tests:
//   * budget >= n_blocks selects everything and the result is EXACTLY the dense
//     attention, bit for bit -- so the sparse path cannot be a different model
//     when it is told to be dense;
//   * every selected set contains the local block, so a query can always see
//     the tokens it is sitting on.
// ============================================================================
#ifndef KRK_MSA_HPP
#define KRK_MSA_HPP

#include <vector>

#include "krk/common.hpp"

namespace krk {

// Pool `n_kv` key rows of width `head_dim` into one row per `block` keys.
// The trailing partial block is pooled over the keys it has (not padded), so a
// context that is not a whole number of blocks still scores correctly.
// `out` must hold ceil(n_kv / block) * head_dim floats.
i32 msa_pool(f32 *out, const f32 *keys, i32 head_dim, i32 n_kv, i32 block);

// Select up to `budget` KV blocks for one query block.
//
//   pooled      ceil(n_kv/block) x head_dim, from msa_pool()
//   q           the query block's pooled row, head_dim wide
//   n_kv        number of keys in the cache
//   n_qpos      the position of the query block's last row (the causality bound:
//               a block is visible when its first key is <= n_qpos)
//   block       keys per block
//   budget      maximum blocks to return; <= 0 means "all visible blocks"
//   out         at least (n_kv/block + 1) entries, written in ASCENDING block
//               order so the run list they turn into is canonical
//
// Returns the number of blocks written, or 0 when nothing is visible.
i32 msa_select(i64 *out, const f32 *pooled, const f32 *q, i32 head_dim,
               i32 n_kv, i32 n_qpos, i32 block, i32 budget);

// Turn a block list into the half-open key runs `AttnDesc::key_runs` takes.
// Adjacent blocks merge into one run, so a selected prefix costs one run, not
// one per block. Returns the number of runs written (out holds 2 * n_blocks
// i64s, which is always enough).
i32 msa_runs(i64 *out, const i64 *blocks, i32 n_blocks, i32 block, i32 n_kv);

} // namespace krk

#endif // KRK_MSA_HPP
