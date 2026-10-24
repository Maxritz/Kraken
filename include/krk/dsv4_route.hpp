// ============================================================================
//  dsv4_route.hpp — DeepSeek-V4's hash route (first N layers), host oracle.
//
//  The first `hash_layer_count` layers of a V4 file do not pick their experts
//  from the router at all: `blk.N.ffn_gate_tid2eid.weight` names them per
//  TOKEN ID. The tensor is int32 DATA, not a weight -- one column per token id,
//  `n_expert_used` ids down each column:
//
//      ids[slot] = tid2eid[token * per_token + slot]
//
//  GGUF's ne[0] is the fastest axis and the loader reads the flat array as the
//  file lays it out, so the stride is TOKEN-major. A column-major read is the
//  defect this header's gate exists for: it is in range, it is stable, and it
//  sends every token to another token's experts -- plausible output, wrong
//  model. Nothing downstream can detect that, because the table IS the ground
//  truth for the route.
//
//  What this oracle deliberately does NOT do: decide the mixing weights. The
//  router still runs on a hash layer (`blk.N.ffn_gate_inp.weight` is present in
//  the file) and the weights are the selected experts' scores through whatever
//  gating the layer declares (`expert_gating_func`, `expert_weights_scale`,
//  `expert_weights_norm`). Those transforms already live in the engine's
//  routing path; duplicating them here would be a second definition of the
//  same thing, which is how the two drift. `dsv4_hash_gather` only reads the
//  scores of the ids the table named.
//
//  Wiring note: `Engine::moe_ffn` calls `dsv4_hash_route` / `dsv4_hash_gather`
//  for `layer < cfg.dsv4_hash_layers` and skips the top-k selection there.
//  Wired in `src/engine.cpp` (Engine::moe_ffn). Today it is reachable-but-inactive
//  because `deepseek4` still refuses at `Engine::init` (`ArchSpec::attention_implemented`
//  is false), so the call site only runs when that verdict flips. The GPU kernels come
//  later.
// ============================================================================
#pragma once

#include "krk/common.hpp"

namespace krk {

// The hash route of one layer for one token.
//
// Returns false, writing nothing, when the request cannot be honoured:
//   * `layer` is at or past `n_hash_layers` (that layer routes by gate),
//   * the table is empty / `per_token <= 0` / `n_tokens <= 0`,
//   * `token` is outside [0, n_tokens),
//   * any selected id is outside [0, n_expert) -- a corrupt or misread table.
//
// `out_ids` must hold `per_token` entries. Duplicate ids are PRESERVED: the
// table is authority, and a route that names an expert twice is what the file
// says, not something to deduplicate.
bool dsv4_hash_route(i32 *out_ids, i32 layer, i32 n_hash_layers,
                     const i32 *tid2eid, i32 per_token, i32 n_tokens,
                     i32 token, i32 n_expert);

// The router scores of the experts a hash route selected, in TABLE order
// (`out[j] = scores[ids[j]]`), so the caller can mix them with the same
// arithmetic the other layers use. Returns false on a bad index.
//
// Order matters and is pinned by the suite: a caller that gathers by sorted id
// instead of by slot pairs expert A's weight with expert B's output -- finite,
// plausible, wrong. `scores` is the layer's router row, already transformed by
// whatever gating the layer declares.
bool dsv4_hash_gather(f32 *out, const i32 *ids, i32 k, const f32 *scores,
                      i32 n_expert);

} // namespace krk
