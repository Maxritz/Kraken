// ============================================================================
//  moe_route.hpp — group-limited expert selection (DeepSeek-V3 / MiniMax),
//  the one rule every grouped-MoE family in this collection shares.
//
//  The router scores every expert and keeps the top k. That is the plain rule.
//  A grouped model narrows it first: the experts are partitioned into
//  `n_group` CONTIGUOUS blocks, each group is scored by the sum of its two
//  best selection scores, only the `n_group_used` best groups stay eligible,
//  and the top-k then runs over the survivors.
//
//  Three things make this load-bearing rather than decorative:
//
//   * the group score is the sum of the group's TOP TWO, not its best one and
//     not its mean -- so a group holding two strong experts outranks a group
//     holding one that is slightly stronger;
//   * the partition is CONTIGUOUS, `n_expert / n_group` each, with expert e in
//     group `e / (n_expert/n_group)`. A router that grouped by `e % n_group`
//     would select a plausible, disjoint set of experts and decode;
//   * the bias enters the SELECTION score only (the weights are the unbiased
//     probabilities), so this helper takes the already-biased `sel` vector and
//     returns which experts survive -- it never touches the weights.
//
//  Reference: llama.cpp src/llama-graph.cpp ("select top n_group_used expert
//  groups", the DeepSeek-V3 block) and modeling_deepseek.py L440-L457.
// ============================================================================
#pragma once

#include "krk/common.hpp"

namespace krk {

// Marks which experts survive the group rule: `keep[e] = 1` when expert e's
// group is among the `group_used` best groups, 0 otherwise.
//
// `sel` is the per-expert SELECTION score (probabilities plus the routing bias
// where the family has one) for ONE token. Ties are broken by the lower group
// index, which is what an argsort over the groups does.
//
// Returns false (and writes nothing) when the geometry is contradictory:
// n_expert <= 0, n_group <= 0, group_used <= 0, n_group > n_expert, group_used
// > n_group, or n_expert % n_group != 0. The caller refuses in that case
// rather than routing with a partition nobody declared.
bool moe_group_mask(u8 *keep, const f32 *sel, i32 n_expert, i32 n_group,
                    i32 group_used);

// `group_used <= 0` or `n_group <= 1` means "no grouping": every expert is
// kept, and `keep` is all ones. Kept separate so a caller can call it
// unconditionally without an `if` that some future file forgets.
bool moe_group_mask_or_all(u8 *keep, const f32 *sel, i32 n_expert, i32 n_group,
                           i32 group_used);

} // namespace krk
