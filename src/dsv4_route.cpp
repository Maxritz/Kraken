// ============================================================================
//  dsv4_route.cpp — DeepSeek-V4 hash route (tid2eid), host implementation.
//
//  A lookup, an index check and a gather. The checks are the point: a misread
//  table produces in-range ids and a stable route, so the only place a layout
//  mistake can be caught before it reaches the mixer is here.
// ============================================================================
#include "krk/dsv4_route.hpp"

namespace krk {
namespace {

bool route_args_ok(i32 layer, i32 n_hash_layers, const i32 *tid2eid,
                   i32 per_token, i32 n_tokens) {
    if (!tid2eid || per_token <= 0 || n_tokens <= 0) return false;
    if (layer < 0 || layer >= n_hash_layers) return false;
    return true;
}

} // namespace

bool dsv4_hash_route(i32 *out_ids, i32 layer, i32 n_hash_layers,
                     const i32 *tid2eid, i32 per_token, i32 n_tokens,
                     i32 token, i32 n_expert) {
    if (!out_ids) return false;
    if (!route_args_ok(layer, n_hash_layers, tid2eid, per_token, n_tokens))
        return false;
    if (token < 0 || token >= n_tokens) return false;
    if (n_expert <= 0) return false;
    // TOKEN-major: the token selects the row, the slot runs down it.
    const i32 *row = tid2eid + static_cast<i64>(token) * per_token;
    for (i32 s = 0; s < per_token; s++) {
        if (row[s] < 0 || row[s] >= n_expert) return false;
    }
    // Written only once every id is known good, so a refused call cannot leave
    // half a route behind for a caller that ignores the return value.
    for (i32 s = 0; s < per_token; s++) out_ids[s] = row[s];
    return true;
}

bool dsv4_hash_gather(f32 *out, const i32 *ids, i32 k, const f32 *scores,
                      i32 n_expert) {
    if (!out || !ids || !scores || k <= 0 || n_expert <= 0) return false;
    for (i32 s = 0; s < k; s++)
        if (ids[s] < 0 || ids[s] >= n_expert) return false;
    for (i32 s = 0; s < k; s++) out[s] = scores[ids[s]];
    return true;
}

} // namespace krk
