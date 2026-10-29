// ============================================================================
//  engram.hpp — DeepSeek-V4.1's Engram: conditional memory addressed by an
//  n-gram hash.
//
//  Engram is not an expert and not PLE. Each Engram layer hashes the last
//  `n_gram` tokens into `n_heads` row ids per token, gathers those rows from a
//  table SHARED by every Engram layer (`per_layer_tok_embd`), projects the
//  gathered block through `engram_kv`, and adds a gated copy of the value half
//  into the lane stack. The hash is the whole interface:
//
//      toks[0] = token_map[t]                       the token itself
//      toks[s] = token_map[prev[t - s]]             s = 1..n_gram-1
//      rolling = toks[0] * mult[0]
//      for o = 1..n_gram-1:
//          rolling ^= toks[o] * mult[o]             ROLLING, not re-summed
//          for h in 0..heads_per_ngram-1:
//              col = (o-1) * heads_per_ngram + h
//              row[col] = rolling % vocab_sizes[col] + offsets[col]
//
//  Four things here are load-bearing and each is a silent-wrong-answer if it is
//  missed, which is why the hash lives in its own file with its own tests:
//
//  * ROLLING. `rolling` accumulates across `o`; the id for column `o` hashes
//    every token from 0..o, and each new token is XOR-ed into the running
//    value. Recomputing per column from scratch gives n_gram-1 unrelated ids
//    that are still in range.
//  * the table is `per_layer_tok_embd`, shared across layers, and `engram_kv`
//    is per layer. A per-layer table is the obvious wrong reading and it is
//    indistinguishable by shape, since both are [ple_head_dim, n_rows].
//  * an id that is not in `token_map` -- or a position before the sequence
//    starts -- is DEAD, and once one shift is dead EVERY LONGER SHIFT IS TOO.
//    A dead shift substitutes the PAD id, so the n-gram window degrades to a
//    shorter one rather than hashing a zero as if it were token 0.
//  * `n_cols = (n_gram - 1) * heads_per_ngram`, not `n_gram * heads`. The
//    token at shift 0 contributes to the running hash but produces no column,
//    which is why the loop above starts emitting at o = 1.
// ============================================================================
#pragma once

#include "krk/common.hpp"

namespace krk {

// Where one Engram module's constants live in the file's flattened arrays. All
// three are `module_index * stride` into the model-level vectors, exactly as
// the metadata lays them out: `layer_multipliers` is [n_e, n_gram],
// `head_offsets` and `head_vocab_sizes` are [n_e, n_cols].
struct EngramModule {
    const u64 *mult = nullptr;          // [n_gram]
    const u32 *offsets = nullptr;       // [n_cols]
    const u32 *vocab_sizes = nullptr;   // [n_cols]
    i32 n_gram = 0;
    i32 heads_per_ngram = 0;

    i32 n_cols() const { return n_gram > 1 ? (n_gram - 1) * heads_per_ngram : 0; }
    bool valid() const {
        return mult && offsets && vocab_sizes && n_gram > 1 && heads_per_ngram > 0;
    }
};

// The ids one token's window hashes to. `token_map` is the file's
// `engram.token_map` ([n_vocab_file]); `pad` is `token_map[engram.pad_token_id]`.
// `prev` is the n_gram-1 tokens BEFORE position t (oldest first); a position
// whose window runs off the start of the sequence passes `n_prev` < n_gram-1 and
// the missing shifts are dead.
//
// `out` is written with `n_cols()` ids. Returns false (and writes nothing) when
// the module is malformed, so a corrupt file cannot half-fill a route.
bool engram_hash(i32 *out, const EngramModule &m, const u32 *token_map,
                 i64 token_map_len, i64 pad, i32 token, const i32 *prev,
                 i32 n_prev);

// The table rows for a batch: one call per (token, column). Row `r` of
// `table` (row_bytes wide, `row_bytes` supplied by the caller because the table
// can be any dtype) is copied to `out + (t * n_cols + c) * row_bytes`.
//
// Kept separate from the hash because it is the only part that touches the
// table, and the table is mmap'd rather than resident.
void engram_gather(u8 *out, const u8 *table, i64 n_rows, i64 row_bytes,
                   const i32 *ids, i32 n_tokens, i32 n_cols);

} // namespace krk
