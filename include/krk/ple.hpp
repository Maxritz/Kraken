// ============================================================================
//  ple.hpp — qwen4exp's PLE (per-layer embedding) n-gram hash table.
//
//  Every token gathers `ple_n_heads` rows out of one flat table
//  (`per_layer_token_embd.weight`, 320,001,536 rows of 160 in the REAP-288
//  file — tens of GB, so it is read row by row out of the mapping and never
//  uploaded). Which rows is a hash over the token's own n-gram window:
//
//      mixed(n) = t[p]*m[0] ^ (t[p-1]*m[1] ^ (t[p-2]*m[2] ... ))   for n = 2..ngram
//      row[h]   = mixed(n) % vocab[h] + offset[h]                   h in that n-gram's heads
//
//  The first `heads_per_ngram` heads hash a 2-gram, the next a 3-gram, and so on,
//  which is why a 3-gram file has (3-1)*8 = 16 heads. An EOS anywhere in the
//  window cuts it: every position at or before the EOS reads as EOS, so a
//  sequence never hashes across a boundary. A missing predecessor (before the
//  start, or no cached cell) reads as EOS too.
//
//  The hash is 64-bit and runs on the host because ggml has no 64-bit integer
//  and no xor — the same reason the reference does it here. Getting it wrong is
//  not a crash: the wrong-but-valid rows are gathered and the model decodes
//  fluent text that is subtly not the model.
// ============================================================================
#pragma once

#include "krk/common.hpp"
#include "krk/quant.hpp"

#include <vector>

namespace krk {

struct PleConfig {
    i32 ngram = 0;           // ple.ngram_size, 3
    i32 heads_per_ngram = 0; // ple.heads_per_ngram, 8
    i32 head_dim = 0;        // embedding_length_per_layer_input, 160
    i32 conv_kernel = 0;     // ple.conv_kernel, 4
    i32 eos_token = -1;      // the cut token
    // (ngram - 1) * heads_per_ngram entries, one row range each. The file
    // stores them as uint64; they are narrowed here after the range check the
    // loader does, so a head can never index past the table.
    std::vector<i64> head_offsets;
    std::vector<i64> head_vocab_sizes;
    std::vector<i64> multipliers; // [ngram], the 2^61-style odd constants

    i32 n_heads() const { return ngram > 0 ? (ngram - 1) * heads_per_ngram : 0; }
    // The gathered embedding is one `n_heads * head_dim` row per token, and it
    // feeds a projection whose input is the whole wide stream: those two have
    // to be the same number or the block cannot run at all.
    i64 row_width() const {
        return static_cast<i64>(n_heads()) * head_dim;
    }
    bool valid() const {
        return ngram >= 2 && heads_per_ngram > 0 && head_dim > 0 &&
               static_cast<i32>(head_offsets.size()) >= n_heads() &&
               static_cast<i32>(head_vocab_sizes.size()) >= n_heads() &&
               static_cast<i32>(multipliers.size()) >= ngram;
    }
};

// Fills `out[0..n_heads())` with the table row each head gathers for one token.
//
// `window` is oldest-first and holds the token's predecessors followed by the
// token itself, so `window[n-1]` is the current token and `window[n-2]` the one
// before it. Ten entries are enough for any n-gram this format allows; `n_window`
// is how many are valid — a shorter window (the start of a sequence) reads as
// EOS. `eos_token < 0` disables the cut, which a fixture may want.
void ple_rows(const PleConfig &cfg, const i64 *window, i32 n_window, i64 *out);

// Every token of one forward pass, hashed in place.
//
// `seq` is the sequence so far, position-indexed, and must cover pos0 + n
// entries: the window before the pass is the same token array the reference
// reads out of the KV cells, so nothing is re-derived from the KV and nothing
// is remembered here. The first tokens of a sequence simply have a shorter
// window, which reads as EOS (see ple_rows). `out` is [n, n_heads), head
// fastest, matching the reference's `idx[i*n_heads + h]`.
void ple_rows_for(const PleConfig &cfg, const i64 *seq, i64 pos0, i32 n,
                  i64 *out);

// Gathers the hashed rows out of the mapping into host f32, head-slowest inside
// a token:
//   dst[t * n_heads * head_dim + h * head_dim + d]
// which is the order the reference's ggml_get_rows produces and the order the
// key projection expects. Rows are read straight out of the file's bytes
// (`row_bytes` apart, in the file's own type) — the table is tens of GB and is
// never uploaded, which is the whole point of computing the hash on the host.
void ple_gather(f32 *dst, const u8 *table, DType type, u64 row_bytes,
                i32 head_dim, i32 n_heads, const i64 *rows);

} // namespace krk
