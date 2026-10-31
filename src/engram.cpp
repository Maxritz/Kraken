// ============================================================================
//  engram.cpp — DeepSeek-V4.1's n-gram hash (see krk/engram.hpp).
// ============================================================================
#include "krk/engram.hpp"

#include <cstring>
#include <vector>

namespace krk {
namespace {
// The value a dead shift substitutes. Anything outside the table is
// inexpressible here (the caller picks the pad id), so this is a sentinel used
// only inside the hash, where it is replaced by `pad` before arithmetic.
constexpr i64 kDead = -1;
} // namespace

bool engram_hash(i32 *out, const EngramModule &m, const u32 *token_map,
                 i64 token_map_len, i64 pad, i32 token, const i32 *prev,
                 i32 n_prev) {
    if (!out || !m.valid() || !token_map || token_map_len <= 0) return false;
    const i32 n_cols = m.n_cols();
    if (n_cols <= 0) return false;

    auto map_id = [&](i32 t) -> i64 {
        if (t < 0 || static_cast<i64>(t) >= token_map_len) return kDead;
        return static_cast<i64>(token_map[static_cast<size_t>(t)]);
    };

    // Shift 0 is the token itself. Once a shift is dead, every longer one is
    // too -- the window is a suffix of the sequence, so a gap at shift s means
    // there is nothing valid at s+1 either.
    i64 c = map_id(token);
    bool blocked = c == kDead;
    std::vector<u64> toks(static_cast<size_t>(m.n_gram));
    toks[0] = blocked ? static_cast<u64>(pad) : static_cast<u64>(c);
    for (i32 s = 1; s < m.n_gram; s++) {
        // `prev` is the window oldest-first: shift s is the token s positions
        // back, which is prev[n_prev - s] when the window is long enough.
        i64 id = kDead;
        if (s <= n_prev && prev) id = map_id(prev[n_prev - s]);
        blocked = blocked || id == kDead;
        toks[static_cast<size_t>(s)] =
            blocked ? static_cast<u64>(pad) : static_cast<u64>(id);
    }

    // The rolling accumulation. Each shift contributes ONE column group, and
    // the group's ids all come from the same running value -- that is what
    // makes a 3-gram's second column depend on tokens 0 AND 1.
    u64 rolling = toks[0] * m.mult[0];
    for (i32 o = 1; o < m.n_gram; o++) {
        rolling ^= toks[static_cast<size_t>(o)] * m.mult[o];
        for (i32 h = 0; h < m.heads_per_ngram; h++) {
            const i32 col = (o - 1) * m.heads_per_ngram + h;
            if (col >= n_cols) break;
            const u32 prime = m.vocab_sizes[col];
            if (prime < 2) return false;   // a 0/1 bucket count traps the hash
            out[col] = static_cast<i32>(rolling % prime +
                                        static_cast<u64>(m.offsets[col]));
        }
    }
    return true;
}

void engram_gather(u8 *out, const u8 *table, i64 n_rows, i64 row_bytes,
                   const i32 *ids, i32 n_tokens, i32 n_cols) {
    if (!out || !table || !ids || row_bytes <= 0) return;
    for (i32 t = 0; t < n_tokens; t++)
        for (i32 c = 0; c < n_cols; c++) {
            const i32 id = ids[static_cast<i64>(t) * n_cols + c];
            u8 *dst = out + (static_cast<i64>(t) * n_cols + c) * row_bytes;
            if (id < 0 || static_cast<i64>(id) >= n_rows) {
                // A row past the table gets zeros, the same rule the embedding
                // lookup now follows: a defined value, not the next tensor's
                // bytes.
                std::memset(dst, 0, static_cast<size_t>(row_bytes));
                continue;
            }
            std::memcpy(dst, table + static_cast<i64>(id) * row_bytes,
                        static_cast<size_t>(row_bytes));
        }
}

} // namespace krk
