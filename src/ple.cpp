// ============================================================================
//  ple.cpp — qwen4exp's n-gram hash, host side (see include/krk/ple.hpp).
// ============================================================================
#include "krk/ple.hpp"

namespace krk {

void ple_rows(const PleConfig &cfg, const i64 *window, i32 n_window, i64 *out) {
    const i32 n_heads = cfg.n_heads();
    if (!cfg.valid() || !window || !out || n_heads <= 0) return;

    // The window the hash reads, oldest first, with every position at or before
    // an EOS pinned to EOS. `ctx[0]` is the current token, matching the
    // reference's t[p] * m[0] term.
    i64 ctx[16];
    const i32 ng = cfg.ngram > 16 ? 16 : cfg.ngram;
    for (i32 s = 0; s < ng; s++) {
        // s positions back from the end of the window
        const i32 idx = n_window - 1 - s;
        if (idx < 0) {
            ctx[s] = cfg.eos_token >= 0 ? cfg.eos_token : 0;
            continue;
        }
        const i64 t = window[idx];
        ctx[s] = (cfg.eos_token >= 0 && t == cfg.eos_token) ? cfg.eos_token : t;
    }
    // The cut is positional: once a predecessor is EOS, everything older is EOS.
    if (cfg.eos_token >= 0) {
        bool cut = false;
        for (i32 s = 1; s < ng; s++) {
            if (cut) {
                ctx[s] = cfg.eos_token;
            } else if (ctx[s] == cfg.eos_token) {
                cut = true;
            }
        }
    }

    for (i32 nn = 2; nn <= ng; nn++) {
        // mixed = t[p]*m[0] ^ t[p-1]*m[1] ^ ... ^ t[p-(n-1)]*m[n-1]
        u64 mixed = static_cast<u64>(ctx[0]) *
                    static_cast<u64>(cfg.multipliers[0]);
        for (i32 j = 1; j < nn; j++)
            mixed ^= static_cast<u64>(ctx[j]) *
                     static_cast<u64>(cfg.multipliers[j]);
        const i32 base = (nn - 2) * cfg.heads_per_ngram;
        for (i32 g = 0; g < cfg.heads_per_ngram; g++) {
            const i32 h = base + g;
            if (h >= n_heads) break;
            const i64 vocab = cfg.head_vocab_sizes[static_cast<size_t>(h)];
            if (vocab <= 0) continue;
            out[h] = static_cast<i64>(mixed % static_cast<u64>(vocab)) +
                     cfg.head_offsets[static_cast<size_t>(h)];
        }
    }
}

void ple_rows_for(const PleConfig &cfg, const i64 *seq, i64 pos0, i32 n,
                  i64 *out) {
    const i32 nh = cfg.n_heads();
    if (!cfg.valid() || !seq || !out || nh <= 0 || n <= 0) return;
    const i32 ng = cfg.ngram > 16 ? 16 : cfg.ngram;
    for (i32 i = 0; i < n; i++) {
        const i64 pos = pos0 + i;
        if (pos < 0) continue;
        // The window ends at this token and is shorter at the start of the
        // sequence; ple_rows reads a short window as EOS, which is what the
        // reference's missing KV cells hash as.
        const i32 nw = (pos + 1 < static_cast<i64>(ng))
                           ? static_cast<i32>(pos + 1)
                           : ng;
        ple_rows(cfg, seq + (pos - nw + 1), nw, out + static_cast<i64>(i) * nh);
    }
}

void ple_gather(f32 *dst, const u8 *table, DType type, u64 row_bytes,
                i32 head_dim, i32 n_heads, const i64 *rows) {
    if (!dst || !table || !rows || head_dim <= 0 || n_heads <= 0 || !row_bytes)
        return;
    for (i32 h = 0; h < n_heads; h++) {
        const i64 row = rows[h];
        if (row < 0) continue;
        const u8 *src = table + static_cast<u64>(row) * row_bytes;
        dequant_row(type, src, dst + static_cast<i64>(h) * head_dim, head_dim);
    }
}

} // namespace krk
