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

} // namespace krk
