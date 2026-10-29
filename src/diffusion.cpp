// ============================================================================
//  diffusion.cpp — the block-diffusion schedule (see krk/diffusion.hpp).
// ============================================================================
#include "krk/diffusion.hpp"

#include <cmath>

namespace krk {

void diffusion_transfer_plan(i32 *out, i32 masks, i32 steps) {
    if (!out || steps <= 0) return;
    if (masks <= 0) {
        for (i32 i = 0; i < steps; i++) out[i] = 0;
        return;
    }
    // Base plus remainder: the remainder goes to the FIRST steps, so a block of
    // 10 masks over 3 steps commits 4, 3, 3 -- never 3, 3, 3 (which would leave
    // one mask behind and never finish the block) and never 4, 4, 4 (which
    // would over-run the block and commit positions nobody selected).
    const i32 base = masks / steps;
    const i32 rem = masks % steps;
    for (i32 i = 0; i < steps; i++) out[i] = base + (i < rem ? 1 : 0);
}

i32 diffusion_generate(std::vector<i32> &seq, i32 prompt_len, i32 n_vocab,
                       const DiffusionParams &p,
                       const std::function<bool(const i32 *, i32, f32 *)> &forward) {
    if (p.gen_length <= 0 || p.block_length <= 0 || p.steps <= 0) return -1;
    if (prompt_len <= 0 || n_vocab <= 0) return -1;
    const i32 total = prompt_len + p.gen_length;
    seq.assign(static_cast<size_t>(total), p.mask_id);

    // A generation length that is not a whole number of blocks denoises the
    // remainder as a short final block rather than dropping it -- dropping it
    // promises gen_length tokens and returns fewer, which then reads as a
    // truncated answer instead of a schedule choice.
    const i32 n_blocks = (p.gen_length + p.block_length - 1) / p.block_length;

    std::vector<f32> logits(static_cast<size_t>(total) * n_vocab);
    std::vector<f32> conf(static_cast<size_t>(total));
    std::vector<i32> x0(static_cast<size_t>(total));
    std::vector<i32> plan(static_cast<size_t>(p.steps));
    std::vector<char> chosen(static_cast<size_t>(total));
    i32 forwards = 0;

    for (i32 b = 0; b < n_blocks; b++) {
        const i32 lo = prompt_len + b * p.block_length;
        const i32 hi = std::min(total, lo + p.block_length);
        const i32 n_masks = hi - lo;
        // Per BLOCK budget. The caller passes the per-block step count.
        diffusion_transfer_plan(plan.data(), n_masks, p.steps);
        for (i32 s = 0; s < p.steps; s++) {
            if (plan[static_cast<size_t>(s)] <= 0) continue;
            if (!forward(seq.data(), total, logits.data())) return -1;
            forwards++;
            // argmax + its probability, per position.
            for (i32 t = 0; t < total; t++) {
                const f32 *row = logits.data() + static_cast<i64>(t) * n_vocab;
                i32 best = 0;
                f32 bestv = row[0];
                for (i32 v = 1; v < n_vocab; v++)
                    if (row[v] > bestv) { bestv = row[v]; best = v; }
                if (p.eos_forbid_prefix && p.eos_id >= 0 && t < hi - 1) {
                    // The reference's logits_eos_inf: EOS may only be chosen at
                    // the last position the model is allowed to end on.
                    if (best == p.eos_id) {
                        best = 0;
                        bestv = row[0];
                        for (i32 v = 1; v < n_vocab; v++)
                            if (v != p.eos_id && row[v] > bestv) { bestv = row[v]; best = v; }
                    }
                }
                x0[static_cast<size_t>(t)] = best;
                // softmax probability of that token, max-shifted.
                f32 mx = row[0];
                for (i32 v = 1; v < n_vocab; v++) mx = std::max(mx, row[v]);
                f32 sum = 0.0f;
                for (i32 v = 0; v < n_vocab; v++) sum += std::exp(row[v] - mx);
                conf[static_cast<size_t>(t)] =
                    sum > 0.0f ? std::exp(bestv - mx) / sum : 0.0f;
            }
            // Candidates: masked positions inside THIS block, nothing else.
            for (i32 t = 0; t < total; t++) {
                const bool in_block = t >= lo && t < hi;
                chosen[static_cast<size_t>(t)] =
                    (in_block && seq[static_cast<size_t>(t)] == p.mask_id) ? 1 : 0;
                if (!chosen[static_cast<size_t>(t)])
                    conf[static_cast<size_t>(t)] = -1e30f;
            }
            // Commit `plan[s]` of the most confident candidates. A selection
            // sort, so an exact confidence tie resolves to the LOWER position --
            // deterministic, which is what makes a diffusion run reproducible.
            for (i32 c = 0; c < plan[static_cast<size_t>(s)]; c++) {
                i32 pick = -1;
                f32 pickv = 0.0f;
                for (i32 t = 0; t < total; t++) {
                    if (!chosen[static_cast<size_t>(t)]) continue;
                    if (pick < 0 || conf[static_cast<size_t>(t)] > pickv) {
                        pick = t;
                        pickv = conf[static_cast<size_t>(t)];
                    }
                }
                if (pick < 0) break;   // fewer candidates than the plan asked for
                seq[static_cast<size_t>(pick)] = x0[static_cast<size_t>(pick)];
                chosen[static_cast<size_t>(pick)] = 0;
            }
        }
    }
    return forwards;
}

} // namespace krk
