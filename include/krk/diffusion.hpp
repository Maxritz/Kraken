// ============================================================================
//  diffusion.hpp — the block-diffusion sampler for `diffuse` (LLaDA2.x).
//
//  A masked discrete diffusion model does not decode left to right. The
//  sequence starts as `[prompt | MASK x gen_length]` and is denoised in
//  PARALLEL: each step runs ONE bidirectional forward over the whole sequence
//  and commits the tokens it is most confident about. There is no causal mask
//  and no KV reuse across steps, so "generation" is a schedule around repeated
//  full-sequence forwards, not a decode loop.
//
//  The schedule is the load-bearing part and it is small enough to state
//  exactly (ML-GSAI/LLaDA generate.py, the reference every implementation
//  follows):
//
//      num_blocks  = gen_length / block_length
//      steps_block = steps / num_blocks
//      for block in num_blocks:
//          transfer = get_num_transfer_tokens(masks_in_block, steps_block)
//          for step in steps_block:
//              logits = model(x)                 # full sequence, bidirectional
//              x0     = argmax(logits)
//              conf   = softmax(logits)[x0]
//              conf[beyond this block] = -inf     # only this block is a candidate
//              x0     = where(is_mask(x), x0, x)  # never overwrite a committed token
//              conf   = where(is_mask(x), conf, -inf)
//              sel    = topk(conf, transfer[step])
//              x[sel] = x0[sel]
//
//  Three things here are easy to get wrong and are what the tests pin:
//
//  * `transfer` is BASE PLUS REMAINDER, not a constant: `masks / steps` tokens
//    every step, and the first `masks % steps` steps take one extra. A flat
//    `masks / steps` leaves the remainder masked forever (the block never
//    finishes) and a flat ceil over-commits and then runs steps with nothing
//    left to commit.
//  * candidates are MASKED positions only. A committed token is not re-scored
//    and never re-decided -- `conf = -inf` off the mask is what enforces it, and
//    without it the top-k selection can pick a committed position and "commit"
//    it again from a different step's logits.
//  * candidates are THIS BLOCK only. The semi-autoregressive schedule gives
//    block k the whole sequence as context but only its own positions as
//    targets; letting a later block be committed early makes the schedule
//    depend on the model rather than on the plan.
//
//  Everything in this file is pure: it takes a forward callback and touches no
//  backend, which is what lets test_diffusion_sampler drive it with a model it
//  can predict exactly.
// ============================================================================
#pragma once

#include "krk/common.hpp"

#include <functional>
#include <vector>

namespace krk {

// How many tokens a step of a block commits: `masks / steps` every step, plus
// one extra on the first `masks % steps` steps. Writes `steps` entries.
// A step count of 0 commits everything at once (one full block in one step),
// which is the degenerate-but-legal arm.
void diffusion_transfer_plan(i32 *out, i32 masks, i32 steps);

// The sampler's knobs. `steps` is the budget per BLOCK, not per sequence --
// the reference divides the user's total by the block count before this runs,
// and doing it the other way round silently gives each block the whole budget.
struct DiffusionParams {
    i32 gen_length = 0;      // tokens to denoise, after the prompt
    i32 block_length = 0;    // semi-autoregressive block size
    i32 steps = 0;           // denoising steps per block
    i32 mask_id = 0;         // the file's `diffuse.mask_token_id`
    i32 eos_id = -1;         // -1 disables the EOS rule
    // The reference forces EOS logits to +inf at positions that must not end
    // the sequence. Off by default so a test can see the raw model.
    bool eos_forbid_prefix = false;
};

// `forward` is called with the CURRENT sequence (prompt + generated) and must
// fill `logits` with `n_tok * n_vocab` floats, row-major [token][vocab]. It is
// called `steps_block * num_blocks` times, which is the number the metrics
// report.
//
// Returns the number of forwards it ran, or -1 on a malformed schedule.
i32 diffusion_generate(std::vector<i32> &seq, i32 prompt_len, i32 n_vocab,
                       const DiffusionParams &p,
                       const std::function<bool(const i32 *tokens, i32 n_tok,
                                                f32 *logits)> &forward);

} // namespace krk
