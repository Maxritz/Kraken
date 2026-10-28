// ============================================================================
//  dsv4_compress.hpp — DeepSeek-V4 compressed attention (CSA / HCA), host side.
//
//  A V4 layer either attends the raw latent window (compress_ratio 0) or the
//  window PLUS a stream of POOLED keys, one per block of `ratio` tokens:
//
//      ratio 128 -> HCA: every block is visible (dense over the blocks)
//      ratio   4 -> CSA: blocks are selected per query token by a small
//                        "lightning indexer", and the pool OVERLAPS: block b
//                        is a weighted average over the tokens of blocks b-1
//                        and b, with a different score channel for each
//
//  The pooled key is not an average of latents. Each token contributes two
//  projections of its embedding, and the weights are a per-channel softmax over
//  the block's tokens:
//
//      kv_t    = Wkv   x_t                       [coff*hd]
//      score_t = Wgate x_t + APE[:, p % ratio]   [coff*hd]
//      w       = softmax_over_t(score[:, c])     (per channel c)
//      pooled  = sum_t w_t * kv_t[:, c]          [hd]
//      pooled  = rms_norm(pooled, comp_norm)     <- the row the cache stores
//      K       = rope_tail(copy of pooled, pos=block_start, base=compress_base)
//
//  Three details that are invisible in the shapes and wrong in the output:
//
//   * The softmax is over the BLOCK AXIS, PER CHANNEL. `ggml_soft_max` on a
//     tensor viewed [hd, ratio] normalizes each channel over the block's rows.
//     A softmax over the whole row instead (or a plain average) produces a
//     finite pooled key and a model that reads as merely worse.
//   * The rope is on the TAIL (offset hd - rope) and uses the file's
//     `attention.compress_rope_base`, not the attention base: the compression
//     happens at a coarser scale than the per-token keys, and the file says so.
//   * The rope POSITION is the block's first token (`source_start`), not the
//     block index: two runs over the same prefix must land on the same angle.
//
//  The CSA overlap is the other silent one. `coff` is 2 for ratio 4, so a
//  state row is [cur | prev] halves; block b's pool reads its own block's
//  SECOND half and the previous block's FIRST half, in that value order:
//
//      values = [ prev rows' kv[0..hd) | cur rows' kv[hd..2hd) ]
//      scores = [ prev rows' sc[0..hd) | cur rows' sc[hd..2hd) ]
//
//  and the softmax runs over all 2*ratio of them together. Reading the halves
//  the other way round is in range, finite, and pools the wrong block.
//
//  Everything here is plain f32 and caller-supplied buffers: the engine owns
//  the state rows (one per token, per compressed layer) and the pooled keys;
//  this file owns the arithmetic, so there is one definition for the CPU path,
//  the suite and any kernel A/B.
// ============================================================================
#pragma once

#include "krk/common.hpp"

namespace krk {

// The two ratios a V4 file uses. Anything else is refused by the loader
// (llama.cpp's own loader supports exactly these three values, 0 included).
constexpr i32 kDsv4CsaRatio = 4;
constexpr i32 kDsv4HcaRatio = 128;

// One compressed layer. Weights are row-major [in, out], the orientation
// mla_shared.hpp uses, so `proj` means the same thing everywhere.
struct Dsv4Compressor {
    i32 ratio = 0;             // kDsv4CsaRatio or kDsv4HcaRatio
    i32 n_embd = 0;
    i32 hd = 0;                // pooled key width (== the attention latent)
    i32 rope = 0;              // rotary TAIL of the pooled key
    f32 rope_base = 0.0f;      // attention.compress_rope_base
    f32 eps = 0.0f;            // attention.layer_norm_rms_epsilon
    const f32 *wkv = nullptr;    // [n_embd, coff*hd]
    const f32 *wgate = nullptr;  // [n_embd, coff*hd]
    const f32 *ape = nullptr;    // [coff*hd, ratio]  (column = pos % ratio)
    const f32 *norm = nullptr;   // [hd]

    i32 coff() const { return ratio == kDsv4CsaRatio ? 2 : 1; }
    i32 state_width() const { return coff() * hd; }
    // Rows one pool consumes: the block, plus the previous block when the
    // layout overlaps.
    i32 pool_rows() const { return coff() * ratio; }
    bool valid() const {
        // `rope <= hd`, not `<`: the reference only asserts that the head is at
        // least as wide as the rotary part, so an all-rotary pooled key is a
        // shape this engine must accept rather than one it invents as invalid.
        return (ratio == kDsv4CsaRatio || ratio == kDsv4HcaRatio) &&
               n_embd > 0 && hd > 0 && rope > 0 && rope <= hd &&
               rope_base > 0.0f && eps > 0.0f && wkv && wgate && ape && norm;
    }
};

// One token's compressor state row. `pos_in_block` is `position % ratio` and
// selects the APE column -- getting it from the block index instead is the
// defect this parameter exists to prevent.
//
// `kv_out` and `score_out` each receive `state_width()` floats.
bool dsv4_comp_state(const Dsv4Compressor &c, const f32 *x, i32 pos_in_block,
                     f32 *kv_out, f32 *score_out);

// Pool one block into `out` (hd floats, normed and UNROPED).
//
// `kv` and `score` hold `pool_rows()` state rows IN POOL ORDER: for the
// overlapping layout the previous block's `ratio` rows come first, then the
// current block's. For the dense layout they are just the block's rows.
//
// Unroped on purpose, the same split mla_shared.hpp uses for the raw latent:
// the cached row is the value, and K is a rotated copy made on read. The angle
// is the block's FIRST TOKEN position, so two runs over the same prefix land on
// the same angle -- `dsv4_comp_rope(out, hd, rope, block_start, base)`.
bool dsv4_comp_pool(const Dsv4Compressor &c, const f32 *kv, const f32 *score,
                    f32 *out);

// Rotary embedding of a `rope`-wide TAIL of an `hd`-wide row, at `pos`, with an
// explicit base. Consecutive pairs (the NORM convention V4's latent uses).
void dsv4_comp_rope(f32 *v, i32 hd, i32 rope, i64 pos, f32 base);

// The frequency table `dsv4_comp_rope` uses, `rope/2` entries:
// inv_freq[j] = base^(-2j/rope). Built once per layer for the compression base,
// which is a different table from the attention one.
void dsv4_comp_inv_freq(f32 *out, i32 rope, f32 base);

// ---- the lightning indexer (CSA layers only) -------------------------------

// Score every pooled indexer key against one query token:
//
//      score[b] = sum_h relu(q_h . k_b) * w_h        (+ mask[b] when given)
//
// `q` is [n_head, head] already roped (compress base) and permuted to
// head-major; `k` is [n_blocks, head] likewise; `w` is [n_head]. `mask` is the
// additive block mask (0 visible, -inf not) or null. A negative score is
// impossible except through the mask, which is the point of the relu: the
// weights are a sum of non-negative per-head similarities, so the selection
// cannot be dominated by one head's sign flip.
bool dsv4_indexer_scores(i32 n_head, i32 head, const f32 *q, const f32 *k,
                         const f32 *w, const f32 *mask, i32 n_blocks,
                         f32 *out);

// Descending top-k over `scores`, written to `idx`. Ties go to the LOWER index:
// the selection must be a function of the scores, not of the sort's history, or
// two runs of the same prompt pick different blocks (the determinism contract).
// `k` is clamped to `n`; returns the number written.
i32 dsv4_top_k(i32 *idx, i32 k, const f32 *scores, i32 n);

} // namespace krk
