// The MTP head's dense prefix (plan step 1.5).
//
// The block's first stage, before any attention: fuse the trunk's multi-stream
// residual with the proposed token's embedding and produce the block input.
//
//   e       = rms(embed) * enorm                      (2560 -> 2560, one lane)
//   h_lane  = rms(R_lane) * hnorm[..lane..]           (10240 -> 2560 per lane)
//   x_lane  = eh_proj( cat([e, h_lane]) )             or fc_embedding(e) + fc_hidden(h_lane)
//
// It runs on the HOST, in f32, and it reads the prefix's matrices straight out
// of the GGUF one output row at a time. Two reasons for that shape:
//
//   * it is checkable before any device buffer exists, which is the point of
//     doing it first -- a wrong pairing or a wrong norm here is a drafter that
//     runs and is rejected every time, and nothing downstream can tell you that;
//   * `eh_proj` is 5120 x 2560 (52 MB in f32). Dequantizing one row per output
//     keeps the resident cost at the two norms plus one 5120-float scratch.
//
// The scan order and the norm contract are the reference's
// (`src/models/qwen4exp.cpp`, `build_hc_mix` / the MTP graph): a per-lane RMS
// with one concatenated [n_lane * n_embd] gamma, `e` repeated across lanes, and
// the concat ordered `[e | h]` because that is the column order of the fused
// matrix the split form is a column split of.
#pragma once

#include <string>
#include <vector>

#include "krk/quant.hpp"

namespace krk {

class Gguf;
struct GgufTensor;

// What the prefix reads. The matrices stay in the mapping; only the two norms
// are dequantized up front, because they are multiplied against every lane.
struct MtpHostPrefix {
    i32 n_embd = 0;
    i32 n_lane = 0;
    bool split = false;   // fc_embedding + fc_hidden instead of one eh_proj
    f32 eps = 1e-5f;      // the trunk's rms eps

    std::vector<f32> enorm;   // [n_embd]
    std::vector<f32> hnorm;   // [n_lane * n_embd], lane-major

    // `eh_proj` is [n_embd, 2*n_embd] in ggml order: dst-major, so output row
    // `o` is `2*n_embd` consecutive values starting with `e`'s column block.
    const GgufTensor *eh_proj = nullptr;
    const GgufTensor *fc_embedding = nullptr;  // [n_embd, n_embd]
    const GgufTensor *fc_hidden = nullptr;     // [n_embd, n_embd]

    bool ready() const {
        return n_embd > 0 && n_lane > 0 && !enorm.empty() && !hnorm.empty() &&
               (split ? (fc_embedding && fc_hidden) : eh_proj != nullptr);
    }
};

// Fill `out` from an HF `mtp.*` head. `layer_prefix` is `mtp.layers.0.` for the
// block-scoped roles; the prefix tensors are all at the head's top level, so a
// caller that already knows the geometry can pass an empty prefix.
bool mtp_host_prefix_load(const Gguf &g, i32 n_embd, i32 n_lane, f32 eps,
                          MtpHostPrefix *out, std::string *err);

// The prefix itself. `embed` is [n_embd]; `R` is [n_lane * n_embd] lane-major
// (the trunk's final multi-stream residual at the cell being drafted); `x` is
// [n_lane * n_embd] lane-major on return.
//
// `per_stream` selects the two hnorm modes the reference carries: a separate
// RMS per lane (the checkpoint's own, and the default) or one RMS over the whole
// 4 x 2560 vector. The two differ, the artifact decides which, and the
// acceptance rate is the only oracle that can settle it.
bool mtp_dense_prefix(const MtpHostPrefix &w, const f32 *embed, const f32 *R,
                      bool per_stream, f32 *x, std::string *err);

}  // namespace krk
