// ============================================================================
//  dspark.hpp -- the DSpark draft head: DFlash's block, plus two heads on top.
//
//  Read off the reference implementation
//  (spiritbuun/buun-llama-cpp, src/models/dflash.cpp: `build_dspark_markov_head`),
//  which settles what a DSpark file IS:
//
//   * It is a **DFlash** head set. `general.architecture` is `dflash`; the
//     presence of `markov_w1.weight` is what makes it DSpark. There is no
//     separate schema, which is why an `--draft` probe on a real file must ask
//     about tensors and not about the arch string.
//   * The Markov head is a bigram bias, applied LEFT TO RIGHT over the block:
//         w1_prev = row(markov_w1, prev)          # [rank], a row gather
//         bias    = markov_w2 @ w1_prev           # [n_vocab]
//         col     = logits_row + bias
//         prev    = argmax(col)                   # greedy, by construction
//     No activation between w1 and w2. An earlier version of this file put a
//     ReLU there, which is not in the reference and would have quietly biased
//     every proposal.
//   * The confidence head reads the CONCATENATION of the block's hidden row and
//     w1_prev ([n_embd + rank]) through a [n_embd + rank, 1] projection, plus
//     an optional bias, then a sigmoid. An earlier version read the hidden row
//     alone against a [n_embd, 1] projection -- the wrong width, so a real file
//     would have been refused as "inconsistent shapes".
//
//  The block decode itself is DFlash's and is not reimplemented here: this class
//  owns a DflashDraft and adds the two heads over the rows it returns. That is
//  the reference's structure too, and it means a fix to the block reaches both.
// ============================================================================
#ifndef KRK_DSPARK_HPP
#define KRK_DSPARK_HPP

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "krk/backend.hpp"
#include "krk/dflash.hpp"
#include "krk/draft_block.hpp"
#include "krk/gguf.hpp"

namespace krk {

// True for arch strings that *can* carry a DSpark head. A DSpark file is a
// DFlash file, so the answer is "anything the DFlash loader accepts"; it still
// needs markov_w1 before it is actually DSpark -- see dspark_head_set().
bool dspark_arch(std::string_view arch);

// True when this mapping carries the DSpark head regardless of its arch string.
bool dspark_head_set(const Gguf &g);

class DsparkDraft : public DraftBlock {
public:
    DsparkDraft() = default;
    ~DsparkDraft() override;
    DsparkDraft(const DsparkDraft &) = delete;
    DsparkDraft &operator=(const DsparkDraft &) = delete;

    bool load(Backend &be, const std::string &path, i64 chunk, i64 ctx,
              i32 target_embd, i32 target_layer_count, i32 target_vocab,
              std::string *err);

    // ---- DraftBlock (the block half delegates to the DFlash drafter) -----
    const std::vector<i32> &target_layers() const override {
        return base_.target_layers();
    }
    void *capture_slot(i32 aux) const override {
        return base_.capture_slot(aux);
    }
    bool commit(i32 pos0, i32 n) override { return base_.commit(pos0, n); }
    i32 block_size() const override { return base_.block_size(); }
    i64 kv_capacity() const override { return base_.kv_capacity(); }
    i32 draft_block(const QuantTensor &tok_embd, const QuantTensor &lm_head,
                    i32 n_vocab, i32 id_last, i32 n_draft, i32 pos,
                    f32 *rows_out) override;
    void unload() override;
    std::string describe() const override;
    const char *kind() const override { return "dspark"; }

    // ---- facts a test and the run report ask about -----------------------
    bool has_markov() const { return w1_.data != nullptr && w2_.data != nullptr; }
    bool has_confidence() const { return conf_.data != nullptr; }
    i32 markov_rank() const { return rank_; }
    const std::vector<f32> &last_confidence() const { return last_conf_; }
    // The loop's view of the same vector: null unless the head is on and the
    // block produced rows.
    const f32 *row_confidence() const override {
        return has_confidence() && !last_conf_.empty() ? last_conf_.data()
                                                       : nullptr;
    }
    const std::vector<i32> &markov_prev() const { return markov_prev_; }
    i32 last_rows() const { return last_rows_; }
    const DflashDraft &base() const { return base_; }

private:
    bool load_head(std::string *err);

    Backend *be_ = nullptr;
    std::unique_ptr<Gguf> gguf_;   // our own mapping: the base opens the file too
    DflashDraft base_;

    QuantTensor w1_;               // [rank, n_vocab]
    QuantTensor w2_;               // [rank, n_vocab]
    QuantTensor conf_;             // [n_embd + rank, 1]
    f32 *conf_bias_ = nullptr;     // optional scalar bias
    i32 rank_ = 0;
    f32 w2_scale_ = 1.0f;          // markov_w2.scale, when a quantised file has one

    void *ws_row_ = nullptr;       // [rank]
    void *ws_bias_ = nullptr;      // [n_vocab]
    void *ws_feat_ = nullptr;      // [n_embd + rank]

    std::vector<f32> tmp_row_;     // [n_vocab]
    std::vector<f32> tmp_bias_;    // [n_vocab]
    std::vector<f32> hidden_;      // [rows][n_embd]
    std::vector<f32> last_conf_;
    std::vector<i32> markov_prev_;
    i32 last_rows_ = 0;

    bool markov_on_ = true;
    f32 conf_cut_ = 0.0f;
};

} // namespace krk

#endif // KRK_DSPARK_HPP
