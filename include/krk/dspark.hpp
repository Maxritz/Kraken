// ============================================================================
//  dspark.hpp -- the DSpark block drafter.
//
//  DSpark is a *parallel block* drafter: a small stack of `blk.N` transformer
//  blocks whose attention is NON-CAUSAL over the draft block, so every
//  candidate row is read out of one pass instead of a token loop. It is the
//  same position in the stack as DFlash -- a head set that borrows the target's
//  embedding and head -- but the block maths and the two side heads are its own.
//
//  The file (schema read off the arch table and the tensors the family ships;
//  there is no validated public GGUF for it yet, which is why the arch entry
//  says so rather than saying "supported"):
//
//    fc.weight                 [n_embd, n_aux * target_embd]
//    output_norm.weight        [n_embd]
//    blk.N.{attn_norm, ffn_norm}.weight
//    blk.N.{attn_q, attn_k, attn_v, attn_output}.weight
//    blk.N.{attn_q_norm, attn_k_norm}.weight       (optional)
//    blk.N.{ffn_gate, ffn_up, ffn_down}.weight
//    markov.w1 / markov.w2      the bigram head: w1 [rank, n_embd], w2 [n_vocab, rank]
//    confidence.proj            the per-candidate confidence head
//
//  Three parts, and the third is what makes it a *block* drafter rather than a
//  second DFlash:
//
//   1. Fusion. The target copies the residual stream entering a few chosen
//      layers; each block is RMS-normed, scaled by its own gain, concatenated
//      per token, and pushed through `fc`. Same contract as DFlash, because it
//      is the same job -- the drafter's context arrives as features.
//   2. Block decode. `block_size` rows: the last committed token followed by
//      mask tokens, one non-causal attention stack over them, one readout
//      through the target's head. Row k is a candidate for pos+k.
//   3. Markov. The transformer pass is parallel, but the candidates are not
//      independent: the token chosen at row k-1 conditions row k. `markov.w1/w2`
//      is that recurrence -- a rank-`markov.rank` bigram logit bias, applied
//      left to right over the block, starting from the last committed token.
//      `confidence.proj` scores each row so a caller can cut the block where it
//      stops being sure. Both are optional and both announce themselves.
//
//  What this file does NOT do, and says so rather than guessing: the MinMax-M3
//  MSA attention and the qwen4exp 512-expert head block are different objects
//  with their own schemas; see the arch table for what each still needs.
// ============================================================================
#ifndef KRK_DSPARK_HPP
#define KRK_DSPARK_HPP

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "krk/backend.hpp"
#include "krk/draft_block.hpp"
#include "krk/expert_cache.hpp"
#include "krk/gguf.hpp"
#include "krk/model.hpp"

namespace krk {

// True for the arch strings `DsparkDraft::load` accepts.
bool dspark_arch(std::string_view arch);

class DsparkDraft : public DraftBlock {
public:
    DsparkDraft() = default;
    ~DsparkDraft() override;
    DsparkDraft(const DsparkDraft &) = delete;
    DsparkDraft &operator=(const DsparkDraft &) = delete;

    bool load(Backend &be, const std::string &path, i64 chunk, i64 ctx,
              i32 target_embd, i32 target_layer_count, i32 target_vocab,
              std::string *err);

    // ---- DraftBlock ------------------------------------------------------
    const std::vector<i32> &target_layers() const override {
        return target_layers_;
    }
    void *capture_slot(i32 aux) const override;
    bool commit(i32 pos0, i32 n) override;
    i32 block_size() const override { return block_size_; }
    i64 kv_capacity() const override { return kv_cap_; }
    i32 draft_block(const QuantTensor &tok_embd, const QuantTensor &lm_head,
                    i32 n_vocab, i32 id_last, i32 n_draft, i32 pos,
                    f32 *rows_out) override;
    void unload() override;
    std::string describe() const override;
    const char *kind() const override { return "dspark"; }

    // ---- what the run report and the tests ask about ----------------------
    i32 n_embd() const { return cfg_.n_embd; }
    i32 n_layer() const { return cfg_.n_layer; }
    i32 n_embd_enc() const { return n_embd_enc_; }
    i32 mask_id() const { return mask_id_; }
    bool causal() const { return causal_; }
    bool has_markov() const { return markov_w1_.data != nullptr &&
                                     markov_w2_.data != nullptr; }
    bool has_confidence() const { return conf_.data != nullptr; }
    i32 markov_rank() const { return markov_rank_; }
    // The confidence each row of the most recent block scored, in row order.
    const std::vector<f32> &last_confidence() const { return last_conf_; }
    // The tokens the Markov recurrence walked, row by row: row k's bias came
    // from markov_prev()[k]. Exposed because "the block proposed these tokens"
    // and "the block walked this chain" are different facts and a test needs
    // the second one.
    const std::vector<i32> &markov_prev() const { return markov_prev_; }
    const std::string &arch() const { return cfg_.arch; }
    i32 mask_rows() const { return mask_rows_; }
    i64 committed() const { return committed_; }

private:
    bool load_weights(std::string *err);
    void free_dev();

    Backend *be_ = nullptr;
    std::unique_ptr<Gguf> gguf_;
    ModelConfig cfg_;
    std::vector<LayerWeights> layers_;

    QuantTensor fc_;
    f32 *out_norm_ = nullptr;
    f32 *aux_norm_ = nullptr;      // [n_embd, n_aux], stacked
    std::vector<f32> aux_norm_host_;
    // The two side heads. Both may be absent, and absent is not an error: a
    // head set without them is still a block drafter.
    QuantTensor markov_w1_;        // [rank, n_embd]
    QuantTensor markov_w2_;        // [n_vocab, rank]
    QuantTensor conf_;             // [1, n_embd]
    i32 markov_rank_ = 0;

    std::vector<i32> target_layers_;
    std::vector<f32> inv_freq_;
    i32 n_embd_enc_ = 0;
    i32 block_size_ = 0;
    i32 mask_id_ = -1;
    i32 mask_rows_ = 0;
    bool causal_ = false;

    i64 chunk_ = 0;
    i64 kv_cap_ = 0;
    i64 committed_ = 0;

    // device
    void *cap_ = nullptr;
    void *feat_ = nullptr;
    void *enc_out_ = nullptr;
    void *ws_x_ = nullptr;
    void *ws_xn_ = nullptr;
    void *ws_q_ = nullptr;
    void *ws_k_ = nullptr;
    void *ws_v_ = nullptr;
    void *ws_attn_ = nullptr;
    void *ws_gate_ = nullptr;
    void *ws_up_ = nullptr;
    void *ws_logits_ = nullptr;
    void *ws_markov_ = nullptr;   // scratch: embedding / rank / confidence
    void *ws_mbias_ = nullptr;    // [n_vocab] one Markov bias row
    void *kcache_ = nullptr;
    void *vcache_ = nullptr;
    void *ws_gs_ = nullptr;
    i64 gs_in_ = 0;

    // host
    std::vector<f32> feat_host_;
    std::vector<f32> tmp_host_;
    std::vector<f32> markov_bias_;
    std::vector<i32> block_tok_;
    std::vector<f32> last_conf_;
    std::vector<i32> markov_prev_;

    bool markov_on_ = true;
    f32 conf_cut_ = 0.0f;  // 0 disables the cut
};

} // namespace krk

#endif // KRK_DSPARK_HPP
