// An external draft head for speculative decoding, in GGUF.
//
// The published EAGLE-3 / DFlash / Arctic speculators for gpt-oss, Qwen and
// Llama are all *heads*, not models: one or two blocks that take the target's
// own hidden state (and the last token's embedding) and propose k tokens, which
// the target then verifies. None of them ships as GGUF and none of them is a
// nextn block living inside the target file -- Tiel's `blk.40.*` is a different
// thing, loaded by Model::load_mtp_block. This is the other shape: a file
// beside the model, converted from the published safetensors by
// tools/eagle_to_gguf.py, whose tensors are named the way that converter writes
// them and whose geometry is checked against the target before it is used.
//
// The maths is the EAGLE-3 block: `fc` over the concatenated features, one
// decoder layer (RMSNorm, rope attention with its own KV, SwiGLU MLP, both
// residual), a final RMSNorm, and a projection to the DRAFT vocabulary -- which
// is often smaller than the target's, with a `d2t` map back to target ids. A
// draft logit row is over draft ids; proposing a token means argmax there and
// then mapping through d2t.
//
// It runs on the HOST in f32 arithmetic over f16 weights. A head is 1-2 blocks;
// the cost of a proposal is one block forward (milliseconds on this box), and
// the alternative -- another device tier competing with the target's expert
// cache for VRAM -- is worse on every card this engine targets.
#pragma once

#include <string>
#include <vector>

#include "krk/common.hpp"

namespace krk {

struct DraftHeadConfig {
    std::string kind = "eagle3"; // "eagle3" today; the file states it
    std::string target_arch;
    i32 hidden = 0;        // width the head works in
    i32 target_hidden = 0; // one slot of target hidden state
    i32 n_in = 1;          // how many target hidden slots `fc` consumes
    i32 n_layer = 0;
    i32 n_head = 0;
    i32 n_kv = 0;
    i32 head_dim = 0;
    i32 n_ff = 0;
    i32 draft_vocab = 0;
    i32 target_vocab = 0;
    f32 rope_theta = 10000.0f;
    f32 rms_eps = 1e-5f;
    // True when the file carries a `d2t` map, i.e. its vocabulary is a subset.
    bool has_d2t = false;
};

class DraftHead {
public:
    // Reads the file and validates its own geometry. `err` says which key or
    // tensor was wrong -- a head that loads but disagrees with the target about
    // the vocabulary proposes tokens the verifier can never accept, which reads
    // as a bad drafter rather than a wrong file.
    bool load(const std::string &path, std::string *err);
    bool ok() const { return loaded_; }
    const DraftHeadConfig &cfg() const { return cfg_; }
    const std::string &path() const { return path_; }

    // The target it is about to draft for. Refuses a head whose vocabulary or
    // hidden width is a different model's.
    bool check_target(i32 target_hidden, i32 target_vocab, std::string *err) const;
    // The head's own KV, as bytes: [ctx, n_kv * head_dim] x 2 in f32.
    size_t kv_bytes() const {
        return (k_cache_.size() + v_cache_.size()) * sizeof(f32);
    }

    // Start a new proposal round at `pos` (the position of the last verified
    // token). Clears the head's own KV.
    void reset(i32 ctx);

    // Keep the head's KV for positions [0, pos) and forget the rest. A rejected
    // proposal suffix is exactly what this drops; the rows are re-fed from the
    // target's captured hidden state afterwards, never kept.
    void truncate(i32 pos) { if (pos >= 0 && pos <= pos_) pos_ = pos; }
    // Attention ignores positions below `p`. This is how a bounded prime window
    // is expressed: the KV rows before the window are still in the buffer, and a
    // forward pass over them would be the cost the window exists to avoid.
    void drop_before(i32 p) { base_ = p > 0 ? p : 0; }
    i32 position() const { return pos_; }

    // One step. `feat` is n_in * target_hidden floats of the target's hidden
    // state for the current position; `tok_embd` is the target's embedding row
    // for the token at that position. Returns the draft-vocabulary logits row.
    //
    // `want_logits = false` stops after the K/V append: priming a window needs
    // the keys and values of every position but no output, and the block's
    // attention + MLP + head are most of its cost.
    bool step(const f32 *feat, const f32 *tok_embd, i32 pos, std::vector<f32> *logits,
              bool want_logits = true);

    // The hidden state this step wrote, i.e. the head's input for the NEXT
    // position. A proposal chain feeds it back with the proposed token's
    // embedding, which is what lets one head propose k tokens from one target
    // forward -- the same contract the nextn block has.
    const f32 *last_hidden() const { return out_.data(); }
    i32 hidden_width() const { return cfg_.hidden; }
    // Bytes of f16 weights this head holds, for the load line: a head that is
    // unexpectedly 6 GiB is a different file than the one that was converted.
    size_t weight_bytes() const {
        size_t n = 0;
        for (const std::vector<u16> *v : {&fc_, &ln1_, &ln2_, &lnf_, &wq_, &wk_,
                                          &wv_, &wo_, &gate_, &up_, &down_, &lm_})
            n += v->size() * sizeof(u16);
        return n + d2t_.size() * sizeof(i32);
    }

    // Draft id (index into the draft vocabulary) of the best next token.
    static i32 argmax(const std::vector<f32> &logits);
    // Draft id -> target id. Identity when the file carries no `d2t` map.
    i32 to_target(i32 draft_id) const;
    // The map itself, for a caller that has to move a whole distribution into
    // target space (speculative sampling compares distributions, not ids).
    // nullptr when the file carries no map, i.e. the identity.
    const i32 *d2t_data() const { return cfg_.has_d2t ? d2t_.data() : nullptr; }

private:
    bool loaded_ = false;
    std::string path_;
    DraftHeadConfig cfg_;
    // f16 weights, row-major [n_out, n_in], converted per matvec: a 1.7 GiB
    // artifact stays 1.7 GiB in RAM instead of doubling.
    std::vector<u16> fc_;
    std::vector<u16> ln1_, ln2_, lnf_; // RMSNorm gains
    std::vector<u16> wq_, wk_, wv_, wo_;
    std::vector<u16> gate_, up_, down_;
    std::vector<u16> lm_;
    std::vector<i32> d2t_;
    std::vector<f32> k_cache_, v_cache_; // [ctx, n_kv * head_dim] per layer
    i32 ctx_ = 0;
    i32 pos_ = 0;
    i32 base_ = 0; // first position attention reads
    // Scratch, sized on load.
    std::vector<f32> x_, h_, hn_, q_, k_, v_, att_, proj_, ff_, out_;
};

} // namespace krk
