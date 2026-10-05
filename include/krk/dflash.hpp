// ============================================================================
//  dflash.hpp — Poolside's DFlash speculative head set, loaded as a drafter.
//
//  A DFlash GGUF is not a model. It has no token embeddings and no output head,
//  and it cannot be asked to continue a sequence on its own: every token it
//  predicts arrives through the target's own weights. What it does have is a
//  small transformer stack plus the two projections that let it read the
//  target's *hidden states* instead of its tokens.
//
//  The contract, read off poolside's fork (src/models/dflash.cpp, branch
//  `laguna`) and the converters that write these files:
//
//   1. Feature fusion (the "encoder"). The target copies the residual stream
//      entering each of a few chosen layers, plus — when one of the ids equals
//      the target's layer count — the pre-final-norm state. The drafter RMS
//      norms each of those feature blocks, scales it by that block's own weight,
//      concatenates them per token, and runs one `fc` followed by one RMS norm.
//      The result is not a token embedding and never re-enters the target; it is
//      the K/V source for step 2.
//
//   2. KV injection. The fused features are projected into per-layer K and V
//      (K carried through the layer's k-norm and rope) and written straight into
//      a KV cache of the drafter's own, at the positions of the tokens they came
//      from. This is why the drafter needs no embedding: its context lives in
//      that cache, as fused features, and is what the block below attends to.
//
//   3. Masked block decode. One block per draft step: the last committed token
//      followed by `n_draft` mask tokens. The block's own K/V is appended to the
//      cache and every mask row is read out as a candidate for its own position.
//
//  Two spellings of the same file exist and both are read here. The current one
//  (`general.architecture = dflash`, `enc.*`, `dflash.decoder_arch`) is what
//  poolside's own converter writes today; the older one
//  (`qwen35-dflash-draft`, `dflash.fc/hidden_norm/aux_hidden_norm.*`) says the
//  same three things under different names, with the per-aux norms stored
//  separately instead of stacked. Where the two disagree about a *behaviour*
//  rather than a name, the file's own flag decides: `decoder_arch = "laguna"`
//  and `dflash.context_kv_layer_norm` are the two statements of "norm the
//  injection input with the layer's attn_norm", and the laguna flag is also what
//  makes the block causal.
//
//  Everything in this file is driven by the target: `commit()` is called from
//  inside the target's forward pass, with the chunk it just computed, and writes
//  the injection into the drafter's cache. Nothing here allocates on the target's
//  behalf, so a run with no DFlash drafter spends nothing on any of it.
// ============================================================================
#ifndef KRK_DFLASH_HPP
#define KRK_DFLASH_HPP

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "krk/backend.hpp"
#include "krk/expert_cache.hpp"
#include "krk/gguf.hpp"
#include "krk/model.hpp"

namespace krk {

// Is this architecture a DFlash head set, i.e. something `DflashDraft::load`
// will accept? Kept next to the loader so a caller that only has the arch string
// (Engine::load_draft, deciding whether `--draft` names a model or a head set)
// does not have to re-derive the list.
bool dflash_arch(std::string_view arch);

class DflashDraft {
public:
    DflashDraft() = default;
    ~DflashDraft();
    DflashDraft(const DflashDraft &) = delete;
    DflashDraft &operator=(const DflashDraft &) = delete;

    // `chunk` and `ctx` come from the target: the capture buffer has to hold one
    // target prefill chunk, and the drafter's cache is trimmed to the target's
    // context so the two cannot disagree about how far a position can reach.
    // `target_embd` is the target's embedding width, which is what the feature
    // blocks in the file are sized against; a mismatch is a refusal, not a
    // silent reinterpretation.
    bool load(Backend &be, const std::string &path, i64 chunk, i64 ctx,
              i32 target_embd, i32 target_layer_count, std::string *err);
    void unload();

    // ---- what the target side needs to drive this -------------------------
    // Capture ids, in the target's own layer numbering: id l means "the
    // residual stream entering block l", and an id equal to the target's layer
    // count means the pre-final-norm state. Both are what `forward_core` has in
    // ws_x_ at those two moments.
    const std::vector<i32> &target_layers() const { return target_layers_; }
    i32 n_aux() const { return static_cast<i32>(target_layers_.size()); }
    i32 block_size() const { return block_size_; }
    i32 mask_id() const { return mask_id_; }
    i32 n_embd() const { return cfg_.n_embd; }
    i32 n_embd_enc() const { return n_embd_enc_; }
    i32 n_layer() const { return cfg_.n_layer; }
    i64 chunk() const { return chunk_; }
    i64 kv_capacity() const { return kv_cap_; }
    bool causal() const { return causal_; }
    const std::string &name() const { return cfg_.name; }
    const std::string &arch() const { return cfg_.arch; }
    // The per-file behaviour flags, for the run report: a drafter whose block is
    // causal and whose injection is normed is a different object from one that
    // is neither, and the two are told apart by these, not by the file name.
    bool decoder_laguna() const { return decoder_laguna_; }
    bool kv_norm() const { return kv_norm_; }
    bool zero_mask_embedding() const { return zero_mask_embed_; }
    const std::vector<i32> &layer_window() const { return layer_window_; }

    // ---- the capture buffer ----------------------------------------------
    // [n_aux][chunk][n_embd], layer-major, so the target fills one aux slot with
    // a single contiguous copy and the assembly below reads it back the same way.
    void *capture_slot(i32 aux) const;
    i64 capture_stride() const { return chunk_ * cfg_.n_embd; }

    // ---- step 1 + 2: fuse the captured features and inject their K/V ------
    // `n` tokens at positions pos0..pos0+n-1, which is the shape of one target
    // forward call: a prefill chunk, a speculative verification block, or a
    // single decode row.
    bool commit(i32 pos0, i32 n);

    // ---- step 3: one masked block ----------------------------------------
    // Writes up to n_draft logit rows, four bytes of f32 each, one per candidate.
    // Row k is the candidate for position pos+k, which is what the caller's
    // greedy speculation window expects. Returns the number of rows written.
    i32 draft_block(const QuantTensor &tok_embd, const QuantTensor &lm_head,
                    i32 n_vocab, i32 id_last, i32 n_draft, i32 pos,
                    f32 *rows_out);

private:
    bool load_weights(std::string *err);

    Backend *be_ = nullptr;
    // In a pointer so unload() can drop the mapping without needing Gguf to be
    // assignable; the head set is small enough that holding it is not the point,
    // but a rebuilt drafter in the same process must not keep a Windows file
    // handle open on the file it is about to overwrite.
    std::unique_ptr<Gguf> gguf_;
    ModelConfig cfg_;
    std::vector<LayerWeights> layers_;

    // The tensors that are not part of a layer.
    QuantTensor fc_;        // [n_embd_enc, n_embd]
    f32 *enc_out_norm_ = nullptr; // enc.output_norm / dflash.hidden_norm
    f32 *out_norm_ = nullptr;     // output_norm: the norm before the target's head
    f32 *aux_norm_ = nullptr;     // [n_embd, n_aux], stacked
    std::vector<f32> aux_norm_host_;
    // One rope table for every layer: DFlash has a single rotor on purpose, so
    // there is no inv_freq_swa_ and no per-layer choice to make.
    std::vector<f32> inv_freq_;

    std::vector<i32> target_layers_;
    std::vector<i32> layer_window_; // per-layer sliding window, 0 = full
    std::vector<i32> layer_causal_; // per-layer noise-block mask, 0 = block-visible
    i32 n_embd_enc_ = 0;
    i32 block_size_ = 0;
    i32 mask_id_ = -1;
    bool decoder_laguna_ = false;
    bool kv_norm_ = false;
    bool causal_ = false;
    bool zero_mask_embed_ = false;
    i32 n_causal_layers_ = 0;

    i64 chunk_ = 0;   // capture rows == the target's prefill chunk
    i64 kv_cap_ = 0;
    // Where block row 0 (the last committed token) sits relative to the next
    // free position. The reference builds the block at `n_past` and reads the
    // target's logits from `n_past + i` after incrementing n_past once, so row 0
    // lands on the free position and mask row k is the candidate for pos+k-1 --
    // which is what the caller wants: `pos` is the free position and pos+k-1 is
    // the k-th token this round can emit. KRK_DFLASH_BLOCK_POS shifts it,
    // because the reference's own arithmetic is only unambiguous up to this
    // choice and a wrong one costs every candidate in the round.
    i32 block_base_ = 0;

    // device
    void *cap_ = nullptr;      // [n_aux][chunk][n_embd] activations
    void *feat_ = nullptr;     // [chunk][n_embd_enc] assembled encoder input
    void *enc_out_ = nullptr;  // [chunk][n_embd]   encoder output (injection input)
    void *ws_x_ = nullptr;     // [chunk][n_embd]
    void *ws_xn_ = nullptr;    // [chunk][n_embd]
    void *ws_x2_ = nullptr;    // [chunk][n_embd]
    void *ws_q_ = nullptr;     // [chunk][q_dim]
    void *ws_k_ = nullptr;     // [chunk][kv_dim]
    void *ws_v_ = nullptr;     // [chunk][kv_dim]
    void *ws_attn_ = nullptr;  // [chunk][q_dim]
    void *ws_gate_ = nullptr;  // [chunk][n_ff]
    void *ws_up_ = nullptr;    // [chunk][n_ff]
    void *ws_agate_ = nullptr; // [chunk][q_dim] output gate
    void *ws_logits_ = nullptr; // [block_size-1][n_vocab]
    void *kcache_ = nullptr;
    void *vcache_ = nullptr;

    // host
    std::vector<f32> feat_host_;
    std::vector<f32> tmp_host_;
    std::vector<i32> block_tok_;

    // Bookkeeping for the run report: how many tokens have been injected and how
    // many block rows are in the cache, so a caller can tell a drafter that is
    // being fed from one that is running on an empty context.
    i64 committed_ = 0;
    i64 blocks_ = 0;
    friend struct DflashStats;
};

struct DflashStats {
    i64 committed = 0;
    i64 blocks = 0;
};
DflashStats dflash_stats(const DflashDraft &d);

} // namespace krk

#endif // KRK_DFLASH_HPP
