// ============================================================================
//  draft_block.hpp -- the interface the speculative loop drives.
//
//  The engine's generate loop knows a drafter by six calls and nothing else:
//  which target layers to capture, where the capture slots are, how many draft
//  rows a block can hold, how much context its cache can take, how to inject a
//  committed chunk, and how to propose a block. DFlash and DSpark are different
//  files with different heads and different block maths, but they answer those
//  six questions in the same shape, so the loop asks the interface and not the
//  class.
//
//  This exists because the alternative -- a second concrete pointer beside
//  `dflash_` and a second branch at every one of the seven call sites inside
//  the generate loop -- is how a speculative loop grows a second copy of
//  itself. A new drafter is a subclass and one line in `Engine::load_draft`.
//
//  Nothing here allocates on the target's behalf. A run with no drafter holds a
//  null pointer and spends nothing on any of it.
// ============================================================================
#ifndef KRK_DRAFT_BLOCK_HPP
#define KRK_DRAFT_BLOCK_HPP

#include <string>
#include <vector>

#include "krk/common.hpp"
#include "krk/expert_cache.hpp"

namespace krk {

class Backend;

struct DraftBlock {
    virtual ~DraftBlock() = default;

    // Capture ids in the TARGET's layer numbering: id l is the residual stream
    // entering block l, and an id equal to the target's layer count is the
    // pre-final-norm state.
    virtual const std::vector<i32> &target_layers() const = 0;
    // [n_aux][chunk][n_embd], layer-major.
    virtual void *capture_slot(i32 aux) const = 0;
    // Inject the chunk the target just computed: n tokens from pos0.
    virtual bool commit(i32 pos0, i32 n) = 0;
    virtual i32 block_size() const = 0;
    virtual i64 kv_capacity() const = 0;
    // Up to n_draft logit rows, n_vocab floats each, one per candidate. Row k
    // is the candidate for position pos+k. Returns the rows written.
    virtual i32 draft_block(const QuantTensor &tok_embd, const QuantTensor &lm_head,
                            i32 n_vocab, i32 id_last, i32 n_draft, i32 pos,
                            f32 *rows_out) = 0;
    virtual void unload() = 0;

    // One line for the run report. Every drafter says what it is in its own
    // words, because "a drafter is loaded" is not a fact a run can be read on.
    virtual std::string describe() const = 0;
    // The kind, for switches that must tell the two apart (DSpark can resample
    // a row's logits through its Markov head; DFlash has no such head).
    virtual const char *kind() const = 0;
};

} // namespace krk

#endif // KRK_DRAFT_BLOCK_HPP
