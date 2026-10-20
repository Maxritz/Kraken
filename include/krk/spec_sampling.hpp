// Speculative sampling: the distribution-preserving accept/reject rule.
//
// Greedy speculation is a prefix match -- the target's argmax either equals the
// proposal or it does not -- so "did the target agree" is the whole test. Once
// the sampler is stochastic that test is wrong: a proposal the target merely
// found unlikely still has to be accepted sometimes, or the output distribution
// is not the target's. Leviathan et al. (arXiv:2302.01318) and Chen et al.
// ("Accelerating Large Language Model Decoding with Speculative Sampling")
// give the rule that keeps it exact:
//
//   accept proposal x with probability min(1, p(x) / q(x)), else draw the
//   replacement from the residual max(0, p - q), normalised.
//
// The proof is one line of arithmetic, and the reason a wrong implementation is
// hard to spot: P(out = x) = q(x) * min(1, p(x)/q(x)) + P(reject) * r(x)
//                          = min(p(x), q(x)) + (p(x) - q(x))+
//                          = p(x),
// because sum(p - q)+ = 1 - sum min(p, q) when both are distributions. A
// sampler that always accepts, or that draws the replacement from p, still
// produces fluent text -- it just produces the DRAFT's distribution, or a
// slightly wrong mixture, and nothing in the output says so. That is why
// spec_verify_identity() exists: it enumerates the identity above, so the test
// can fail on the arithmetic rather than on a text comparison.
//
// These functions are pure float maths on host vectors: no backend, no device,
// which is what lets the suite pin the rule on a CPU-only fixture.
#pragma once

#include <string>
#include <vector>

#include "krk/common.hpp"

namespace krk {

// One verification decision. `accepted` says whether the proposal survived; on
// a rejection `token` is the residual draw that replaces it.
struct SpecStep {
    i32 token = -1;
    bool accepted = false;
};

// The rule above, for one position. `p` and `q` are dense target-vocabulary
// distributions (both sum to 1 within rounding), `x` the proposed token,
// `u_accept` and `u_residual` uniforms in [0,1) drawn by the caller -- passing
// them in is what makes a round replayable from a seed.
//
// A proposal with q(x) == 0 is accepted: the rule's min(1, p/q) is undefined
// there, and the correct limit is 1 (the draft could not have proposed it, so
// there is nothing to correct -- the mass that reaches it is p(x) itself, which
// is what the residual arm would produce anyway).
SpecStep spec_accept_step(const std::vector<f32> &p, const std::vector<f32> &q, i32 x,
                          f32 u_accept, f32 u_residual);

// max(0, p - q), normalised. False when p and q differ in size, or when the
// residual has no mass at all (p == q), in which case the caller must fall back
// to drawing from p -- there is no correction left to make.
bool spec_residual(const std::vector<f32> &p, const std::vector<f32> &q,
                   std::vector<f32> *out);

// The exactness identity, enumerated over the vocabulary rather than sampled:
// for every token x, q(x)*min(1, p(x)/q(x)) + P(reject)*(p(x)-q(x))+ equals
// p(x). Returns false and fills `why` with the worst token when it does not, so
// a broken rule names the token whose mass is wrong instead of "not equal".
bool spec_verify_identity(const std::vector<f32> &p, const std::vector<f32> &q,
                          f64 tol, std::string *why);

// softmax(logits / temp) into a dense target-vocabulary row. `temp <= 0` is
// one-hot argmax, which is the greedy limit; `logits` of width `n` and `prob`
// of width `n`. Used for the DRAFT side of a round, where the only thing that
// matters for exactness is that the proposal was drawn from the row this
// function describes.
void spec_softmax(const f32 *logits, i32 n, f32 temp, std::vector<f32> *prob);

// A draft-vocabulary distribution (size `n_draft`, indexed by draft id) mapped
// into target space through `d2t` (null means the identity map). Ids in `d2t`
// outside [0, n_target) are skipped rather than clamped: a clamped id would put
// the draft's probability on a token it never proposed. The result is
// renormalised, because a map with duplicate ids would otherwise sum past 1.
void spec_map_dist(const f32 *draft_prob, i32 n_draft, const i32 *d2t, i32 n_target,
                   std::vector<f32> *out);

// Draw one token from a dense distribution with `u` in [0,1). The caller owns
// the uniformity; this only walks the cumulative sum, and returns the last
// non-zero index when rounding leaves the sum a hair short of `u`.
i32 spec_draw(const std::vector<f32> &dist, f32 u);

} // namespace krk
