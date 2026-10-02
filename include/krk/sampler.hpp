// ============================================================================
//  sampler.hpp — deterministic, seedable sampling.
//
//  Chain: repetition penalty -> temperature -> top-k -> top-p -> min-p ->
//  inverse-CDF draw. Greedy mode short-circuits everything to an argmax.
// ============================================================================
#ifndef KRK_SAMPLER_HPP
#define KRK_SAMPLER_HPP

#include <vector>

#include "krk/common.hpp"

namespace krk {

struct SampleParams {
    f32 temp = 0.8f;
    i32 top_k = 40;
    f32 top_p = 0.95f;
    f32 min_p = 0.05f;
    f32 repeat_penalty = 1.1f;
    i32 repeat_last_n = 64;
    bool greedy = false;
    u64 seed = 0;
};

// splitmix64 + xorshift128+ — cheap, deterministic, no libc dependency.
class Rng {
public:
    void seed(u64 s);
    u32 next_u32();
    u64 next_u64();
    f32 next_f32(); // [0,1)

private:
    u64 s_[2] = {0x9E3779B97F4A7C15ull, 0xBF58476D1CE4E5B9ull};
};

class Sampler {
public:
    void reset(const SampleParams &p);

    // Records an accepted token for the repetition penalty.
    void accept(i32 token);
    void accept(const i32 *tokens, int n);

    // Draws the next token from logits[n_vocab].
    i32 sample(const f32 *logits, i32 n_vocab);

    const SampleParams &params() const { return p_; }

private:
    void apply_repetition(std::vector<f32> &logits);

    SampleParams p_;
    Rng rng_;
    std::vector<i32> recent_;
    int recent_pos_ = 0;
};

} // namespace krk

#endif // KRK_SAMPLER_HPP
