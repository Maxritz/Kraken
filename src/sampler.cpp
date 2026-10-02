// sampler.cpp — deterministic sampling: rep-penalty -> temp -> top-k -> top-p
// -> min-p -> inverse-CDF draw.
#include "krk/sampler.hpp"

#include <algorithm>
#include <cmath>

namespace krk {

// ---------------------------------------------------------------------------
// Rng — splitmix64 seeding, xorshift128+ stream
// ---------------------------------------------------------------------------

void Rng::seed(u64 s) {
    auto splitmix = [](u64 &x) {
        x += 0x9E3779B97F4A7C15ull;
        u64 z = x;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    };
    u64 x = s ? s : 0x123456789ABCDEFull;
    s_[0] = splitmix(x);
    s_[1] = splitmix(x);
    if (s_[0] == 0 && s_[1] == 0) s_[0] = 1;
}

u64 Rng::next_u64() {
    u64 x = s_[0];
    const u64 y = s_[1];
    s_[0] = y;
    x ^= x << 23;
    s_[1] = x ^ y ^ (x >> 17) ^ (y >> 26);
    return s_[1] + y;
}

u32 Rng::next_u32() { return static_cast<u32>(next_u64() >> 32); }

f32 Rng::next_f32() {
    // 24 random mantissa bits -> [0,1)
    return static_cast<f32>(next_u32() >> 8) * (1.0f / 16777216.0f);
}

// ---------------------------------------------------------------------------
// Sampler
// ---------------------------------------------------------------------------

void Sampler::reset(const SampleParams &p) {
    p_ = p;
    rng_.seed(p.seed ? p.seed : 0xC0FFEEull);
    recent_.assign(static_cast<size_t>(std::max(1, p.repeat_last_n > 0 ? p.repeat_last_n : 64)), -1);
    recent_pos_ = 0;
}

void Sampler::accept(i32 token) {
    if (recent_.empty()) return;
    recent_[static_cast<size_t>(recent_pos_)] = token;
    recent_pos_ = (recent_pos_ + 1) % static_cast<int>(recent_.size());
}

void Sampler::accept(const i32 *tokens, int n) {
    for (int i = 0; i < n; i++) accept(tokens[i]);
}

void Sampler::apply_repetition(std::vector<f32> &logits) {
    if (p_.repeat_penalty <= 1.0f || p_.repeat_last_n <= 0) return;
    for (i32 tok : recent_) {
        if (tok < 0 || static_cast<size_t>(tok) >= logits.size()) continue;
        f32 &l = logits[static_cast<size_t>(tok)];
        l = l > 0 ? l / p_.repeat_penalty : l * p_.repeat_penalty;
    }
}

i32 Sampler::sample(const f32 *logits, i32 n_vocab) {
    if (n_vocab <= 0) return 0;
    std::vector<f32> work(logits, logits + n_vocab);

    // greedy: exact argmax, no repetition penalty. A penalty would
    // bias the argmax away from tokens that merely appeared in the
    // prompt (e.g. <|im_end|>), diverging from reference greedy
    // decoding; sampling modes still apply it below.
    if (p_.greedy || p_.temp <= 0.0f) {
        i32 best = 0;
        f32 bv = work[0];
        for (i32 i = 1; i < n_vocab; i++)
            if (work[i] > bv) { bv = work[i]; best = i; }
        return best;
    }

    apply_repetition(work);

    const f32 inv_temp = 1.0f / p_.temp;
    // Candidate list of (index, score). k-sort is enough: only the top
    // min(top_k, n) survive, so a full O(n log n) sort is unnecessary work.
    std::vector<i32> idx(static_cast<size_t>(n_vocab));
    for (i32 i = 0; i < n_vocab; i++) idx[static_cast<size_t>(i)] = i;

    int keep = n_vocab;
    if (p_.top_k > 0 && p_.top_k < keep) keep = p_.top_k;

    const auto better = [&work](i32 a, i32 b) { return work[a] > work[b]; };
    if (keep < n_vocab) {
        std::partial_sort(idx.begin(), idx.begin() + keep, idx.end(), better);
    } else {
        std::sort(idx.begin(), idx.end(), better);
    }

    // softmax over the survivors
    std::vector<f32> prob(static_cast<size_t>(keep));
    f32 mx = work[idx[0]];
    f32 sum = 0;
    for (int i = 0; i < keep; i++) {
        const f32 e = std::exp((work[idx[static_cast<size_t>(i)]] - mx) * inv_temp);
        prob[static_cast<size_t>(i)] = e;
        sum += e;
    }
    if (sum <= 0) return idx[0];
    for (int i = 0; i < keep; i++) prob[static_cast<size_t>(i)] /= sum;

    // top-p (nucleus)
    int cut = keep;
    if (p_.top_p < 1.0f && p_.top_p > 0.0f) {
        f32 c = 0;
        for (int i = 0; i < keep; i++) {
            c += prob[static_cast<size_t>(i)];
            if (c >= p_.top_p) { cut = i + 1; break; }
        }
    }
    // min-p relative to the peak probability
    if (p_.min_p > 0.0f) {
        const f32 thr = p_.min_p * prob[0];
        int c = 0;
        while (c < cut && prob[static_cast<size_t>(c)] >= thr) c++;
        if (c > 0) cut = c;
    }
    if (cut < 1) cut = 1;

    // renormalize the surviving slice and draw
    f32 tot = 0;
    for (int i = 0; i < cut; i++) tot += prob[static_cast<size_t>(i)];
    if (tot <= 0) return idx[0];
    f32 r = rng_.next_f32() * tot;
    f32 acc = 0;
    for (int i = 0; i < cut; i++) {
        acc += prob[static_cast<size_t>(i)];
        if (r <= acc) return idx[static_cast<size_t>(i)];
    }
    return idx[static_cast<size_t>(cut - 1)];
}

} // namespace krk
