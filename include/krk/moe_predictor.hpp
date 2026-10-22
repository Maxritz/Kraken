#ifndef KRK_MOE_PREDICTOR_HPP
#define KRK_MOE_PREDICTOR_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <vector>

namespace krk {

// Online, request-local prediction for MoE expert-cache staging.
//
// The predictor sees only expert IDs selected by the model's native router.
// It never edits those IDs or their weights. Temporal candidates come from a
// target layer's recent routes; cross-layer candidates are the current token's
// routes at the preceding MoE layer. Per-layer precision EWMAs gate staging, so
// the policy can fall back to no prefetch when either signal is inaccurate.
class MoeRoutePredictor {
public:
    void configure(int layers, int experts, int history) {
        layers_ = layers > 0 ? layers : 0;
        experts_ = experts > 0 ? experts : 0;
        history_ = history < 2 ? 2 : (history > 16 ? 16 : history);
        state_.assign(static_cast<std::size_t>(layers_), LayerState{});
        reset_counters();
    }

    void reset() {
        for (LayerState &s : state_) s = LayerState{};
        reset_counters();
    }

    // Record the native route for a decode row. This scores the preceding
    // target-layer prediction, updates its method confidence, and saves the
    // current route for a later adjacent-layer prediction.
    void observe(int layer, const int *ids, int n) {
        if (!valid_layer(layer) || !ids || n <= 0) return;
        LayerState &s = state_[static_cast<std::size_t>(layer)];
        const std::vector<int> actual = unique_valid(ids, n);
        // Score the prior token's separate signals. EMA precision is a staging
        // gate, not a claim of probability calibration.
        update_accuracy(s.temporal_accuracy, s.temporal_seen,
                        s.pending_temporal, actual);
        update_accuracy(s.cross_accuracy, s.cross_seen,
                        s.pending_cross, actual);
        s.pending_temporal.clear();
        s.pending_cross.clear();
        // Candidate counters describe only sets actually submitted to the
        // cache. A non-issued (confidence-gated) set is still used above to
        // train its signal but is not included in the reported precision.
        if (!s.pending_issued.empty()) {
            for (int e : s.pending_issued)
                if (contains(actual, e)) ++candidate_hits_;
            candidate_total_ += s.pending_issued.size();
        }
        s.pending_issued.clear();
        s.current = actual;
        s.history.push_back(actual);
        while (static_cast<int>(s.history.size()) > history_)
            s.history.pop_front();
    }

    // Propose a bounded, unique set for `target_layer`. policy: 1 temporal,
    // 2 adjacent-layer, 3 combine both, preferring the more accurate signal.
    // `min_precision` gates a signal; outputs are hints only.
    void predict(int target_layer, int source_layer, int limit, float min_precision,
                 int policy, std::vector<int> &out) {
        out.clear();
        if (!valid_layer(target_layer) || limit <= 0) return;
        LayerState &target = state_[static_cast<std::size_t>(target_layer)];
        std::vector<int> temporal;
        temporal_candidates(target, limit, temporal);
        std::vector<int> cross;
        if (valid_layer(source_layer) && source_layer != target_layer)
            cross = state_[static_cast<std::size_t>(source_layer)].current;
        trim(cross, limit);

        // Save candidate lists even when confidence gates staging off: the
        // current token's authoritative route provides online feedback.
        target.pending_temporal = temporal;
        target.pending_cross = cross;
        target.pending_issued.clear();
        const float tc = target.temporal_seen ? target.temporal_accuracy : 0.0f;
        const float cc = target.cross_seen ? target.cross_accuracy : 0.0f;
        // A signal must have enough feedback to be evaluated. This prevents
        // cold-start guesses from staging on the first token; the second token
        // can bootstrap from a just-observed native route, then the EMA gates it.
        const bool t_ok = !temporal.empty() &&
                          (!target.temporal_seen || tc >= min_precision);
        const bool c_ok = !cross.empty() &&
                          (!target.cross_seen || cc >= min_precision);
        if (policy == 1) {
            if (t_ok) out = temporal;
        } else if (policy == 2) {
            if (c_ok) out = cross;
        } else if (t_ok && c_ok) {
            const bool temporal_first = tc >= cc;
            const std::vector<int> &first = temporal_first ? temporal : cross;
            const std::vector<int> &second = temporal_first ? cross : temporal;
            append_bounded(out, first, limit);
            append_bounded(out, second, limit);
        } else if (t_ok) {
            out = temporal;
        } else if (c_ok) {
            out = cross;
        }
        trim(out, limit);
        target.pending_issued = out;
        if (!out.empty()) ++predictions_issued_;
    }

    void note_staged(std::size_t n) { staged_ += n; }
    std::uint64_t predictions_issued() const { return predictions_issued_; }
    std::uint64_t candidates() const { return candidate_total_; }
    std::uint64_t candidate_hits() const { return candidate_hits_; }
    std::uint64_t staged() const { return staged_; }
    float candidate_precision() const {
        return candidate_total_ ? static_cast<float>(candidate_hits_) /
                                     static_cast<float>(candidate_total_)
                               : 0.0f;
    }

private:
    struct LayerState {
        std::deque<std::vector<int>> history;
        std::vector<int> current;
        std::vector<int> pending_temporal;
        std::vector<int> pending_cross;
        std::vector<int> pending_issued;
        float temporal_accuracy = 0.0f;
        float cross_accuracy = 0.0f;
        bool temporal_seen = false;
        bool cross_seen = false;
    };

    bool valid_layer(int layer) const {
        return layer >= 0 && layer < layers_;
    }

    static bool contains(const std::vector<int> &ids, int value) {
        return std::find(ids.begin(), ids.end(), value) != ids.end();
    }

    std::vector<int> unique_valid(const int *ids, int n) const {
        std::vector<int> out;
        out.reserve(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) {
            const int e = ids[i];
            if (e >= 0 && e < experts_ && !contains(out, e)) out.push_back(e);
        }
        return out;
    }

    void temporal_candidates(const LayerState &s, int limit,
                             std::vector<int> &out) const {
        out.clear();
        if (s.history.empty() || experts_ <= 0) return;
        std::vector<unsigned> counts(static_cast<std::size_t>(experts_), 0);
        // A recency-weighted frequency ordering adapts quickly while retaining
        // repeated experts across the configured window.
        unsigned weight = 1;
        for (auto it = s.history.rbegin(); it != s.history.rend(); ++it, ++weight)
            for (int e : *it)
                if (e >= 0 && e < experts_)
                    counts[static_cast<std::size_t>(e)] +=
                        static_cast<unsigned>(history_ + 1) -
                        (weight < static_cast<unsigned>(history_)
                             ? weight
                             : static_cast<unsigned>(history_));
        // Select at most `limit` candidates directly, avoiding a full sort of
        // all experts on every decode token.
        for (int pick = 0; pick < (limit < experts_ ? limit : experts_); ++pick) {
            int best = -1;
            unsigned best_count = 0;
            for (int e = 0; e < experts_; ++e) {
                const unsigned count = counts[static_cast<std::size_t>(e)];
                if (count > best_count && !contains(out, e)) {
                    best = e;
                    best_count = count;
                }
            }
            if (best < 0) break;
            out.push_back(best);
        }
    }

    static void trim(std::vector<int> &ids, int limit) {
        if (limit < 0) limit = 0;
        if (static_cast<int>(ids.size()) > limit)
            ids.resize(static_cast<std::size_t>(limit));
    }

    static void append_bounded(std::vector<int> &dst,
                               const std::vector<int> &src, int limit) {
        for (int e : src) {
            if (static_cast<int>(dst.size()) >= limit) break;
            if (!contains(dst, e)) dst.push_back(e);
        }
    }

    static float precision(const std::vector<int> &predicted,
                           const std::vector<int> &actual) {
        if (predicted.empty()) return 0.0f;
        unsigned hits = 0;
        for (int e : predicted)
            if (contains(actual, e)) ++hits;
        return static_cast<float>(hits) / static_cast<float>(predicted.size());
    }

    static void update_accuracy(float &ema, bool &seen,
                                const std::vector<int> &predicted,
                                const std::vector<int> &actual) {
        if (predicted.empty()) return;
        const float sample = precision(predicted, actual);
        ema = seen ? (0.5f * ema + 0.5f * sample) : sample;
        seen = true;
    }

    void reset_counters() {
        predictions_issued_ = 0;
        candidate_total_ = 0;
        candidate_hits_ = 0;
        staged_ = 0;
    }

    int layers_ = 0;
    int experts_ = 0;
    int history_ = 4;
    std::vector<LayerState> state_;
    std::uint64_t predictions_issued_ = 0;
    std::uint64_t candidate_total_ = 0;
    std::uint64_t candidate_hits_ = 0;
    std::uint64_t staged_ = 0;
};

} // namespace krk

#endif // KRK_MOE_PREDICTOR_HPP
