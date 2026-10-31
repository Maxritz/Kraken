#include "krk/moe_route.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace krk {

bool moe_group_mask(u8 *keep, const f32 *sel, i32 n_expert, i32 n_group,
                    i32 group_used) {
    if (!keep || !sel) return false;
    if (n_expert <= 0 || n_group <= 1 || group_used <= 0) return false;
    if (n_group > n_expert || group_used > n_group) return false;
    if (n_expert % n_group != 0) return false;

    const i32 per_group = n_expert / n_group;
    // A group's score is the sum of its best two, in SELECTION-score order.
    std::vector<f32> group_score(static_cast<size_t>(n_group), 0.0f);
    for (i32 g = 0; g < n_group; g++) {
        const i32 base = g * per_group;
        f32 best = -std::numeric_limits<f32>::infinity();
        f32 second = -std::numeric_limits<f32>::infinity();
        for (i32 i = 0; i < per_group; i++) {
            const f32 v = sel[base + i];
            if (v > best) {
                second = best;
                best = v;
            } else if (v > second) {
                second = v;
            }
        }
        // A one-expert group has no second term; adding -inf would make every
        // such group unselectable, which is not what the reference means.
        group_score[static_cast<size_t>(g)] =
            best + (per_group > 1 ? second : 0.0f);
    }

    // The best `group_used` groups, by score, ties going to the lower index.
    std::vector<i32> order(static_cast<size_t>(n_group));
    for (i32 g = 0; g < n_group; g++) order[static_cast<size_t>(g)] = g;
    std::stable_sort(order.begin(), order.end(), [&](i32 a, i32 b) {
        return group_score[static_cast<size_t>(a)] >
               group_score[static_cast<size_t>(b)];
    });

    for (i32 e = 0; e < n_expert; e++) keep[e] = 0;
    for (i32 s = 0; s < group_used; s++) {
        const i32 g = order[static_cast<size_t>(s)];
        for (i32 i = 0; i < per_group; i++)
            keep[g * per_group + i] = 1;
    }
    return true;
}

bool moe_group_mask_or_all(u8 *keep, const f32 *sel, i32 n_expert, i32 n_group,
                           i32 group_used) {
    if (!keep || n_expert <= 0) return false;
    if (n_group <= 1 || group_used <= 0) {
        for (i32 e = 0; e < n_expert; e++) keep[e] = 1;
        return true;
    }
    return moe_group_mask(keep, sel, n_expert, n_group, group_used);
}

} // namespace krk
