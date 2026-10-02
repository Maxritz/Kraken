// ============================================================================
//  expert_cache.hpp — bounded, lazy residency for Mixture-of-Experts weights.
//
//  Expert tensors (`blk.N.ffn_{gate,up,down}_exps.weight`) dominate the size of
//  an MoE model: Qwen3-235B-A22B carries ~235B parameters but only ~22B are
//  touched per token. Uploading every expert at load time would demand the whole
//  model resident and defeat the architecture.
//
//  So experts are never eagerly loaded. The loader keeps only the GGUF mapping;
//  the router decides per token which experts run, and ExpertCache materializes
//  exactly those three matrices on demand into a byte-budgeted residency set.
//  Evicted slots are released, so a machine with far less memory than the model
//  still decodes — just with more reload traffic.
//
//  The residency policy is LFU with aging, not LRU. Expert popularity is
//  extremely skewed and roughly stable across a conversation: a handful of
//  experts serve most tokens. Plain LRU evicts a long-serving hot expert the
//  first time a cold token brushes past it, which is precisely backwards. So:
//
//    * every hit increments the slot's counter; a miss that loads also counts
//    * a slot whose counter reaches the pin threshold becomes pinned — it is
//      never chosen for eviction while pinned (pins still count, so a pin is
//      always earned by traffic, never granted for free)
//    * the counters decay by DECAY every DECAY_PERIOD acquires, so a favourite
//      of the first thousand tokens cools off and releases its pin when the
//      conversation moves on
//
//  A victim is chosen by: unpinned first, then the lowest counter, then the
//  oldest load. Guarantee: if at least one slot can hold the request, a pinned
//  expert is never evicted; and the cache always makes progress (see acquire).
//
//  Pointers returned by acquire() stay valid until the next acquire() call.
// ============================================================================
#ifndef KRK_EXPERT_CACHE_HPP
#define KRK_EXPERT_CACHE_HPP

#include <list>
#include <unordered_map>

#include "krk/backend.hpp"
#include "krk/gguf.hpp"
#include "krk/quant.hpp"

namespace krk {

// A quantized weight matrix resident in backend memory. Defined here because the
// cache is what materializes these for experts; the model loader uses the same
// descriptor for its eagerly-loaded matrices.
struct QuantTensor {
    void *data = nullptr;        // device pointer (or host, on the CPU backend)
    DType type = DType::Unknown; // Unknown => tensor absent
    i64 n_in = 0;                // contiguous inner dimension
    i64 n_out = 1;               // outer dimension

    bool present() const { return data != nullptr && type != DType::Unknown; }
};

// One layer's three expert tensors, still pointing into the GGUF mapping. The
// loader records these instead of uploading, which is what makes loading lazy.
struct ExpertSource {
    const GgufTensor *gate = nullptr;
    const GgufTensor *up = nullptr;
    const GgufTensor *down = nullptr;
    i32 n_expert = 0;
    i64 n_embd = 0;   // inner dimension of the gate/up matrices
    i64 n_ff_exp = 0; // expert hidden width

    bool present() const { return gate != nullptr && up != nullptr && down != nullptr; }
    // Bytes of one expert's slice in each of the three tensors.
    size_t per_expert(const GgufTensor *t) const {
        return t && n_expert > 0 ? t->n_bytes / static_cast<size_t>(n_expert) : 0;
    }
    size_t expert_bytes() const {
        return per_expert(gate) + per_expert(up) + per_expert(down);
    }
};

// The three resident matrices for one (layer, expert) pair.
struct ResidentExpert {
    QuantTensor gate, up, down;
};

class ExpertCache {
public:
    // Counters reaching this value pin a slot; every PIN_DECAY acquires all
    // counters are halved, so a pin must be continuously re-earned.
    static constexpr u32 kPinThreshold = 24;
    static constexpr u32 kPinDecay = 64;
    // Per-decay subtraction, applied when a counter exceeds it (keeps small
    // counts meaningful instead of driving everything to the same floor).
    static constexpr u32 kPinDecayStep = 8;

    // budget_bytes is a soft cap on resident expert memory. At least one slot is
    // always allowed, even if a single expert exceeds the budget.
    void configure(Backend *be, size_t budget_bytes);
    void clear();

    // Returns the resident matrices for (layer, expert), loading on a miss.
    // nullptr when the source has no expert tensors or the id is out of range.
    // The returned pointer is valid until the next acquire().
    const ResidentExpert *acquire(const ExpertSource &src, i32 layer, i32 expert);

    size_t budget_bytes() const { return budget_; }
    size_t resident_bytes() const { return bytes_; }
    size_t resident_slots() const { return slots_.size(); }
    size_t capacity_slots() const { return capacity_; }
    u64 loads() const { return loads_; }
    u64 evictions() const { return evictions_; }
    u64 hits() const { return hits_; }
    u64 acquires() const { return acquires_; }
    u64 decay_events() const { return decays_; }
    size_t pinned_slots() const;
    // Current LFU counter of one slot, or 0 when it is not resident. Because a
    // pin is granted at kPinThreshold and dropped below it, a count below the
    // threshold proves the slot is unpinned.
    u32 touch_count(i32 layer, i32 expert) const;

private:
    struct Slot {
        i32 layer = -1;
        i32 expert = -1;
        ResidentExpert m;
        size_t bytes = 0;
        u32 count = 0;      // LFU counter, decayed
        u64 seq = 0;        // load order, the tie-break for victims
        bool pinned = false;
    };

    void drop(Slot &s);
    // Halves counters past the step and releases pins that no longer qualify.
    void decay();
    // Picks the slot to sacrifice: unpinned, then rarest, then oldest.
    u64 pick_victim();

    Backend *be_ = nullptr;
    size_t budget_ = 0;
    size_t bytes_ = 0;
    size_t capacity_ = 0;
    u64 acquires_ = 0; // since the last decay
    u64 loads_ = 0;
    u64 evictions_ = 0;
    u64 hits_ = 0;
    u64 decays_ = 0;
    u64 seq_ = 0;
    std::unordered_map<u64, Slot> slots_;
};

} // namespace krk

#endif // KRK_EXPERT_CACHE_HPP
