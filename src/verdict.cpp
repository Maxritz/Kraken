// ============================================================================
//  verdict.cpp — see include/krk/verdict.hpp.
//
//  The pass is deliberately shallow: it reads the arch string and the two keys
//  that describe the stack, then walks the tensor directory three times (arch
//  gaps, formats, head tensors). No tensor data is touched, so the answer for a
//  60 GB file costs the same as for a 100 MB one, which is the point — a refusal
//  should arrive before the load.
// ============================================================================
#include "krk/verdict.hpp"

#include <algorithm>

namespace krk {
namespace {

// Append `what` unless an identical reason is already recorded. Forty layers of
// the same unimplemented gate are one cause, not forty.
void add_unique(std::vector<std::string> &v, const std::string &what) {
    if (std::find(v.begin(), v.end(), what) == v.end()) v.push_back(what);
}

std::string named_dtype(const GgufTensor &t) {
    const std::string n = dtype_name(t.type);
    if (n != "UNKNOWN") return n;
    return "type #" + std::to_string(t.type_id);
}

// MTP blocks appended to the stack. The converter records how many in
// `<arch>.nextn_predict_layers`, as an array of one entry per head in the
// files seen so far and as a scalar in others; both spellings mean the same
// count, and the loader used to read this key itself.
i32 mtp_block_count(const Gguf &g, const std::string &arch, i32 block_count) {
    const std::string key = arch + ".nextn_predict_layers";
    i32 n = 0;
    if (const std::vector<i32> *arr = g.get_i32_array(key)) {
        if (!arr->empty()) n = (*arr)[0];
    } else {
        n = static_cast<i32>(g.get_i64(key, 0));
    }
    if (n < 0 || n > block_count) n = 0;
    return n;
}

} // namespace

std::string ModelVerdict::headline() const {
    const std::string name = arch.empty() ? std::string("<absent>") : arch;
    if (!known) return name + " — unknown (llama-like defaults)";
    if (spec && spec->role == ArchRole::Draft)
        return name + " — draft file (not a model)";
    if (spec && spec->support == ArchSupport::Yes)
        return name + " — supported (" + arch_shape_name(spec->shape) + ")";
    if (spec && spec->support == ArchSupport::Partial)
        return name + " — partial (" + arch_shape_name(spec->shape) + ")";
    return name + " — not supported";
}

ModelVerdict assess_model(const Gguf &g) {
    ModelVerdict v;
    v.arch = g.get_str("general.architecture");

    // Some converters omit the label but keep the per-arch keys, and the key
    // prefix is then the arch in everything but name: `<arch>.block_count`
    // cannot mean anything else. Trust the keys over the missing label.
    if (v.arch.empty()) {
        const std::string suffix = ".block_count";
        for (const auto &kv : g.entries()) {
            const std::string &k = kv.first;
            if (k.size() > suffix.size() &&
                k.compare(k.size() - suffix.size(), suffix.size(), suffix) == 0) {
                v.arch = k.substr(0, k.size() - suffix.size());
                add_unique(v.notes, "the file declares no general.architecture; "
                                    "read as '" + v.arch +
                                        "' from its key prefix");
                break;
            }
        }
    }
    v.spec = &arch_spec(v.arch, &v.known);

    if (v.arch.empty()) {
        add_unique(v.notes,
                   "the file declares no general.architecture; read against "
                   "llama-like defaults");
    } else if (!v.known) {
        add_unique(v.notes, "not in the architecture table; read against " +
                                std::string(v.spec->family) + " defaults");
    }

    const ArchShape shape = v.spec->shape;

    // What kind of file this is, before what arch it is. A shard of a split
    // model and an importance matrix both carry tensors and metadata and are not
    // models; saying so beats loading 45 GB of half a model and failing on a
    // missing layer.
    {
        // Split (`split.*`) sets are loaded as one model: the loader derives
        // the sibling files from the `-NNNNN-of-MMMMM.gguf` naming and merges
        // their tensor directories. Refuse only a set that is actually short of
        // tensors, which is a real download error rather than a format one.
        const i32 split_count = static_cast<i32>(g.get_i64("split.count", 0));
        if (split_count > 1) {
            const i64 declared = g.get_i64("split.tensors.count", 0);
            const i64 have = static_cast<i64>(g.tensor_count());
            if (declared > 0 && have != declared)
                add_unique(v.blockers,
                           format("split model is incomplete: %lld of %lld tensors "
                                  "across %d shards (split.tensors.count)",
                                  static_cast<long long>(have),
                                  static_cast<long long>(declared),
                                  static_cast<int>(split_count)));
        }
        const std::string type = g.get_str("general.type", "");
        if (!type.empty() && type != "model")
            add_unique(v.blockers,
                       "general.type is '" + type + "'; this file is not a model");
        // The effective key prefix: the arch's own name, or llama's when the file
        // names no arch at all, which is how the loader reads it too.
        // Only for files that should be models: a draft shard or an unlisted
        // arch is judged on its own terms above, and the loader will have its
        // own say about geometry.
        const bool expect_stack = v.arch.empty() ||
                                  (v.known && v.spec->role == ArchRole::Target);
        const std::string prefix = v.arch.empty() ? "llama" : v.arch;
        if (expect_stack && g.get_i64(prefix + ".block_count", 0) <= 0)
            add_unique(v.blockers, "declares no " + prefix +
                                       ".block_count, so there is no stack of "
                                       "layers to load");
    }

    if (v.known) {
        if (v.spec->role == ArchRole::Draft) {
            // A head set is refused as a model, and the reason says what it is
            // for instead, because "not supported" alone would read as a bug.
            add_unique(v.blockers, std::string(v.spec->why) +
                                       "; kraken's speculation drafts with "
                                       "--draft, a full model of the family");
        } else if (v.spec->support == ArchSupport::No) {
            add_unique(v.blockers, v.spec->why);
        } else if (v.spec->support == ArchSupport::Partial) {
            add_unique(v.notes, v.spec->why);
        }
    }

    // What the file carries past its layers: MTP blocks, and any speculative
    // head tensors left inside an otherwise ordinary model. Neither is fatal —
    // the model runs — but a user should know the file was built with them.
    {
        const i32 block_count =
            static_cast<i32>(g.get_i64(v.arch + ".block_count", 0));
        v.mtp_blocks = v.arch.empty() ? 0 : mtp_block_count(g, v.arch, block_count);
        if (v.mtp_blocks > 0) {
            const i32 first = block_count - v.mtp_blocks;
            add_unique(v.notes,
                       format("carries %d multi-token-prediction block(s) "
                              "(blk.%d%s) that this engine skips",
                              v.mtp_blocks, first,
                              v.mtp_blocks > 1 ? ".." : ""));
        }
    }
    {
        std::vector<std::string> heads;
        for (const GgufTensor &t : g.tensors()) {
            if (const char *h = arch_tensor_head(t.name))
                add_unique(heads, h);
        }
        for (const std::string &h : heads)
            add_unique(v.notes, "carries " + h + " tensors, which this engine "
                                             "does not use");
    }

    // Unimplemented computations, by tensor name. This is what makes the
    // table's Partial/No verdicts hold for a checkpoint whose arch string is
    // correct but whose tensors are from a variant nobody listed.
    for (const GgufTensor &t : g.tensors()) {
        if (const char *gap = arch_tensor_gap(t.name, shape, v.arch)) {
            add_unique(v.blockers,
                       "uses " + std::string(gap) +
                           ", which this engine does not implement");
        }
    }

    // Quantization formats this build cannot dequantize, and rows that are not
    // a whole number of blocks (an exotic producer, or a corrupt file).
    std::vector<std::string> bad_formats;
    for (const GgufTensor &t : g.tensors()) {
        if (!dtype_supported(t.type)) {
            const std::string name = named_dtype(t);
            if (std::find(bad_formats.begin(), bad_formats.end(), name) ==
                bad_formats.end())
                bad_formats.push_back(name);
        } else if (t.n_dims >= 1 &&
                   !dtype_row_aligned(t.type, static_cast<i64>(t.ne[0]))) {
            add_unique(v.blockers, "tensor '" + t.name +
                                       "' has a row length that is not a whole "
                                       "number of quantization blocks");
        }
    }
    for (const std::string &f : bad_formats)
        add_unique(v.blockers,
                   "uses quantization " + f +
                       ", which this build cannot dequantize");

    v.runnable = v.blockers.empty();
    return v;
}

} // namespace krk
