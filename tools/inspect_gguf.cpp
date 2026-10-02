// inspect_gguf.cpp — dump a GGUF file's metadata and tensor directory.
//
//   kraken-inspect model.gguf            metadata, format histogram, tensors
//   kraken-inspect model.gguf --meta     metadata only
//   kraken-inspect model.gguf --quant    format histogram only
#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include "krk/gguf.hpp"
#include "krk/verdict.hpp"

namespace {

using namespace krk;

constexpr f64 kGiB = 1024.0 * 1024.0 * 1024.0;

std::string human(size_t bytes) {
    if (bytes >= 1024ull * 1024ull * 1024ull)
        return format("%.2f GiB", static_cast<f64>(bytes) / kGiB);
    if (bytes >= 1024ull * 1024ull)
        return format("%.1f MiB", static_cast<f64>(bytes) / (1024.0 * 1024.0));
    if (bytes >= 1024ull) return format("%.1f KiB", static_cast<f64>(bytes) / 1024.0);
    return format("%zu B", bytes);
}

std::string value_str(const GgufValue &v) {
    switch (v.type) {
        case GgufType::String:
            return "\"" + (v.str.size() > 60 ? v.str.substr(0, 57) + "..." : v.str) + "\"";
        case GgufType::Bool: return v.b ? "true" : "false";
        case GgufType::F32:
        case GgufType::F64: return format("%g", v.f);
        case GgufType::Array: {
            if (!v.strs.empty()) return format("[%zu strings]", v.strs.size());
            if (!v.f32s.empty()) return format("[%zu f32]", v.f32s.size());
            if (!v.i32s.empty()) return format("[%zu i32]", v.i32s.size());
            return format("[%zu bytes]", v.raw.size());
        }
        default: return format("%lld", static_cast<long long>(v.i));
    }
}

} // namespace

int main(int argc, char **argv) {
    std::string path;
    bool meta_only = false;
    bool quant_only = false;
    for (int i = 1; i < argc; i++) {
        const std::string f = argv[i];
        if (f == "--meta") meta_only = true;
        else if (f == "--quant") quant_only = true;
        else if (f == "-h" || f == "--help") {
            std::printf("usage: kraken-inspect model.gguf [--meta] [--quant]\n");
            return 0;
        } else {
            path = f;
        }
    }
    if (path.empty()) {
        std::fprintf(stderr, "usage: kraken-inspect model.gguf [--meta] [--quant]\n");
        return 2;
    }

    log_set_level(Log::Warn);
    Gguf g;
    std::string err;
    if (!g.load(path, &err)) {
        std::fprintf(stderr, "kraken-inspect: %s\n", err.c_str());
        return 1;
    }

    std::printf("file       %s\n", path.c_str());
    std::printf("version    GGUF v%u\n", g.version());
    std::printf("alignment  %llu\n", static_cast<unsigned long long>(g.alignment()));
    std::printf("tensors    %zu\n", g.tensors().size());

    size_t total = 0;
    size_t unknown = 0; // tensors in formats this build does not implement
    std::map<int, std::pair<int, size_t>> by_type; // type id -> (count, bytes)
    for (const GgufTensor &t : g.tensors()) {
        total += t.n_bytes;
        if (!t.dtype_known) unknown++;
        auto &e = by_type[static_cast<int>(t.type)];
        e.first++;
        e.second += t.n_bytes;
    }
    if (unknown)
        std::printf("payload    %s in known formats + %zu tensor(s) of unknown size\n",
                    human(total).c_str(), unknown);
    else
        std::printf("payload    %s\n", human(total).c_str());

    // What the engine would decide about this file, before any device work:
    // the arch table's verdict plus the formats and tensors that stop it.
    {
        const ModelVerdict v = assess_model(g);
        std::printf("arch       %s\n", v.headline().c_str());
        std::printf("verdict    %s\n", v.runnable ? "runnable" : "refused");
        for (const std::string &b : v.blockers)
            std::printf("           - %s\n", b.c_str());
        for (const std::string &n : v.notes)
            std::printf("           . %s\n", n.c_str());
    }

    if (!quant_only) {
        std::printf("\nmetadata:\n");
        for (const auto &kv : g.entries())
            std::printf("  %-46s = %s\n", kv.first.c_str(), value_str(kv.second).c_str());
    }

    if (!meta_only) {
        std::printf("\nformat histogram:\n");
        for (const auto &kv : by_type) {
            // A format this build does not implement has no size in the
            // histogram, so print the on-disk type id instead of a name.
            std::string name = dtype_name(static_cast<DType>(kv.first));
            if (name == "UNKNOWN")
                name = format("#%d", kv.first);
            std::printf("  %-8s %6d tensors  %s\n", name.c_str(), kv.second.first,
                        kv.second.second ? human(kv.second.second).c_str()
                                         : "(unknown format)");
        }
    }

    if (!meta_only && !quant_only) {
        std::printf("\ntensors:\n");
        std::printf("  %-46s %-7s %-24s %s\n", "name", "type", "shape", "bytes");
        for (const GgufTensor &t : g.tensors()) {
            std::string shape;
            for (i32 d = t.n_dims - 1; d >= 0; d--) {
                shape += format("%llu", static_cast<unsigned long long>(t.ne[d]));
                if (d) shape += "x";
            }
            std::printf("  %-46s %-7s %-24s %s\n", t.name.c_str(), dtype_name(t.type),
                        shape.c_str(), human(t.n_bytes).c_str());
        }
    }
    return 0;
}
