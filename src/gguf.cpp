// gguf.cpp — GGUF v2/v3 container reader.
#include "krk/gguf.hpp"

#include <algorithm>

namespace krk {

namespace {

constexpr u32 kGgufMagic = 0x46554747u; // "GGUF" little-endian

// Byte width of a scalar metadata element type; 0 for String/Array.
int scalar_size(GgufType t) {
    switch (t) {
        case GgufType::U8:
        case GgufType::I8:
        case GgufType::Bool: return 1;
        case GgufType::U16:
        case GgufType::I16: return 2;
        case GgufType::U32:
        case GgufType::I32:
        case GgufType::F32: return 4;
        case GgufType::U64:
        case GgufType::I64:
        case GgufType::F64: return 8;
        default: return 0;
    }
}

bool read_scalar(Cursor &c, GgufType t, GgufValue &v) {
    v.type = t;
    switch (t) {
        case GgufType::U8: v.u = c.u8v(); v.i = static_cast<i64>(v.u); break;
        case GgufType::I8: v.i = c.i8v(); v.u = static_cast<u64>(v.i); break;
        case GgufType::Bool: v.b = c.u8v() != 0; v.u = v.b; v.i = v.b; break;
        case GgufType::U16: v.u = c.u16v(); v.i = static_cast<i64>(v.u); break;
        case GgufType::I16: v.i = c.i16v(); v.u = static_cast<u64>(v.i); break;
        case GgufType::U32: v.u = c.u32v(); v.i = static_cast<i64>(v.u); break;
        case GgufType::I32: v.i = c.i32v(); v.u = static_cast<u64>(v.i); break;
        case GgufType::F32: v.f = c.f32v(); v.i = static_cast<i64>(v.f); break;
        case GgufType::U64: v.u = c.u64v(); v.i = static_cast<i64>(v.u); break;
        case GgufType::I64: v.i = c.i64v(); v.u = static_cast<u64>(v.i); break;
        case GgufType::F64: v.f = c.f64v(); v.i = static_cast<i64>(v.f); break;
        case GgufType::String: v.str = c.str(); break;
        default: return false;
    }
    return c.ok();
}

bool read_value(Cursor &c, GgufValue &v) {
    const u32 raw = c.u32v();
    if (!c.ok() || raw > static_cast<u32>(GgufType::F64)) return false;
    const auto t = static_cast<GgufType>(raw);
    if (t != GgufType::Array) return read_scalar(c, t, v);

    v.type = GgufType::Array;
    const u32 elem_raw = c.u32v();
    const u64 count = c.u64v();
    if (!c.ok() || elem_raw > static_cast<u32>(GgufType::F64)) return false;
    const auto et = static_cast<GgufType>(elem_raw);
    v.elem = et;

    if (et == GgufType::String) {
        v.strs.reserve(static_cast<size_t>(std::min<u64>(count, 1u << 20)));
        for (u64 i = 0; i < count; i++) {
            v.strs.push_back(c.str());
            if (!c.ok()) return false;
        }
        return true;
    }
    if (et == GgufType::Array) return false; // nested arrays are not legal

    const int esz = scalar_size(et);
    if (esz <= 0) return false;
    const u64 total = count * static_cast<u64>(esz);
    if (total > c.remaining()) return false;
    const u8 *base = c.bytes(static_cast<size_t>(total));
    if (!base) return false;

    if (et == GgufType::F32) {
        v.f32s.resize(static_cast<size_t>(count));
        std::memcpy(v.f32s.data(), base, static_cast<size_t>(count) * 4);
    } else if (et == GgufType::I32 || et == GgufType::U32) {
        v.i32s.resize(static_cast<size_t>(count));
        std::memcpy(v.i32s.data(), base, static_cast<size_t>(count) * 4);
    } else {
        v.raw.assign(base, base + total);
    }
    return true;
}

// "model-00001-of-00003.gguf" -> file_no=1, count=3. False for any other name.
//
// NOTE the numbering convention, which is llama.cpp's and is easy to get
// backwards: the number *written in the filename* is 1-based ("00001" is the
// first shard), while the `split.no` metadata key is 0-based (0 is the first
// shard). `file_no` here is the filename's, so callers convert with -1/+1.
bool parse_split_name(const std::string &path, i32 *file_no, i32 *count) {
    const std::string ext = ".gguf";
    if (path.size() <= ext.size() + 12) return false;
    if (path.compare(path.size() - ext.size(), ext.size(), ext) != 0) return false;
    const size_t of = path.rfind("-of-");
    if (of == std::string::npos) return false;
    const size_t dash = path.rfind('-', of - 1);
    if (dash == std::string::npos || of + 4 >= path.size() - ext.size()) return false;
    const std::string a = path.substr(dash + 1, of - dash - 1);
    const std::string b = path.substr(of + 4, path.size() - ext.size() - of - 4);
    for (char ch : a) if (ch < '0' || ch > '9') return false;
    for (char ch : b) if (ch < '0' || ch > '9') return false;
    *file_no = std::atoi(a.c_str());
    *count = std::atoi(b.c_str());
    return *count > 0 && *file_no >= 1;
}

// The same name with the `-NNNNN-` field replaced, so a shard set is found
// from whichever shard the caller happened to name. `want_no` is the 0-based
// split index; the field written is 1-based.
std::string split_sibling_path(const std::string &path, i32 want_no) {
    const size_t of = path.rfind("-of-");
    if (of == std::string::npos) return {};
    const size_t dash = path.rfind('-', of - 1);
    if (dash == std::string::npos) return {};
    const size_t width = of - dash - 1;
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%0*d", static_cast<int>(width), want_no + 1);
    return path.substr(0, dash + 1) + buf + path.substr(of);
}

} // namespace

// ---------------------------------------------------------------------------
// one file: header, metadata, tensor directory
// ---------------------------------------------------------------------------

bool Gguf::parse_shard(MappedFile &f, i32 shard_index, bool keep_all_kv,
                       std::vector<GgufTensor> *out,
                       std::vector<std::pair<std::string, GgufValue>> *kvs_out,
                       size_t *data_off_out, std::string *err) {
    Cursor c(f.data(), f.size());
    if (c.u32v() != kGgufMagic) {
        if (err) *err = "'" + f.path() + "' is not a GGUF file";
        return false;
    }
    version_ = c.u32v();
    if (version_ < 2 || version_ > 3) {
        if (err) *err = format("unsupported GGUF version %u", version_);
        return false;
    }
    const u64 n_tensors = c.u64v();
    const u64 n_kv = c.u64v();
    if (!c.ok() || n_tensors > (1ull << 32) || n_kv > (1ull << 28)) {
        if (err) *err = "corrupt GGUF header";
        return false;
    }

    kvs_out->reserve(static_cast<size_t>(std::min<u64>(n_kv, 4096)));
    for (u64 i = 0; i < n_kv; i++) {
        std::string key = c.str();
        if (!c.ok()) {
            if (err) *err = "corrupt GGUF metadata";
            return false;
        }
        GgufValue v;
        if (!read_value(c, v)) {
            if (err) *err = "corrupt GGUF metadata value for '" + key + "'";
            return false;
        }
        // A non-zero shard of a split set carries only the split.* keys; the
        // full metadata lives in shard 0 and is what the loader reads.
        if (!keep_all_kv && key.compare(0, 6, "split.") != 0) continue;
        kvs_out->emplace_back(std::move(key), std::move(v));
    }

    out->reserve(out->size() + static_cast<size_t>(n_tensors));
    for (u64 i = 0; i < n_tensors; i++) {
        GgufTensor t;
        t.name = c.str();
        t.n_dims = c.i32v();
        if (!c.ok() || t.n_dims < 1 || t.n_dims > 4) {
            if (err) *err = "corrupt GGUF tensor descriptor";
            return false;
        }
        for (i32 d = 0; d < t.n_dims; d++) t.ne[d] = c.u64v();
        t.type_id = c.u32v();
        t.offset = c.u64v();
        if (!c.ok()) {
            if (err) *err = "corrupt GGUF tensor descriptor for '" + t.name + "'";
            return false;
        }
        t.type = static_cast<DType>(t.type_id);
        t.dtype_known = dtype_block_bytes(t.type) != 0;
        t.n_elements = 1;
        for (i32 d = 0; d < t.n_dims; d++)
            t.n_elements *= static_cast<i64>(t.ne[d]);
        t.n_bytes = !t.dtype_known
                        ? 0
                        : (t.n_dims >= 1)
                              ? dtype_row_bytes(t.type, static_cast<i64>(t.ne[0])) *
                                    static_cast<size_t>(t.n_elements /
                                                        std::max<i64>(1, static_cast<i64>(t.ne[0])))
                              : 0;
        t.shard = static_cast<u32>(shard_index);
        out->push_back(std::move(t));
    }

    // alignment is declared by general.alignment and defaults to 32. A shard
    // that carries no metadata of its own (every data shard of a split set)
    // takes the value already resolved from the metadata shard. The lookup is
    // against the KV just read, not find(): kvs_ is only assigned afterwards.
    for (const auto &kv : *kvs_out)
        if (kv.first == "general.alignment" && kv.second.u > 0)
            alignment_ = kv.second.u;
    if (alignment_ == 0) alignment_ = 32;

    // the tensor data section begins at the next alignment boundary
    size_t off = align_up_s(c.pos(), static_cast<size_t>(alignment_));
    if (off > f.size()) {
        if (err)
            *err = "GGUF tensor data section is out of bounds in '" + f.path() + "'";
        return false;
    }
    const u8 *data_base = f.data() + off;
    *data_off_out = off;

    for (auto &t : *out) {
        // A format this build does not implement has no known payload size, so
        // there is nothing to bounds-check and no pointer to hand out. The
        // entry stays in the directory (its name and id are what tell the
        // caller which format is missing) and the verdict refuses the file.
        if (!t.dtype_known) continue;
        // bounds check the tensor payload
        size_t bytes = 0;
        if (t.n_dims >= 1) {
            const i64 inner = static_cast<i64>(t.ne[0]);
            const size_t row = dtype_row_bytes(t.type, inner);
            const i64 rows = t.n_elements / std::max<i64>(1, inner);
            bytes = row * static_cast<size_t>(rows);
        }
        if (t.offset + bytes > f.size() - off) {
            if (err) *err = "tensor '" + t.name + "' overruns '" + f.path() + "'";
            return false;
        }
        t.data = data_base + t.offset;
        t.n_bytes = bytes;
    }
    return true;
}

// ---------------------------------------------------------------------------
// a model, possibly spanning several files
// ---------------------------------------------------------------------------

bool Gguf::load(const std::string &path, std::string *err) {
    shards_.clear();
    kvs_.clear();
    tensors_.clear();
    data_base_ = nullptr;
    data_section_off_ = 0;

    auto first = std::unique_ptr<MappedFile>(new MappedFile());
    if (!first->open(path, err)) return false;

    std::vector<std::pair<std::string, GgufValue>> kvs;
    std::vector<GgufTensor> ters;
    size_t data_off = 0;
    if (!parse_shard(*first, 0, true, &ters, &kvs, &data_off, err)) return false;

    // Is this file one shard of a split set? The filename is the reliable
    // signal (a producer may omit the split.* keys); the keys are the fallback
    // and carry the same count.
    i64 split_count = 1, split_no = 0;
    for (const auto &kv : kvs) {
        if (kv.first == "split.count") split_count = kv.second.as_i64();
        else if (kv.first == "split.no") split_no = kv.second.as_i64();
    }

    i32 named_file_no = 0, named_count = 0;
    const bool named_split = parse_split_name(path, &named_file_no, &named_count) &&
                             named_count > 1;

    // `here` is the 0-based split index. split.no already is one; the filename
    // number is 1-based, so it converts by subtracting one.
    const i32 here = named_split ? named_file_no - 1
                                 : static_cast<i32>(split_no);
    if (named_split) split_count = named_count;

    if (split_count <= 1) {
        // Single file: the directory and mapping as parsed.
        shards_.push_back({std::move(first), data_off, 0, static_cast<int>(ters.size())});
        kvs_ = std::move(kvs);
        tensors_ = std::move(ters);
        data_base_ = shards_.front().file->data() + data_off;
        data_section_off_ = data_off;
        return true;
    }

    // Split set: open every sibling by name and merge the directories. Only
    // split.no == 0 carries the full KV block; that is the metadata shard.
    const i32 total = static_cast<i32>(split_count);
    if (total > 4096) {
        if (err) *err = format("split.count %d is implausible", total);
        return false;
    }

    for (i32 want = 0; want < total; want++) {
        std::unique_ptr<MappedFile> mf;
        std::vector<GgufTensor> mine;
        size_t off = 0;
        if (want == here) {
            mf = std::move(first);
            mine = std::move(ters);
            off = data_off;
        } else {
            const std::string sp = split_sibling_path(path, want);
            if (sp.empty()) {
                if (err)
                    *err = format("cannot derive shard %d of %d from '%s'", want,
                                  total, path.c_str());
                return false;
            }
            mf = std::unique_ptr<MappedFile>(new MappedFile());
            if (!mf->open(sp, err)) return false;
            std::vector<std::pair<std::string, GgufValue>> ignored;
            if (!parse_shard(*mf, want, false, &mine, &ignored, &off, err)) return false;
        }

        const int n = static_cast<int>(mine.size());
        const u32 idx = static_cast<u32>(shards_.size());
        for (GgufTensor &t : mine) {
            t.shard = idx;
            tensors_.push_back(t);
        }
        shards_.push_back({std::move(mf), off, want, n});
    }

    if (here == 0) {
        kvs_ = std::move(kvs);
    } else {
        // The caller named a data shard; read the metadata from shard 0.
        const std::string p0 = split_sibling_path(path, 0);
        auto mf0 = std::unique_ptr<MappedFile>(new MappedFile());
        if (!mf0->open(p0, err)) return false;
        std::vector<GgufTensor> ignored;
        std::vector<std::pair<std::string, GgufValue>> kv0;
        size_t off0 = 0;
        if (!parse_shard(*mf0, 0, true, &ignored, &kv0, &off0, err)) return false;
        kvs_ = std::move(kv0);
    }

    // `data_base()` / `file()` keep meaning "the metadata shard".
    for (const GgufShard &sh : shards_) {
        if (sh.split_no == 0) {
            data_base_ = sh.file->data() + sh.data_off;
            data_section_off_ = sh.data_off;
            break;
        }
    }
    if (!data_base_) {
        data_base_ = shards_.front().file->data() + shards_.front().data_off;
        data_section_off_ = shards_.front().data_off;
    }
    return true;
}

const GgufValue *Gguf::find(const std::string &key) const {
    for (const auto &kv : kvs_)
        if (kv.first == key) return &kv.second;
    return nullptr;
}

const GgufTensor *Gguf::tensor(const std::string &name) const {
    for (const auto &t : tensors_)
        if (t.name == name) return &t;
    return nullptr;
}

std::string Gguf::get_str(const std::string &key, const std::string &def) const {
    const GgufValue *v = find(key);
    return (v && v->type == GgufType::String) ? v->str : def;
}

i64 Gguf::get_i64(const std::string &key, i64 def) const {
    const GgufValue *v = find(key);
    return v ? v->as_i64() : def;
}

f64 Gguf::get_f64(const std::string &key, f64 def) const {
    const GgufValue *v = find(key);
    return v ? v->as_f64() : def;
}

bool Gguf::get_bool(const std::string &key, bool def) const {
    const GgufValue *v = find(key);
    return v ? (v->b || v->u != 0) : def;
}

const std::vector<f32> *Gguf::get_f32_array(const std::string &key) const {
    const GgufValue *v = find(key);
    if (!v || v->type != GgufType::Array || v->f32s.empty()) return nullptr;
    return &v->f32s;
}

const std::vector<i32> *Gguf::get_i32_array(const std::string &key) const {
    const GgufValue *v = find(key);
    if (!v || v->type != GgufType::Array || v->i32s.empty()) return nullptr;
    return &v->i32s;
}

const std::vector<std::string> *Gguf::get_str_array(const std::string &key) const {
    const GgufValue *v = find(key);
    if (!v || v->type != GgufType::Array || v->strs.empty()) return nullptr;
    return &v->strs;
}

} // namespace krk
