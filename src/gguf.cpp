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

} // namespace

bool Gguf::load(const std::string &path, std::string *err) {
    if (!file_.open(path, err)) return false;

    Cursor c(file_.data(), file_.size());
    if (c.u32v() != kGgufMagic) {
        if (err) *err = "'" + path + "' is not a GGUF file";
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

    kvs_.reserve(static_cast<size_t>(std::min<u64>(n_kv, 4096)));
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
        kvs_.emplace_back(std::move(key), std::move(v));
    }

    tensors_.reserve(static_cast<size_t>(n_tensors));
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
        tensors_.push_back(std::move(t));
    }

    // alignment is declared by general.alignment and defaults to 32
    if (const GgufValue *a = find("general.alignment"))
        alignment_ = a->u > 0 ? a->u : 32;
    if (alignment_ == 0) alignment_ = 32;

    // the tensor data section begins at the next alignment boundary
    size_t off = align_up_s(c.pos(), static_cast<size_t>(alignment_));
    if (off > file_.size()) {
        if (err) *err = "GGUF tensor data section is out of bounds";
        return false;
    }
    data_base_ = file_.data() + off;
    data_section_off_ = off;

    for (auto &t : tensors_) {
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
        if (t.offset + bytes > file_.size() - off) {
            if (err) *err = "tensor '" + t.name + "' overruns the file";
            return false;
        }
        t.data = data_base_ + t.offset;
        t.n_bytes = bytes;
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
