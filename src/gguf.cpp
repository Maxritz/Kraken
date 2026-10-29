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
        // A u32 array is kept verbatim as well: i32s sign-extends it, which
        // loses the top half of an offset that legitimately reaches 300M.
        if (et == GgufType::U32) v.raw.assign(base, base + total);
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

    // ---- ggml type id 42: which 2-bit dialect is this file? -------------
    //
    // Id 42 has two live meanings and they share no block geometry:
    //
    //   Q2_0     128 values / 34 bytes = 2.125 bpw   (llama-dx fork,
    //             Ternary-Bonsai exports -- the id's own meaning)
    //   Q2_0_64   64 values / 18 bytes = 2.25 bpw    (upstream llama.cpp's
    //             Q2_0; files declaring 42 measure 2.25 bpw, e.g. the Swift
    //             Qwen3.8-Flash shards)
    //
    // and the same id is the BitNet fork's TL2. The file does not carry the
    // name anywhere, but it does carry the STRIDE: the distance from this
    // tensor's offset to the next one's is exactly what this file's producer
    // thought the block was. So the dialect is resolved per file from that,
    // and a stride that fits neither is refused BY NAME rather than decoded
    // as Q2_0 -- decoding a 2.25 bpw tensor at 2.125 bpw shifts every row by
    // two bytes per 128 values and returns a different model, silently.
    //
    // Every id-42 tensor in the shard votes, and the file gets ONE verdict:
    //   * a tensor whose stride fits only 128/34 is a Q2_0 vote;
    //   * one whose stride fits only 64/18 is a Q2_0_64 vote;
    //   * one that fits both (small tensors: 2*(n/128) < alignment) is
    //     ambiguous and abstains -- its own id wins;
    //   * one that fits neither is not a dialect this build knows, and the
    //     file is refused.
    // A file that votes both ways is refused too: no producer emits a stack
    // whose tensors disagree about their own block layout.
    {
        const size_t data_end = f.size() - off;
        std::vector<size_t> order;
        order.reserve(out->size());
        for (size_t i = 0; i < out->size(); i++)
            if ((*out)[i].offset <= data_end) order.push_back(i);
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            return (*out)[a].offset < (*out)[b].offset;
        });
        int vote128 = 0, vote64 = 0;
        for (size_t k = 0; k < order.size(); k++) {
            const GgufTensor &t = (*out)[order[k]];
            if (t.type_id != 42u || t.n_elements <= 0) continue;
            if (t.n_elements % 128 != 0) continue;  // geometry cannot speak
            const size_t stride = k + 1 < order.size()
                                      ? static_cast<size_t>((*out)[order[k + 1]].offset -
                                                            t.offset)
                                      : static_cast<size_t>(data_end - t.offset);
            const size_t b128 = static_cast<size_t>(t.n_elements / 128) * 34;
            const size_t b64 = static_cast<size_t>(t.n_elements / 64) * 18;
            // A tensor may be followed by alignment padding, so the stride
            // fits a hypothesis when it covers it by less than one alignment
            // unit of slack.
            const bool f128 = stride >= b128 && stride - b128 < alignment_;
            const bool f64 = stride >= b64 && stride - b64 < alignment_;
            if (f128 && !f64) {
                vote128++;
            } else if (f64 && !f128) {
                vote64++;
            } else if (!f128 && !f64) {
                if (err)
                    *err = "tensor '" + t.name + "' declares ggml type id 42, "
                           "but its stride in '" + f.path() + "' is " +
                           std::to_string(stride) + " bytes; Q2_0 (128 values/"
                           "34 bytes) needs " + std::to_string(b128) +
                           " and Q2_0_64 (64 values/18 bytes) needs " +
                           std::to_string(b64) +
                           ". Id 42 is also the BitNet fork's TL2, and that "
                           "block layout is not known to this build -- "
                           "refusing rather than guessing";
                return false;
            }
        }
        if (vote128 && vote64) {
            if (err)
                *err = "'" + f.path() +
                       "' declares ggml type id 42 with two incompatible "
                       "strides -- some tensors measure 2.125 bpw (Q2_0) and "
                       "others 2.25 bpw (Q2_0_64 / the fork's TL2); no "
                       "producer emits that, so the file is refused";
            return false;
        }
        if (vote64) {
            for (auto &t : *out)
                if (t.type_id == 42u) t.type = DType::Q2_0_64;
        }
    }

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
        // I2_S carries a 32-byte scale tail past the payload. It is inside the
        // declared tensor range only by construction, so check it here: reading
        // it past the mapping would be the silent kind of wrong (a scale of
        // 4.6e-41 -- i.e. every weight zero) rather than a crash.
        size_t suffix = 0;
        if (t.type == DType::I2_S) {
            if (t.offset + bytes + 32 > f.size() - off) {
                if (err)
                    *err = "tensor '" + t.name +
                           "' is I2_S but its 32-byte scale tail is past the end "
                           "of '" + f.path() + "'";
                return false;
            }
            suffix = 32;
        }
        t.data = data_base + t.offset;
        t.n_bytes = bytes;
        if (suffix)
            t.tail = data_base + t.offset + bytes, t.tail_bytes = suffix;
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

bool Gguf::add_alias(const std::string &to, const std::string &from) {
    if (to.empty() || from.empty()) return false;
    const GgufTensor *src = tensor(from);
    if (!src) return false;
    if (tensor(to)) return true;   // the file's own spelling wins
    GgufTensor t = *src;
    t.name = to;
    tensors_.push_back(std::move(t));
    return true;
}

bool Gguf::add_row_slice(const std::string &to, const std::string &from,
                         i64 row0, i64 rows, i64 n_in) {
    if (to.empty() || from.empty()) return false;
    const GgufTensor *src = tensor(from);
    if (!src) return false;
    if (tensor(to)) return true;
    if (row0 < 0 || rows <= 0 || n_in <= 0) return false;
    if (n_in != static_cast<i64>(src->ne[0])) return false;
    // Total rows as the loader reads them: ne[1] for a matrix, and the packed
    // element count over the row width for anything wider (a 3-D expert bank is
    // not a row-slice source, so this only ever sees 1-D and 2-D).
    const i64 total = src->n_dims >= 2
                          ? static_cast<i64>(src->ne[1])
                          : (n_in > 0 ? src->n_elements / n_in : 0);
    if (row0 + rows > total) return false;
    const size_t rb = dtype_row_bytes(src->type, static_cast<u64>(n_in));
    if (rb == 0) return false;
    GgufTensor t = *src;
    t.name = to;
    t.data = src->data + static_cast<size_t>(row0) * rb;
    t.ne[0] = static_cast<u64>(n_in);
    t.ne[1] = static_cast<u64>(rows);
    t.n_dims = 2;
    t.n_elements = n_in * rows;
    t.n_bytes = static_cast<u64>(rb) * static_cast<u64>(rows);
    tensors_.push_back(std::move(t));
    return true;
}

bool Gguf::add_tensor_entry(GgufTensor t) {
    if (t.name.empty() || !t.data) return false;
    if (tensor(t.name)) return false;
    tensors_.push_back(std::move(t));
    return true;
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

bool Gguf::get_i64_array(const std::string &key, std::vector<i64> *out) const {
    if (!out) return false;
    const GgufValue *v = find(key);
    if (!v || v->type != GgufType::Array) return false;
    // The i32/u32 case is decoded at parse time into i32s. Nothing else is,
    // so every other integral width arrives as `raw` bytes and is widened here
    // from the declared element type -- reading `raw` as the wrong width is
    // exactly the silent wrong-answer shape this function exists to avoid.
    if (v->elem == GgufType::I32 ||
        (v->elem == GgufType::U32 && v->raw.empty())) {
        out->assign(v->i32s.begin(), v->i32s.end());
        return true;
    }
    const int esz = scalar_size(v->elem);
    if (esz <= 0 || v->raw.empty()) {
        if (!v->i32s.empty()) {
            out->assign(v->i32s.begin(), v->i32s.end());
            return true;
        }
        return false;
    }
    const size_t n = v->raw.size() / static_cast<size_t>(esz);
    out->resize(n);
    for (size_t i = 0; i < n; i++) {
        const u8 *p = v->raw.data() + i * static_cast<size_t>(esz);
        i64 x = 0;
        switch (v->elem) {
            case GgufType::U8: x = p[0]; break;
            case GgufType::I8: x = static_cast<i8>(p[0]); break;
            case GgufType::U16: { u16 t; std::memcpy(&t, p, 2); x = t; break; }
            case GgufType::I16: { i16 t; std::memcpy(&t, p, 2); x = t; break; }
            case GgufType::U32: { u32 t; std::memcpy(&t, p, 4); x = t; break; }
            case GgufType::I32: { i32 t; std::memcpy(&t, p, 4); x = t; break; }
            case GgufType::Bool: x = p[0] != 0; break;
            case GgufType::U64: { u64 t; std::memcpy(&t, p, 8); x = static_cast<i64>(t); break; }
            case GgufType::I64: { std::memcpy(&x, p, 8); break; }
            default: return false;
        }
        (*out)[i] = x;
    }
    return true;
}

const std::vector<std::string> *Gguf::get_str_array(const std::string &key) const {
    const GgufValue *v = find(key);
    if (!v || v->type != GgufType::Array || v->strs.empty()) return nullptr;
    return &v->strs;
}

} // namespace krk
