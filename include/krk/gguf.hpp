// ============================================================================
//  gguf.hpp — GGUF v2/v3 container reader.
//
//  The file is mapped, never copied. Tensor entries point directly into the
//  mapping; the model loader then copies (or repacks) them into device memory.
// ============================================================================
#ifndef KRK_GGUF_HPP
#define KRK_GGUF_HPP

#include "krk/common.hpp"
#include "krk/quant.hpp"

namespace krk {

enum class GgufType : u32 {
    U8 = 0,
    I8 = 1,
    U16 = 2,
    I16 = 3,
    U32 = 4,
    I32 = 5,
    F32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    U64 = 10,
    I64 = 11,
    F64 = 12,
};

struct GgufValue {
    GgufType type = GgufType::U32;
    u64 u = 0;   // integral payload (sign-extended into i)
    i64 i = 0;
    f64 f = 0;
    bool b = false;
    std::string str;

    // array payload
    GgufType elem = GgufType::U8;
    std::vector<f32> f32s;
    std::vector<i32> i32s;
    std::vector<std::string> strs;
    std::vector<u8> raw; // numeric arrays of other widths, verbatim

    i64 as_i64() const { return type == GgufType::F32 || type == GgufType::F64
                                    ? static_cast<i64>(f)
                                    : i; }
    f64 as_f64() const { return type == GgufType::F32 || type == GgufType::F64
                                    ? f
                                    : static_cast<f64>(i); }
};

struct GgufTensor {
    std::string name;
    DType type = DType::Unknown;
    u32 type_id = 0xFFFFFFFFu;
    // False for a format this build does not implement (Q1_0, ...). The
    // container still reads and describes the entry so a caller can say
    // *which* format it cannot decode; only the loader has to refuse it, and
    // `data`/`n_bytes` stay unset because the payload size is unknown.
    bool dtype_known = true;
    i32 n_dims = 0;
    u64 ne[4] = {1, 1, 1, 1};
    u64 offset = 0;      // relative to the tensor data section
    const u8 *data = nullptr;
    i64 n_elements = 0;
    size_t n_bytes = 0;
};

class Gguf {
public:
    Gguf() = default;
    ~Gguf() = default;
    Gguf(const Gguf &) = delete;
    Gguf &operator=(const Gguf &) = delete;

    bool load(const std::string &path, std::string *err);

    u32 version() const { return version_; }
    const std::vector<GgufTensor> &tensors() const { return tensors_; }
    const std::vector<std::pair<std::string, GgufValue>> &entries() const {
        return kvs_;
    }
    const u8 *data_base() const { return data_base_; }

    const GgufValue *find(const std::string &key) const;
    const GgufTensor *tensor(const std::string &name) const;

    std::string get_str(const std::string &key, const std::string &def = {}) const;
    i64 get_i64(const std::string &key, i64 def = 0) const;
    f64 get_f64(const std::string &key, f64 def = 0) const;
    bool get_bool(const std::string &key, bool def = false) const;
    const std::vector<f32> *get_f32_array(const std::string &key) const;
    const std::vector<i32> *get_i32_array(const std::string &key) const;
    const std::vector<std::string> *get_str_array(const std::string &key) const;

    const MappedFile &file() const { return file_; }
    // File offset of the tensor data section. `GgufTensor::data` points into the
    // mapping, so `(t.data - file().data())` is the same number; this exposes it
    // once instead of at every call site.
    size_t data_section_off() const { return data_section_off_; }
    u64 alignment() const { return alignment_; }

private:
    MappedFile file_;
    size_t data_section_off_ = 0;
    u32 version_ = 0;
    u64 alignment_ = 32;
    std::vector<std::pair<std::string, GgufValue>> kvs_;
    std::vector<GgufTensor> tensors_;
    const u8 *data_base_ = nullptr;
};

} // namespace krk

#endif // KRK_GGUF_HPP
