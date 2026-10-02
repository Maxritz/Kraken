// ============================================================================
//  common.hpp — aliases, logging, allocation, fp16/bf16, MappedFile, Cursor.
// ============================================================================
#ifndef KRK_COMMON_HPP
#define KRK_COMMON_HPP

#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace krk {

using u8 = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;
using i8 = int8_t;
using i16 = int16_t;
using i32 = int32_t;
using i64 = int64_t;
using f32 = float;
using f64 = double;

// ---------------------------------------------------------------------------
// logging
// ---------------------------------------------------------------------------

enum class Log : int { Debug = 0, Info = 1, Warn = 2, Error = 3 };

void log_set_level(Log lvl);
Log log_level();
void log_printf(Log lvl, const char *fmt, ...);

#define KRK_DEBUG(...) ::krk::log_printf(::krk::Log::Debug, __VA_ARGS__)
#define KRK_INFO(...) ::krk::log_printf(::krk::Log::Info, __VA_ARGS__)
#define KRK_WARN(...) ::krk::log_printf(::krk::Log::Warn, __VA_ARGS__)
#define KRK_ERROR(...) ::krk::log_printf(::krk::Log::Error, __VA_ARGS__)

// printf-style string formatting (no libc dependency at call sites).
std::string format(const char *fmt, ...);

// Product identity, printed by the CLI and the server banner.
constexpr const char *kEngineName = "KRAKEN";
constexpr const char *kEngineVersion = "0.1.0";

// ---------------------------------------------------------------------------
// allocation (64-byte aligned by default)
// ---------------------------------------------------------------------------

void *host_alloc(size_t bytes, size_t alignment = 64);
void *host_realloc(void *p, size_t bytes, size_t alignment = 64);
void host_free(void *p);

// ---------------------------------------------------------------------------
// half / bfloat16 conversion
// ---------------------------------------------------------------------------

f32 fp16_to_fp32(u16 h);
u16 fp32_to_fp16(f32 f);
f32 bf16_to_fp32(u16 h);
u16 fp32_to_bf16(f32 f);

// ---------------------------------------------------------------------------
// MappedFile — read-only file view (mmap on POSIX, CreateFileMapping on Win32)
// ---------------------------------------------------------------------------

class MappedFile {
public:
    MappedFile() = default;
    ~MappedFile();
    MappedFile(const MappedFile &) = delete;
    MappedFile &operator=(const MappedFile &) = delete;

    // Maps the file read-only. On failure returns false and, when `err` is
    // given, a human-readable reason.
    bool open(const std::string &path, std::string *err = nullptr);
    void close();

    bool ok() const { return data_ != nullptr; }
    const u8 *data() const { return data_; }
    size_t size() const { return size_; }

private:
    const u8 *data_ = nullptr;
    size_t size_ = 0;
    void *native_handle_ = nullptr; // fd (POSIX) or file HANDLE (Win32)
    void *native_map_ = nullptr;    // mapping handle (Win32 only)
};

// ---------------------------------------------------------------------------
// Cursor — bounds-checked little-endian reader. Out-of-range reads set ok()
// false and return zero, so loaders can check once per section.
// ---------------------------------------------------------------------------

class Cursor {
public:
    Cursor() = default;
    Cursor(const u8 *base, size_t size) : base_(base), size_(size) {}

    bool ok() const { return ok_; }
    size_t pos() const { return pos_; }
    size_t size() const { return size_; }
    size_t remaining() const { return size_ - pos_; }

    u8 u8v();
    i8 i8v() { return static_cast<i8>(u8v()); }
    u16 u16v();
    i16 i16v() { return static_cast<i16>(u16v()); }
    u32 u32v();
    i32 i32v() { return static_cast<i32>(u32v()); }
    u64 u64v();
    i64 i64v() { return static_cast<i64>(u64v()); }
    f32 f32v();
    f64 f64v();

    // Reads n raw bytes; returns nullptr on overrun.
    const u8 *bytes(size_t n);
    // Reads a u64-length-prefixed string.
    std::string str();
    void skip(size_t n);
    void seek(size_t p);

private:
    const u8 *base_ = nullptr;
    size_t size_ = 0;
    size_t pos_ = 0;
    bool ok_ = true;
};

// ---------------------------------------------------------------------------
// misc
// ---------------------------------------------------------------------------

inline size_t align_up_s(size_t v, size_t a) { return (v + a - 1) / a * a; }

i64 now_us();
i64 now_ms();

// Scoped stopwatch. .ms() / .us() may be read repeatedly and return the
// elapsed time since construction.
class Timer {
public:
    Timer() : start_(now_us()) {}
    f64 ms() const { return static_cast<f64>(now_us() - start_) / 1000.0; }
    f64 us() const { return static_cast<f64>(now_us() - start_); }

private:
    i64 start_;
};

} // namespace krk

#endif // KRK_COMMON_HPP
