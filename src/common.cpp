// common.cpp — logging, allocation, half conversion, file mapping, timing.
#include "krk/common.hpp"

#include <cmath>
#include <ctime>

#if defined(_WIN32)
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace krk {

// ---------------------------------------------------------------------------
// logging
// ---------------------------------------------------------------------------

namespace {
Log g_level = Log::Info;
} // namespace

void log_set_level(Log lvl) { g_level = lvl; }
Log log_level() { return g_level; }

void log_printf(Log lvl, const char *fmt, ...) {
    if (static_cast<int>(lvl) < static_cast<int>(g_level)) return;
    static const char *tag[] = {"debug", "info ", "warn ", "error"};
    std::fprintf(stderr, "[%s] ", tag[static_cast<int>(lvl) & 3]);
    va_list ap;
    va_start(ap, fmt);
    std::vfprintf(stderr, fmt, ap);
    va_end(ap);
    std::fputc('\n', stderr);
    std::fflush(stderr);
}

// ---------------------------------------------------------------------------
// allocation
// ---------------------------------------------------------------------------

void *host_alloc(size_t bytes, size_t alignment) {
    if (bytes == 0) bytes = 1;
#if defined(_WIN32)
    void *p = _aligned_malloc(bytes, alignment);
#else
    void *p = nullptr;
    if (posix_memalign(&p, alignment, bytes) != 0) p = nullptr;
#endif
    if (!p) {
        KRK_ERROR("out of memory: %zu bytes", bytes);
        std::abort();
    }
    return p;
}

void *host_realloc(void *p, size_t bytes, size_t alignment) {
    if (bytes == 0) bytes = 1;
#if defined(_WIN32)
    void *q = _aligned_realloc(p, bytes, alignment);
#else
    (void)alignment;
    void *q = std::realloc(p, bytes);
#endif
    if (!q) {
        KRK_ERROR("out of memory: %zu bytes", bytes);
        std::abort();
    }
    return q;
}

size_t host_available_bytes() {
#if defined(_WIN32)
    MEMORYSTATUSEX st;
    st.dwLength = sizeof(st);
    if (GlobalMemoryStatusEx(&st)) return static_cast<size_t>(st.ullAvailPhys);
    return 0;
#else
    const long pages = sysconf(_SC_AVPHYS_PAGES);
    const long page = sysconf(_SC_PAGESIZE);
    if (pages <= 0 || page <= 0) return 0;
    return static_cast<size_t>(pages) * static_cast<size_t>(page);
#endif
}

size_t host_total_bytes() {
#if defined(_WIN32)
    MEMORYSTATUSEX st;
    st.dwLength = sizeof(st);
    if (GlobalMemoryStatusEx(&st)) return static_cast<size_t>(st.ullTotalPhys);
    return 0;
#else
    const long pages = sysconf(_SC_PHYS_PAGES);
    const long page = sysconf(_SC_PAGESIZE);
    if (pages <= 0 || page <= 0) return 0;
    return static_cast<size_t>(pages) * static_cast<size_t>(page);
#endif
}

void host_free(void *p) {
    if (!p) return;
#if defined(_WIN32)
    _aligned_free(p);
#else
    std::free(p);
#endif
}

// ---------------------------------------------------------------------------
// half / bf16
// ---------------------------------------------------------------------------

f32 fp16_to_fp32(u16 h) {
    const u32 sign = static_cast<u32>(h & 0x8000u) << 16;
    const u32 e = (h >> 10) & 0x1Fu;
    const u32 m = h & 0x3FFu;
    u32 bits;
    if (e == 0) {
        if (m == 0) {
            bits = sign;
        } else {
            u32 mm = m;
            int ee = -1;
            do {
                mm <<= 1;
                ee++;
            } while (!(mm & 0x400u));
            mm &= 0x3FFu;
            bits = sign | (static_cast<u32>(127 - 15 - ee) << 23) | (mm << 13);
        }
    } else if (e == 31) {
        bits = sign | 0x7F800000u | (m << 13);
    } else {
        bits = sign | ((e - 15 + 127) << 23) | (m << 13);
    }
    f32 f;
    std::memcpy(&f, &bits, 4);
    return f;
}

u16 fp32_to_fp16(f32 f) {
    u32 x;
    std::memcpy(&x, &f, 4);
    const u32 sign = (x >> 16) & 0x8000u;
    const i32 e = static_cast<i32>((x >> 23) & 0xFFu) - 127 + 15;
    const u32 m = x & 0x7FFFFFu;
    if (((x >> 23) & 0xFFu) == 0xFFu)
        return static_cast<u16>(sign | 0x7C00u | (m ? 0x200u : 0));
    if (e >= 31) return static_cast<u16>(sign | 0x7C00u);
    if (e <= 0) {
        if (e < -10) return static_cast<u16>(sign);
        const int sh = 14 - e;
        const u32 mm = m | 0x800000u;
        u32 half = mm >> sh;
        const u32 rem = mm & ((1u << sh) - 1);
        const u32 thresh = 1u << (sh - 1);
        if (rem > thresh || (rem == thresh && (half & 1))) half++;
        return static_cast<u16>(sign | (half & 0x7FFu));
    }
    u32 half = (static_cast<u32>(e) << 10) | (m >> 13);
    if ((m & 0x1FFFu) > 0x1000u || ((m & 0x3FFFu) == 0x1000u && (half & 1))) half++;
    return static_cast<u16>(sign | (half & 0x7FFFu));
}

f32 bf16_to_fp32(u16 h) {
    u32 bits = static_cast<u32>(h) << 16;
    f32 f;
    std::memcpy(&f, &bits, 4);
    return f;
}

u16 fp32_to_bf16(f32 f) {
    u32 x;
    std::memcpy(&x, &f, 4);
    const u32 lsb = (x >> 16) & 1u;
    x += 0x7FFFu + lsb;
    return static_cast<u16>(x >> 16);
}

// ---------------------------------------------------------------------------
// MappedFile
// ---------------------------------------------------------------------------

#if defined(_WIN32)

MappedFile::~MappedFile() { close(); }

bool MappedFile::open(const std::string &path, std::string *err) {
    close();
    HANDLE fh = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (fh == INVALID_HANDLE_VALUE) {
        if (err) *err = "cannot open '" + path + "'";
        return false;
    }
    LARGE_INTEGER li;
    if (!GetFileSizeEx(fh, &li) || li.QuadPart <= 0) {
        CloseHandle(fh);
        if (err) *err = "'" + path + "' is empty";
        return false;
    }
    HANDLE mh = CreateFileMappingA(fh, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!mh) {
        CloseHandle(fh);
        if (err) *err = "CreateFileMapping failed for '" + path + "'";
        return false;
    }
    const void *view = MapViewOfFile(mh, FILE_MAP_READ, 0, 0, 0);
    if (!view) {
        CloseHandle(mh);
        CloseHandle(fh);
        if (err) *err = "MapViewOfFile failed for '" + path + "'";
        return false;
    }
    native_handle_ = fh;
    native_map_ = mh;
    data_ = static_cast<const u8 *>(view);
    size_ = static_cast<size_t>(li.QuadPart);
    path_ = path;
    return true;
}

void MappedFile::close() {
    if (data_) {
        UnmapViewOfFile(data_);
        data_ = nullptr;
    }
    if (native_map_) {
        CloseHandle(static_cast<HANDLE>(native_map_));
        native_map_ = nullptr;
    }
    if (native_handle_) {
        CloseHandle(static_cast<HANDLE>(native_handle_));
        native_handle_ = nullptr;
    }
    size_ = 0;
    path_.clear();
}

FileReader::~FileReader() { close(); }

bool FileReader::open(const std::string &path) {
    close();
    HANDLE fh = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (fh == INVALID_HANDLE_VALUE) return false;
    handle_ = fh;
    return true;
}

void FileReader::close() {
    if (handle_) {
        CloseHandle(static_cast<HANDLE>(handle_));
        handle_ = nullptr;
    }
}

bool FileReader::at(size_t off, void *dst, size_t len) {
    if (!handle_) return false;
    HANDLE fh = static_cast<HANDLE>(handle_);
    OVERLAPPED ov{};
    ov.Offset = static_cast<DWORD>(off & 0xffffffffull);
    ov.OffsetHigh = static_cast<DWORD>(off >> 32);
    char *p = static_cast<char *>(dst);
    size_t got = 0;
    while (got < len) {
        DWORD want = static_cast<DWORD>(len - got > 0x40000000u ? 0x40000000u : len - got);
        DWORD n = 0;
        if (!ReadFile(fh, p + got, want, &n, &ov)) return false;
        if (n == 0) return false;
        got += n;
        ov.Offset += n;
        if (ov.Offset < n) ov.OffsetHigh++;
    }
    return true;
}

// See the header: a cold mapping faults per page, read() does not.
bool MappedFile::read_at(size_t off, void *dst, size_t len) const {
    if (!native_handle_) return false;
    if (off > size_ || len > size_ - off) return false;
    HANDLE fh = static_cast<HANDLE>(native_handle_);
    OVERLAPPED ov{};
    ov.Offset = static_cast<DWORD>(off & 0xffffffffull);
    ov.OffsetHigh = static_cast<DWORD>(off >> 32);
    char *p = static_cast<char *>(dst);
    size_t got = 0;
    while (got < len) {
        DWORD want = static_cast<DWORD>(len - got > 0x40000000u ? 0x40000000u : len - got);
        DWORD n = 0;
        if (!ReadFile(fh, p + got, want, &n, &ov)) return false;
        if (n == 0) return false;
        got += n;
        ov.Offset += n;
        if (ov.Offset < n) ov.OffsetHigh++;
    }
    return true;
}

#else

MappedFile::~MappedFile() { close(); }

bool MappedFile::open(const std::string &path, std::string *err) {
    close();
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        if (err) *err = "cannot open '" + path + "'";
        return false;
    }
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size <= 0) {
        ::close(fd);
        if (err) *err = "cannot stat '" + path + "'";
        return false;
    }
    void *p = mmap(nullptr, static_cast<size_t>(st.st_size), PROT_READ,
                   MAP_PRIVATE, fd, 0);
    if (p == MAP_FAILED) {
        ::close(fd);
        if (err) *err = "mmap failed for '" + path + "'";
        return false;
    }
    native_handle_ = reinterpret_cast<void *>(static_cast<intptr_t>(fd));
    data_ = static_cast<const u8 *>(p);
    size_ = static_cast<size_t>(st.st_size);
    path_ = path;
    return true;
}

void MappedFile::close() {
    if (data_) {
        munmap(const_cast<u8 *>(data_), size_);
        data_ = nullptr;
    }
    if (native_handle_) {
        ::close(static_cast<int>(reinterpret_cast<intptr_t>(native_handle_)));
        native_handle_ = nullptr;
    }
    size_ = 0;
    path_.clear();
}

FileReader::~FileReader() { close(); }

bool FileReader::open(const std::string &path) {
    close();
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) return false;
    handle_ = reinterpret_cast<void *>(static_cast<intptr_t>(fd));
    return true;
}

void FileReader::close() {
    if (handle_) {
        ::close(static_cast<int>(reinterpret_cast<intptr_t>(handle_)));
        handle_ = nullptr;
    }
}

bool FileReader::at(size_t off, void *dst, size_t len) {
    if (!handle_) return false;
    const int fd = static_cast<int>(reinterpret_cast<intptr_t>(handle_));
    char *p = static_cast<char *>(dst);
    size_t got = 0;
    while (got < len) {
        const ssize_t n = pread(fd, p + got, len - got, static_cast<off_t>(off + got));
        if (n <= 0) return false;
        got += static_cast<size_t>(n);
    }
    return true;
}

bool MappedFile::read_at(size_t off, void *dst, size_t len) const {
    if (!native_handle_) return false;
    if (off > size_ || len > size_ - off) return false;
    const int fd = static_cast<int>(reinterpret_cast<intptr_t>(native_handle_));
    char *p = static_cast<char *>(dst);
    size_t got = 0;
    while (got < len) {
        const ssize_t n = pread(fd, p + got, len - got, static_cast<off_t>(off + got));
        if (n <= 0) return false;
        got += static_cast<size_t>(n);
    }
    return true;
}

#endif

// ---------------------------------------------------------------------------
// Cursor
// ---------------------------------------------------------------------------

u8 Cursor::u8v() {
    if (pos_ + 1 > size_) { ok_ = false; return 0; }
    return base_[pos_++];
}

u16 Cursor::u16v() {
    if (pos_ + 2 > size_) { ok_ = false; return 0; }
    const u16 v = static_cast<u16>(base_[pos_] | (base_[pos_ + 1] << 8));
    pos_ += 2;
    return v;
}

u32 Cursor::u32v() {
    if (pos_ + 4 > size_) { ok_ = false; return 0; }
    const u32 v = static_cast<u32>(base_[pos_]) |
                  (static_cast<u32>(base_[pos_ + 1]) << 8) |
                  (static_cast<u32>(base_[pos_ + 2]) << 16) |
                  (static_cast<u32>(base_[pos_ + 3]) << 24);
    pos_ += 4;
    return v;
}

u64 Cursor::u64v() {
    const u64 lo = u32v();
    const u64 hi = u32v();
    return lo | (hi << 32);
}

f32 Cursor::f32v() {
    const u32 bits = u32v();
    f32 f;
    std::memcpy(&f, &bits, 4);
    return f;
}

f64 Cursor::f64v() {
    const u64 bits = u64v();
    f64 d;
    std::memcpy(&d, &bits, 8);
    return d;
}

const u8 *Cursor::bytes(size_t n) {
    if (pos_ + n > size_) { ok_ = false; return nullptr; }
    const u8 *p = base_ + pos_;
    pos_ += n;
    return p;
}

std::string Cursor::str() {
    const u64 n = u64v();
    if (!ok_ || n > remaining()) { ok_ = false; return {}; }
    std::string s(reinterpret_cast<const char *>(base_ + pos_), static_cast<size_t>(n));
    pos_ += static_cast<size_t>(n);
    return s;
}

void Cursor::skip(size_t n) {
    if (pos_ + n > size_) { ok_ = false; return; }
    pos_ += n;
}

void Cursor::seek(size_t p) {
    if (p > size_) { ok_ = false; return; }
    pos_ = p;
}

// ---------------------------------------------------------------------------
// time
// ---------------------------------------------------------------------------

i64 now_us() {
#if defined(_WIN32)
    static LARGE_INTEGER freq = [] {
        LARGE_INTEGER f;
        QueryPerformanceFrequency(&f);
        return f;
    }();
    LARGE_INTEGER c;
    QueryPerformanceCounter(&c);
    return static_cast<i64>(c.QuadPart * 1000000LL / freq.QuadPart);
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<i64>(ts.tv_sec) * 1000000LL + ts.tv_nsec / 1000;
#endif
}

i64 now_ms() { return now_us() / 1000; }

std::string format(const char *fmt, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    const int n = std::vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    if (n <= 0) return {};
    if (static_cast<size_t>(n) < sizeof(buf)) return std::string(buf, static_cast<size_t>(n));
    std::string big(static_cast<size_t>(n) + 1, '\0');
    va_start(ap, fmt);
    std::vsnprintf(&big[0], big.size(), fmt, ap);
    va_end(ap);
    big.resize(static_cast<size_t>(n));
    return big;
}

} // namespace krk
