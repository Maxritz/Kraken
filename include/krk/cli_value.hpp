#pragma once

#include <cerrno>
#include <climits>
#include <cstdlib>

// A command-line value that is not the number it claims to be used to be read
// as 0 or 0.0: `std::atoi("abc")` is 0 and `std::atoi("4096x")` is 4096, so
// `--ctx abc` ran with a zero-token context while `--ctx 4096x` ran with 4096 --
// a typo read as a choice, silently, with nothing on stderr. These parse the
// WHOLE string, which is the contract a caller can act on.
//
// One definition, because kraken and kraken-server take the same flags and a
// second copy is how the two drift: the CLI refused `--prefill bogus` while the
// server accepted it for exactly as long as it took someone to notice.
//
// They REPORT rather than exit. Each entry point owns its message text and its
// exit status, and a helper that calls std::exit cannot be exercised from a
// test binary at all.
namespace krk {

// A whole number, base 10, no trailing characters, inside i32.
inline bool whole_int(const char *v, long long *out) {
    if (!v || !*v) return false;
    errno = 0;
    char *end = nullptr;
    const long long n = std::strtoll(v, &end, 10);
    if (end == v || *end != '\0' || errno == ERANGE || n < INT_MIN ||
        n > INT_MAX) {
        return false;
    }
    *out = n;
    return true;
}

// A number as strtod reads it, but with the whole string consumed. "1e3" and
// "0.5" are accepted; "1.5x" and "" are not.
inline bool whole_float(const char *v, double *out) {
    if (!v || !*v) return false;
    errno = 0;
    char *end = nullptr;
    const double d = std::strtod(v, &end);
    if (end == v || *end != '\0' || errno == ERANGE) return false;
    *out = d;
    return true;
}

// A non-negative whole number. strtoull accepts a leading '-' and wraps it,
// which would turn `--seed -3` into 18446744073709551613 without a word.
inline bool whole_uint(const char *v, unsigned long long *out) {
    if (!v || !*v || v[0] == '-') return false;
    errno = 0;
    char *end = nullptr;
    const unsigned long long n = std::strtoull(v, &end, 10);
    if (end == v || *end != '\0' || errno == ERANGE) return false;
    *out = n;
    return true;
}

} // namespace krk
