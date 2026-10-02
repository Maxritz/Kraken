// json.cpp — recursive-descent JSON parser and compact writer.
#include "krk/json.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace krk {

namespace {

// ---------------------------------------------------------------------------
// parser
// ---------------------------------------------------------------------------

class JsonParser {
public:
    JsonParser(const char *p, const char *end, std::string *err)
        : p_(p), end_(end), err_(err) {}

    JsonValue run() {
        skip_ws();
        if (at_end()) { fail("empty input"); return {}; }
        JsonValue v = parse_value(0);
        if (!failed_) {
            skip_ws();
            if (!at_end()) fail("trailing characters after JSON value");
        }
        return v;
    }

    bool ok() const { return !failed_; }

private:
    const char *p_;
    const char *end_;
    std::string *err_;
    bool failed_ = false;
    int depth_ = 0;

    static constexpr int kMaxDepth = 128;

    bool at_end() const { return p_ >= end_; }
    char peek() const { return at_end() ? '\0' : *p_; }

    void fail(const char *what) {
        if (!failed_ && err_) *err_ = what;
        failed_ = true;
    }

    void skip_ws() {
        while (!at_end()) {
            const char c = *p_;
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') ++p_;
            else break;
        }
    }

    bool eat(char c) {
        if (!at_end() && *p_ == c) { ++p_; return true; }
        return false;
    }

    bool expect(char c) {
        if (eat(c)) return true;
        fail("unexpected character");
        return false;
    }

    JsonValue parse_value(int depth) {
        if (failed_ || depth > kMaxDepth) { fail("nesting too deep"); return {}; }
        skip_ws();
        if (at_end()) { fail("unexpected end of input"); return {}; }
        switch (*p_) {
            case '{': return parse_object(depth + 1);
            case '[': return parse_array(depth + 1);
            case '"': return parse_string_value();
            case 't': return parse_literal("true", JsonValue(true));
            case 'f': return parse_literal("false", JsonValue(false));
            case 'n': return parse_literal("null", JsonValue(nullptr));
            default: return parse_number();
        }
    }

    JsonValue parse_literal(const char *lit, JsonValue v) {
        for (const char *q = lit; *q; ++q, ++p_) {
            if (at_end() || *p_ != *q) { fail("invalid literal"); return {}; }
        }
        return v;
    }

    JsonValue parse_object(int depth) {
        ++p_; // {
        JsonMembers members;
        skip_ws();
        if (eat('}')) return JsonValue(std::move(members));
        while (!failed_) {
            skip_ws();
            if (peek() != '"') { fail("object key must be a string"); break; }
            std::string key = parse_string_raw();
            if (failed_) break;
            skip_ws();
            if (!expect(':')) break;
            JsonValue v = parse_value(depth);
            if (failed_) break;
            members.emplace_back(std::move(key), std::move(v));
            skip_ws();
            if (eat(',')) continue;
            if (eat('}')) break;
            fail("expected ',' or '}' in object");
            break;
        }
        if (failed_) return {};
        return JsonValue(std::move(members));
    }

    JsonValue parse_array(int depth) {
        ++p_; // [
        JsonArray items;
        skip_ws();
        if (eat(']')) return JsonValue(std::move(items));
        while (!failed_) {
            JsonValue v = parse_value(depth);
            if (failed_) break;
            items.push_back(std::move(v));
            skip_ws();
            if (eat(',')) continue;
            if (eat(']')) break;
            fail("expected ',' or ']' in array");
            break;
        }
        if (failed_) return {};
        return JsonValue(std::move(items));
    }

    JsonValue parse_string_value() { return JsonValue(parse_string_raw()); }

    std::string parse_string_raw() {
        std::string out;
        if (!expect('"') || failed_) return out;
        while (!failed_) {
            if (at_end()) { fail("unterminated string"); break; }
            const char c = *p_++;
            if (c == '"') return out;
            if (c == '\\') {
                if (at_end()) { fail("unterminated escape"); break; }
                const char e = *p_++;
                switch (e) {
                    case '"': out += '"'; break;
                    case '\\': out += '\\'; break;
                    case '/': out += '/'; break;
                    case 'b': out += '\b'; break;
                    case 'f': out += '\f'; break;
                    case 'n': out += '\n'; break;
                    case 'r': out += '\r'; break;
                    case 't': out += '\t'; break;
                    case 'u': {
                        u32 cp = 0;
                        if (!read_hex4(&cp)) { fail("bad \\u escape"); break; }
                        if (cp >= 0xD800 && cp <= 0xDBFF) {
                            // high surrogate: require a low surrogate to follow
                            if (p_ + 1 < end_ && p_[0] == '\\' && p_[1] == 'u') {
                                p_ += 2;
                                u32 lo = 0;
                                if (!read_hex4(&lo)) { fail("bad surrogate pair"); break; }
                                if (lo < 0xDC00 || lo > 0xDFFF) {
                                    fail("invalid low surrogate");
                                    break;
                                }
                                cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                            } else {
                                fail("lone high surrogate");
                                break;
                            }
                        } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
                            fail("lone low surrogate");
                            break;
                        }
                        append_utf8(out, cp);
                        break;
                    }
                    default:
                        fail("invalid escape sequence");
                        break;
                }
                continue;
            }
            if (static_cast<u8>(c) < 0x20) { fail("control character in string"); break; }
            out += c;
        }
        return out;
    }

    bool read_hex4(u32 *out) {
        if (end_ - p_ < 4) return false;
        u32 v = 0;
        for (int i = 0; i < 4; i++) {
            const char c = *p_++;
            v <<= 4;
            if (c >= '0' && c <= '9') v |= static_cast<u32>(c - '0');
            else if (c >= 'a' && c <= 'f') v |= static_cast<u32>(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') v |= static_cast<u32>(c - 'A' + 10);
            else return false;
        }
        *out = v;
        return true;
    }

    static void append_utf8(std::string &out, u32 cp) {
        if (cp < 0x80) {
            out += static_cast<char>(cp);
        } else if (cp < 0x800) {
            out += static_cast<char>(0xC0 | (cp >> 6));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            out += static_cast<char>(0xE0 | (cp >> 12));
            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        } else {
            out += static_cast<char>(0xF0 | (cp >> 18));
            out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        }
    }

    JsonValue parse_number() {
        const char *start = p_;
        if (eat('-')) {}
        // integer part; RFC 8259 forbids leading zeros ("01" is invalid)
        if (at_end() || !(*p_ >= '0' && *p_ <= '9')) {
            fail("invalid number");
            return {};
        }
        if (peek() == '0') {
            ++p_;
        } else {
            while (!at_end() && *p_ >= '0' && *p_ <= '9') ++p_;
        }
        // fraction
        if (peek() == '.') {
            ++p_;
            if (at_end() || !(*p_ >= '0' && *p_ <= '9')) { fail("invalid number"); return {}; }
            while (!at_end() && *p_ >= '0' && *p_ <= '9') ++p_;
        }
        // exponent
        if (peek() == 'e' || peek() == 'E') {
            ++p_;
            if (peek() == '+' || peek() == '-') ++p_;
            if (at_end() || !(*p_ >= '0' && *p_ <= '9')) { fail("invalid number"); return {}; }
            while (!at_end() && *p_ >= '0' && *p_ <= '9') ++p_;
        }
        const std::string tok(start, p_);
        return JsonValue(std::strtod(tok.c_str(), nullptr));
    }
};

// ---------------------------------------------------------------------------
// writer
// ---------------------------------------------------------------------------

void write_number(std::string &out, f64 v) {
    if (std::isnan(v) || std::isinf(v)) {
        out += "null"; // JSON has no NaN/Inf
        return;
    }
    char buf[40];
    const f64 r = std::nearbyint(v);
    if (std::fabs(v) < 1e15 && r == v) {
        // integral value: print without a decimal point
        std::snprintf(buf, sizeof(buf), "%lld", static_cast<long long>(v));
    } else {
        std::snprintf(buf, sizeof(buf), "%.17g", v);
    }
    out += buf;
}

} // namespace

void json_escape_to(std::string &out, const std::string &s) {
    out += '"';
    for (unsigned char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += static_cast<char>(c);
                }
        }
    }
    out += '"';
}

const JsonValue *JsonValue::find(const std::string &key) const {
    if (type_ != Type::Object) return nullptr;
    for (const auto &kv : members_)
        if (kv.first == key) return &kv.second;
    return nullptr;
}

f64 JsonValue::get_number(const std::string &key, f64 def) const {
    const JsonValue *v = find(key);
    return (v && v->is_number()) ? v->num_ : def;
}

i64 JsonValue::get_int(const std::string &key, i64 def) const {
    const JsonValue *v = find(key);
    return (v && v->is_number()) ? static_cast<i64>(v->num_) : def;
}

bool JsonValue::get_bool(const std::string &key, bool def) const {
    const JsonValue *v = find(key);
    return (v && v->is_bool()) ? v->bool_ : def;
}

bool JsonValue::get_string(const std::string &key, std::string *out) const {
    const JsonValue *v = find(key);
    if (!v || !v->is_string() || v->str_.empty()) return false;
    if (out) *out = v->str_;
    return true;
}

void JsonValue::set(const std::string &key, JsonValue v) {
    if (type_ != Type::Object) {
        type_ = Type::Object;
        members_.clear();
    }
    for (auto &kv : members_) {
        if (kv.first == key) {
            kv.second = std::move(v);
            return;
        }
    }
    members_.emplace_back(key, std::move(v));
}

void JsonValue::push_back(JsonValue v) {
    if (type_ != Type::Array) {
        type_ = Type::Array;
        arr_.clear();
    }
    arr_.push_back(std::move(v));
}

void JsonValue::dump_to(std::string &out) const {
    switch (type_) {
        case Type::Null: out += "null"; break;
        case Type::Bool: out += bool_ ? "true" : "false"; break;
        case Type::Number: write_number(out, num_); break;
        case Type::String: json_escape_to(out, str_); break;
        case Type::Array: {
            out += '[';
            bool first = true;
            for (const JsonValue &v : arr_) {
                if (!first) out += ',';
                first = false;
                v.dump_to(out);
            }
            out += ']';
            break;
        }
        case Type::Object: {
            out += '{';
            bool first = true;
            for (const auto &kv : members_) {
                if (!first) out += ',';
                first = false;
                json_escape_to(out, kv.first);
                out += ':';
                kv.second.dump_to(out);
            }
            out += '}';
            break;
        }
    }
}

std::string JsonValue::dump() const {
    std::string out;
    out.reserve(256);
    dump_to(out);
    return out;
}

bool json_parse(std::string_view text, JsonValue *out, std::string *err) {
    JsonParser p(text.data(), text.data() + text.size(), err);
    *out = p.run();
    return p.ok();
}

} // namespace krk
