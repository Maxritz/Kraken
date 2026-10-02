// ============================================================================
//  json.hpp — a small, dependency-free JSON value tree.
//
//  Written for the OpenAI-compatible server: parse a request body into a
//  tree, build responses with escaping handled correctly. The parser is a
//  straightforward recursive descent over UTF-8; \uXXXX escapes decode into
//  UTF-8 (surrogate pairs included). Numbers round-trip through f64, which is
//  what every field this API carries needs.
// ============================================================================
#ifndef KRK_JSON_HPP
#define KRK_JSON_HPP

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "krk/common.hpp"

namespace krk {

class JsonValue;
using JsonMembers = std::vector<std::pair<std::string, JsonValue>>;
using JsonArray = std::vector<JsonValue>;

class JsonValue {
public:
    enum class Type { Null, Bool, Number, String, Array, Object };

    JsonValue() : type_(Type::Null) {}
    JsonValue(std::nullptr_t) : type_(Type::Null) {}
    JsonValue(bool b) : type_(Type::Bool), bool_(b) {}
    JsonValue(f64 n) : type_(Type::Number), num_(n) {}
    JsonValue(i64 n) : type_(Type::Number), num_(static_cast<f64>(n)) {}
    JsonValue(i32 n) : type_(Type::Number), num_(static_cast<f64>(n)) {}
    JsonValue(u64 n) : type_(Type::Number), num_(static_cast<f64>(n)) {}
    JsonValue(const char *s) : type_(Type::String), str_(s ? s : "") {}
    JsonValue(std::string s) : type_(Type::String), str_(std::move(s)) {}
    JsonValue(JsonArray a) : type_(Type::Array), arr_(std::move(a)) {}
    JsonValue(JsonMembers m) : type_(Type::Object), members_(std::move(m)) {}

    static JsonValue object() { return JsonValue(JsonMembers{}); }
    static JsonValue array() { return JsonValue(JsonArray{}); }

    Type type() const { return type_; }
    bool is_null() const { return type_ == Type::Null; }
    bool is_bool() const { return type_ == Type::Bool; }
    bool is_number() const { return type_ == Type::Number; }
    bool is_string() const { return type_ == Type::String; }
    bool is_array() const { return type_ == Type::Array; }
    bool is_object() const { return type_ == Type::Object; }

    // Accessors: safe but type-checked. as_bool() on a non-bool returns false.
    bool as_bool(bool def = false) const { return is_bool() ? bool_ : def; }
    f64 as_number(f64 def = 0) const { return is_number() ? num_ : def; }
    i64 as_int(i64 def = 0) const { return is_number() ? static_cast<i64>(num_) : def; }
    const std::string &as_string() const { return str_; }

    // ---- object interface --------------------------------------------------
    // First match wins; duplicate keys are not produced by the parser.
    const JsonValue *find(const std::string &key) const;
    // get_* sugar over find().
    f64 get_number(const std::string &key, f64 def = 0) const;
    i64 get_int(const std::string &key, i64 def = 0) const;
    bool get_bool(const std::string &key, bool def = false) const;
    // Present only when the key holds a non-empty string.
    bool get_string(const std::string &key, std::string *out) const;

    void set(const std::string &key, JsonValue v);
    void push_back(JsonValue v);

    const JsonArray &items() const { return arr_; }
    const JsonMembers &members() const { return members_; }

    // Serializes compactly (no spaces). Numbers print with enough digits to
    // round-trip a f64, integers without a decimal point.
    std::string dump() const;
    void dump_to(std::string &out) const;

private:
    Type type_ = Type::Null;
    bool bool_ = false;
    f64 num_ = 0;
    std::string str_;
    JsonArray arr_;
    JsonMembers members_;
};

// Parses `text`. Returns false and fills `err` on malformed input; the
// returned value is then Null. Accepts trailing whitespace, nothing else.
bool json_parse(std::string_view text, JsonValue *out, std::string *err);

// Escapes a string as a JSON string literal (with quotes).
void json_escape_to(std::string &out, const std::string &s);

} // namespace krk

#endif // KRK_JSON_HPP
