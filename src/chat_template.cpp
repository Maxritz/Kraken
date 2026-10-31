// chat_template.cpp -- a scoped Jinja interpreter for `tokenizer.chat_template`.
//
// WHY an interpreter and not a table of formats: the layout a model was trained
// on is authored by the model's authors, ships inside the GGUF, and changes
// between point releases of the same family. A table of seven formats is wrong
// the eighth time. The file is the authority, and it speaks Jinja -- so this
// reads Jinja.
//
// SCOPE. The language is implemented to the depth real templates use and no
// further: `if`/`for`/`set`/`break`/`continue`, expressions with the usual
// operators and precedence, filters, tests, `loop.*`, `namespace()`,
// `raise_exception()`, and the standard method calls on strings, lists and
// dicts. Anything outside that -- `macro`, `include`, `import`, `call`,
// `filter` blocks, `with` -- is a NAMED failure with the construct in the
// message, never a best-effort guess. A caller that gets the name back can fall
// back to a builtin; a caller that got a plausible-looking wrong prompt could
// not tell. That distinction is the whole design.
//
// WHITESPACE. `{%-`/`-%}` and `{{-`/`-}}` are honoured, because templates use
// them to control exactly the newlines that end up in the prompt -- and a stray
// newline is a token the model did not expect. No other trimming is applied:
// no `trim_blocks`, no `lstrip_blocks`, because a GGUF's template was authored
// against llama.cpp's defaults and those are off.
#include "krk/chat_template.hpp"

#include "krk/common.hpp"   // i64 / f64

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace krk {
namespace {

// ---------------------------------------------------------------------------
// Values
//
// A deliberate subset of Python's: none, bool, int, float, str, list, dict, and
// an `undefined` that is distinct from none (Jinja's `is defined` turns on it,
// and a template that guards with it must see the guard fail rather than see a
// null). Lists and dicts are shared pointers: `{% set x = messages %}` and
// `{% for m in x %}` must not deep-copy a thousand turns.
// ---------------------------------------------------------------------------
struct Val;
using VP = std::shared_ptr<Val>;

struct Val {
    enum T { Und, Nul, Boo, Int, Num, Str, Arr, Obj } t = Und;
    bool b = false;
    i64 i = 0;
    f64 n = 0.0;
    std::string s;
    std::shared_ptr<std::vector<VP>> a;
    // Insertion-ordered, because `|tojson` and `{% for k, v in d.items() %}`
    // both render in the order the template built the object.
    std::shared_ptr<std::vector<std::pair<std::string, VP>>> o;

    static VP undef() { return std::make_shared<Val>(); }
    static VP none() {
        VP v = std::make_shared<Val>();
        v->t = Nul;
        return v;
    }
    static VP boolean(bool x) {
        VP v = std::make_shared<Val>();
        v->t = Boo;
        v->b = x;
        return v;
    }
    static VP integer(i64 x) {
        VP v = std::make_shared<Val>();
        v->t = Int;
        v->i = x;
        v->n = static_cast<f64>(x);
        return v;
    }
    static VP number(f64 x) {
        VP v = std::make_shared<Val>();
        v->t = Num;
        v->n = x;
        v->i = static_cast<i64>(x);
        return v;
    }
    static VP str(std::string x) {
        VP v = std::make_shared<Val>();
        v->t = Str;
        v->s = std::move(x);
        return v;
    }
    static VP arr() {
        VP v = std::make_shared<Val>();
        v->t = Arr;
        v->a = std::make_shared<std::vector<VP>>();
        return v;
    }
    static VP obj() {
        VP v = std::make_shared<Val>();
        v->t = Obj;
        v->o = std::make_shared<std::vector<std::pair<std::string, VP>>>();
        return v;
    }
    bool is_num() const { return t == Int || t == Num; }
};

// The one exception type. `unsupported` distinguishes "this template is outside
// the language subset" from "the template ran and refused" -- the first is a
// reason to fall back, the second is the template's own verdict and should be
// shown to the user.
struct TmplError : std::runtime_error {
    explicit TmplError(const std::string &m, bool unsupported = false)
        : std::runtime_error(m), unsupported(unsupported) {}
    bool unsupported;
};

std::string fmt_num(f64 x) {
    if (std::isnan(x)) return "nan";
    if (std::isinf(x)) return x < 0 ? "-inf" : "inf";
    if (x == std::floor(x) && std::fabs(x) < 1e15)
        return std::to_string(static_cast<i64>(x));
    char b[40];
    std::snprintf(b, sizeof b, "%.10g", x);
    return b;
}

bool truthy(const VP &v) {
    if (!v) return false;
    switch (v->t) {
        case Val::Und:
        case Val::Nul: return false;
        case Val::Boo: return v->b;
        case Val::Int: return v->i != 0;
        case Val::Num: return v->n != 0.0;
        case Val::Str: return !v->s.empty();
        case Val::Arr: return v->a && !v->a->empty();
        case Val::Obj: return v->o && !v->o->empty();
    }
    return false;
}

std::string to_str(const VP &v) {
    if (!v) return {};
    switch (v->t) {
        case Val::Und:
        case Val::Nul: return {};
        case Val::Boo: return v->b ? "True" : "False";
        case Val::Int: return std::to_string(v->i);
        case Val::Num: return fmt_num(v->n);
        case Val::Str: return v->s;
        case Val::Arr: {
            std::string r;
            if (!v->a) return r;
            for (size_t k = 0; k < v->a->size(); k++) {
                if (k) r += ", ";
                r += to_str((*v->a)[k]);
            }
            return r;
        }
        case Val::Obj: {
            std::string r;
            if (!v->o) return r;
            for (size_t k = 0; k < v->o->size(); k++) {
                if (k) r += ", ";
                r += "'" + (*v->o)[k].first + "': " + to_str((*v->o)[k].second);
            }
            return r;
        }
    }
    return {};
}

i64 to_int(const VP &v) {
    if (!v) return 0;
    if (v->t == Val::Int) return v->i;
    if (v->t == Val::Num) return static_cast<i64>(v->n);
    if (v->t == Val::Boo) return v->b ? 1 : 0;
    if (v->t == Val::Str) return std::strtoll(v->s.c_str(), nullptr, 10);
    throw TmplError("cannot convert this value to an integer");
}

f64 to_float(const VP &v) {
    if (!v) return 0.0;
    if (v->is_num()) return v->n;
    if (v->t == Val::Boo) return v->b ? 1.0 : 0.0;
    if (v->t == Val::Str) return std::strtod(v->s.c_str(), nullptr);
    throw TmplError("cannot convert this value to a number");
}

std::string lower_ascii(std::string s) {
    for (char &c : s)
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
    return s;
}

std::string upper_ascii(std::string s) {
    for (char &c : s)
        if (c >= 'a' && c <= 'z') c = static_cast<char>(c - 'a' + 'A');
    return s;
}

bool is_space(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' ||
           c == '\v';
}

std::string trim_str(const std::string &s) {
    size_t b = 0, e = s.size();
    while (b < e && is_space(s[b])) b++;
    while (e > b && is_space(s[e - 1])) e--;
    return s.substr(b, e - b);
}

std::string json_escape(const std::string &s) {
    std::string r;
    r.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) {
            case '"': r += "\\\""; break;
            case '\\': r += "\\\\"; break;
            case '\n': r += "\\n"; break;
            case '\r': r += "\\r"; break;
            case '\t': r += "\\t"; break;
            case '\b': r += "\\b"; break;
            case '\f': r += "\\f"; break;
            default:
                if (c < 0x20) {
                    char b[8];
                    std::snprintf(b, sizeof b, "\\u%04x", c);
                    r += b;
                } else {
                    r += static_cast<char>(c);
                }
        }
    }
    return r;
}

std::string to_json(const VP &v) {
    if (!v) return "null";
    switch (v->t) {
        case Val::Und:
        case Val::Nul: return "null";
        case Val::Boo: return v->b ? "true" : "false";
        case Val::Int: return std::to_string(v->i);
        case Val::Num: return fmt_num(v->n);
        case Val::Str: return "\"" + json_escape(v->s) + "\"";
        case Val::Arr: {
            std::string r = "[";
            if (v->a)
                for (size_t k = 0; k < v->a->size(); k++) {
                    if (k) r += ", ";
                    r += to_json((*v->a)[k]);
                }
            return r + "]";
        }
        case Val::Obj: {
            std::string r = "{";
            if (v->o)
                for (size_t k = 0; k < v->o->size(); k++) {
                    if (k) r += ", ";
                    r += "\"" + json_escape((*v->o)[k].first) + "\": " +
                         to_json((*v->o)[k].second);
                }
            return r + "}";
        }
    }
    return "null";
}

// ---------------------------------------------------------------------------
// Expression tokens
// ---------------------------------------------------------------------------
enum class TK { Name, Num, Str, Op, End };

struct Tok {
    TK k = TK::End;
    std::string s;   // name / operator / string body
    f64 n = 0.0;
};

bool name_start(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
}
bool name_char(char c) {
    return name_start(c) || (c >= '0' && c <= '9');
}

std::vector<Tok> lex_expr(const std::string &src) {
    std::vector<Tok> out;
    size_t i = 0;
    const size_t n = src.size();
    while (i < n) {
        const char c = src[i];
        if (is_space(c)) {
            i++;
            continue;
        }
        if (name_start(c)) {
            size_t j = i;
            while (j < n && name_char(src[j])) j++;
            Tok t;
            t.k = TK::Name;
            t.s = src.substr(i, j - i);
            out.push_back(std::move(t));
            i = j;
            continue;
        }
        if ((c >= '0' && c <= '9') ||
            (c == '.' && i + 1 < n && src[i + 1] >= '0' && src[i + 1] <= '9')) {
            size_t j = i;
            while (j < n && src[j] >= '0' && src[j] <= '9') j++;
            if (j < n && src[j] == '.') {
                j++;
                while (j < n && src[j] >= '0' && src[j] <= '9') j++;
            }
            if (j < n && (src[j] == 'e' || src[j] == 'E')) {
                size_t k = j + 1;
                if (k < n && (src[k] == '+' || src[k] == '-')) k++;
                if (k < n && src[k] >= '0' && src[k] <= '9') {
                    j = k;
                    while (j < n && src[j] >= '0' && src[j] <= '9') j++;
                }
            }
            Tok t;
            t.k = TK::Num;
            t.s = src.substr(i, j - i);
            t.n = std::strtod(t.s.c_str(), nullptr);
            out.push_back(std::move(t));
            i = j;
            continue;
        }
        if (c == '\'' || c == '"') {
            const char q = c;
            size_t j = i + 1;
            std::string body;
            while (j < n && src[j] != q) {
                if (src[j] == '\\' && j + 1 < n) {
                    const char e = src[j + 1];
                    switch (e) {
                        case 'n': body += '\n'; break;
                        case 't': body += '\t'; break;
                        case 'r': body += '\r'; break;
                        case '\\': body += '\\'; break;
                        case '\'': body += '\''; break;
                        case '"': body += '"'; break;
                        case '0': body += '\0'; break;
                        default: body += e;
                    }
                    j += 2;
                    continue;
                }
                body += src[j++];
            }
            if (j >= n)
                throw TmplError("unterminated string literal in an expression",
                                true);
            Tok t;
            t.k = TK::Str;
            t.s = std::move(body);
            out.push_back(std::move(t));
            i = j + 1;
            continue;
        }
        // Two-character operators. Augmented assignment (`+=`, `//=`) is not
        // in the subset -- no template in the target collection writes one,
        // and Jinja's `{% set x = x + 1 %}` is what the rest use.
        static const char *ops2[] = {"==", "!=", "<=", ">=", "//", "**", nullptr};
        bool matched = false;
        for (int k = 0; ops2[k]; k++) {
            if (src.compare(i, 2, ops2[k]) == 0) {
                Tok t;
                t.k = TK::Op;
                t.s = ops2[k];
                out.push_back(std::move(t));
                i += 2;
                matched = true;
                break;
            }
        }
        if (matched) continue;
        Tok t;
        t.k = TK::Op;
        t.s = std::string(1, c);
        out.push_back(std::move(t));
        i++;
    }
    Tok end;
    end.k = TK::End;
    out.push_back(std::move(end));
    return out;
}

// ---------------------------------------------------------------------------
// Expression AST
// ---------------------------------------------------------------------------
struct Expr;
using EP = std::shared_ptr<Expr>;

struct Expr {
    enum K { Lit, Name, Attr, Item, Slice, Call, Filter, Test, Unary, Bin, Cond,
             ListE, DictE } k = Lit;
    VP lit;                       // Lit
    std::string name;             // Name / Attr / Call / Filter / Test / op
    std::vector<EP> args;         // see the parser for each node's layout
    // Keyword arguments, for calls only. `namespace(v=0)` is the one templates
    // reach for constantly -- it is Jinja's way to get a mutable flag out of a
    // loop, and the Qwen/Hermes tool templates build their state with it.
    std::vector<std::pair<std::string, EP>> kwargs;
};

struct ExprParser {
    std::vector<Tok> t;
    size_t p = 0;

    const Tok &peek(size_t off = 0) const {
        const size_t i = p + off;
        return i < t.size() ? t[i] : t.back();
    }
    bool is_op(const char *s, size_t off = 0) const {
        return peek(off).k == TK::Op && peek(off).s == s;
    }
    bool is_name(const char *s, size_t off = 0) const {
        return peek(off).k == TK::Name && peek(off).s == s;
    }
    void expect_op(const char *s) {
        if (!is_op(s))
            throw TmplError(std::string("expected '") + s + "' in an expression",
                            true);
        p++;
    }

    EP make(Expr::K k) {
        EP e = std::make_shared<Expr>();
        e->k = k;
        return e;
    }

    // conditional:  or_expr ['if' or_expr ['else' or_expr]]
    EP parse_expr() { return parse_cond(); }

    EP parse_cond() {
        EP a = parse_or();
        if (is_name("if")) {
            p++;
            EP c = parse_or();
            EP d;
            if (is_name("else")) {
                p++;
                d = parse_cond();
            }
            EP e = make(Expr::Cond);
            e->args = {a, c, d};
            return e;
        }
        return a;
    }

    EP parse_or() {
        EP a = parse_and();
        while (is_name("or")) {
            p++;
            EP b = parse_and();
            EP e = make(Expr::Bin);
            e->name = "or";
            e->args = {a, b};
            a = e;
        }
        return a;
    }

    EP parse_and() {
        EP a = parse_not();
        while (is_name("and")) {
            p++;
            EP b = parse_not();
            EP e = make(Expr::Bin);
            e->name = "and";
            e->args = {a, b};
            a = e;
        }
        return a;
    }

    EP parse_not() {
        if (is_name("not")) {
            p++;
            EP a = parse_not();
            EP e = make(Expr::Unary);
            e->name = "not";
            e->args = {a};
            return e;
        }
        return parse_cmp();
    }

    // Comparisons chain like Python's: `a < b < c` is `(a<b) and (b<c)`.
    EP parse_cmp() {
        EP a = parse_concat();
        while (true) {
            std::string op;
            if (peek().k == TK::Op &&
                (peek().s == "==" || peek().s == "!=" || peek().s == "<" ||
                 peek().s == ">" || peek().s == "<=" || peek().s == ">=")) {
                op = peek().s;
                p++;
            } else if (is_name("in")) {
                op = "in";
                p++;
            } else if (is_name("not") && is_name("in", 1)) {
                op = "notin";
                p += 2;
            } else {
                break;
            }
            EP b = parse_concat();
            EP e = make(Expr::Bin);
            e->name = op;
            e->args = {a, b};
            a = e;
        }
        return a;
    }

    EP parse_concat() {
        EP a = parse_add();
        while (is_op("~")) {
            p++;
            EP b = parse_add();
            EP e = make(Expr::Bin);
            e->name = "~";
            e->args = {a, b};
            a = e;
        }
        return a;
    }

    EP parse_add() {
        EP a = parse_mul();
        while (is_op("+") || is_op("-")) {
            const std::string op = peek().s;
            p++;
            EP b = parse_mul();
            EP e = make(Expr::Bin);
            e->name = op;
            e->args = {a, b};
            a = e;
        }
        return a;
    }

    EP parse_mul() {
        EP a = parse_unary();
        while (is_op("*") || is_op("/") || is_op("//") || is_op("%")) {
            const std::string op = peek().s;
            p++;
            EP b = parse_unary();
            EP e = make(Expr::Bin);
            e->name = op;
            e->args = {a, b};
            a = e;
        }
        return a;
    }

    EP parse_unary() {
        if (is_op("-")) {
            p++;
            EP a = parse_unary();
            EP e = make(Expr::Unary);
            e->name = "-";
            e->args = {a};
            return e;
        }
        if (is_op("+")) {
            p++;
            return parse_unary();
        }
        return parse_filter();
    }

    // Filters and tests bind tighter than arithmetic (Jinja's grammar), which
    // is why `'x' + y | trim` means `'x' + (y|trim)`.
    EP parse_filter() {
        EP a = parse_postfix();
        while (true) {
            if (is_op("|")) {
                p++;
                if (peek().k != TK::Name)
                    throw TmplError("expected a filter name after '|'", true);
                EP f = make(Expr::Filter);
                f->name = peek().s;
                p++;
                f->args.push_back(a);
                if (is_op("(")) {
                    p++;
                    while (!is_op(")")) {
                        f->args.push_back(parse_expr());
                        if (is_op(",")) {
                            p++;
                            continue;
                        }
                        break;
                    }
                    expect_op(")");
                }
                a = f;
                continue;
            }
            if (is_name("is")) {
                p++;
                bool negate = false;
                if (is_name("not")) {
                    negate = true;
                    p++;
                }
                if (peek().k != TK::Name)
                    throw TmplError("expected a test name after 'is'", true);
                EP t = make(Expr::Test);
                t->name = peek().s;
                p++;
                t->args.push_back(a);
                if (is_op("(")) {
                    p++;
                    while (!is_op(")")) {
                        t->args.push_back(parse_expr());
                        if (is_op(",")) {
                            p++;
                            continue;
                        }
                        break;
                    }
                    expect_op(")");
                }
                if (negate) {
                    EP u = make(Expr::Unary);
                    u->name = "not";
                    u->args = {t};
                    a = u;
                } else {
                    a = t;
                }
                continue;
            }
            break;
        }
        return a;
    }

    EP parse_postfix() {
        EP a = parse_primary();
        while (true) {
            if (is_op(".")) {
                p++;
                if (peek().k != TK::Name)
                    throw TmplError("expected a name after '.'", true);
                EP e = make(Expr::Attr);
                e->name = peek().s;
                p++;
                e->args.push_back(a);
                a = e;
                continue;
            }
            if (is_op("[")) {
                p++;
                EP e = make(Expr::Item);
                e->args.push_back(a);
                if (is_op(":")) {
                    EP sl = make(Expr::Slice);
                    sl->args.push_back(nullptr);
                    p++;
                    if (!is_op("]") && !is_op(":"))
                        sl->args[0] = parse_expr();
                    if (is_op(":")) {
                        p++;
                        if (!is_op("]")) sl->args.push_back(parse_expr());
                        else sl->args.push_back(nullptr);
                    } else {
                        sl->args.push_back(nullptr);
                    }
                    e->args.push_back(sl);
                } else {
                    e->args.push_back(parse_expr());
                    if (is_op(":")) {
                        EP sl = make(Expr::Slice);
                        sl->args.push_back(e->args.back());
                        p++;
                        if (!is_op("]")) sl->args.push_back(parse_expr());
                        else sl->args.push_back(nullptr);
                        e->args[1] = sl;
                    }
                }
                expect_op("]");
                a = e;
                continue;
            }
            if (is_op("(")) {
                // A call: the callee is whatever we just built. `Attr` covering
                // a method call is resolved at eval time.
                p++;
                EP e = make(Expr::Call);
                e->args.push_back(a);
                while (!is_op(")")) {
                    if (peek().k == TK::Name && is_op("=", 1)) {
                        const std::string kw = peek().s;
                        p += 2;
                        e->kwargs.push_back({kw, parse_expr()});
                    } else {
                        e->args.push_back(parse_expr());
                    }
                    if (is_op(",")) {
                        p++;
                        continue;
                    }
                    break;
                }
                expect_op(")");
                a = e;
                continue;
            }
            break;
        }
        return a;
    }

    EP parse_primary() {
        const Tok &t0 = peek();
        if (t0.k == TK::Num) {
            EP e = make(Expr::Lit);
            e->lit = Val::number(t0.n);
            p++;
            return e;
        }
        if (t0.k == TK::Str) {
            EP e = make(Expr::Lit);
            e->lit = Val::str(t0.s);
            p++;
            // Adjacent string literals concatenate, as in Python.
            while (peek().k == TK::Str) {
                e->lit->s += peek().s;
                p++;
            }
            return e;
        }
        if (t0.k == TK::Name) {
            const std::string &nm = t0.s;
            if (nm == "true" || nm == "True") {
                p++;
                EP e = make(Expr::Lit);
                e->lit = Val::boolean(true);
                return e;
            }
            if (nm == "false" || nm == "False") {
                p++;
                EP e = make(Expr::Lit);
                e->lit = Val::boolean(false);
                return e;
            }
            if (nm == "none" || nm == "None") {
                p++;
                EP e = make(Expr::Lit);
                e->lit = Val::none();
                return e;
            }
            p++;
            EP e = make(Expr::Name);
            e->name = nm;
            return e;
        }
        if (is_op("(")) {
            p++;
            EP a = parse_expr();
            expect_op(")");
            return a;
        }
        if (is_op("[")) {
            p++;
            EP e = make(Expr::ListE);
            while (!is_op("]")) {
                e->args.push_back(parse_expr());
                if (is_op(",")) {
                    p++;
                    continue;
                }
                break;
            }
            expect_op("]");
            return e;
        }
        if (is_op("{")) {
            p++;
            EP e = make(Expr::DictE);
            while (!is_op("}")) {
                EP key = parse_expr();
                expect_op(":");
                EP val = parse_expr();
                e->args.push_back(key);
                e->args.push_back(val);
                if (is_op(",")) {
                    p++;
                    continue;
                }
                break;
            }
            expect_op("}");
            return e;
        }
        throw TmplError("unexpected token in an expression: '" + peek().s + "'",
                        true);
    }
};

// ---------------------------------------------------------------------------
// Statement tree
// ---------------------------------------------------------------------------
struct Stmt;
using SP = std::shared_ptr<Stmt>;

struct Stmt {
    enum K { Text, Out, If, For, Set, SetAttr, Break, Continue } k = Text;
    std::string text;             // Text
    EP expr;                      // Out value / If cond / For iterable / Set value
    std::string var, var2;        // Set name / For targets
    std::vector<SP> body, alt;
    bool have_alt = false;
};

struct Chunk {
    enum K { Text, Out, Stmt } k = Text;
    std::string s;
};

// Split the template into text / `{{ }}` / `{% %}`, applying `-` whitespace
// control. Comments are dropped here so no later stage has to think about them.
std::vector<Chunk> split_chunks(const std::string &src) {
    std::vector<Chunk> out;
    size_t i = 0;
    const size_t n = src.size();
    std::string text;
    bool strip_next_text = false;

    auto flush_text = [&](bool strip_trailing) {
        std::string t = std::move(text);
        text.clear();
        if (strip_next_text) {
            size_t b = 0;
            while (b < t.size() && is_space(t[b])) b++;
            t.erase(0, b);
            strip_next_text = false;
        }
        if (strip_trailing) {
            size_t e = t.size();
            while (e > 0 && is_space(t[e - 1])) e--;
            t.erase(e);
        }
        if (!t.empty()) {
            Chunk c;
            c.k = Chunk::Text;
            c.s = std::move(t);
            out.push_back(std::move(c));
        }
    };

    while (i < n) {
        if (src[i] == '{' && i + 1 < n &&
            (src[i + 1] == '{' || src[i + 1] == '%' || src[i + 1] == '#')) {
            const char kind = src[i + 1];
            size_t j = i + 2;
            bool strip_before = false;
            if (j < n && src[j] == '-') {
                strip_before = true;
                j++;
            }
            const char *close = kind == '{' ? "}}" : (kind == '%' ? "%}" : "#}");
            const size_t clen = 2;
            size_t e = src.find(close, j);
            if (e == std::string::npos)
                throw TmplError(std::string("unterminated '") + src[i + 1] +
                                    "' tag in the template",
                                true);
            bool strip_after = false;
            size_t body_end = e;
            if (body_end > j && src[body_end - 1] == '-') {
                strip_after = true;
                body_end--;
            }
            const std::string body = src.substr(j, body_end - j);
            flush_text(strip_before);
            if (kind == '#') {
                // comment: dropped
            } else {
                Chunk c;
                c.k = kind == '{' ? Chunk::Out : Chunk::Stmt;
                c.s = trim_str(body);
                out.push_back(std::move(c));
            }
            strip_next_text = strip_after;
            i = e + clen;
            continue;
        }
        text += src[i++];
    }
    flush_text(false);
    return out;
}

struct TemplateParser {
    std::vector<Chunk> ch;
    size_t p = 0;

    SP make(Stmt::K k) {
        SP s = std::make_shared<Stmt>();
        s->k = k;
        return s;
    }

    // Keyword of a `{% ... %}` chunk, and the rest of its text.
    static std::string kw_of(const std::string &body, std::string *rest) {
        size_t b = 0;
        while (b < body.size() && name_char(body[b])) b++;
        if (rest) *rest = trim_str(body.substr(b));
        return body.substr(0, b);
    }

    EP parse_expr_str(const std::string &s) {
        ExprParser ep;
        ep.t = lex_expr(s);
        EP e = ep.parse_expr();
        if (ep.peek().k != TK::End)
            throw TmplError("trailing input in an expression: '" + s + "'", true);
        return e;
    }

    // Parse statements until one of `stop` keywords (which is NOT consumed) or
    // the end of the chunk list.
    std::vector<SP> parse_block(const std::vector<std::string> &stop,
                                std::string *stopped_at) {
        std::vector<SP> out;
        while (p < ch.size()) {
            const Chunk &c = ch[p];
            if (c.k == Chunk::Text) {
                SP s = make(Stmt::Text);
                s->text = c.s;
                out.push_back(s);
                p++;
                continue;
            }
            if (c.k == Chunk::Out) {
                SP s = make(Stmt::Out);
                s->expr = parse_expr_str(c.s);
                out.push_back(s);
                p++;
                continue;
            }
            std::string rest;
            const std::string kw = kw_of(c.s, &rest);
            for (const std::string &s : stop)
                if (kw == s) {
                    if (stopped_at) *stopped_at = kw;
                    return out;
                }
            p++;
            if (kw == "if") {
                SP s = make(Stmt::If);
                s->expr = parse_expr_str(rest);
                s->body = parse_block({"elif", "else", "endif"}, nullptr);
                // elif/else chain
                while (p < ch.size() && ch[p].k == Chunk::Stmt) {
                    std::string r2;
                    const std::string kw2 = kw_of(ch[p].s, &r2);
                    if (kw2 == "elif") {
                        p++;
                        SP nested = make(Stmt::If);
                        nested->expr = parse_expr_str(r2);
                        nested->body = parse_block({"elif", "else", "endif"},
                                                   nullptr);
                        if (!s->have_alt) {
                            s->alt.push_back(nested);
                            s->have_alt = true;
                        } else {
                            // append to the tail of the chain
                            SP cur = s->alt.back();
                            while (cur->have_alt && !cur->alt.empty() &&
                                   cur->alt.back()->k == Stmt::If)
                                cur = cur->alt.back();
                            cur->alt.push_back(nested);
                            cur->have_alt = true;
                        }
                        continue;
                    }
                    if (kw2 == "else") {
                        p++;
                        std::vector<SP> e =
                            parse_block({"elif", "else", "endif"}, nullptr);
                        SP cur = s;
                        while (cur->have_alt && !cur->alt.empty() &&
                               cur->alt.back()->k == Stmt::If)
                            cur = cur->alt.back();
                        cur->alt = std::move(e);
                        cur->have_alt = true;
                        continue;
                    }
                    if (kw2 == "endif") {
                        p++;
                        break;
                    }
                    break;
                }
                out.push_back(s);
                continue;
            }
            if (kw == "for") {
                // `for a, b in expr` / `for x in expr`
                size_t in_pos = std::string::npos;
                for (size_t k = 0; k + 1 < rest.size(); k++) {
                    if (rest.compare(k, 4, " in ") == 0) {
                        in_pos = k;
                        break;
                    }
                }
                if (in_pos == std::string::npos)
                    throw TmplError("malformed 'for' tag: " + c.s, true);
                const std::string targets = trim_str(rest.substr(0, in_pos));
                const std::string iterable = trim_str(rest.substr(in_pos + 4));
                SP s = make(Stmt::For);
                s->expr = parse_expr_str(iterable);
                const size_t comma = targets.find(',');
                if (comma == std::string::npos) {
                    s->var = targets;
                } else {
                    s->var = trim_str(targets.substr(0, comma));
                    s->var2 = trim_str(targets.substr(comma + 1));
                }
                if (s->var.empty())
                    throw TmplError("malformed 'for' tag: " + c.s, true);
                s->body = parse_block({"else", "endfor"}, nullptr);
                while (p < ch.size() && ch[p].k == Chunk::Stmt) {
                    std::string r2;
                    const std::string kw2 = kw_of(ch[p].s, &r2);
                    if (kw2 == "else") {
                        p++;
                        s->alt = parse_block({"endfor"}, nullptr);
                        s->have_alt = true;
                        continue;
                    }
                    if (kw2 == "endfor") {
                        p++;
                        break;
                    }
                    break;
                }
                out.push_back(s);
                continue;
            }
            if (kw == "set") {
                // `set x = expr` or `set ns.attr = expr`
                const size_t eq = rest.find('=');
                if (eq == std::string::npos || (eq + 1 < rest.size() &&
                                                rest[eq + 1] == '='))
                    throw TmplError("malformed 'set' tag: " + c.s, true);
                const std::string lhs = trim_str(rest.substr(0, eq));
                const std::string rhs = trim_str(rest.substr(eq + 1));
                const size_t dot = lhs.find('.');
                SP s = (dot == std::string::npos) ? make(Stmt::Set)
                                                  : make(Stmt::SetAttr);
                if (dot == std::string::npos) {
                    s->var = lhs;
                } else {
                    s->var = trim_str(lhs.substr(0, dot));
                    s->var2 = trim_str(lhs.substr(dot + 1));
                }
                s->expr = parse_expr_str(rhs);
                out.push_back(s);
                continue;
            }
            if (kw == "break") {
                out.push_back(make(Stmt::Break));
                continue;
            }
            if (kw == "continue") {
                out.push_back(make(Stmt::Continue));
                continue;
            }
            if (kw == "raw") {
                // `{% raw %}` ... `{% endraw %}`: literal, by definition.
                std::string lit;
                while (p < ch.size()) {
                    std::string r2;
                    const std::string kw2 = kw_of(ch[p].s, &r2);
                    if (ch[p].k == Chunk::Stmt && kw2 == "endraw") {
                        p++;
                        break;
                    }
                    if (ch[p].k == Chunk::Text) lit += ch[p].s;
                    else if (ch[p].k == Chunk::Out) lit += "{{" + ch[p].s + "}}";
                    else lit += "{%" + ch[p].s + "%}";
                    p++;
                }
                SP s = make(Stmt::Text);
                s->text = lit;
                out.push_back(s);
                continue;
            }
            // Everything else is outside the subset, and named so the caller can
            // fall back rather than ship a prompt nobody authored.
            throw TmplError("unsupported template tag '{% " + kw +
                                " %}' (this engine implements if/for/set/"
                                "raw/break/continue)",
                            true);
        }
        if (stopped_at) *stopped_at = {};
        return out;
    }
};

// ---------------------------------------------------------------------------
// Evaluation
// ---------------------------------------------------------------------------
enum class Flow { Normal, Break, Continue };

struct LoopCtl {
    i64 index = 0, length = 0;
};

struct Renderer {
    std::string out;
    std::vector<std::map<std::string, VP>> scopes;
    std::vector<const LoopCtl *> loops;
    std::string bos, eos;

    VP lookup(const std::string &n) const {
        for (size_t i = scopes.size(); i-- > 0;) {
            auto it = scopes[i].find(n);
            if (it != scopes[i].end()) return it->second;
        }
        // The names every template expects from the caller.
        if (n == "bos_token") return Val::str(bos);
        if (n == "eos_token") return Val::str(eos);
        return Val::undef();
    }

    void set(const std::string &n, VP v) {
        if (!scopes.empty()) scopes.back()[n] = std::move(v);
    }

    static bool val_eq(const VP &a, const VP &b) {
        if (!a || !b) return false;
        if (a->is_num() && b->is_num()) return a->n == b->n;
        if (a->t != b->t) {
            if (a->t == Val::Und || b->t == Val::Und) return false;
            if ((a->t == Val::Nul) != (b->t == Val::Nul)) return false;
            return false;
        }
        switch (a->t) {
            case Val::Und:
            case Val::Nul: return true;
            case Val::Boo: return a->b == b->b;
            case Val::Str: return a->s == b->s;
            case Val::Arr: return a.get() == b.get();
            case Val::Obj: return a.get() == b.get();
            default: return false;
        }
    }

    static int cmp_vals(const VP &a, const VP &b) {
        if (a && b && a->is_num() && b->is_num())
            return a->n < b->n ? -1 : (a->n > b->n ? 1 : 0);
        const std::string x = to_str(a), y = to_str(b);
        return x < y ? -1 : (x > y ? 1 : 0);
    }

    VP eval(const EP &e) {
        if (!e) return Val::undef();
        switch (e->k) {
            case Expr::Lit: return e->lit;
            case Expr::Name: {
                if (e->name == "loop") {
                    VP o = Val::obj();
                    if (!loops.empty()) {
                        const LoopCtl &L = *loops.back();
                        auto put = [&](const char *k, VP v) {
                            o->o->push_back({k, std::move(v)});
                        };
                        put("index", Val::integer(L.index + 1));
                        put("index0", Val::integer(L.index));
                        put("revindex", Val::integer(L.length - L.index));
                        put("revindex0",
                            Val::integer(L.length - L.index - 1));
                        put("first", Val::boolean(L.index == 0));
                        put("last", Val::boolean(L.index + 1 >= L.length));
                        put("length", Val::integer(L.length));
                    }
                    return o;
                }
                return lookup(e->name);
            }
            case Expr::Attr: {
                VP base = eval(e->args[0]);
                if (!base) return Val::undef();
                if (base->t == Val::Obj && base->o)
                    for (const auto &kv : *base->o)
                        if (kv.first == e->name) return kv.second;
                return Val::undef();
            }
            case Expr::Item: return eval_item(eval(e->args[0]), e->args[1]);
            case Expr::Slice: throw TmplError("slice used outside a subscript", true);
            case Expr::Call: return eval_call(e);
            case Expr::Filter: {
                std::vector<VP> args;
                for (size_t i = 1; i < e->args.size(); i++)
                    args.push_back(eval(e->args[i]));
                return apply_filter(e->name, eval(e->args[0]), args);
            }
            case Expr::Test: {
                std::vector<VP> args;
                for (size_t i = 1; i < e->args.size(); i++)
                    args.push_back(eval(e->args[i]));
                return Val::boolean(apply_test(e->name, eval(e->args[0]), args));
            }
            case Expr::Unary: {
                VP a = eval(e->args[0]);
                if (e->name == "not") return Val::boolean(!truthy(a));
                if (e->name == "-") {
                    if (a && a->t == Val::Int) return Val::integer(-a->i);
                    return Val::number(-to_float(a));
                }
                return a;
            }
            case Expr::Bin: return eval_bin(e->name, eval(e->args[0]),
                                            eval(e->args[1]));
            case Expr::Cond: {
                if (truthy(eval(e->args[1]))) return eval(e->args[0]);
                return e->args[2] ? eval(e->args[2]) : Val::undef();
            }
            case Expr::ListE: {
                VP a = Val::arr();
                for (const EP &x : e->args) a->a->push_back(eval(x));
                return a;
            }
            case Expr::DictE: {
                VP o = Val::obj();
                for (size_t i = 0; i + 1 < e->args.size(); i += 2)
                    o->o->push_back({to_str(eval(e->args[i])),
                                     eval(e->args[i + 1])});
                return o;
            }
        }
        return Val::undef();
    }

    VP eval_item(const VP &base, const EP &key) {
        if (!base) return Val::undef();
        if (key && key->k == Expr::Slice) {
            // An OMITTED bound is not -1: `a[1:]` runs to the end and `a[:-1]`
            // stops one short. Treating "absent" and "-1" as the same number
            // made `a[1:]` return an empty list, which is the kind of quiet
            // off-by-one that would have shipped a truncated transcript.
            const bool has_start = !key->args.empty() && key->args[0];
            const bool has_stop = key->args.size() > 1 && key->args[1];
            i64 b = has_start ? to_int(eval(key->args[0])) : 0;
            i64 e = has_stop ? to_int(eval(key->args[1])) : 0;
            if (base->t == Val::Arr && base->a) {
                const i64 n = static_cast<i64>(base->a->size());
                if (b < 0) b += n;
                if (!has_stop) e = n;
                else if (e < 0) e += n;
                b = std::max<i64>(0, std::min<i64>(b, n));
                e = std::max<i64>(b, std::min<i64>(e, n));
                VP r = Val::arr();
                for (i64 i = b; i < e; i++)
                    r->a->push_back((*base->a)[static_cast<size_t>(i)]);
                return r;
            }
            if (base->t == Val::Str) {
                const i64 n = static_cast<i64>(base->s.size());
                if (!has_stop) e = n;
                else if (e < 0) e += n;
                b = std::max<i64>(0, std::min<i64>(b, n));
                e = std::max<i64>(b, std::min<i64>(e, n));
                return Val::str(base->s.substr(static_cast<size_t>(b),
                                               static_cast<size_t>(e - b)));
            }
            return Val::undef();
        }
        VP k = eval(key);
        if (base->t == Val::Arr && base->a) {
            i64 idx = to_int(k);
            const i64 n = static_cast<i64>(base->a->size());
            if (idx < 0) idx += n;
            if (idx < 0 || idx >= n) return Val::undef();
            return (*base->a)[static_cast<size_t>(idx)];
        }
        if (base->t == Val::Obj && base->o) {
            const std::string key_s = to_str(k);
            for (const auto &kv : *base->o)
                if (kv.first == key_s) return kv.second;
            return Val::undef();
        }
        if (base->t == Val::Str) {
            i64 idx = to_int(k);
            const i64 n = static_cast<i64>(base->s.size());
            if (idx < 0) idx += n;
            if (idx < 0 || idx >= n) return Val::undef();
            return Val::str(std::string(1, base->s[static_cast<size_t>(idx)]));
        }
        return Val::undef();
    }

    VP eval_bin(const std::string &op, const VP &a, const VP &b) {
        if (op == "and") return truthy(a) ? b : a;
        if (op == "or") return truthy(a) ? a : b;
        if (op == "==") return Val::boolean(val_eq(a, b));
        if (op == "!=") return Val::boolean(!val_eq(a, b));
        if (op == "<") return Val::boolean(cmp_vals(a, b) < 0);
        if (op == ">") return Val::boolean(cmp_vals(a, b) > 0);
        if (op == "<=") return Val::boolean(cmp_vals(a, b) <= 0);
        if (op == ">=") return Val::boolean(cmp_vals(a, b) >= 0);
        if (op == "~") return Val::str(to_str(a) + to_str(b));
        if (op == "in" || op == "notin") {
            bool found = false;
            if (b && b->t == Val::Arr && b->a)
                for (const VP &x : *b->a)
                    if (val_eq(a, x)) found = true;
            else if (b && b->t == Val::Str)
                found = b->s.find(to_str(a)) != std::string::npos;
            else if (b && b->t == Val::Obj && b->o)
                for (const auto &kv : *b->o)
                    if (kv.first == to_str(a)) found = true;
            return Val::boolean(op == "in" ? found : !found);
        }
        if (op == "+") {
            if (a && b && a->t == Val::Str && b->t == Val::Str)
                return Val::str(a->s + b->s);
            if (a && b && a->t == Val::Arr && b->t == Val::Arr) {
                VP r = Val::arr();
                for (const VP &x : *a->a) r->a->push_back(x);
                for (const VP &x : *b->a) r->a->push_back(x);
                return r;
            }
            if (a && a->t == Val::Int && b && b->t == Val::Int)
                return Val::integer(a->i + b->i);
            return Val::number(to_float(a) + to_float(b));
        }
        if (op == "-") {
            if (a && a->t == Val::Int && b && b->t == Val::Int)
                return Val::integer(a->i - b->i);
            return Val::number(to_float(a) - to_float(b));
        }
        if (op == "*") {
            if (a && a->t == Val::Int && b && b->t == Val::Int)
                return Val::integer(a->i * b->i);
            return Val::number(to_float(a) * to_float(b));
        }
        if (op == "/") {
            const f64 d = to_float(b);
            if (d == 0.0)
                throw TmplError("division by zero in the template");
            return Val::number(to_float(a) / d);
        }
        if (op == "//") {
            const i64 d = to_int(b);
            if (d == 0) throw TmplError("division by zero in the template");
            i64 x = to_int(a);
            i64 q = x / d;
            if ((x % d != 0) && ((x < 0) != (d < 0))) q--;
            return Val::integer(q);
        }
        if (op == "%") {
            const i64 d = to_int(b);
            if (d == 0) throw TmplError("modulo by zero in the template");
            i64 r = to_int(a) % d;
            if (r != 0 && ((r < 0) != (d < 0))) r += d;
            return Val::integer(r);
        }
        throw TmplError("unsupported operator '" + op + "'", true);
    }

    VP eval_call(const EP &e) {
        const EP &callee = e->args[0];
        std::vector<VP> args;
        for (size_t i = 1; i < e->args.size(); i++)
            args.push_back(eval(e->args[i]));
        std::vector<std::pair<std::string, VP>> kwargs;
        for (const auto &kv : e->kwargs)
            kwargs.push_back({kv.first, eval(kv.second)});
        if (callee->k == Expr::Name)
            return call_global(callee->name, args, kwargs);
        if (callee->k == Expr::Attr) {
            VP base = eval(callee->args[0]);
            return call_method(base, callee->name, args);
        }
        throw TmplError("unsupported call target in the template", true);
    }

    VP call_global(const std::string &name, const std::vector<VP> &args,
                   const std::vector<std::pair<std::string, VP>> &kwargs) {
        if (name == "raise_exception") {
            throw TmplError("the template rejected this conversation: " +
                            (args.empty() ? std::string("(no message)")
                                          : to_str(args[0])));
        }
        if (name == "range") {
            i64 a = 0, b = 0, step = 1;
            if (args.size() == 1) {
                b = to_int(args[0]);
            } else if (args.size() >= 2) {
                a = to_int(args[0]);
                b = to_int(args[1]);
                if (args.size() >= 3) step = to_int(args[2]);
            }
            VP r = Val::arr();
            if (step == 0) throw TmplError("range() with a zero step");
            for (i64 i = a; step > 0 ? i < b : i > b; i += step)
                r->a->push_back(Val::integer(i));
            return r;
        }
        if (name == "namespace") {
            // Jinja's mutable bag: `namespace(v=0)` starts with v=0 and
            // `{% set ns.v = ... %}` writes into it. Objects are shared, so the
            // write is visible after the loop -- which is the whole point.
            VP o = Val::obj();
            for (const auto &kv : kwargs) o->o->push_back({kv.first, kv.second});
            return o;
        }
        if (name == "strftime_now") {
            std::time_t now = std::time(nullptr);
            std::tm tmv{};
#if defined(_WIN32)
            localtime_s(&tmv, &now);
#else
            localtime_r(&now, &tmv);
#endif
            char buf[128];
            const std::string fmt = args.empty() ? "%Y-%m-%d" : to_str(args[0]);
            const size_t k =
                std::strftime(buf, sizeof buf, fmt.c_str(), &tmv);
            return Val::str(std::string(buf, k));
        }
        if (name == "dict") {
            VP o = Val::obj();
            for (const auto &kv : kwargs) o->o->push_back({kv.first, kv.second});
            return o;
        }
        throw TmplError("unsupported function '" + name + "()'", true);
    }

    VP call_method(const VP &base, const std::string &m,
                   const std::vector<VP> &args) {
        if (!base) throw TmplError("method '" + m + "' on an undefined value");
        if (base->t == Val::Str) {
            const std::string &s = base->s;
            if (m == "strip") return Val::str(trim_str(s));
            if (m == "lstrip") {
                size_t b = 0;
                while (b < s.size() && is_space(s[b])) b++;
                return Val::str(s.substr(b));
            }
            if (m == "rstrip") {
                size_t e = s.size();
                while (e > 0 && is_space(s[e - 1])) e--;
                return Val::str(s.substr(0, e));
            }
            if (m == "lower") return Val::str(lower_ascii(s));
            if (m == "upper") return Val::str(upper_ascii(s));
            if (m == "strip_newlines")
                return Val::str(trim_str(s));
            if (m == "startswith")
                return Val::boolean(!args.empty() &&
                                    s.rfind(to_str(args[0]), 0) == 0);
            if (m == "endswith") {
                const std::string suf = args.empty() ? "" : to_str(args[0]);
                return Val::boolean(s.size() >= suf.size() &&
                                    s.compare(s.size() - suf.size(), suf.size(),
                                              suf) == 0);
            }
            if (m == "split") {
                std::vector<std::string> parts;
                if (args.empty() || to_str(args[0]).empty()) {
                    std::string cur;
                    for (char c : s) {
                        if (is_space(c)) {
                            if (!cur.empty()) {
                                parts.push_back(cur);
                                cur.clear();
                            }
                        } else {
                            cur += c;
                        }
                    }
                    if (!cur.empty()) parts.push_back(cur);
                } else {
                    const std::string sep = to_str(args[0]);
                    size_t b = 0;
                    while (true) {
                        const size_t e = s.find(sep, b);
                        if (e == std::string::npos) {
                            parts.push_back(s.substr(b));
                            break;
                        }
                        parts.push_back(s.substr(b, e - b));
                        b = e + sep.size();
                    }
                }
                VP r = Val::arr();
                for (auto &x : parts) r->a->push_back(Val::str(x));
                return r;
            }
            if (m == "splitlines") {
                VP r = Val::arr();
                std::string cur;
                for (char c : s) {
                    if (c == '\n') {
                        r->a->push_back(Val::str(cur));
                        cur.clear();
                    } else if (c != '\r') {
                        cur += c;
                    }
                }
                if (!cur.empty()) r->a->push_back(Val::str(cur));
                return r;
            }
            if (m == "replace") {
                if (args.size() < 2) throw TmplError("replace() needs 2 arguments");
                const std::string from = to_str(args[0]);
                const std::string to = to_str(args[1]);
                if (from.empty()) return Val::str(s);
                std::string r;
                size_t b = 0;
                while (true) {
                    const size_t e = s.find(from, b);
                    if (e == std::string::npos) {
                        r += s.substr(b);
                        break;
                    }
                    r += s.substr(b, e - b);
                    r += to;
                    b = e + from.size();
                }
                return Val::str(r);
            }
            if (m == "title") {
                std::string r = s;
                bool start = true;
                for (char &c : r) {
                    if (is_space(c)) {
                        start = true;
                    } else {
                        c = start ? static_cast<char>(std::toupper(
                                        static_cast<unsigned char>(c)))
                                  : static_cast<char>(std::tolower(
                                        static_cast<unsigned char>(c)));
                        start = false;
                    }
                }
                return Val::str(r);
            }
            if (m == "capitalize") {
                std::string r = lower_ascii(s);
                if (!r.empty())
                    r[0] = static_cast<char>(
                        std::toupper(static_cast<unsigned char>(r[0])));
                return Val::str(r);
            }
            if (m == "count") {
                const std::string what = args.empty() ? "" : to_str(args[0]);
                if (what.empty()) return Val::integer(0);
                i64 c = 0;
                size_t b = 0;
                while ((b = s.find(what, b)) != std::string::npos) {
                    c++;
                    b += what.size();
                }
                return Val::integer(c);
            }
            if (m == "format") {
                // `"{}".format(x)` -- positional replacement only. Real
                // templates use it for tool payloads; the subset covers the
                // empty-brace spelling and refuses the rest by name.
                std::string r;
                size_t next = 0;
                for (size_t i = 0; i < s.size(); i++) {
                    if (s[i] == '{' && i + 1 < s.size() && s[i + 1] == '}') {
                        r += next < args.size() ? to_str(args[next]) : "";
                        next++;
                        i++;
                    } else if (s[i] == '{') {
                        throw TmplError("str.format with named fields is "
                                        "outside the supported subset",
                                        true);
                    } else {
                        r += s[i];
                    }
                }
                return Val::str(r);
            }
            throw TmplError("unsupported string method '." + m + "()'", true);
        }
        if (base->t == Val::Arr) {
            if (m == "append") {
                if (!args.empty()) base->a->push_back(args[0]);
                return Val::none();
            }
            if (m == "extend") {
                if (!args.empty() && args[0] && args[0]->t == Val::Arr)
                    for (const VP &x : *args[0]->a) base->a->push_back(x);
                return Val::none();
            }
            if (m == "pop") {
                if (base->a->empty()) throw TmplError("pop() on an empty list");
                VP v = base->a->back();
                base->a->pop_back();
                return v;
            }
            if (m == "count") {
                i64 c = 0;
                for (const VP &x : *base->a)
                    if (val_eq(x, args.empty() ? Val::undef() : args[0])) c++;
                return Val::integer(c);
            }
            if (m == "index") {
                for (size_t i = 0; i < base->a->size(); i++)
                    if (val_eq((*base->a)[i],
                               args.empty() ? Val::undef() : args[0]))
                        return Val::integer(static_cast<i64>(i));
                return Val::integer(-1);
            }
            if (m == "join") {
                const std::string sep = args.empty() ? "" : to_str(args[0]);
                std::string r;
                for (size_t i = 0; i < base->a->size(); i++) {
                    if (i) r += sep;
                    r += to_str((*base->a)[i]);
                }
                return Val::str(r);
            }
            throw TmplError("unsupported list method '." + m + "()'", true);
        }
        if (base->t == Val::Obj) {
            if (m == "items") {
                VP r = Val::arr();
                for (const auto &kv : *base->o) {
                    VP pair = Val::arr();
                    pair->a->push_back(Val::str(kv.first));
                    pair->a->push_back(kv.second);
                    r->a->push_back(pair);
                }
                return r;
            }
            if (m == "keys") {
                VP r = Val::arr();
                for (const auto &kv : *base->o) r->a->push_back(Val::str(kv.first));
                return r;
            }
            if (m == "values") {
                VP r = Val::arr();
                for (const auto &kv : *base->o) r->a->push_back(kv.second);
                return r;
            }
            if (m == "get") {
                const std::string key = args.empty() ? "" : to_str(args[0]);
                for (const auto &kv : *base->o)
                    if (kv.first == key) return kv.second;
                return args.size() > 1 ? args[1] : Val::none();
            }
            if (m == "update") {
                if (!args.empty() && args[0] && args[0]->t == Val::Obj)
                    for (const auto &kv : *args[0]->o) {
                        bool found = false;
                        for (auto &mine : *base->o)
                            if (mine.first == kv.first) {
                                mine.second = kv.second;
                                found = true;
                            }
                        if (!found) base->o->push_back(kv);
                    }
                return Val::none();
            }
            throw TmplError("unsupported dict method '." + m + "()'", true);
        }
        throw TmplError("method '." + m + "()' on a value that has no methods",
                        true);
    }

    VP apply_filter(const std::string &name, const VP &v,
                    const std::vector<VP> &args) {
        if (name == "trim") return Val::str(trim_str(to_str(v)));
        if (name == "lower") return Val::str(lower_ascii(to_str(v)));
        if (name == "upper") return Val::str(upper_ascii(to_str(v)));
        if (name == "capitalize") {
            std::string r = lower_ascii(to_str(v));
            if (!r.empty())
                r[0] = static_cast<char>(
                    std::toupper(static_cast<unsigned char>(r[0])));
            return Val::str(r);
        }
        if (name == "length" || name == "count") {
            if (!v) return Val::integer(0);
            if (v->t == Val::Str) return Val::integer(
                static_cast<i64>(v->s.size()));
            if (v->t == Val::Arr) return Val::integer(
                static_cast<i64>(v->a ? v->a->size() : 0));
            if (v->t == Val::Obj) return Val::integer(
                static_cast<i64>(v->o ? v->o->size() : 0));
            return Val::integer(0);
        }
        if (name == "first") {
            if (v && v->t == Val::Arr && v->a && !v->a->empty())
                return v->a->front();
            if (v && v->t == Val::Str && !v->s.empty())
                return Val::str(std::string(1, v->s[0]));
            return Val::undef();
        }
        if (name == "last") {
            if (v && v->t == Val::Arr && v->a && !v->a->empty())
                return v->a->back();
            if (v && v->t == Val::Str && !v->s.empty())
                return Val::str(std::string(1, v->s.back()));
            return Val::undef();
        }
        if (name == "join") {
            const std::string sep = args.empty() ? "" : to_str(args[0]);
            if (v && v->t == Val::Arr && v->a) {
                std::string r;
                for (size_t i = 0; i < v->a->size(); i++) {
                    if (i) r += sep;
                    r += to_str((*v->a)[i]);
                }
                return Val::str(r);
            }
            return Val::str(to_str(v));
        }
        if (name == "default" || name == "d") {
            const bool use_default =
                v && v->t != Val::Und &&
                (args.size() > 1 ? truthy(args[1]) : false);
            if (!v || v->t == Val::Und || use_default)
                return args.empty() ? Val::str("") : args[0];
            return v;
        }
        if (name == "tojson" || name == "to_json") return Val::str(to_json(v));
        if (name == "string") return Val::str(to_str(v));
        if (name == "int") return Val::integer(to_int(v));
        if (name == "float") return Val::number(to_float(v));
        if (name == "list") {
            if (v && v->t == Val::Arr) return v;
            VP r = Val::arr();
            if (v && v->t == Val::Obj && v->o)
                for (const auto &kv : *v->o) r->a->push_back(Val::str(kv.first));
            else if (v && v->t == Val::Str)
                for (char c : v->s)
                    r->a->push_back(Val::str(std::string(1, c)));
            return r;
        }
        if (name == "items") {
            if (v && v->t == Val::Obj && v->o) {
                VP r = Val::arr();
                for (const auto &kv : *v->o) {
                    VP pair = Val::arr();
                    pair->a->push_back(Val::str(kv.first));
                    pair->a->push_back(kv.second);
                    r->a->push_back(pair);
                }
                return r;
            }
            return Val::arr();
        }
        if (name == "replace") {
            if (args.size() < 2) throw TmplError("replace() needs 2 arguments");
            const std::string from = to_str(args[0]);
            const std::string to = to_str(args[1]);
            std::string s = to_str(v), r;
            if (from.empty()) return Val::str(s);
            size_t b = 0;
            while (true) {
                const size_t e = s.find(from, b);
                if (e == std::string::npos) {
                    r += s.substr(b);
                    break;
                }
                r += s.substr(b, e - b);
                r += to;
                b = e + from.size();
            }
            return Val::str(r);
        }
        if (name == "safe" || name == "escape" || name == "e") return v;
        if (name == "reverse") {
            if (v && v->t == Val::Arr && v->a) {
                VP r = Val::arr();
                for (size_t i = v->a->size(); i-- > 0;)
                    r->a->push_back((*v->a)[i]);
                return r;
            }
            std::string s = to_str(v);
            std::reverse(s.begin(), s.end());
            return Val::str(s);
        }
        if (name == "sum") {
            f64 acc = 0;
            if (v && v->t == Val::Arr && v->a)
                for (const VP &x : *v->a) acc += to_float(x);
            return Val::number(acc);
        }
        if (name == "unique" || name == "sort" || name == "map" ||
            name == "select" || name == "reject" || name == "selectattr" ||
            name == "rejectattr" || name == "attr" || name == "batch" ||
            name == "slice" || name == "groupby" || name == "indent" ||
            name == "wordwrap" || name == "truncate" || name == "urlencode" ||
            name == "striptags" || name == "center" || name == "filesizeformat" ||
            name == "round" || name == "abs" || name == "min" || name == "max")
            throw TmplError("unsupported filter '" + name + "'", true);
        throw TmplError("unknown filter '" + name + "'", true);
    }

    bool apply_test(const std::string &name, const VP &v,
                    const std::vector<VP> &args) {
        if (name == "defined") return v && v->t != Val::Und;
        if (name == "undefined") return !v || v->t == Val::Und;
        if (name == "none" || name == "none_") return v && v->t == Val::Nul;
        if (name == "string") return v && v->t == Val::Str;
        if (name == "mapping") return v && v->t == Val::Obj;
        if (name == "sequence" || name == "iterable")
            return v && (v->t == Val::Arr || v->t == Val::Str);
        if (name == "number") return v && v->is_num();
        if (name == "integer") return v && v->t == Val::Int;
        if (name == "float") return v && v->t == Val::Num;
        if (name == "boolean") return v && v->t == Val::Boo;
        if (name == "true") return truthy(v);
        if (name == "false") return !truthy(v);
        if (name == "even") return to_int(v) % 2 == 0;
        if (name == "odd") return to_int(v) % 2 != 0;
        if (name == "divisibleby")
            return !args.empty() && to_int(v) % to_int(args[0]) == 0;
        if (name == "equalto" || name == "eq")
            return !args.empty() && val_eq(v, args[0]);
        if (name == "sameas") return !args.empty() && v.get() == args[0].get();
        throw TmplError("unsupported test '" + name + "'", true);
    }

    // ---- statements ------------------------------------------------------
    Flow run(const std::vector<SP> &body) {
        for (const SP &s : body) {
            switch (s->k) {
                case Stmt::Text: out += s->text; break;
                case Stmt::Out: out += to_str(eval(s->expr)); break;
                case Stmt::Set: set(s->var, eval(s->expr)); break;
                case Stmt::SetAttr: {
                    VP base = lookup(s->var);
                    if (base && base->t == Val::Obj && base->o) {
                        VP nv = eval(s->expr);
                        bool found = false;
                        for (auto &kv : *base->o)
                            if (kv.first == s->var2) {
                                kv.second = nv;
                                found = true;
                            }
                        if (!found) base->o->push_back({s->var2, nv});
                    }
                    break;
                }
                case Stmt::Break: return Flow::Break;
                case Stmt::Continue: return Flow::Continue;
                case Stmt::If: {
                    bool taken = false;
                    if (truthy(eval(s->expr))) {
                        const Flow f = run(s->body);
                        if (f != Flow::Normal) return f;
                        taken = true;
                    } else if (s->have_alt) {
                        // elif chains are nested Ifs inside alt[0]; a plain
                        // else is the whole alt vector.
                        if (!s->alt.empty() && s->alt[0]->k == Stmt::If) {
                            const Flow f = run_chain(s->alt[0]);
                            if (f != Flow::Normal) return f;
                        } else {
                            const Flow f = run(s->alt);
                            if (f != Flow::Normal) return f;
                        }
                        taken = true;
                    }
                    (void)taken;
                    break;
                }
                case Stmt::For: {
                    VP it = eval(s->expr);
                    std::vector<VP> items;
                    bool is_pairs = false;
                    if (it && it->t == Val::Arr) {
                        items = *it->a;
                    } else if (it && it->t == Val::Obj) {
                        for (const auto &kv : *it->o) items.push_back(kv.second);
                    } else if (it && it->t == Val::Str) {
                        for (char c : it->s)
                            items.push_back(
                                Val::str(std::string(1, c)));
                    }
                    // `for k, v in d.items()`: items() already yields pairs.
                    is_pairs = !s->var2.empty();
                    (void)is_pairs;
                    if (items.empty()) {
                        if (s->have_alt) {
                            const Flow f = run(s->alt);
                            if (f != Flow::Normal) return f;
                        }
                        break;
                    }
                    LoopCtl lc;
                    lc.length = static_cast<i64>(items.size());
                    for (i64 idx = 0; idx < lc.length; idx++) {
                        lc.index = idx;
                        scopes.emplace_back();
                        const VP &item = items[static_cast<size_t>(idx)];
                        if (!s->var2.empty()) {
                            // pair destructuring: a 2-list, or a dict entry
                            if (item && item->t == Val::Arr && item->a &&
                                item->a->size() >= 2) {
                                set(s->var, (*item->a)[0]);
                                set(s->var2, (*item->a)[1]);
                            } else {
                                set(s->var, item);
                                set(s->var2, Val::undef());
                            }
                        } else {
                            set(s->var, item);
                        }
                        loops.push_back(&lc);
                        const Flow f = run(s->body);
                        loops.pop_back();
                        scopes.pop_back();
                        if (f == Flow::Break) break;
                        if (f == Flow::Continue) continue;
                    }
                    break;
                }
            }
        }
        return Flow::Normal;
    }

    // An elif chain: the node's own condition, then its alt.
    Flow run_chain(const SP &s) {
        if (truthy(eval(s->expr))) return run(s->body);
        if (!s->have_alt) return Flow::Normal;
        if (!s->alt.empty() && s->alt[0]->k == Stmt::If) return run_chain(s->alt[0]);
        return run(s->alt);
    }
};

std::string render_raw(const std::string &tmpl,
                       const std::vector<ChatMessage> &msgs,
                       bool add_generation_prompt, const std::string &bos,
                       const std::string &eos) {
    TemplateParser tp;
    tp.ch = split_chunks(tmpl);
    tp.parse_block({}, nullptr);   // validate structure from the top
    tp.p = 0;
    std::vector<SP> body = tp.parse_block({}, nullptr);

    Renderer r;
    r.bos = bos;
    r.eos = eos;
    r.scopes.emplace_back();
    VP list = Val::arr();
    for (const ChatMessage &m : msgs) {
        VP o = Val::obj();
        o->o->push_back({"role", Val::str(m.role)});
        o->o->push_back({"content", Val::str(m.content)});
        list->a->push_back(o);
    }
    r.scopes.back()["messages"] = list;
    r.scopes.back()["add_generation_prompt"] = Val::boolean(add_generation_prompt);
    // Templates that iterate `tools`/`documents` see an empty list rather than
    // undefined: `{% for t in tools %}` on undefined is a Jinja error, and an
    // empty loop is what "this request has no tools" means.
    r.scopes.back()["tools"] = Val::arr();
    r.scopes.back()["documents"] = Val::arr();
    r.scopes.back()["enable_thinking"] = Val::boolean(false);
    r.run(body);
    return r.out;
}

} // namespace

// ---------------------------------------------------------------------------
// Builtin templates
//
// These are the layouts the families in this collection actually train on. A
// file that carries its own template never reaches them -- they exist for files
// that were converted without one, and for `--chat-template NAME`.
// ---------------------------------------------------------------------------
const char *kBuiltinDefault = "chatml";

namespace {

const char *kChatml =
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + "
    "'<|im_end|>' + '\\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}";

const char *kLlama3 =
    "{% set loop_messages = messages %}"
    "{% for message in loop_messages %}"
    "{% set content = '<|start_header_id|>' + message['role'] + "
    "'<|end_header_id|>\\n\\n' + message['content'] | trim + '<|eot_id|>' %}"
    "{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}"
    "{{ content }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}{% endif %}";

const char *kLlama2 =
    "{{ bos_token }}"
    "{% for message in messages %}"
    "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
    "{{ raise_exception('Conversation roles must alternate "
    "user/assistant/user/assistant/...') }}"
    "{% endif %}"
    "{% if message['role'] == 'user' %}"
    "{{ '[INST] ' + message['content'] + ' [/INST]' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ message['content'] + eos_token }}"
    "{% endif %}"
    "{% endfor %}";

const char *kGemma =
    "{{ bos_token }}"
    "{% if messages[0]['role'] == 'system' %}"
    "{{ raise_exception('System role not supported') }}{% endif %}"
    "{% for message in messages %}"
    "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
    "{{ raise_exception('Conversation roles must alternate "
    "user/assistant/user/assistant/...') }}"
    "{% endif %}"
    "{% if message['role'] == 'user' %}"
    "{{ '<start_of_turn>user\\n' + message['content'] | trim + "
    "'<end_of_turn>\\n' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ '<start_of_turn>model\\n' + message['content'] | trim + "
    "'<end_of_turn>\\n' }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<start_of_turn>model\\n' }}{% endif %}";

const char *kPhi3 =
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "{{ '<|system|>\\n' + message['content'] + '<|end|>\\n' }}"
    "{% elif message['role'] == 'user' %}"
    "{{ '<|user|>\\n' + message['content'] + '<|end|>\\n' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ '<|assistant|>\\n' + message['content'] + '<|end|>\\n' }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|assistant|>\\n' }}"
    "{% else %}{{ eos_token }}{% endif %}";

const char *kDeepseek =
    "{{ bos_token }}"
    "{% if messages[0]['role'] == 'system' %}"
    "{{ messages[0]['content'] + '\\n\\n' }}{% endif %}"
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}"
    "{{ 'User: ' + message['content'] + '\\n\\n' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ 'Assistant: ' + message['content'] + eos_token }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}";

const char *kGlm4 =
    "[gMASK]<sop>"
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "{{ '<|system|>\\n' + message['content'] }}"
    "{% elif message['role'] == 'user' %}"
    "{{ '<|user|>\\n' + message['content'] }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ '<|assistant|>\\n' + message['content'] }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|assistant|>' }}{% endif %}";

struct BuiltinEntry {
    const char *name;
    const char *src;
};

const BuiltinEntry kBuiltins[] = {
    {"chatml", kChatml},
    {"llama3", kLlama3},
    {"llama2", kLlama2},
    {"mistral", kLlama2},
    {"gemma", kGemma},
    {"phi3", kPhi3},
    {"deepseek", kDeepseek},
    {"glm4", kGlm4},
};

bool has_prefix(const std::string &s, const char *p) {
    return s.rfind(p, 0) == 0;
}

} // namespace

std::string builtin_chat_template_by_name(const std::string &name) {
    for (const BuiltinEntry &e : kBuiltins)
        if (name == e.name) return e.src;
    return {};
}

std::vector<std::string> chat_turn_markers(const std::string &rendered) {
    // Only the markers this model's own prompt contains are returned, so a
    // Llama-3 transcript never hides ChatML's marker by accident.
    static const char *const kCandidates[] = {
        "<|im_end|>",    "<|eot_id|>",   "<end_of_turn>", "<|end|>",
        "<|endoftext|>", "<|eot|>",      "<|end_of_turn|>", "</s>",
        "<\xEF\xBD\x9C" "end\xE2\x96\x81of\xE2\x96\x81sentence\xEF\xBD\x9C>",
    };
    std::vector<std::string> out;
    for (const char *c : kCandidates) {
        const std::string m(c);
        if (rendered.find(m) != std::string::npos) out.push_back(m);
    }
    return out;
}

const std::vector<std::string> &builtin_chat_template_names() {
    static const std::vector<std::string> names = [] {
        std::vector<std::string> v;
        std::string last;
        for (const BuiltinEntry &e : kBuiltins)
            if (last != e.name) {
                v.push_back(e.name);
                last = e.name;
            }
        return v;
    }();
    return names;
}

const char *builtin_chat_template_name(const std::string &arch,
                                       const std::string &pre) {
    // The tokenizer's pre-tokenizer name is the more reliable signal when it is
    // present: `llama-bpe`/`llama3` is the Llama-3 family whatever the arch
    // string says, and a dozen finetunes reuse `llama`.
    const std::string p = lower_ascii(pre);
    const std::string a = lower_ascii(arch);
    if (has_prefix(p, "llama3") || has_prefix(p, "llama-bpe")) return "llama3";
    if (has_prefix(p, "gemma")) return "gemma";
    if (has_prefix(p, "deepseek")) return "deepseek";
    if (has_prefix(a, "llama3")) return "llama3";
    if (has_prefix(a, "gemma")) return "gemma";
    if (has_prefix(a, "phi3") || a == "phimoe" || a == "phi3moe") return "phi3";
    if (has_prefix(a, "glm4")) return "glm4";
    if (has_prefix(a, "deepseek")) return "deepseek";
    if (a == "mistral" || a == "mixtral") return "mistral";
    // Qwen, ChatGLM, and everything this engine has no better guess for take
    // ChatML: it is the most common template in the wild and the family most of
    // this collection's files belong to.
    return kBuiltinDefault;
}

std::string builtin_chat_template(const std::string &arch,
                                 const std::string &pre) {
    return builtin_chat_template_by_name(builtin_chat_template_name(arch, pre));
}

bool render_chat_prompt(const std::string &tmpl,
                        const std::vector<ChatMessage> &msgs,
                        bool add_generation_prompt,
                        const std::string &bos_token,
                        const std::string &eos_token, std::string *out,
                        std::string *err) {
    auto fail = [&](const std::string &m) {
        if (err) *err = m;
        return false;
    };
    if (tmpl.empty()) return fail("the template is empty");
    try {
        std::string text = render_raw(tmpl, msgs, add_generation_prompt,
                                      bos_token, eos_token);
        if (out) *out = std::move(text);
        return true;
    } catch (const TmplError &e) {
        // The one retry the reference makes too: a template that refuses a
        // system message (Gemma's, Mistral's) gets it folded into the first
        // user turn. Refusing the conversation would be worse than delivering
        // it in the shape the model expects -- and every caller of this
        // function is a chat UI whose user did nothing wrong.
        const std::string what = e.what();
        const bool system_rejected =
            what.find("System role not supported") != std::string::npos ||
            what.find("system") != std::string::npos;
        const bool has_system = !msgs.empty() && msgs[0].role == "system";
        if (system_rejected && has_system) {
            std::vector<ChatMessage> merged;
            merged.reserve(msgs.size() - 1);
            std::string sys = msgs[0].content;
            bool placed = false;
            for (size_t i = 1; i < msgs.size(); i++) {
                ChatMessage m = msgs[i];
                if (!placed && m.role == "user") {
                    if (!sys.empty()) {
                        m.content = sys + "\n\n" + m.content;
                        sys.clear();
                    }
                    placed = true;
                }
                merged.push_back(std::move(m));
            }
            if (!placed && !sys.empty())
                merged.insert(merged.begin(), ChatMessage{"system", sys});
            try {
                std::string text = render_raw(tmpl, merged, add_generation_prompt,
                                              bos_token, eos_token);
                if (out) *out = std::move(text);
                return true;
            } catch (const TmplError &e2) {
                return fail(e2.what());
            }
        }
        return fail(what);
    } catch (const std::exception &e) {
        return fail(std::string("template render failed: ") + e.what());
    }
}

bool chat_template_supported(const std::string &tmpl, std::string *why) {
    if (tmpl.empty()) {
        if (why) *why = "the file carries no template";
        return false;
    }
    // Parse first: structural problems are the ones worth naming precisely.
    try {
        TemplateParser tp;
        tp.ch = split_chunks(tmpl);
        (void)tp.parse_block({}, nullptr);
    } catch (const TmplError &e) {
        if (why) *why = e.what();
        return false;
    } catch (const std::exception &e) {
        if (why) *why = e.what();
        return false;
    }
    // Then run it. A template can be structurally fine and still reach an
    // unknown filter or function on a path the parse never exercises, and that
    // only shows up when it runs.
    std::vector<ChatMessage> probe = {{"user", "Hello"},
                                      {"assistant", "Hi"},
                                      {"user", "How are you?"}};
    std::string sink, err;
    if (!render_chat_prompt(tmpl, probe, true, "<s>", "</s>", &sink, &err)) {
        if (why) *why = err;
        return false;
    }
    return true;
}

} // namespace krk
