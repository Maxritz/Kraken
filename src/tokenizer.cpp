// tokenizer.cpp — SentencePiece unigram (LLaMA) and byte-level BPE (GPT-2/Qwen).
//
// Two independent encoders share one vocab table:
//   * Spm  — the GGUF "llama" model. SentencePiece convention: the text is
//            prefixed with U+2581 (▁), spaces are folded to ▁, then a Viterbi
//            pass maximizes the total unigram score over the symbol lattice.
//   * Bpe  — the GGUF "gpt2" model (Qwen, SmolLM, Granite, ...). Bytes are
//            mapped through the GPT-2 byte<->unicode permutation, pre-split by
//            a GPT-2/Qwen2-style scanner, then merged by rank.
#include "krk/tokenizer.hpp"

#include <algorithm>
#include <cctype>
#include <climits>
#include <cmath>
#include <cstring>

namespace krk {

namespace {

// ---------------------------------------------------------------------------
// UTF-8
// ---------------------------------------------------------------------------

// Decodes one codepoint at s[i]; returns the byte length consumed (>=1).
int utf8_decode(const std::string &s, size_t i, u32 *cp) {
    const u8 c = static_cast<u8>(s[i]);
    if (c < 0x80) { *cp = c; return 1; }
    int n = 0;
    u32 v = 0;
    if ((c & 0xE0) == 0xC0) { n = 1; v = c & 0x1F; }
    else if ((c & 0xF0) == 0xE0) { n = 2; v = c & 0x0F; }
    else if ((c & 0xF8) == 0xF0) { n = 3; v = c & 0x07; }
    else { *cp = 0xFFFD; return 1; }
    if (i + static_cast<size_t>(n) >= s.size()) { *cp = 0xFFFD; return 1; }
    for (int k = 1; k <= n; k++) {
        const u8 cc = static_cast<u8>(s[i + static_cast<size_t>(k)]);
        if ((cc & 0xC0) != 0x80) { *cp = 0xFFFD; return 1; }
        v = (v << 6) | (cc & 0x3F);
    }
    *cp = v;
    return n + 1;
}

void utf8_append(std::string &s, u32 cp) {
    if (cp < 0x80) {
        s.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
        s.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        s.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        s.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        s.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

// ---------------------------------------------------------------------------
// GPT-2 byte <-> unicode permutation
// ---------------------------------------------------------------------------

struct ByteCodec {
    u32 to_cp[256];
    i32 from_cp[512]; // sparse: codepoint -> byte, or -1

    ByteCodec() {
        for (int i = 0; i < 512; i++) from_cp[i] = -1;
        int n = 0;
        // printable ASCII plus the Latin-1 tails GPT-2 keeps verbatim
        auto keep = [](int b) {
            return (b >= 33 && b <= 126) || (b >= 161 && b <= 172) ||
                   (b >= 174 && b <= 255);
        };
        bool used[256] = {false};
        for (int b = 0; b < 256; b++) {
            if (keep(b)) {
                to_cp[b] = static_cast<u32>(b);
                used[b] = true;
            }
        }
        for (int b = 0; b < 256; b++) {
            if (!used[b]) to_cp[b] = static_cast<u32>(256 + n++);
        }
        for (int b = 0; b < 256; b++) {
            const u32 cp = to_cp[b];
            if (cp < 512) from_cp[cp] = static_cast<i32>(b);
        }
    }

    // Expands one raw byte into its mapped UTF-8 sequence.
    void encode_byte(std::string &out, u8 b) const { utf8_append(out, to_cp[b]); }

    // Maps one mapped codepoint back to a raw byte; -1 when not mapped.
    i32 decode_cp(u32 cp) const {
        return cp < 512 ? from_cp[cp] : -1;
    }
};

const ByteCodec &byte_codec() {
    static const ByteCodec c;
    return c;
}

// ---------------------------------------------------------------------------
// pre-tokenization
// ---------------------------------------------------------------------------

enum class CpKind { Space, Newline, Digit, Letter, Other };

bool is_cjk(u32 cp) {
    return (cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0x3400 && cp <= 0x4DBF) ||
           (cp >= 0xF900 && cp <= 0xFAFF) || (cp >= 0x3040 && cp <= 0x30FF) ||
           (cp >= 0xAC00 && cp <= 0xD7AF);
}

CpKind classify(u32 cp) {
    if (cp == ' ' || cp == '\t' || cp == 0x0B || cp == 0x0C) return CpKind::Space;
    if (cp == '\r' || cp == '\n') return CpKind::Newline;
    if (cp >= '0' && cp <= '9') return CpKind::Digit;
    if (cp >= 0xFF10 && cp <= 0xFF19) return CpKind::Digit;
    if ((cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z')) return CpKind::Letter;
    if (cp >= 0x80 && !is_cjk(cp)) return CpKind::Letter; // Latin/Greek/Cyrillic runs
    return CpKind::Other;
}

struct Sym {
    size_t begin = 0, len = 0;
    u32 cp = 0;
    CpKind kind = CpKind::Other;
};

std::vector<Sym> split_symbols(const std::string &s) {
    std::vector<Sym> out;
    size_t i = 0;
    while (i < s.size()) {
        Sym sy;
        sy.begin = i;
        const int n = utf8_decode(s, i, &sy.cp);
        sy.len = static_cast<size_t>(n);
        sy.kind = classify(sy.cp);
        out.push_back(sy);
        i += static_cast<size_t>(n);
    }
    return out;
}

// the GPT-2 contraction set: 's 't 're 've 'm 'll 'd
int contraction_len(const std::vector<Sym> &sy, size_t i) {
    if (i + 1 >= sy.size() || sy[i].cp != '\'') return 0;
    const u32 a = sy[i + 1].cp | 0x20;
    if (a == 's' || a == 't' || a == 'm' || a == 'd') return 2;
    if (i + 2 < sy.size()) {
        const u32 b = sy[i + 2].cp | 0x20;
        if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') ||
            (a == 'l' && b == 'l'))
            return 3;
    }
    return 0;
}

enum class PreMode { Gpt2, Qwen2, Llama3 };

// Appends the [begin,end) byte range of `s` as one pre-token chunk.
void emit_chunk(const std::string &s, size_t begin, size_t end,
                std::vector<std::string> &out) {
    if (end > begin) out.emplace_back(s, begin, end - begin);
}

std::vector<std::string> pre_tokenize(const std::string &text, PreMode mode) {
    const std::vector<Sym> sy = split_symbols(text);
    std::vector<std::string> chunks;
    size_t i = 0;
    const size_t n = sy.size();

    while (i < n) {
        size_t begin = sy[i].begin;

        // 1. contractions
        if (int cl = contraction_len(sy, i)) {
            const Sym &last = sy[i + static_cast<size_t>(cl) - 1];
            emit_chunk(text, begin, last.begin + last.len, chunks);
            i += static_cast<size_t>(cl);
            continue;
        }

        // 2. whitespace runs, with the (?<!\S) trailing-space rule: a run of
        //    spaces is emitted as one chunk unless it is the final run.
        if (sy[i].kind == CpKind::Space || sy[i].kind == CpKind::Newline) {
            if (sy[i].kind == CpKind::Newline) {
                size_t j = i;
                while (j < n && sy[j].kind == CpKind::Newline) j++;
                const Sym &last = sy[j - 1];
                emit_chunk(text, begin, last.begin + last.len, chunks);
                i = j;
                continue;
            }
            size_t j = i;
            while (j < n && sy[j].kind == CpKind::Space) j++;
            if (j < n && mode == PreMode::Llama3) {
                // llama3: `\s+` — word chunks never absorb a leading
                // space, so the whole run is its own chunk
                const Sym &last = sy[j - 1];
                emit_chunk(text, begin, last.begin + last.len, chunks);
                i = j;
                continue;
            }
            if (j < n) {
                // gpt2/qwen2: all but the last space form their own
                // chunk; the last space is absorbed by the word run
                // below (steps 3-5), which is why this falls through
                // instead of continuing at the space again
                if (j - i > 1) {
                    const Sym &last = sy[j - 2];
                    emit_chunk(text, begin, last.begin + last.len, chunks);
                }
                i = j - 1;
                // the word chunk below starts at the absorbed
                // space, not at the start of the original run
                begin = sy[i].begin;
            } else {
                const Sym &last = sy[j - 1];
                emit_chunk(text, begin, last.begin + last.len, chunks);
                i = j;
                continue;
            }
        }

        // 3. optional single leading space / non-alnum prefix + letter run
        size_t k = i;
        if (sy[k].kind == CpKind::Space && k + 1 < n) k++; // ` ?` prefix
        if (mode != PreMode::Gpt2 && sy[k].kind == CpKind::Other &&
            sy[k].cp != '\r' && sy[k].cp != '\n') {
            // [^\r\n\p{L}\p{N}]? prefix
            if (k + 1 < n && sy[k + 1].kind == CpKind::Letter) k++;
        }
        if (sy[k].kind == CpKind::Letter) {
            size_t j = k;
            while (j < n && sy[j].kind == CpKind::Letter) j++;
            const Sym &last = sy[j - 1];
            emit_chunk(text, begin, last.begin + last.len, chunks);
            i = j;
            continue;
        }

        // 4. digits: [0-9]+ (gpt2) | [0-9] (qwen2) | [0-9]{1,3} (llama3)
        if (sy[k].kind == CpKind::Digit) {
            size_t j = k;
            const int limit = (mode == PreMode::Gpt2) ? INT_MAX
                              : (mode == PreMode::Llama3) ? 3 : 1;
            while (j < n && sy[j].kind == CpKind::Digit &&
                   static_cast<int>(j - k) < limit)
                j++;
            const Sym &last = sy[j - 1];
            emit_chunk(text, begin, last.begin + last.len, chunks);
            i = j;
            continue;
        }

        // 5. everything else: run of non-space, non-letter, non-digit symbols
        {
            size_t j = k;
            while (j < n && sy[j].kind == CpKind::Other) j++;
            if (j == k) { // lone CJK symbol
                emit_chunk(text, begin, sy[i].begin + sy[i].len, chunks);
                i++;
                continue;
            }
            // trailing newlines attach to the punctuation run
            while (j < n && sy[j].kind == CpKind::Newline) j++;
            const Sym &last = sy[j - 1];
            emit_chunk(text, begin, last.begin + last.len, chunks);
            i = j;
        }
    }
    return chunks;
}

PreMode pre_mode_from_name(const std::string &pre) {
    // "laguna" (Poolside) tokenizes like qwen2, not like GPT-2. The two
    // pre-tokenizer patterns differ on punctuation that abuts a word: GPT-2
    // cuts `_start` into `_` + `start`, while qwen2 keeps an optional single
    // non-letter prefix with the following letters, so the `_start` merge can
    // fire. That is the whole ChatML prompt: under GPT-2 rules `<|im_start|>`
    // reached the model as `im` + `_` + `start`, and a model asked a prompt it
    // was never trained on answers a different question than the reference
    // does (token ids compared in docs/traces/laguna-chat-tokenizer.txt).
    if (pre.find("laguna") != std::string::npos) return PreMode::Qwen2;
    if (pre.find("qwen2") != std::string::npos) return PreMode::Qwen2;
    if (pre.find("llama3") != std::string::npos ||
        pre.find("llama-bpe") != std::string::npos)
        return PreMode::Gpt2;
    if (pre.find("deepseek") != std::string::npos) return PreMode::Llama3;
    return PreMode::Gpt2;
}

} // namespace

// ---------------------------------------------------------------------------

bool Tokenizer::load(const Gguf &g, std::string *err) {
    const std::string model = g.get_str("tokenizer.ggml.model");
    const std::vector<std::string> *toks = g.get_str_array("tokenizer.ggml.tokens");
    if (!toks || toks->empty()) {
        if (err) *err = "GGUF has no tokenizer.ggml.tokens array";
        return false;
    }
    tokens_ = *toks;

    if (model == "llama" || model == "spm") {
        model_ = TokModel::Spm;
    } else if (model == "gpt2" || model == "bpe" || model == "qwen2") {
        model_ = TokModel::Bpe;
    } else if (!model.empty()) {
        // Many converters emit no model string but do emit merges/scores.
        model_ = g.get_str_array("tokenizer.ggml.merges") ? TokModel::Bpe
                                                          : TokModel::Spm;
    } else {
        model_ = g.get_str_array("tokenizer.ggml.merges") ? TokModel::Bpe
                                                          : TokModel::Spm;
    }

    ids_.reserve(tokens_.size() * 2);
    for (size_t i = 0; i < tokens_.size(); i++) {
        // first definition wins; duplicate pieces keep their lowest id
        ids_.emplace(tokens_[i], static_cast<i32>(i));
    }

    if (const std::vector<f32> *sc = g.get_f32_array("tokenizer.ggml.scores"))
        scores_ = *sc;
    else
        scores_.assign(tokens_.size(), 0.0f);

    if (const std::vector<std::string> *merges =
            g.get_str_array("tokenizer.ggml.merges")) {
        i32 rank = 0;
        for (const std::string &m : *merges) {
            ranks_.emplace(m, rank++);
        }
    }

    bos_ = static_cast<i32>(g.get_i64("tokenizer.ggml.bos_token_id", -1));
    eos_ = static_cast<i32>(g.get_i64("tokenizer.ggml.eos_token_id", -1));
    pad_ = static_cast<i32>(g.get_i64("tokenizer.ggml.padding_token_id", -1));
    add_bos_ = g.get_bool("tokenizer.ggml.add_bos_token",
                          model_ == TokModel::Spm);
    byte_fallback_ = g.get_bool("tokenizer.ggml.byte_fallback", false);
    pre_ = g.get_str("tokenizer.ggml.pre");

    // Whole-word "special" tokens (chat markers like <|im_start|>, control
    // tokens, added tokens). The backend encoders would otherwise shred these
    // into ordinary sub-pieces: the regex pre-tokenizer splits on them and the
    // SPM lattice only finds them by score. Segment the input on the longest
    // match at each position and emit the id directly instead.
    const std::vector<i32> *types = g.get_i32_array("tokenizer.ggml.token_type");
    for (size_t i = 0; i < tokens_.size(); i++) {
        const std::string &p = tokens_[i];
        if (p.empty() || is_byte_piece(p)) continue;
        bool special = false;
        if (types && i < types->size()) {
            // GGUF token_type: 0=normal, 1=normal in the (observed) 1-based
            // variant used by some producers; 2=control, 3=control, 4=unused.
            const i32 t = (*types)[i];
            special = (t >= 2);
        }
        if (!special && p.size() >= 4 && p.front() == '<' && p.back() == '>'
            && p.find('|') != std::string::npos) {
            special = true; // <|...|> convention, also covers missing metadata
        }
        if (special) specials_.push_back({p, (i32)i});
    }
    std::sort(specials_.begin(), specials_.end(),
              [](const SpecialToken &a, const SpecialToken &b) {
                  return a.text.size() > b.text.size();
              });
    for (size_t i = 0; i < specials_.size(); i++)
        special_index_[specials_[i].text[0]].push_back(i);

    return true;
}

bool Tokenizer::is_byte_piece(const std::string &p) {
    return p.size() == 6 && p[0] == '<' && p[1] == '0' && p[2] == 'x'
        && std::isxdigit((unsigned char)p[3]) && std::isxdigit((unsigned char)p[4])
        && p[5] == '>';
}

int Tokenizer::encode_segmented(const char *text, i32 *out, int max_out) const {
    int n = 0;
    const size_t len = std::strlen(text);
    size_t i = 0, run_start = 0;
    auto flush_run = [&](size_t upto) {
        if (upto <= run_start || n >= max_out) return;
        const std::string seg(text + run_start, upto - run_start);
        const int produced = model_ == TokModel::Bpe
                                 ? encode_bpe(seg, out + n, max_out - n)
                                 : encode_spm(seg, out + n, max_out - n);
        if (produced < 0) return;
        n += produced;
    };
    while (i < len) {
        bool matched = false;
        auto it = special_index_.find(text[i]);
        if (it != special_index_.end()) {
            for (size_t idx : it->second) {
                const SpecialToken &sp = specials_[idx];
                if (sp.text.size() <= len - i
                    && std::memcmp(text + i, sp.text.data(), sp.text.size()) == 0) {
                    flush_run(i);
                    if (n < max_out) out[n++] = sp.id;
                    i += sp.text.size();
                    run_start = i;
                    matched = true;
                    break;
                }
            }
        }
        if (!matched) i++;
    }
    flush_run(len);
    return n;
}

const std::string &Tokenizer::piece(i32 id) const {
    static const std::string kEmpty;
    if (id < 0 || id >= static_cast<i32>(tokens_.size())) return kEmpty;
    return tokens_[static_cast<size_t>(id)];
}

i32 Tokenizer::piece_id(const std::string &p) const {
    auto it = ids_.find(p);
    return it == ids_.end() ? -1 : it->second;
}

// ---------------------------------------------------------------------------
// byte-level BPE
// ---------------------------------------------------------------------------

int Tokenizer::encode_bpe(const std::string &text, i32 *out, int max_out) const {
    const ByteCodec &codec = byte_codec();
    int n_out = 0;

    const std::vector<std::string> chunks =
        pre_tokenize(text, pre_mode_from_name(pre_));

    std::string mapped;
    std::vector<std::string> syms;
    for (const std::string &chunk : chunks) {
        mapped.clear();
        for (char rc : chunk)
            codec.encode_byte(mapped, static_cast<u8>(rc));
        // split into mapped codepoints, then merge by rank
        syms.clear();
        size_t i = 0;
        while (i < mapped.size()) {
            u32 cp = 0;
            const int len = utf8_decode(mapped, i, &cp);
            syms.emplace_back(mapped, i, static_cast<size_t>(len));
            i += static_cast<size_t>(len);
        }

        while (syms.size() > 1) {
            int best_rank = INT_MAX;
            size_t best_at = 0;
            for (size_t j = 0; j + 1 < syms.size(); j++) {
                const std::string key = syms[j] + " " + syms[j + 1];
                auto it = ranks_.find(key);
                if (it != ranks_.end() && it->second < best_rank) {
                    best_rank = it->second;
                    best_at = j;
                }
            }
            if (best_rank == INT_MAX) break;
            syms[best_at] += syms[best_at + 1];
            syms.erase(syms.begin() + static_cast<ptrdiff_t>(best_at + 1));
        }

        for (const std::string &s : syms) {
            i32 id = piece_id(s);
            if (id < 0) {
                // byte fallback: emit each mapped codepoint as <0xNN>
                size_t p = 0;
                while (p < s.size()) {
                    u32 cp = 0;
                    const int len = utf8_decode(s, p, &cp);
                    p += static_cast<size_t>(len);
                    const i32 b = codec.decode_cp(cp);
                    if (b < 0) continue;
                    char buf[16];
                    std::snprintf(buf, sizeof(buf), "<0x%02X>", b);
                    const i32 bid = piece_id(buf);
                    if (bid >= 0) {
                        if (n_out >= max_out) return -1;
                        out[n_out++] = bid;
                    }
                }
                continue;
            }
            if (n_out >= max_out) return -1;
            out[n_out++] = id;
        }
    }
    return n_out;
}

// ---------------------------------------------------------------------------
// sentencepiece unigram
// ---------------------------------------------------------------------------

int Tokenizer::encode_spm(const std::string &text, i32 *out, int max_out) const {
    // SPM convention: prepend the sentence-start marker, fold spaces to ▁.
    std::string t;
    t.reserve(text.size() + 4);
    if (spm_leading_space_) t += "\xE2\x96\x81";
    for (char c : text) {
        if (c == ' ')
            t += "\xE2\x96\x81";
        else
            t.push_back(c);
    }

    // split into codepoint-sized symbols
    std::vector<std::string> syms;
    {
        size_t i = 0;
        while (i < t.size()) {
            u32 cp = 0;
            const int len = utf8_decode(t, i, &cp);
            syms.emplace_back(t, i, static_cast<size_t>(len));
            i += static_cast<size_t>(len);
        }
    }
    const size_t n = syms.size();
    if (n == 0) return 0;

    // worst-case token length in symbols (bounded so the DP stays linear-ish)
    size_t max_sym = 1;
    for (const std::string &p : tokens_)
        max_sym = std::max<size_t>(max_sym, p.size());
    max_sym = std::min<size_t>(max_sym, 64);

    const f32 kNegInf = -1e30f;
    std::vector<f32> best(n + 1, kNegInf);
    std::vector<i32> back(n + 1, -1);
    best[0] = 0.0f;

    for (size_t i = 0; i < n; i++) {
        if (best[i] == kNegInf) continue;
        std::string piece;
        for (size_t j = i; j < n && j - i < max_sym; j++) {
            piece += syms[j];
            auto it = ids_.find(piece);
            if (it == ids_.end()) continue;
            const i32 id = it->second;
            const f32 score = (static_cast<size_t>(id) < scores_.size())
                                  ? scores_[static_cast<size_t>(id)]
                                  : 0.0f;
            const f32 cand = best[i] + score;
            if (cand > best[j + 1]) {
                best[j + 1] = cand;
                back[j + 1] = id;
            }
        }
    }

    // Reconstruct; any unreachable span falls back to byte tokens.
    std::vector<i32> rev;
    size_t pos = n;
    while (pos > 0) {
        if (back[pos] < 0) {
            // emit a byte-fallback for the symbol ending at pos-1
            const std::string &sy = syms[pos - 1];
            u32 cp = 0;
            utf8_decode(sy, 0, &cp);
            char buf[16];
            std::snprintf(buf, sizeof(buf), "<0x%02X>", static_cast<unsigned>(cp & 0xFF));
            const i32 bid = piece_id(buf);
            rev.push_back(bid >= 0 ? bid : 0);
            pos--;
            continue;
        }
        rev.push_back(back[pos]);
        // walk back over the matched span
        size_t len = 0;
        {
            // recompute the matched piece length by scanning forward from a
            // candidate start; cheaper than storing it per position
            const i32 id = rev.back();
            const std::string &p = piece(id);
            size_t pbytes = p.size();
            // find the start k such that syms[k..pos) concatenated == p
            size_t k = pos;
            size_t acc = 0;
            while (k > 0 && acc < pbytes) {
                k--;
                acc += syms[k].size();
            }
            len = pos - k;
        }
        pos -= len;
    }
    std::reverse(rev.begin(), rev.end());

    if (static_cast<int>(rev.size()) > max_out) return -1;
    for (size_t i = 0; i < rev.size(); i++) out[i] = rev[i];
    return static_cast<int>(rev.size());
}

int Tokenizer::encode(const char *text, i32 *out, int max_out, bool add_special) const {
    int n = 0;
    if (add_special && add_bos_ && bos_ >= 0) {
        if (max_out < 1) return -1;
        out[n++] = bos_;
    }
    const int produced = !specials_.empty()
                             ? encode_segmented(text ? text : "", out + n, max_out - n)
                             : (model_ == TokModel::Bpe
                                    ? encode_bpe(text ? text : "", out + n, max_out - n)
                                    : encode_spm(text ? text : "", out + n, max_out - n));
    if (produced < 0) return -1;
    return n + produced;
}

int Tokenizer::encode(const std::string &text, std::vector<i32> &out,
                      bool add_special) const {
    std::vector<i32> tmp(text.size() * 4 + 16);
    const int n = encode(text.c_str(), tmp.data(), static_cast<int>(tmp.size()),
                         add_special);
    if (n < 0) return -1;
    out.assign(tmp.begin(), tmp.begin() + n);
    return n;
}

std::string Tokenizer::decode(const i32 *toks, int n) const {
    std::string out;
    for (int i = 0; i < n; i++) {
        const std::string &p = piece(toks[i]);
        if (p.empty()) continue;

        // <0xNN> byte token -> raw byte (both SPM and byte-fallback BPE)
        if (p.size() == 6 && p[0] == '<' && p[1] == '0' && p[2] == 'x') {
            unsigned v = 0;
            bool ok = true;
            for (int k = 3; k < 5; k++) {
                const char c = p[static_cast<size_t>(k)];
                unsigned d;
                if (c >= '0' && c <= '9') d = static_cast<unsigned>(c - '0');
                else if (c >= 'A' && c <= 'F') d = static_cast<unsigned>(c - 'A' + 10);
                else if (c >= 'a' && c <= 'f') d = static_cast<unsigned>(c - 'a' + 10);
                else { ok = false; break; }
                v = v * 16 + d;
            }
            if (ok) { out.push_back(static_cast<char>(v)); continue; }
        }

        if (model_ == TokModel::Spm) {
            size_t i2 = 0;
            while (i2 < p.size()) {
                u32 cp = 0;
                const int len = utf8_decode(p, i2, &cp);
                if (cp == 0x2581) out.push_back(' ');
                else out.append(p, i2, static_cast<size_t>(len));
                i2 += static_cast<size_t>(len);
            }
        } else {
            // byte-level BPE: map mapped codepoints back to raw bytes
            const ByteCodec &codec = byte_codec();
            size_t i2 = 0;
            while (i2 < p.size()) {
                u32 cp = 0;
                const int len = utf8_decode(p, i2, &cp);
                const i32 b = codec.decode_cp(cp);
                if (b >= 0) out.push_back(static_cast<char>(b));
                else out.append(p, i2, static_cast<size_t>(len));
                i2 += static_cast<size_t>(len);
            }
        }
    }
    return out;
}

std::string Tokenizer::Streamer::push(i32 token) {
    pending_ += tok_->decode(&token, 1);
    // hand back only complete UTF-8 sequences
    size_t cut = pending_.size();
    for (size_t i = 0; i < pending_.size(); i++) {
        const u8 c = static_cast<u8>(pending_[i]);
        int need = 0;
        if ((c & 0x80) == 0) need = 1;
        else if ((c & 0xE0) == 0xC0) need = 2;
        else if ((c & 0xF0) == 0xE0) need = 3;
        else if ((c & 0xF8) == 0xF0) need = 4;
        else need = 1;
        if (i + static_cast<size_t>(need) > pending_.size()) {
            cut = i;
            break;
        }
        i += static_cast<size_t>(need) - 1;
    }
    std::string out = pending_.substr(0, cut);
    pending_.erase(0, cut);
    return out;
}

} // namespace krk
