// ============================================================================
//  tokenizer.hpp — SentencePiece-style unigram (LLaMA) and byte-level BPE
//  (GPT-2 / Qwen) tokenizers, both driven purely by GGUF metadata.
// ============================================================================
#ifndef KRK_TOKENIZER_HPP
#define KRK_TOKENIZER_HPP

#include <unordered_map>

#include "krk/common.hpp"
#include "krk/gguf.hpp"

namespace krk {

enum class TokModel { Unknown, Bpe, Spm };

class Tokenizer {
public:
    Tokenizer() = default;

    bool load(const Gguf &g, std::string *err);

    i32 n_vocab() const { return static_cast<i32>(tokens_.size()); }
    i32 bos() const { return bos_; }
    i32 eos() const { return eos_; }
    i32 pad() const { return pad_; }
    bool add_bos() const { return add_bos_; }
    TokModel model() const { return model_; }

    const std::string &piece(i32 id) const;
    i32 piece_id(const std::string &piece) const;

    // Encodes text; returns the token count or -1 on overflow.
    int encode(const char *text, i32 *out, int max_out, bool add_special) const;
    int encode(const std::string &text, std::vector<i32> &out, bool add_special) const;

    // Decodes to a std::string (invalid sequences replaced with U+FFFD).
    std::string decode(const i32 *toks, int n) const;

    // Streaming detokenizer: holds back incomplete UTF-8 sequences.
    class Streamer {
    public:
        explicit Streamer(const Tokenizer *t) : tok_(t) {}
        // Appends token text; returns the complete bytes now available.
        std::string push(i32 token);
        void reset() { pending_.clear(); }

    private:
        const Tokenizer *tok_ = nullptr;
        std::string pending_;
    };

private:
    int encode_bpe(const std::string &text, i32 *out, int max_out) const;
    int encode_spm(const std::string &text, i32 *out, int max_out) const;

    // Whole-word tokens (chat-template markers, control tokens) that are
    // emitted as a single id instead of being split by the backend encoder.
    struct SpecialToken { std::string text; i32 id; };
    int encode_segmented(const char *text, i32 *out, int max_out) const;
    static bool is_byte_piece(const std::string &p);

    TokModel model_ = TokModel::Unknown;
    std::vector<std::string> tokens_;      // id -> piece
    std::unordered_map<std::string, i32> ids_;   // piece -> id
    std::unordered_map<std::string, i32> ranks_; // bpe merge rank
    std::vector<f32> scores_;              // spm unigram log-probabilities
    std::string pre_;                      // tokenizer.ggml.pre (pre-tokenizer family)
    std::unordered_map<std::string, i32> byte_to_char_; // gpt2 byte<->unicode
    i32 bos_ = -1, eos_ = -1, pad_ = -1;
    bool add_bos_ = false;
    bool byte_fallback_ = false;
    bool spm_leading_space_ = true;

    std::vector<SpecialToken> specials_;   // sorted by piece length, desc
    std::unordered_map<char, std::vector<size_t>> special_index_; // first byte -> specials_
};


} // namespace krk

#endif // KRK_TOKENIZER_HPP
