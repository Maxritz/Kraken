// chat_template.hpp -- turning a conversation into a prompt, the way the file
// says to.
//
// A chat model was trained on one exact byte layout around its turns:
//
//   ChatML     <|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n
//   Llama-3    <|start_header_id|>user<|end_header_id|>\n\n...<|eot_id|>
//   Gemma      <start_of_turn>user\n...<end_of_turn>\n<start_of_turn>model\n
//   Phi-3      <|user|>\n...<|end|>\n<|assistant|>\n
//   Mistral    [INST] ... [/INST]
//
// These are not interchangeable and they are not cosmetic. The markers are
// SPECIAL tokens -- single ids the model saw during training -- and the model
// attends to them as structure. Feeding a Gemma checkpoint ChatML markers does
// not produce a Gemma answer with strange formatting; it produces an answer to
// a prompt the model never saw, which reads as a much dumber model rather than
// as a bug. So this is not a formatting convenience, it is part of correctness,
// and it is why the template has to come from the file.
//
// GGUF carries it: `tokenizer.chat_template` is a Jinja program, written by the
// model's authors, and llama.cpp renders it with a bundled Jinja engine. This
// is that, scoped to the language real templates use -- statements (if/for/
// set), expressions, filters, tests, `loop`, namespaces -- with a named failure
// for anything outside it, so a caller can fall back to a builtin instead of
// emitting a prompt that is silently wrong.
#pragma once

#include <string>
#include <vector>

namespace krk {

// One turn. `content` is the text; `role` is the OpenAI spelling (system, user,
// assistant, tool). Extra fields some templates read (name, tool_calls) are not
// carried yet: a template that needs them sees `undefined`, which is falsy and
// renders as empty, so it degrades to a template without them rather than
// failing.
struct ChatMessage {
    std::string role;
    std::string content;
};

// Render `msgs` with `tmpl`. On success writes the prompt to `out` and returns
// true. On failure returns false and writes WHY to `err` -- the message names
// the construct when the template is outside the supported subset, and quotes
// the template's own `raise_exception` when it rejected the conversation.
//
// `add_generation_prompt` is the standard flag: true when the prompt is about
// to be continued by the model (an ordinary chat turn), false when the caller
// wants the transcript alone (e.g. to score a completion).
//
// The renderer implements ONE fallback the reference does too: a template that
// raises "System role not supported" (Gemma's, Mistral's) gets the system
// message folded into the first user turn and one retry, because refusing the
// conversation is worse than delivering it in the shape the model expects.
bool render_chat_prompt(const std::string &tmpl,
                        const std::vector<ChatMessage> &msgs,
                        bool add_generation_prompt, const std::string &bos_token,
                        const std::string &eos_token, std::string *out,
                        std::string *err);

// Can this template be rendered by the subset above? Parses, then trial-renders
// a canonical two-turn conversation. `why` names the first unsupported thing
// found, so a caller choosing between the file's template and a builtin can say
// which one it used and why.
bool chat_template_supported(const std::string &tmpl, std::string *why);

// The turn-end markers a rendered prompt actually uses, in the order they were
// offered. A CLI streaming a reply must hide these from the transcript --
// ChatML's `<|im_end|>` was hardcoded once, which printed `<|eot_id|>` at the
// end of every Llama-3 reply. Reading them off the prompt keeps the display in
// step with whatever layout the model declared.
std::vector<std::string> chat_turn_markers(const std::string &rendered);

// The builtin template for a family, by `general.architecture` and
// `tokenizer.ggml.pre`. Empty when the family is not one of the known ones --
// the caller then decides, and `kBuiltinDefault` is what a file with no
// template at all gets.
std::string builtin_chat_template(const std::string &arch,
                                  const std::string &pre);

// The name of the builtin a family maps to ("chatml", "llama3", ...), for a
// log line that has to say which template produced the prompt.
const char *builtin_chat_template_name(const std::string &arch,
                                      const std::string &pre);

// A builtin by name, for --chat-template. Empty for an unknown name; the names
// are the ones `builtin_chat_template_names()` lists, and they double as the
// documented vocabulary of `--chat-template`.
std::string builtin_chat_template_by_name(const std::string &name);
const std::vector<std::string> &builtin_chat_template_names();

// The name used when nothing else applies. ChatML: it is the most common
// template in the wild, this engine's historical behaviour, and the one every
// Qwen-derived family in this collection uses.
extern const char *kBuiltinDefault;

} // namespace krk
