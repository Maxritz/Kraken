// ============================================================================
//  webui.hpp — the built-in chat UI, served at GET /.
//
//  One self-contained document: no build step, no CDN, no external asset, and
//  no dependency on anything the engine does not already serve. It talks to
//  /health, /v1/models and /v1/chat/completions exactly as an external client
//  does, so the UI cannot paper over a broken endpoint.
// ============================================================================
#ifndef KRK_WEBUI_HPP
#define KRK_WEBUI_HPP

#include <string>

namespace krk {

// The page bytes, NUL-free and already terminated by a newline.
const std::string &web_ui_html();

} // namespace krk

#endif // KRK_WEBUI_HPP
