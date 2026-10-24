// ============================================================================
//  webui.cpp — the built-in chat UI. See webui.hpp for why it is a raw string
//  rather than a bundle: the server ships as one executable, and a page that
//  needs a build step or a CDN is a page that is missing exactly when the
//  machine is offline or the asset server is not the one you think it is.
//
//  Feature set (llama.cpp webui parity where it matters, plus engine facts):
//    * streaming chat over the same /v1/chat/completions SSE an external client
//      uses, with a working Stop (it aborts the request and says so);
//    * system prompt, temperature, top_p, top_k, min_p, repeat_penalty,
//      repeat_last_n, max_tokens, seed, stream toggle -- persisted;
//    * conversation list persisted in localStorage, rename by first message,
//      delete, regenerate, copy, export as JSON;
//    * header shows model, architecture, device, context and a context meter,
//      read from /health and /v1/models rather than hardcoded;
//    * markdown-lite rendering (fenced code, inline code, bold, italics) with
//      everything escaped first, and a dark/light toggle.
// ============================================================================
#include "krk/webui.hpp"

namespace krk {

const std::string &web_ui_html() {
    static const std::string page = R"KRKWEB(<!doctype html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>kraken</title>
<link rel="icon" href="data:,">
<style>
:root{--bg:#12141a;--panel:#181b23;--panel2:#1f232d;--line:#2b3040;--fg:#e8eaf0;
--dim:#9aa2b5;--accent:#5b8cff;--accent2:#38d39f;--warn:#ffb347;--err:#ff6b6b;
--code:#0e1015;--shadow:0 8px 30px rgba(0,0,0,.35)}
[data-theme="light"]{--bg:#f5f6fa;--panel:#ffffff;--panel2:#eef0f6;--line:#d8dce8;
--fg:#161922;--dim:#5b6478;--accent:#2f6bff;--accent2:#0f9f74;--code:#f0f2f8;
--shadow:0 8px 24px rgba(20,25,40,.10)}
*{box-sizing:border-box}
html,body{height:100%}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.55 ui-sans-serif,system-ui,
-apple-system,"Segoe UI",Roboto,sans-serif}
#app{display:grid;grid-template-columns:248px 1fr;height:100vh;overflow:hidden}
aside{background:var(--panel);border-right:1px solid var(--line);display:flex;
flex-direction:column;min-height:0}
.brand{padding:14px 14px 10px;font-weight:700;letter-spacing:.4px;display:flex;
align-items:center;gap:8px}
.brand em{font-style:normal;color:var(--accent)}
.brand small{color:var(--dim);font-weight:500;font-size:11px}
#newchat{margin:0 10px 8px;padding:9px 10px;border-radius:9px;border:1px dashed var(--line);
background:transparent;color:var(--fg);cursor:pointer;font:inherit}
#newchat:hover{border-color:var(--accent);color:var(--accent)}
#chats{flex:1;overflow:auto;padding:4px 8px 8px;min-height:0}
.chatitem{padding:8px 9px;border-radius:8px;cursor:pointer;color:var(--dim);
display:flex;gap:6px;align-items:center;font-size:13px}
.chatitem span{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.chatitem:hover{background:var(--panel2);color:var(--fg)}
.chatitem.on{background:var(--panel2);color:var(--fg);box-shadow:inset 2px 0 0 var(--accent)}
.chatitem b{color:var(--err);font-weight:400;visibility:hidden;padding:0 4px}
.chatitem:hover b{visibility:visible}
.sidefoot{border-top:1px solid var(--line);padding:8px;display:flex;gap:6px}
.sidefoot button{flex:1;padding:7px;border-radius:8px;border:1px solid var(--line);
background:transparent;color:var(--dim);cursor:pointer;font:inherit;font-size:12px}
.sidefoot button:hover{color:var(--fg);border-color:var(--accent)}
main{display:flex;flex-direction:column;min-width:0;min-height:0}
header{display:flex;gap:10px;align-items:center;padding:10px 14px;
border-bottom:1px solid var(--line);background:var(--panel)}
#mname{font-weight:650}
#mmeta{color:var(--dim);font-size:12px;margin-left:8px}
.badges{margin-left:auto;display:flex;gap:6px;flex-wrap:wrap;justify-content:flex-end}
.badge{font-size:11px;color:var(--dim);border:1px solid var(--line);border-radius:999px;
padding:2px 8px;white-space:nowrap}
.badge b{color:var(--fg);font-weight:600}
header button{padding:7px 10px;border-radius:8px;border:1px solid var(--line);
background:var(--panel2);color:var(--fg);cursor:pointer;font:inherit}
#transcript{flex:1;overflow:auto;padding:18px 0 8px;min-height:0}
.wrap{max-width:820px;margin:0 auto;padding:0 18px}
.msg{display:flex;gap:10px;margin:14px 0}
.msg .who{width:26px;height:26px;border-radius:7px;flex:none;display:grid;
place-items:center;font-size:11px;font-weight:700;background:var(--panel2);
color:var(--dim);border:1px solid var(--line)}
.msg.user .who{color:var(--accent);border-color:var(--accent)}
.msg.bot .who{color:var(--accent2);border-color:var(--accent2)}
.body{min-width:0;flex:1}
.role{font-size:11px;color:var(--dim);text-transform:uppercase;letter-spacing:.7px}
.text{white-space:pre-wrap;overflow-wrap:anywhere;margin-top:2px}
.text pre{background:var(--code);border:1px solid var(--line);border-radius:8px;
padding:10px 12px;overflow:auto;white-space:pre;font:12.5px/1.5 ui-monospace,
SFMono-Regular,Consolas,monospace}
.text code{background:var(--code);border:1px solid var(--line);border-radius:5px;
padding:1px 4px;font:12.5px ui-monospace,Consolas,monospace}
.text pre code{border:0;padding:0;background:none}
.acts{display:flex;gap:8px;margin-top:6px;opacity:0;transition:opacity .12s}
.msg:hover .acts{opacity:1}
.acts button{background:none;border:0;color:var(--dim);cursor:pointer;font:inherit;
font-size:12px;padding:0}
.acts button:hover{color:var(--accent)}
.sys{opacity:.0001;position:absolute;inset:0;pointer-events:none}
.meta{font-size:11px;color:var(--dim);margin-top:4px}
.cursor{display:inline-block;width:7px;height:14px;background:var(--accent2);
vertical-align:-2px;animation:bl 1s steps(2) infinite}
@keyframes bl{50%{opacity:0}}
#ctxbar{height:3px;background:var(--panel2)}
#ctxfill{height:100%;width:0;background:var(--accent);transition:width .2s}
#composer{display:flex;gap:8px;align-items:flex-end;padding:12px 14px 14px;
border-top:1px solid var(--line);background:var(--panel)}
#composer .field{flex:1;display:flex;background:var(--panel2);border:1px solid var(--line);
border-radius:12px;padding:6px 8px}
#composer .field:focus-within{border-color:var(--accent)}
#input{flex:1;resize:none;border:0;outline:0;background:transparent;color:var(--fg);
font:inherit;max-height:190px;padding:4px}
#send,#stop{border:0;border-radius:10px;padding:10px 16px;font:inherit;cursor:pointer}
#send{background:var(--accent);color:#fff;font-weight:600}
#send:disabled{opacity:.45;cursor:not-allowed}
#stop{background:var(--err);color:#fff;font-weight:600}
#settings{position:fixed;top:0;right:0;bottom:0;width:330px;background:var(--panel);
border-left:1px solid var(--line);box-shadow:var(--shadow);padding:14px;overflow:auto;
z-index:9}
#settings h3{margin:2px 0 12px;font-size:13px;text-transform:uppercase;
letter-spacing:.8px;color:var(--dim)}
.row{margin:10px 0}
.row label{display:flex;justify-content:space-between;font-size:12px;color:var(--dim)}
.row label b{color:var(--fg);font-weight:600}
.row input[type=range]{width:100%;accent-color:var(--accent)}
.row input[type=number],.row input[type=text],.row textarea{width:100%;background:var(--panel2);
border:1px solid var(--line);border-radius:8px;color:var(--fg);padding:7px 8px;font:inherit}
.row textarea{min-height:74px;resize:vertical}
.hint{color:var(--dim);font-size:11px;margin-top:4px}
#toast{position:fixed;left:50%;bottom:18px;transform:translateX(-50%);background:var(--panel2);
border:1px solid var(--line);color:var(--fg);padding:8px 14px;border-radius:9px;
box-shadow:var(--shadow);opacity:0;pointer-events:none;transition:opacity .2s}
#toast.on{opacity:1}
@media(max-width:760px){#app{grid-template-columns:1fr}aside{display:none}
#settings{width:100%}}
</style>
</head>
<body>
<div id="app">
  <aside>
    <div class="brand">kraken <em>webui</em> <small id="ver"></small></div>
    <button id="newchat">+ New chat</button>
    <div id="chats"></div>
    <div class="sidefoot">
      <button id="theme">Theme</button>
      <button id="export">Export</button>
    </div>
  </aside>
  <main>
    <header>
      <div><b id="mname">connecting…</b><span id="mmeta"></span></div>
      <div class="badges">
        <span class="badge">device <b id="dev">–</b></span>
        <span class="badge">ctx <b id="ctx">–</b></span>
        <span class="badge">tokens <b id="tok">0</b></span>
      </div>
      <button id="settingsBtn" title="settings">⚙</button>
    </header>
    <div id="transcript"><div class="wrap" id="stream"></div></div>
    <div id="ctxbar"><div id="ctxfill"></div></div>
    <form id="composer">
      <div class="field">
        <textarea id="input" rows="1"
          placeholder="Message… (Enter to send, Shift+Enter for a newline)"></textarea>
      </div>
      <button id="send" type="submit">Send</button>
      <button id="stop" type="button" hidden>Stop</button>
    </form>
  </main>
</div>

<section id="settings" hidden>
  <h3>Sampling</h3>
  <div class="row"><label for="temperature">temperature <b id="v_temperature"></b></label>
    <input id="temperature" type="range" min="0" max="2" step="0.05"></div>
  <div class="row"><label for="top_p">top_p <b id="v_top_p"></b></label>
    <input id="top_p" type="range" min="0" max="1" step="0.01"></div>
  <div class="row"><label for="top_k">top_k <b id="v_top_k"></b></label>
    <input id="top_k" type="range" min="0" max="200" step="1"></div>
  <div class="row"><label for="min_p">min_p <b id="v_min_p"></b></label>
    <input id="min_p" type="range" min="0" max="0.9" step="0.01"></div>
  <div class="row"><label for="repeat_penalty">repeat_penalty <b id="v_repeat_penalty"></b></label>
    <input id="repeat_penalty" type="range" min="1" max="2" step="0.01"></div>
  <div class="row"><label for="max_tokens">max_tokens <b id="v_max_tokens"></b></label>
    <input id="max_tokens" type="range" min="16" max="2048" step="16"></div>
  <h3>Request</h3>
  <div class="row"><label for="seed">seed (0 = random)</label>
    <input id="seed" type="number" min="0" step="1"></div>
  <div class="row"><label for="repeat_last_n">repeat_last_n</label>
    <input id="repeat_last_n" type="number" min="1" step="1"></div>
  <div class="row"><label for="system">system prompt</label>
    <textarea id="system" placeholder="You are a helpful assistant."></textarea></div>
  <div class="row"><label for="stream">stream tokens <b id="v_stream"></b></label>
    <input id="stream" type="range" min="0" max="1" step="1"></div>
  <p class="hint">These are the server's per-request knobs; nothing here is stored
  on the server. The UI speaks the same OpenAI endpoints an external client does.</p>
</section>

<div id="toast"></div>
<script>
"use strict";
const $ = (id) => document.getElementById(id);
const LSKEY = "kraken.webui.v1";
const DEFAULTS = { temperature: 0.7, top_p: 0.95, top_k: 40, min_p: 0,
  repeat_penalty: 1.1, repeat_last_n: 64, max_tokens: 512, seed: 0,
  system: "", stream: 1 };

let S = { chats: [], active: null, settings: { ...DEFAULTS }, theme: "dark" };
let META = null;
let ABORT = null;
let BUSY = false;

function load() {
  try {
    const raw = localStorage.getItem(LSKEY);
    if (!raw) return;
    const j = JSON.parse(raw);
    if (Array.isArray(j.chats)) S.chats = j.chats;
    if (j.settings) S.settings = { ...DEFAULTS, ...j.settings };
    if (j.theme) S.theme = j.theme;
    S.active = j.active && S.chats.some((c) => c.id === j.active) ? j.active : null;
  } catch (e) { /* corrupt state must not brick the page */ }
}
function save() {
  try {
    localStorage.setItem(LSKEY, JSON.stringify(
      { chats: S.chats, settings: S.settings, theme: S.theme, active: S.active }));
  } catch (e) { /* private mode: the conversation just is not persisted */ }
}
const uid = () => Math.random().toString(36).slice(2, 10);
function chat() {
  let c = S.chats.find((x) => x.id === S.active);
  if (!c) {
    c = { id: uid(), title: "New chat", messages: [] };
    S.chats.unshift(c);
    S.active = c.id;
  }
  return c;
}
function toast(t) { const el = $("toast"); el.textContent = t; el.classList.add("on");
  clearTimeout(toast.t); toast.t = setTimeout(() => el.classList.remove("on"), 1600); }

// ---- markdown-lite: escape first, then the few forms worth having ----------
const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
function inline(s) {
  return s.replace(/`([^`\n]+)`/g, "<code>$1</code>")
          .replace(/\*\*([^*\n]+)\*\*/g, "<b>$1</b>")
          .replace(/(^|[\s(])\*([^*\n]+)\*/g, "$1<i>$2</i>");
}
function render(text) {
  const parts = esc(text).split(/```/);
  let out = "";
  for (let i = 0; i < parts.length; i++) {
    if (i % 2 === 1) {
      let body = parts[i];
      const nl = body.indexOf("\n");
      if (nl >= 0 && /^[a-zA-Z0-9_+-]*\s*$/.test(body.slice(0, nl))) body = body.slice(nl + 1);
      out += "<pre><code>" + body.replace(/\n$/, "") + "</code></pre>";
    } else out += inline(parts[i]);
  }
  return out;
}

// ---- transcript -----------------------------------------------------------
function paint() {
  const c = chat();
  const root = $("stream");
  root.innerHTML = "";
  if (!c.messages.length) {
    const d = document.createElement("div");
    d.className = "msg bot";
    d.innerHTML = '<div class="who">k</div><div class="body"><div class="role">kraken</div>' +
      '<div class="text">Ready. The model and device above come from /health, and the ' +
      'chat below is the same /v1/chat/completions stream any client gets.</div></div>';
    root.appendChild(d);
  }
  c.messages.forEach((m, idx) => {
    const d = document.createElement("div");
    d.className = "msg " + (m.role === "user" ? "user" : "bot");
    const who = m.role === "user" ? "you" : "k";
    d.innerHTML = '<div class="who">' + who + '</div><div class="body"><div class="role">' +
      m.role + '</div><div class="text"></div><div class="acts"></div></div>';
    const t = d.querySelector(".text");
    t.innerHTML = render(m.content || "") +
      (m.streaming ? '<span class="cursor"></span>' : "");
    if (m.meta) {
      const md = document.createElement("div");
      md.className = "meta"; md.textContent = m.meta;
      d.querySelector(".body").appendChild(md);
    }
    const acts = d.querySelector(".acts");
    const copy = document.createElement("button");
    copy.textContent = "copy";
    copy.onclick = () => { navigator.clipboard.writeText(m.content || "").then(() => toast("copied")); };
    acts.appendChild(copy);
    if (m.role === "assistant" && idx === c.messages.length - 1 && !m.streaming) {
      const rg = document.createElement("button");
      rg.textContent = "regenerate";
      rg.onclick = () => regenerate();
      acts.appendChild(rg);
    }
    if (m.role === "assistant" && idx > 0) {
      const del = document.createElement("button");
      del.textContent = "delete";
      del.onclick = () => { c.messages.splice(idx, 1); save(); paint(); };
      acts.appendChild(del);
    }
    root.appendChild(d);
  });
  $("transcript").scrollTop = $("transcript").scrollHeight;
  $("tok").textContent = c.messages.reduce(
    (n, m) => n + (m.tokens || 0), 0);
  const used = c.messages.length ? (c.messages[c.messages.length - 1].total || 0) : 0;
  const cap = META && META.n_ctx ? META.n_ctx : 1;
  $("ctxfill").style.width = Math.min(100, (used / cap) * 100) + "%";
}
function paintChats() {
  const box = $("chats");
  box.innerHTML = "";
  S.chats.forEach((c) => {
    const d = document.createElement("div");
    d.className = "chatitem" + (c.id === S.active ? " on" : "");
    const s = document.createElement("span");
    s.textContent = c.title || "New chat";
    d.appendChild(s);
    const b = document.createElement("b");
    b.textContent = "×";
    b.title = "delete chat";
    b.onclick = (e) => {
      e.stopPropagation();
      S.chats = S.chats.filter((x) => x.id !== c.id);
      if (S.active === c.id) S.active = S.chats.length ? S.chats[0].id : null;
      save(); paintChats(); paint();
    };
    d.appendChild(b);
    d.onclick = () => { S.active = c.id; save(); paintChats(); paint(); };
    box.appendChild(d);
  });
}

// ---- request --------------------------------------------------------------
function payload(c) {
  const st = S.settings;
  const msgs = [];
  if (st.system.trim()) msgs.push({ role: "system", content: st.system.trim() });
  for (const m of c.messages) {
    if (m.role === "user" || (m.role === "assistant" && m.content)) {
      msgs.push({ role: m.role, content: m.content });
    }
  }
  const body = {
    messages: msgs,
    max_tokens: Number(st.max_tokens),
    temperature: Number(st.temperature),
    top_p: Number(st.top_p),
    top_k: Number(st.top_k),
    min_p: Number(st.min_p),
    repeat_penalty: Number(st.repeat_penalty),
    repeat_last_n: Number(st.repeat_last_n),
    stream: Boolean(Number(st.stream)),
  };
  if (Number(st.seed) > 0) body.seed = Number(st.seed);
  return body;
}

async function send(text) {
  const c = chat();
  c.messages.push({ role: "user", content: text });
  if (c.title === "New chat") c.title = text.slice(0, 42) || "New chat";
  const msg = { role: "assistant", content: "", streaming: true };
  c.messages.push(msg);
  S.active = c.id;
  save(); paintChats(); paint();
  await run(c, msg);
}

async function run(c, msg) {
  BUSY = true;
  $("send").disabled = true;
  $("stop").hidden = false;
  ABORT = new AbortController();
  const t0 = performance.now();
  const body = payload(c);
  const streamed = body.stream;
  msg.content = ""; msg.streaming = true; msg.meta = "";
  paint();
  try {
    const res = await fetch("/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: ABORT.signal,
    });
    if (!res.ok) {
      const raw = await res.text();
      let m = raw;
      try { m = JSON.parse(raw).error.message; } catch (e) { /* keep the bytes */ }
      throw new Error(res.status + ": " + m);
    }
    let finish = "", usage = null;
    if (!streamed) {
      const j = await res.json();
      msg.content = (j.choices && j.choices[0] && j.choices[0].message
        ? j.choices[0].message.content : "") || "";
      finish = j.choices && j.choices[0] ? j.choices[0].finish_reason : "";
      usage = j.usage;
      paint();
    } else {
      const rd = res.body.getReader();
      const dec = new TextDecoder();
      let buf = "";
      for (;;) {
        const r = await rd.read();
        if (r.done) break;
        buf += dec.decode(r.value, { stream: true });
        let at;
        while ((at = buf.indexOf("\n\n")) >= 0) {
          const ev = buf.slice(0, at);
          buf = buf.slice(at + 2);
          for (const line of ev.split("\n")) {
            if (!line.startsWith("data: ")) continue;
            const data = line.slice(6).trim();
            if (!data) continue;
            if (data === "[DONE]") continue;
            let j;
            try { j = JSON.parse(data); } catch (e) { continue; }
            if (j.error) throw new Error(j.error.message || "server error");
            if (j.usage) usage = j.usage;
            const ch = j.choices && j.choices[0];
            if (!ch) continue;
            if (ch.delta && ch.delta.content) {
              msg.content += ch.delta.content;
              const t = document.querySelector("#stream .msg:last-child .text");
              if (t) t.innerHTML = render(msg.content) + '<span class="cursor"></span>';
              const tr = $("transcript");
              const atBottom = tr.scrollHeight - tr.scrollTop - tr.clientHeight < 60;
              if (atBottom) tr.scrollTop = tr.scrollHeight;
            }
            if (ch.finish_reason) finish = ch.finish_reason;
          }
        }
      }
    }
    const dt = (performance.now() - t0) / 1000;
    const toks = usage ? usage.completion_tokens : null;
    msg.tokens = toks || 0;
    msg.total = usage ? usage.total_tokens : 0;
    msg.meta = dt.toFixed(2) + "s" + (toks ? " · " + toks + " tok · " +
      (toks / dt).toFixed(1) + " tok/s" : "") + (finish ? " · " + finish : "");
  } catch (e) {
    if (e.name === "AbortError") {
      msg.content += "\n\n[stopped]";
      msg.meta = "stopped by you";
    } else {
      msg.content = "**error** — " + e.message;
      msg.meta = "";
    }
  } finally {
    msg.streaming = false;
    BUSY = false;
    $("send").disabled = false;
    $("stop").hidden = true;
    ABORT = null;
    save(); paint();
  }
}

function regenerate() {
  if (BUSY) return;
  const c = chat();
  const last = c.messages[c.messages.length - 1];
  if (!last || last.role !== "assistant") return;
  c.messages.pop();
  const msg = { role: "assistant", content: "", streaming: true };
  c.messages.push(msg);
  paint();
  run(c, msg);
}

// ---- settings -------------------------------------------------------------
function bindSetting(id) {
  const el = $(id);
  const apply = () => {
    S.settings[id] = id === "system" ? el.value : Number(el.value);
    if (id === "system") S.settings.system = el.value;
    const lab = $("v_" + id);
    if (lab) lab.textContent = el.value;
    save();
  };
  el.addEventListener("input", apply);
  return apply;
}
function syncSettings() {
  for (const k of Object.keys(DEFAULTS)) {
    const el = $(k);
    if (!el) continue;
    el.value = S.settings[k];
    const lab = $("v_" + k);
    if (lab) lab.textContent = el.value;
  }
}

// ---- boot -----------------------------------------------------------------
async function bootMeta() {
  try {
    const h = await (await fetch("/health")).json();
    META = h;
    $("mname").textContent = h.model || "model";
    $("dev").textContent = h.device || "?";
    $("ctx").textContent = h.n_ctx || "?";
    let line = h.arch || "";
    $("mmeta").textContent = line;
    try {
      const m = await (await fetch("/v1/models")).json();
      const meta = m.data && m.data[0] && m.data[0].meta;
      if (meta) {
        META = { ...h, ...meta };
        $("mmeta").textContent = [meta.arch, meta.n_layer + " layers",
          meta.n_embd + " embd", meta.n_head + "/" + meta.n_head_kv + " heads",
          meta.n_vocab + " vocab"].join(" · ");
        if (meta.n_expert) $("mmeta").textContent += " · " + meta.n_expert + " experts";
      }
    } catch (e) { /* /v1/models is optional for the header */ }
    $("ver").textContent = "· " + (h.arch || "");
  } catch (e) {
    $("mname").textContent = "server unreachable";
    $("mname").style.color = "var(--err)";
  }
}

function boot() {
  load();
  document.documentElement.dataset.theme = S.theme;
  syncSettings();
  paintChats();
  paint();
  bootMeta();

  $("composer").addEventListener("submit", (e) => {
    e.preventDefault();
    const el = $("input");
    const text = el.value.trim();
    if (!text || BUSY) return;
    el.value = ""; el.style.height = "auto";
    send(text);
  });
  $("input").addEventListener("input", (e) => {
    const el = e.target;
    el.style.height = "auto";
    el.style.height = Math.min(190, el.scrollHeight) + "px";
  });
  $("input").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      $("composer").requestSubmit();
    }
  });
  $("stop").onclick = () => { if (ABORT) ABORT.abort(); };
  $("newchat").onclick = () => {
    S.active = null;
    chat();
    save(); paintChats(); paint();
    $("input").focus();
  };
  $("settingsBtn").onclick = () => { $("settings").hidden = !$("settings").hidden; };
  $("theme").onclick = () => {
    S.theme = S.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = S.theme;
    save();
  };
  $("export").onclick = () => {
    const blob = new Blob([JSON.stringify(S.chats, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "kraken-chats.json";
    a.click();
    URL.revokeObjectURL(a.href);
    toast("exported");
  };
  Object.keys(DEFAULTS).forEach(bindSetting);
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") $("settings").hidden = true;
  });
}

boot();
</script>
</body>
</html>
)KRKWEB";
    return page;
}

} // namespace krk
