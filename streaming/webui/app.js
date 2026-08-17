/* DeepSeek-V4-Flash local chat app — vanilla JS, no dependencies. */
"use strict";

const $ = (s, el) => (el || document).querySelector(s);
const $$ = (s, el) => Array.from((el || document).querySelectorAll(s));

const S = {
  config: null,
  sessions: [],
  sid: null,          // current session id
  messages: [],
  generating: false,
  pendingModel: null, // model chosen before a session exists
  fastState: "loading",
};

/* ---------------------------------------------------------------- markdown */

const escapeHtml = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;")
  .replace(/>/g, "&gt;").replace(/"/g, "&quot;");

function mdInline(s) {
  // escape first, then apply span-level rules on the escaped text
  s = escapeHtml(s);
  const codes = [];
  s = s.replace(/`([^`\n]+)`/g, (_, c) => {
    codes.push(c); return `\u0000${codes.length - 1}\u0000`;
  });
  s = s.replace(/\*\*\*([^*]+)\*\*\*/g, "<b><i>$1</i></b>")
       .replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>")
       .replace(/(^|[\s(])\*([^*\s][^*]*)\*/g, "$1<i>$2</i>")
       .replace(/(^|[\s(])_([^_\s][^_]*)_/g, "$1<i>$2</i>")
       .replace(/~~([^~]+)~~/g, "<s>$1</s>")
       .replace(/\[([^\]]+)\]\((https?:[^)\s]+)\)/g,
                '<a href="$2" target="_blank" rel="noopener">$1</a>');
  return s.replace(/\u0000(\d+)\u0000/g, (_, i) => `<code>${codes[+i]}</code>`);
}

function mdRender(src) {
  const out = [];
  const lines = src.replace(/\r\n/g, "\n").split("\n");
  let i = 0, para = [], list = null; // list: {ord, items}

  const flushPara = () => {
    if (para.length) { out.push(`<p>${mdInline(para.join("\n")).replace(/\n/g, "<br>")}</p>`); para = []; }
  };
  const flushList = () => {
    if (list) {
      out.push(`<${list.ord ? "ol" : "ul"}>` +
        list.items.map(it => `<li>${it}</li>`).join("") +
        `</${list.ord ? "ol" : "ul"}>`);
      list = null;
    }
  };

  while (i < lines.length) {
    const line = lines[i];
    const fence = line.match(/^```(\S*)\s*$/);
    if (fence) {
      flushPara(); flushList();
      const lang = fence[1]; const buf = [];
      i++;
      while (i < lines.length && !/^```\s*$/.test(lines[i])) { buf.push(lines[i]); i++; }
      i++; // closing fence (or EOF)
      out.push(
        `<div class="codeblock"><div class="cb-head"><span>${escapeHtml(lang || "code")}</span>` +
        `<button class="code-copy" title="Copy code">` +
        `<svg viewBox="0 0 16 16" width="12" height="12"><rect x="5" y="5" width="8" height="8" rx="1.5" fill="none" stroke="currentColor" stroke-width="1.4"/><path d="M11 5V3.5A1.5 1.5 0 0 0 9.5 2h-6A1.5 1.5 0 0 0 2 3.5v6A1.5 1.5 0 0 0 3.5 11H5" fill="none" stroke="currentColor" stroke-width="1.4"/></svg>` +
        `copy</button></div><pre><code>${escapeHtml(buf.join("\n"))}</code></pre></div>`);
      continue;
    }
    if (/^\s*$/.test(line)) { flushPara(); flushList(); i++; continue; }

    const h = line.match(/^(#{1,4})\s+(.*)$/);
    if (h) { flushPara(); flushList();
      out.push(`<h${h[1].length}>${mdInline(h[2])}</h${h[1].length}>`); i++; continue; }

    if (/^\s*([-*_])\s*\1\s*\1[\s\-*_]*$/.test(line)) {
      flushPara(); flushList(); out.push("<hr>"); i++; continue; }

    const li = line.match(/^\s{0,3}([-*+]|\d+[.)])\s+(.*)$/);
    if (li) {
      flushPara();
      const ord = /\d/.test(li[1][0]);
      if (!list || list.ord !== ord) { flushList(); list = { ord, items: [] }; }
      list.items.push(mdInline(li[2]));
      i++; continue;
    }
    if (list && /^\s{2,}\S/.test(line)) {  // lazy continuation of a list item
      list.items[list.items.length - 1] += "<br>" + mdInline(line.trim());
      i++; continue;
    }

    const bq = line.match(/^>\s?(.*)$/);
    if (bq) {
      flushPara(); flushList();
      const buf = [bq[1]]; i++;
      while (i < lines.length && /^>\s?/.test(lines[i])) { buf.push(lines[i].replace(/^>\s?/, "")); i++; }
      out.push(`<blockquote>${mdRender(buf.join("\n"))}</blockquote>`);
      continue;
    }

    if (/^\s*\|.*\|\s*$/.test(line) && i + 1 < lines.length &&
        /^\s*\|?[\s:|-]+\|?\s*$/.test(lines[i + 1]) && lines[i + 1].includes("-")) {
      flushPara(); flushList();
      const rows = [];
      const parseRow = (l) => l.trim().replace(/^\||\|$/g, "").split("|").map(c => mdInline(c.trim()));
      const head = parseRow(line); i += 2;
      while (i < lines.length && /^\s*\|.*\|\s*$/.test(lines[i])) { rows.push(parseRow(lines[i])); i++; }
      out.push("<table><thead><tr>" + head.map(c => `<th>${c}</th>`).join("") +
        "</tr></thead><tbody>" +
        rows.map(r => "<tr>" + r.map(c => `<td>${c}</td>`).join("") + "</tr>").join("") +
        "</tbody></table>");
      continue;
    }

    para.push(line); i++;
  }
  flushPara(); flushList();
  return out.join("\n");
}

/* -------------------------------------------------------------------- api */

async function api(path, opts = {}) {
  if (opts.body !== undefined && typeof opts.body !== "string") {
    opts.body = JSON.stringify(opts.body);
    opts.headers = Object.assign({ "Content-Type": "application/json" }, opts.headers);
  }
  const r = await fetch(path, opts);
  if (!r.ok) {
    let msg = `${r.status}`;
    try { msg = (await r.json()).detail || msg; } catch (e) { /* keep status */ }
    throw new Error(msg);
  }
  return r.status === 204 ? null : r.json();
}

/* ------------------------------------------------------------ session list */

function fmtWhen(t) {
  const d = new Date(t * 1000), now = new Date();
  const days = Math.floor((now - d) / 864e5);
  if (d.toDateString() === now.toDateString())
    return d.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
  if (days < 7) return d.toLocaleDateString([], { weekday: "short" });
  return d.toLocaleDateString([], { month: "short", day: "numeric" });
}

async function refreshSessions() {
  S.sessions = await api("/api/sessions");
  renderSessions();
}

function renderSessions() {
  const nav = $("#session-list");
  nav.innerHTML = "";
  if (!S.sessions.length) {
    nav.innerHTML = `<div class="sess-empty">No chats yet.<br>Start one below.</div>`;
    return;
  }
  for (const s of S.sessions) {
    const row = document.createElement("div");
    row.className = "sess" + (s.id === S.sid ? " active" : "");
    row.dataset.id = s.id;
    row.innerHTML =
      `<span class="sess-title" title="${escapeHtml(s.title)} · ${fmtWhen(s.updated)}">${escapeHtml(s.title)}</span>` +
      `<span class="sess-tools">` +
      `<button class="t-rename" title="Rename"><svg viewBox="0 0 16 16" width="13" height="13"><path d="m11.1 2.4 2.5 2.5-7.9 7.9-3.1.6.6-3.1 7.9-7.9Z" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linejoin="round"/></svg></button>` +
      `<button class="t-del" title="Delete"><svg viewBox="0 0 16 16" width="13" height="13"><path d="M3 4.5h10M6.5 4V2.8h3V4M5 4.5l.5 8.7h5l.5-8.7" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"/></svg></button>` +
      `</span>`;
    row.addEventListener("click", (e) => {
      if (e.target.closest(".sess-tools") || row.querySelector("input.rename")) return;
      openSession(s.id);
    });
    $(".t-rename", row).addEventListener("click", () => startRename(row, s));
    $(".t-del", row).addEventListener("click", (e) => confirmDelete(e.currentTarget, row, s));
    nav.appendChild(row);
  }
}

function startRename(row, s) {
  const span = $(".sess-title", row);
  const input = document.createElement("input");
  input.className = "rename"; input.value = s.title;
  span.replaceWith(input);
  input.focus(); input.select();
  const commit = async () => {
    const title = input.value.trim();
    if (title && title !== s.title) {
      try { await api(`/api/sessions/${s.id}`, { method: "PATCH", body: { title } }); }
      catch (e) { /* keep old title */ }
    }
    await refreshSessions();
    if (s.id === S.sid) $("#chat-title").textContent = title || s.title;
  };
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") input.blur();
    if (e.key === "Escape") { input.value = s.title; input.blur(); }
  });
  input.addEventListener("blur", commit);
}

function confirmDelete(btn, row, s) {
  if (!btn.classList.contains("confirm-del")) {
    btn.classList.add("confirm-del");
    btn.textContent = "delete?";
    row.classList.add("confirming");
    setTimeout(() => {
      if (document.body.contains(btn)) renderSessions();
    }, 2600);
    return;
  }
  api(`/api/sessions/${s.id}`, { method: "DELETE" }).then(() => {
    if (s.id === S.sid) { S.sid = null; S.messages = []; renderThread(); }
    refreshSessions();
  });
}

/* ----------------------------------------------------------------- thread */

function currentSession() { return S.sessions.find(s => s.id === S.sid) || null; }

function currentModel() {
  const s = currentSession();
  return (s && s.model) || S.pendingModel || (S.config && S.config.default_model) || "dsv4-fast";
}

async function openSession(sid) {
  S.sid = sid;
  S.messages = await api(`/api/sessions/${sid}/messages`);
  renderSessions();
  renderThread();
  renderModelPicker();
  const s = currentSession();
  $("#chat-title").textContent = s ? s.title : "New chat";
  $("#input").focus();
}

function newChat() {
  S.sid = null; S.messages = []; S.pendingModel = currentModel();
  renderSessions(); renderThread(); renderModelPicker();
  $("#chat-title").textContent = "New chat";
  $("#input").focus();
}

function statChips(st) {
  if (!st) return "";
  const chip = (k, v) => v === null || v === undefined ? "" : `<span>${k} <b>${v}</b></span>`;
  const eng = st.engine === "fast" ? "fast · ds4 q2" : "quality · mlx 4/8-bit";
  let html = `<span class="eng-${st.engine}">engine <b>${eng}</b></span>` +
    chip("ttft", st.ttft_s != null ? st.ttft_s + "s" : null) +
    chip("prefill", st.prefill_s != null ? st.prefill_s + "s" : null) +
    chip("decode", st.decode_tok_s != null ? st.decode_tok_s + " tok/s" : null) +
    chip("tokens", (st.tokens_in != null || st.tokens_out != null)
      ? `${st.tokens_in ?? "?"} in / ${st.tokens_out ?? "?"} out` : null) +
    chip("kv reused", st.kv_reused_tokens != null ? st.kv_reused_tokens + " tok" : null) +
    chip("total", st.total_s != null ? st.total_s + "s" : null) +
    chip("peak mem", st.peak_gb != null ? st.peak_gb + " GB" : null);
  if (st.store) {
    const so = st.store;
    html += chip("store", typeof so === "object"
      ? `${Math.round((so.hit_rate || 0) * 100)}% hits · ${so.read_gb ?? "?"} GB read · ${so.cached_gb ?? "?"} GB cache`
      : so);
  }
  if (st.stopped) html += `<span>stopped <b>by user</b></span>`;
  return html;
}

function assistantHtml(m) {
  let html = "";
  if (m.reasoning) {
    html += `<details class="thoughts"><summary>Thoughts</summary>` +
      `<div class="thoughts-body">${escapeHtml(m.reasoning)}</div></details>`;
  }
  html += `<div class="content">${mdRender(m.content || "")}</div>`;
  html += `<div class="dbg">${statChips(m.stats)}</div>`;
  return html;
}

function renderThread() {
  const thread = $("#thread");
  thread.innerHTML = "";
  $("#empty-state").hidden = S.messages.length > 0 || S.generating;
  for (const m of S.messages) {
    const div = document.createElement("div");
    if (m.role === "user") {
      div.className = "turn user";
      div.innerHTML = `<div class="bubble">${escapeHtml(m.content)}</div>`;
    } else {
      div.className = "turn assistant";
      div.innerHTML = assistantHtml(m);
    }
    thread.appendChild(div);
  }
  scrollBottom(true);
}

function scrollBottom(force) {
  const sc = $("#scroller");
  const near = sc.scrollHeight - sc.scrollTop - sc.clientHeight < 160;
  if (force || near) sc.scrollTop = sc.scrollHeight;
}

/* ------------------------------------------------------------- generation */

function setGenerating(on) {
  S.generating = on;
  $("#send-btn").hidden = on;
  $("#stop-btn").hidden = !on;
  updateSendState();
}

function updateSendState() {
  $("#send-btn").disabled = !$("#input").value.trim() || S.generating;
}

async function send(text) {
  const input = $("#input");
  text = (text !== undefined ? text : input.value).trim();
  if (!text || S.generating) return;
  input.value = ""; autosize(); updateSendState();

  try {
    if (!S.sid) {
      const sess = await api("/api/sessions", { method: "POST",
        body: { model: S.pendingModel || currentModel() } });
      S.sid = sess.id; S.pendingModel = null;
      await refreshSessions();
      renderModelPicker();
    }
  } catch (e) { alert("Could not create session: " + e.message); return; }

  S.messages.push({ role: "user", content: text });
  renderThread();
  $("#empty-state").hidden = true;

  // live assistant turn
  const turn = document.createElement("div");
  turn.className = "turn assistant";
  turn.innerHTML =
    `<details class="thoughts" open hidden><summary>Thoughts</summary><div class="thoughts-body"></div></details>` +
    `<div class="content"><div class="pending-note"><span class="dots"><i></i><i></i><i></i></span>` +
    `${currentModel() === "dsv4-quality" ? "prefilling — the quality engine takes a moment" : "thinking"}…</div></div>` +
    `<div class="dbg"></div>`;
  $("#thread").appendChild(turn);
  scrollBottom(true);
  const contentEl = $(".content", turn);
  const thoughtsEl = $(".thoughts", turn);
  const thoughtsBody = $(".thoughts-body", turn);

  setGenerating(true);
  let acc = "", think = "", raf = null, gotFirst = false;
  const paint = () => {
    raf = null;
    contentEl.classList.add("caret");
    contentEl.innerHTML = mdRender(acc);
    scrollBottom(false);
  };
  const queuePaint = () => { if (!raf) raf = requestAnimationFrame(paint); };

  try {
    const r = await fetch(`/api/sessions/${S.sid}/messages`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ content: text }),
    });
    if (!r.ok) {
      let msg = `${r.status}`;
      try { msg = (await r.json()).detail || msg; } catch (e) { /* keep */ }
      throw new Error(msg);
    }
    const reader = r.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf("\n\n")) >= 0) {
        const line = buf.slice(0, idx).trim();
        buf = buf.slice(idx + 2);
        if (!line.startsWith("data: ")) continue;
        const data = line.slice(6);
        if (data === "[DONE]") continue;
        const ev = JSON.parse(data);
        if (ev.type === "thinking") {
          think += ev.text;
          thoughtsEl.hidden = false;
          thoughtsBody.textContent = think;
          if (!gotFirst) { gotFirst = true; contentEl.innerHTML = ""; }
          scrollBottom(false);
        } else if (ev.type === "delta") {
          if (!gotFirst) { gotFirst = true; contentEl.innerHTML = ""; }
          acc += ev.text;
          queuePaint();
        } else if (ev.type === "done") {
          if (raf) cancelAnimationFrame(raf);
          const m = ev.message;
          S.messages.push(m);
          turn.innerHTML = assistantHtml(m);
          await refreshSessions();          // pick up auto-title / ordering
          const s = currentSession();
          if (s) $("#chat-title").textContent = s.title;
        } else if (ev.type === "error") {
          throw new Error(ev.message);
        }
      }
    }
  } catch (e) {
    if (raf) cancelAnimationFrame(raf);
    contentEl.classList.remove("caret");
    if (acc) {
      // keep partial text, add the error under it
      contentEl.innerHTML = mdRender(acc);
      const err = document.createElement("div");
      err.className = "turn error";
      err.innerHTML = `<div class="content">Generation interrupted: ${escapeHtml(e.message)}</div>`;
      turn.after(err);
    } else {
      turn.className = "turn error";
      turn.innerHTML = `<div class="content">${escapeHtml(e.message)}</div>`;
    }
  } finally {
    setGenerating(false);
    scrollBottom(false);
    $("#input").focus();
  }
}

async function stopGen() {
  try { await api("/api/stop", { method: "POST" }); } catch (e) { /* ignore */ }
}

/* ------------------------------------------------------------ model picker */

function renderModelPicker() {
  const box = $("#model-picker");
  if (!S.config) return;
  box.innerHTML = "";
  const cur = currentModel();
  for (const m of S.config.models) {
    const b = document.createElement("button");
    b.className = "model-opt" + (m.id === cur ? " selected" : "");
    b.setAttribute("role", "radio");
    b.setAttribute("aria-checked", m.id === cur);
    const state = m.id === "dsv4-fast" ? S.fastState : "ready";
    b.disabled = state !== "ready";
    b.title = m.tip + (state === "loading" ? " (still loading…)" :
                       state === "error" ? " (unavailable)" : "");
    b.innerHTML = `<b>${m.label}</b><small>${m.sub}</small>`;
    b.addEventListener("click", async () => {
      if (S.sid) {
        try { await api(`/api/sessions/${S.sid}`, { method: "PATCH", body: { model: m.id } }); }
        catch (e) { return; }
        const s = currentSession(); if (s) s.model = m.id;
      } else {
        S.pendingModel = m.id;
      }
      renderModelPicker();
    });
    box.appendChild(b);
  }
}

async function pollFast() {
  try {
    const h = await api("/api/health");
    const st = h.fast === "ready" ? "ready" : (h.fast.startsWith("error") ? "error" : "loading");
    if (st !== S.fastState) { S.fastState = st; renderModelPicker(); }
    if (st === "loading") setTimeout(pollFast, 3000);
  } catch (e) { setTimeout(pollFast, 5000); }
}

/* ---------------------------------------------------------------- modals */

function openModal(html) {
  const bd = $("#modal-backdrop");
  $("#modal").innerHTML = "";
  if (html instanceof Node) $("#modal").appendChild(html);
  else $("#modal").innerHTML = html;
  bd.hidden = false;
  $$("#modal [data-close]").forEach(b => b.addEventListener("click", closeModal));
}
function closeModal() { $("#modal-backdrop").hidden = true; }

function openSettings() {
  const frag = $("#tpl-settings").content.cloneNode(true);
  openModal(frag);
  const t = $("#debug-toggle");
  t.checked = document.body.classList.contains("debug");
  t.addEventListener("change", () => {
    document.body.classList.toggle("debug", t.checked);
    localStorage.setItem("dsv4.debug", t.checked ? "1" : "");
  });
}

function openInfo() {
  const inf = (S.config && S.config.info) || {};
  const engines = (inf.engines || []).map(e =>
    `<div class="spec-row"><b>${escapeHtml(e.name)}</b><small>${escapeHtml(e.detail)}</small></div>`).join("");
  const heads = (inf.headlines || []).map(h => `<li>${escapeHtml(h)}</li>`).join("");
  openModal(`
    <h3>About this rig</h3>
    <p><b>${escapeHtml(inf.model || "")}</b><br>${escapeHtml(inf.machine || "")}</p>
    <h4>Engines</h4>${engines}
    <h4>Headlines</h4><ul>${heads}</ul>
    <h4>Scope</h4><p>${escapeHtml(inf.out_of_scope || "")}</p>
    <p><a href="${inf.repo}" target="_blank" rel="noopener">${inf.repo}</a></p>
    <div class="modal-actions"><button class="ghost-btn" data-close>Close</button></div>`);
}

/* ---------------------------------------------------------------- composer */

function autosize() {
  const t = $("#input");
  t.style.height = "auto";
  t.style.height = Math.min(t.scrollHeight, 200) + "px";
}

/* ------------------------------------------------------------------- init */

function wire() {
  $("#new-chat").addEventListener("click", newChat);
  $("#send-btn").addEventListener("click", () => send());
  $("#stop-btn").addEventListener("click", stopGen);
  $("#settings-btn").addEventListener("click", openSettings);
  $("#info-btn").addEventListener("click", openInfo);
  $("#sb-toggle").addEventListener("click", () =>
    $("#app").classList.toggle("sb-hidden"));
  $("#modal-backdrop").addEventListener("click", (e) => {
    if (e.target === e.currentTarget) closeModal();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeModal();
  });

  const input = $("#input");
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
  });
  input.addEventListener("input", () => { autosize(); updateSendState(); });

  $$(".suggest").forEach(b => b.addEventListener("click", () => send(b.textContent)));

  // copy buttons for code blocks (delegated)
  document.addEventListener("click", (e) => {
    const btn = e.target.closest(".code-copy");
    if (!btn) return;
    const code = $("code", btn.closest(".codeblock"));
    navigator.clipboard.writeText(code.textContent).then(() => {
      btn.classList.add("ok");
      const old = btn.innerHTML;
      btn.textContent = "copied";
      setTimeout(() => { btn.classList.remove("ok"); btn.innerHTML = old; }, 1400);
    });
  });
}

async function init() {
  if (localStorage.getItem("dsv4.debug")) document.body.classList.add("debug");
  wire();
  try {
    S.config = await api("/api/config");
  } catch (e) {
    $("#thread").innerHTML =
      `<div class="turn error"><div class="content">Cannot reach the server: ${escapeHtml(e.message)}</div></div>`;
    return;
  }
  const fast = S.config.models.find(m => m.id === "dsv4-fast");
  S.fastState = fast ? fast.available : "error";
  renderModelPicker();
  if (S.fastState === "loading") pollFast();
  await refreshSessions();
  if (S.sessions.length) await openSession(S.sessions[0].id);
  else { $("#empty-state").hidden = false; }
  updateSendState();
  autosize();
}

init();
