"""Minimal OpenAI-compatible HTTP server for the SSD expert-streaming runtime.

Wraps streaming/chat.py's verified chat machinery (encode_messages ->
generate_turn -> parse_message_from_completion_text) behind two endpoints so
any OpenAI-style chat UI can talk to DeepSeek-V4-Flash:

  POST /v1/chat/completions   {model, messages, temperature, top_p, max_tokens, stream}
                              -> chat.completion JSON, or SSE chunks when stream=true
  GET  /v1/models             -> one entry, "deepseek-v4-flash-streaming"
  GET  /                      -> minimal self-contained chat page (SSE streaming)
  GET  /favicon.ico           -> 204 (log-noise suppression)

Run (model loads once at startup, ~1 s after repack; keep the machine awake):

  caffeinate -i python3 streaming/serve.py --cache-gb 8          # 127.0.0.1:8399

Design decisions:
- One model process, one GPU: generation is strictly serialized on a global
  lock. Concurrent requests QUEUE on that lock (not strictly FIFO); a request
  that cannot start within --queue-timeout seconds (default 600) gets a 429.
- Defaults favor latency: thinking mode OFF (--thinking-mode thinking to
  enable), greedy decoding when temperature is 0/unset, and max_tokens both
  defaults to and is clamped to --max-tokens-cap (512).
- Loopback bind by default (--host 127.0.0.1, --port 8399).
- Streaming reuses chat.generate_turn's echo path verbatim: its stdout deltas
  (already UTF-8-safe re-decode diffs) are captured per-request and forwarded
  as SSE chunks, so serve.py reimplements neither encoding nor generation.

Conversation KV reuse (default on; --no-kv-reuse for the old behavior):
- The server keeps ONE resident conversation slot {token_ids, cache}. After a
  request, the cache already holds KV for prompt+reply, and the next turn of
  the same conversation encodes to a strict token-prefix extension of it
  (verified offline: chat-mode encoding is prefix-stable at the token level;
  thinking mode is NOT — its <think>/</think> glue is rewritten between turns,
  so the guard below simply never matches and every request rebuilds).
- Guard: a request reuses the slot iff its token ids strictly extend the
  slot's ids and the slot cache has capacity; anything else discards the slot
  and prefills fresh. The suffix is fed through the DECODE path one token at a
  time — the port's batched prefill cannot continue from a nonzero cache
  offset (it re-assigns positions from 0 and reseeds window/compressor state).
- The slot is only touched under the generation lock. usage reports
  prompt_tokens_details.cached_tokens; the per-request log line shows
  reused/processed counts.

Limitations (v1):
- One conversation slot: interleaved different conversations rebuild each
  other's cache (each rebuild is just the old full-prefill cost). The expert
  LRU cache and the OS page cache also persist across requests.
- In thinking mode, streamed deltas carry the reasoning inline (dim/ANSI codes
  stripped, reasoning separated from the answer by chat.py's "---" line); the
  non-streaming response separates it properly into "reasoning_content".
- If a streaming client disconnects mid-generation, generation still runs to
  completion before the next request starts (the worker owns the GPU lock).
- Batch size 1; "model" in the request body is accepted and ignored.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import queue
import re
import sys
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "deepseek-v4-mlx"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mlx.core as mx                                     # noqa: E402
from fastapi import FastAPI, HTTPException                # noqa: E402
from fastapi.responses import HTMLResponse, Response, StreamingResponse  # noqa: E402
from pydantic import BaseModel, ConfigDict                # noqa: E402

from chat import generate_turn, make_sampler, parse_turn, stop_ids  # noqa: E402
from deepseek_v4_mlx import encoding_dsv4 as E            # noqa: E402
from deepseek_v4_mlx.cache import make_cache              # noqa: E402
from deepseek_v4_mlx.generate import load_tokenizer       # noqa: E402
from run_streaming import load_streaming                  # noqa: E402

MODEL_ID = "deepseek-v4-flash-streaming"
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

# Populated in main() before uvicorn starts; the model is loaded exactly once.
MODEL = ARGS = STORE = TOK = STOPS = None
CFG: argparse.Namespace = None
GEN_LOCK = threading.Lock()  # one GPU -> strictly one generation at a time

# The resident conversation slot (guarded by GEN_LOCK): token ids whose KV the
# cache already holds, the cache itself, and the cache's max_seq_len capacity.
SLOT: Dict[str, Any] = {"ids": None, "cache": None, "capacity": 0}


def _slot_acquire(ids: List[int], max_new: int):
    """Return (cache, prefix_len) for this request. Caller holds GEN_LOCK.

    Reuse iff the request's ids strictly extend the slot's processed ids and
    the slot cache can hold the finished turn; otherwise discard and rebuild.
    """
    need = len(ids) + max_new + 8
    if not CFG.kv_reuse:
        return None, 0
    if (SLOT["cache"] is not None and SLOT["ids"]
            and len(SLOT["ids"]) < len(ids)
            and need <= SLOT["capacity"]
            and ids[:len(SLOT["ids"])] == SLOT["ids"]):
        return SLOT["cache"], len(SLOT["ids"])
    capacity = max(CFG.kv_slot_len, need)
    SLOT.update(ids=None, cache=make_cache(ARGS, bsz=1, max_seq_len=capacity),
                capacity=capacity)
    return SLOT["cache"], 0


def _slot_commit(ids: List[int], out_ids: List[int], cache):
    """Record what the cache now holds: prompt + generated reply (no stop tok)."""
    if CFG.kv_reuse and cache is SLOT["cache"]:
        SLOT["ids"] = list(ids) + list(out_ids)


def _slot_drop():
    """After a failed generation the cache state is indeterminate — discard."""
    SLOT.update(ids=None, cache=None, capacity=0)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: Optional[str] = None
    messages: List[Dict[str, Any]]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: bool = False


def _content_to_text(content) -> str:
    """Flatten OpenAI content (string or list of parts) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return "" if content is None else str(content)


def _prepare(req: ChatRequest):
    """Validate a request and encode it via chat.py's machinery."""
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages must be non-empty")
    messages = [{"role": m.get("role", "user"),
                 "content": _content_to_text(m.get("content"))}
                for m in req.messages]
    try:
        prompt = E.encode_messages(messages, thinking_mode=CFG.thinking_mode,
                                   reasoning_effort=CFG.reasoning_effort)
    except Exception as err:
        raise HTTPException(status_code=400,
                            detail=f"could not encode messages: {err}")
    ids = TOK(prompt, add_special_tokens=False)["input_ids"]
    temp = req.temperature or 0.0
    sampler = make_sampler(temp, req.top_p if req.top_p is not None else 1.0)
    max_new = min(req.max_tokens or CFG.max_tokens_cap, CFG.max_tokens_cap)
    return ids, sampler, max(1, max_new)


def _finish_reason(stats: dict, max_new: int) -> str:
    return "length" if stats["new_tokens"] >= max_new else "stop"


def _usage(stats: dict) -> dict:
    return {"prompt_tokens": stats["prompt_tokens"],
            "completion_tokens": stats["new_tokens"],
            "total_tokens": stats["prompt_tokens"] + stats["new_tokens"],
            "prompt_tokens_details": {"cached_tokens": stats["reused_tokens"]}}


def _log_request(kind: str, stats: dict):
    print(f"[serve] {kind}: prefill {stats['prefill_s']}s "
          f"(kv reused {stats['reused_tokens']}, processed "
          f"{stats['processed_tokens']} of {stats['prompt_tokens']} tok), "
          f"decode {stats['decode_tok_s']} tok/s "
          f"({stats['new_tokens']} tok) | store {STORE.summary()} | "
          f"peak {mx.get_peak_memory()/1e9:.1f} GB", file=sys.stderr, flush=True)


@contextlib.asynccontextmanager
async def _lifespan(app):
    # Printed once uvicorn has bound the socket and is accepting connections.
    print(f"[serve] READY http://{CFG.host}:{CFG.port} (model={MODEL_ID})",
          file=sys.stderr, flush=True)
    yield


app = FastAPI(title="dsv4-streaming", version="0.1", lifespan=_lifespan)


# --------------------------------------------------------------------- chat page

CHAT_HTML = """<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DeepSeek-V4-Flash (local)</title>
<style>
  * { box-sizing: border-box; margin: 0; }
  body { font: 15px/1.45 -apple-system, system-ui, sans-serif; height: 100vh;
         display: flex; flex-direction: column; background: #f4f4f2; color: #1a1a1a; }
  header { padding: 10px 16px; background: #24292f; color: #eee; display: flex;
           justify-content: space-between; align-items: baseline; }
  header small { color: #999; }
  header button { background: none; border: 1px solid #666; color: #ccc;
                  border-radius: 6px; padding: 2px 10px; cursor: pointer; }
  #log { flex: 1; overflow-y: auto; padding: 16px; }
  .msg { max-width: 46em; margin: 0 auto 10px; padding: 9px 13px;
         border-radius: 10px; white-space: pre-wrap; word-wrap: break-word; }
  .user { background: #d7e3f4; }
  .assistant { background: #fff; border: 1px solid #ddd; }
  .error { background: #fdecea; border: 1px solid #e0b4b4; color: #8b1a1a; }
  footer { display: flex; gap: 8px; padding: 12px 16px; background: #e8e8e5; }
  textarea { flex: 1; resize: none; padding: 8px; border: 1px solid #bbb;
             border-radius: 8px; font: inherit; }
  footer button { padding: 0 18px; border: none; border-radius: 8px;
                  background: #24292f; color: #fff; cursor: pointer; }
  footer button:disabled { opacity: .45; cursor: default; }
</style></head>
<body>
<header><b>DeepSeek-V4-Flash</b> <small>local SSD expert streaming</small>
  <button id="clear">new chat</button></header>
<main id="log"></main>
<footer><textarea id="box" rows="3" autofocus
  placeholder="Message... (Enter to send, Shift+Enter for newline)"></textarea>
  <button id="send">Send</button></footer>
<script>
const log = document.getElementById('log'), box = document.getElementById('box'),
      btn = document.getElementById('send');
let msgs = [], busy = false;

function add(role, text) {
  const d = document.createElement('div');
  d.className = 'msg ' + role;
  d.textContent = text;
  log.appendChild(d);
  log.scrollTop = log.scrollHeight;
  return d;
}

async function send() {
  const text = box.value.trim();
  if (!text || busy) return;
  busy = true; btn.disabled = true; box.value = '';
  msgs.push({role: 'user', content: text});
  add('user', text);
  const d = add('assistant', '\\u2026');   // ellipsis while prefilling
  let acc = '';
  try {
    const r = await fetch('/v1/chat/completions', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({messages: msgs, stream: true})});
    if (!r.ok) throw new Error(r.status + ' ' + (await r.text()).slice(0, 300));
    const reader = r.body.getReader(), dec = new TextDecoder();
    let buf = '';
    for (;;) {
      const {done, value} = await reader.read();
      if (done) break;
      buf += dec.decode(value, {stream: true});
      let i;
      while ((i = buf.indexOf('\\n\\n')) >= 0) {
        const line = buf.slice(0, i).trim();
        buf = buf.slice(i + 2);
        if (!line.startsWith('data: ')) continue;
        const data = line.slice(6);
        if (data === '[DONE]') continue;
        const j = JSON.parse(data);
        if (j.error) throw new Error(j.error.message);
        const delta = j.choices && j.choices[0].delta.content;
        if (delta) { acc += delta; d.textContent = acc; log.scrollTop = log.scrollHeight; }
      }
    }
    msgs.push({role: 'assistant', content: acc});
  } catch (e) {
    d.className = 'msg error';
    d.textContent = 'error: ' + e.message;
    msgs.pop();  // drop the failed user turn so a retry re-sends it cleanly
  }
  busy = false; btn.disabled = false; box.focus();
}

btn.onclick = send;
box.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
});
document.getElementById('clear').onclick = () => { msgs = []; log.innerHTML = ''; box.focus(); };
</script>
</body></html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return CHAT_HTML


@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)


@app.get("/v1/models")
def list_models():
    return {"object": "list",
            "data": [{"id": MODEL_ID, "object": "model",
                      "created": int(time.time()), "owned_by": "local"}]}


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    ids, sampler, max_new = _prepare(req)
    if req.stream:
        return _stream_response(ids, sampler, max_new)
    return _blocking_response(ids, sampler, max_new)


# ---------------------------------------------------------------- non-streaming

def _blocking_response(ids, sampler, max_new):
    if not GEN_LOCK.acquire(timeout=CFG.queue_timeout):
        raise HTTPException(status_code=429,
                            detail="generation busy: queued past --queue-timeout")
    try:
        cache, prefix_len = _slot_acquire(ids, max_new)
        text, hit_eos, stats = generate_turn(
            MODEL, ARGS, TOK, ids, sampler, STOPS,
            max_new=max_new, thinking_mode=CFG.thinking_mode, echo=False,
            cache=cache, prefix_len=prefix_len)
        _slot_commit(ids, stats["out_ids"], cache)
    except HTTPException:
        _slot_drop()
        raise
    except Exception as err:
        _slot_drop()
        raise HTTPException(status_code=500, detail=f"generation failed: {err}")
    finally:
        GEN_LOCK.release()

    msg = parse_turn(text, hit_eos, CFG.thinking_mode)
    _log_request("chat.completion", stats)
    message = {"role": "assistant", "content": msg["content"]}
    if msg.get("reasoning_content"):
        message["reasoning_content"] = msg["reasoning_content"]
    return {"id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": MODEL_ID,
            "choices": [{"index": 0, "message": message,
                         "finish_reason": _finish_reason(stats, max_new)}],
            "usage": _usage(stats)}


# ------------------------------------------------------------------- streaming

class _DeltaWriter:
    """stdout stand-in that forwards chat.generate_turn's echo deltas to a queue.

    generate_turn already emits UTF-8-safe incremental text diffs when echo=True;
    capturing them keeps serve.py from reimplementing the decode/diff logic.
    ANSI dim codes (thinking mode) are stripped.
    """

    def __init__(self, q: queue.Queue):
        self.q = q

    def write(self, s: str):
        s = ANSI_RE.sub("", s)
        if s:
            self.q.put(("delta", s))
        return len(s)

    def flush(self):
        pass


def _gen_worker(q: queue.Queue, ids, sampler, max_new):
    """Owns the GPU lock for the whole generation; always signals start/busy."""
    if not GEN_LOCK.acquire(timeout=CFG.queue_timeout):
        q.put(("busy", None))
        return
    try:
        q.put(("start", None))
        cache, prefix_len = _slot_acquire(ids, max_new)
        with contextlib.redirect_stdout(_DeltaWriter(q)):
            text, hit_eos, stats = generate_turn(
                MODEL, ARGS, TOK, ids, sampler, STOPS,
                max_new=max_new, thinking_mode=CFG.thinking_mode, echo=True,
                cache=cache, prefix_len=prefix_len)
        _slot_commit(ids, stats["out_ids"], cache)
        q.put(("done", (text, hit_eos, stats)))
    except Exception as err:
        import traceback
        traceback.print_exc(file=sys.stderr)
        _slot_drop()
        q.put(("error", f"{type(err).__name__}: {err}"))
    finally:
        GEN_LOCK.release()


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _stream_response(ids, sampler, max_new):
    q: queue.Queue = queue.Queue()
    threading.Thread(target=_gen_worker, args=(q, ids, sampler, max_new),
                     daemon=True).start()
    kind, _ = q.get()  # blocks while queued behind an active generation
    if kind == "busy":
        raise HTTPException(status_code=429,
                            detail="generation busy: queued past --queue-timeout")

    rid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    def chunk(delta: dict, finish=None, usage=None) -> str:
        body = {"id": rid, "object": "chat.completion.chunk", "created": created,
                "model": MODEL_ID,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
        if usage is not None:
            body["usage"] = usage
        return _sse(body)

    def events():
        yield chunk({"role": "assistant", "content": ""})
        pending = None  # one-delta lag so the echo path's final "\n" can be dropped
        while True:
            kind, payload = q.get()
            if kind == "delta":
                if pending is not None:
                    yield chunk({"content": pending})
                pending = payload
            elif kind == "done":
                if pending is not None and pending != "\n":
                    yield chunk({"content": pending})
                _text, _hit_eos, stats = payload
                _log_request("chat.completion.chunk", stats)
                yield chunk({}, finish=_finish_reason(stats, max_new),
                            usage=_usage(stats))
                yield "data: [DONE]\n\n"
                return
            else:  # error
                yield _sse({"error": {"message": payload, "type": "server_error"}})
                yield "data: [DONE]\n\n"
                return

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


# ----------------------------------------------------------------------- main

def main():
    global MODEL, ARGS, STORE, TOK, STOPS, CFG

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repacked", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "repacked"))
    ap.add_argument("--cache-gb", type=float, default=8.0,
                    help="expert LRU budget; 8 GB is the measured optimum")
    ap.add_argument("--store", choices=["lru", "stacked"], default="lru")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8399)
    ap.add_argument("--thinking-mode", choices=["chat", "thinking"],
                    default="chat", help="server-wide; default chat (fast)")
    ap.add_argument("--reasoning-effort", choices=["low", "high", "max"],
                    default=None, help="thinking mode only")
    ap.add_argument("--max-tokens-cap", type=int, default=512,
                    help="default and upper clamp for max_tokens")
    ap.add_argument("--queue-timeout", type=float, default=600.0,
                    help="seconds a request may wait for the GPU before 429")
    ap.add_argument("--no-kv-reuse", dest="kv_reuse", action="store_false",
                    help="disable the conversation KV slot (fresh cache per "
                         "request, the pre-reuse behavior)")
    ap.add_argument("--kv-slot-len", type=int, default=8192,
                    help="minimum max_seq_len for the resident conversation "
                         "cache; longer conversations allocate exactly what "
                         "they need")
    ap.add_argument("--seed", type=int, default=None)
    CFG = ap.parse_args()

    if CFG.seed is not None:
        mx.random.seed(CFG.seed)

    t0 = time.time()
    MODEL, ARGS, STORE = load_streaming(CFG.repacked, CFG.cache_gb, CFG.store)
    TOK = load_tokenizer(CFG.repacked)
    STOPS = stop_ids(TOK)
    print(f"[serve] model loaded in {time.time()-t0:.1f}s | cache {CFG.cache_gb} GB "
          f"| {CFG.thinking_mode} mode | max_tokens cap {CFG.max_tokens_cap} "
          f"| kv reuse {'on' if CFG.kv_reuse else 'off'}",
          file=sys.stderr, flush=True)

    import uvicorn
    uvicorn.run(app, host=CFG.host, port=CFG.port, log_level="info")


if __name__ == "__main__":
    main()
