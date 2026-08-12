"""Minimal OpenAI-compatible HTTP server for the SSD expert-streaming runtime.

Wraps streaming/chat.py's verified chat machinery (encode_messages ->
generate_turn -> parse_message_from_completion_text) behind two endpoints so
any OpenAI-style chat UI can talk to DeepSeek-V4-Flash:

  POST /v1/chat/completions   {model, messages, temperature, top_p, max_tokens, stream}
                              -> chat.completion JSON, or SSE chunks when stream=true
  GET  /v1/models             -> one entry, "deepseek-v4-flash-streaming"

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

Limitations (v1):
- Fresh KV cache per request: no prefix/KV reuse, so every request pays full
  prefill over the entire conversation it posts. The expert LRU cache and the
  OS page cache DO persist across requests (the process stays resident), which
  is where warm-request speedups come from.
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
from fastapi.responses import StreamingResponse           # noqa: E402
from pydantic import BaseModel, ConfigDict                # noqa: E402

from chat import generate_turn, make_sampler, parse_turn, stop_ids  # noqa: E402
from deepseek_v4_mlx import encoding_dsv4 as E            # noqa: E402
from deepseek_v4_mlx.generate import load_tokenizer       # noqa: E402
from run_streaming import load_streaming                  # noqa: E402

MODEL_ID = "deepseek-v4-flash-streaming"
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

# Populated in main() before uvicorn starts; the model is loaded exactly once.
MODEL = ARGS = STORE = TOK = STOPS = None
CFG: argparse.Namespace = None
GEN_LOCK = threading.Lock()  # one GPU -> strictly one generation at a time


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
            "total_tokens": stats["prompt_tokens"] + stats["new_tokens"]}


def _log_request(kind: str, stats: dict):
    print(f"[serve] {kind}: prefill {stats['prefill_s']}s "
          f"({stats['prompt_tokens']} tok), decode {stats['decode_tok_s']} tok/s "
          f"({stats['new_tokens']} tok) | store {STORE.summary()} | "
          f"peak {mx.get_peak_memory()/1e9:.1f} GB", file=sys.stderr, flush=True)


@contextlib.asynccontextmanager
async def _lifespan(app):
    # Printed once uvicorn has bound the socket and is accepting connections.
    print(f"[serve] READY http://{CFG.host}:{CFG.port} (model={MODEL_ID})",
          file=sys.stderr, flush=True)
    yield


app = FastAPI(title="dsv4-streaming", version="0.1", lifespan=_lifespan)


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
        text, hit_eos, stats = generate_turn(
            MODEL, ARGS, TOK, ids, sampler, STOPS,
            max_new=max_new, thinking_mode=CFG.thinking_mode, echo=False)
    except HTTPException:
        raise
    except Exception as err:
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
        with contextlib.redirect_stdout(_DeltaWriter(q)):
            text, hit_eos, stats = generate_turn(
                MODEL, ARGS, TOK, ids, sampler, STOPS,
                max_new=max_new, thinking_mode=CFG.thinking_mode, echo=True)
        q.put(("done", (text, hit_eos, stats)))
    except Exception as err:
        import traceback
        traceback.print_exc(file=sys.stderr)
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
    ap.add_argument("--seed", type=int, default=None)
    CFG = ap.parse_args()

    if CFG.seed is not None:
        mx.random.seed(CFG.seed)

    t0 = time.time()
    MODEL, ARGS, STORE = load_streaming(CFG.repacked, CFG.cache_gb, CFG.store)
    TOK = load_tokenizer(CFG.repacked)
    STOPS = stop_ids(TOK)
    print(f"[serve] model loaded in {time.time()-t0:.1f}s | cache {CFG.cache_gb} GB "
          f"| {CFG.thinking_mode} mode | max_tokens cap {CFG.max_tokens_cap}",
          file=sys.stderr, flush=True)

    import uvicorn
    uvicorn.run(app, host=CFG.host, port=CFG.port, log_level="info")


if __name__ == "__main__":
    main()
