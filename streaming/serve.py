"""Local chat app + OpenAI-compatible server for DeepSeek-V4-Flash, two engines.

Engines (one GLOBAL generation lock across both — they contend for disk/page
cache; memory coexistence is fine, ~5 GB + ~18 GB):

  quality  in-process MLX SSD-expert-streaming stack (mixed 4/8-bit checkpoint,
           ~1-2 tok/s decode). ALL MLX work stays on ONE persistent generation
           thread (MLX lazy arrays are thread-affine; see below).
  fast     antirez's ds4 runtime (q2 GGUF, ~11 tok/s decode) via its own
           ds4-server binary, spawned as a persistent child process on a
           loopback port and proxied. ds4-server is OpenAI-compatible, keeps
           the model resident between requests, and reuses KV prefixes across
           turns (usage.prompt_tokens_details.cached_tokens). It MUST run with
           cwd=ds4-runtime because it loads metal/*.metal relative to CWD.

Endpoints:

  GET  /                        chat app (streaming/webui/, vanilla JS, no CDN)
  GET  /static/*                app assets
  REST /api/sessions...         server-side session persistence (sqlite3 at
                                data/chat.db): list/create/rename/delete,
                                messages, SSE reply streaming, stop.
  GET  /api/config              engine/UI metadata for the app
  POST /v1/chat/completions     OpenAI-compatible; model routes the engine:
                                "dsv4-fast" -> ds4, "dsv4-quality" (default,
                                also the legacy id) -> streaming stack
  GET  /v1/models               both models

Run (quality model loads in ~1 s; ds4-server child becomes ready ~10-60 s
later in the background; keep the machine awake):

  caffeinate -i python3 streaming/serve.py          # http://127.0.0.1:8399

Design decisions kept from v1:
- ALL MLX work (model load, seeding, prefill, decode) happens on ONE persistent
  generation thread started before uvicorn. MLX streams are thread-local: an
  array whose lazy graph nodes were recorded on thread A cannot be evaluated
  from thread B once A is gone ("There is no Stream(gpu, N) in current
  thread" -> uncaught C++ exception -> abort). The resident KV slot holds
  exactly such lazily-updated arrays between requests. Endpoints enqueue jobs
  and wait on a per-request queue; a request that cannot start within
  --queue-timeout seconds gets a 429.
- Conversation KV reuse (quality): one resident slot {token_ids, cache}; a
  request reuses it iff its ids strictly extend the slot's. The suffix goes
  through the decode path one token at a time (the port's batched prefill
  cannot continue from a nonzero offset). Thinking mode is prefix-unstable, so
  its guard never matches and every thinking request rebuilds.
- Defaults favor latency: thinking mode OFF, greedy decoding, 512-token cap.

Limitations:
- One quality KV slot: interleaved quality conversations rebuild each other's
  cache. ds4-server has its own multi-prefix KV store and does not suffer this.
- Stopping a fast generation closes the proxied connection; ds4-server detects
  client disconnect and aborts.
- Batch size 1 per engine; the global lock serializes across engines.
"""

from __future__ import annotations

import argparse
import atexit
import contextlib
import glob as globmod
import http.client
import json
import os
import queue
import re
import sqlite3
import subprocess
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
from fastapi.responses import (FileResponse, Response,    # noqa: E402
                               StreamingResponse)
from fastapi.staticfiles import StaticFiles               # noqa: E402
from pydantic import BaseModel, ConfigDict                # noqa: E402

from chat import generate_turn, make_sampler, parse_turn, stop_ids  # noqa: E402
from deepseek_v4_mlx import encoding_dsv4 as E            # noqa: E402
from deepseek_v4_mlx.cache import make_cache              # noqa: E402
from deepseek_v4_mlx.generate import load_tokenizer       # noqa: E402
from run_streaming import load_streaming                  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEBUI_DIR = os.path.join(ROOT, "streaming", "webui")
DS4_DIR = os.path.join(ROOT, "ds4-runtime")

MODEL_QUALITY = "dsv4-quality"
MODEL_FAST = "dsv4-fast"
LEGACY_ID = "deepseek-v4-flash-streaming"   # v1 clients
ENGINE_BY_MODEL = {MODEL_FAST: "fast", MODEL_QUALITY: "quality",
                   LEGACY_ID: "quality", "deepseek-v4-flash": "fast"}
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

# Populated by the generation worker thread at startup; loaded exactly once.
MODEL = ARGS = STORE = TOK = STOPS = None
CFG: argparse.Namespace = None

# ONE generation at a time across BOTH engines (disk/page-cache contention).
GEN_LOCK = threading.Lock()

# The single persistent MLX generation thread: endpoints enqueue _Job objects
# here and wait on job.out. Only this worker ever touches MLX.
JOBS: "queue.Queue[_Job]" = queue.Queue()
WORKER_READY = threading.Event()
WORKER_LOAD_ERROR: Optional[str] = None

# The resident conversation slot (touched only by the worker thread).
SLOT: Dict[str, Any] = {"ids": None, "cache": None, "capacity": 0}

# The currently running generation's cancel hook (set by whichever engine path
# is generating; /api/stop calls it). Best-effort, guarded by the GIL.
CURRENT: Dict[str, Any] = {"cancel": None}

# ds4-server child state.
FAST = {"proc": None, "ready": False, "error": None, "started_at": None}

DB_LOCK = threading.Lock()
DB: Optional[sqlite3.Connection] = None


# ------------------------------------------------------------------ database

def _db_init():
    global DB
    os.makedirs(os.path.dirname(CFG.db), exist_ok=True)
    DB = sqlite3.connect(CFG.db, check_same_thread=False)
    DB.row_factory = sqlite3.Row
    DB.execute("PRAGMA foreign_keys=ON")
    DB.execute("PRAGMA journal_mode=WAL")
    DB.executescript("""
      CREATE TABLE IF NOT EXISTS sessions(
        id      TEXT PRIMARY KEY,
        title   TEXT NOT NULL DEFAULT 'New chat',
        model   TEXT NOT NULL DEFAULT 'dsv4-fast',
        created REAL NOT NULL,
        updated REAL NOT NULL);
      CREATE TABLE IF NOT EXISTS messages(
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
        role       TEXT NOT NULL,
        content    TEXT NOT NULL,
        reasoning  TEXT NOT NULL DEFAULT '',
        engine     TEXT NOT NULL DEFAULT '',
        stats      TEXT NOT NULL DEFAULT '',
        created    REAL NOT NULL);
      CREATE INDEX IF NOT EXISTS idx_messages_session
        ON messages(session_id, id);
    """)
    DB.commit()


def _q(sql: str, params=(), one=False, write=False):
    with DB_LOCK:
        cur = DB.execute(sql, params)
        if write:
            DB.commit()
            return cur.lastrowid
        rows = cur.fetchall()
    if one:
        return dict(rows[0]) if rows else None
    return [dict(r) for r in rows]


def _session_or_404(sid: str) -> dict:
    row = _q("SELECT * FROM sessions WHERE id=?", (sid,), one=True)
    if row is None:
        raise HTTPException(status_code=404, detail="no such session")
    return row


def _msg_row(mid: int) -> dict:
    m = _q("SELECT * FROM messages WHERE id=?", (mid,), one=True)
    if m and m["stats"]:
        m["stats"] = json.loads(m["stats"])
    else:
        m["stats"] = None
    return m


# ------------------------------------------------------------- fast (ds4) engine

def _fast_model_path() -> Optional[str]:
    hits = globmod.glob(os.path.expanduser(CFG.fast_model))
    return hits[0] if hits else None


def _fast_watchdog(proc, logpath):
    """Poll the child's /v1/models until it answers, then mark ready."""
    deadline = time.time() + CFG.fast_startup_timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            FAST["error"] = (f"ds4-server exited with code {proc.returncode} "
                             f"during startup (see {logpath})")
            print(f"[serve] {FAST['error']}", file=sys.stderr, flush=True)
            return
        try:
            c = http.client.HTTPConnection("127.0.0.1", CFG.fast_port, timeout=2)
            c.request("GET", "/v1/models")
            if c.getresponse().status == 200:
                FAST["ready"] = True
                print(f"[serve] fast engine READY (ds4-server pid {proc.pid}, "
                      f"port {CFG.fast_port}, "
                      f"{time.time()-FAST['started_at']:.1f}s to ready)",
                      file=sys.stderr, flush=True)
                return
        except OSError:
            pass
        finally:
            with contextlib.suppress(Exception):
                c.close()
        time.sleep(0.5)
    FAST["error"] = f"ds4-server not ready after {CFG.fast_startup_timeout}s"
    print(f"[serve] {FAST['error']}", file=sys.stderr, flush=True)


def _fast_start():
    """Spawn ds4-server as a persistent child (cwd=ds4-runtime: metal/*.metal
    is loaded relative to CWD). Readiness is polled in a background thread."""
    if CFG.no_fast:
        FAST["error"] = "disabled with --no-fast"
        return
    binpath = os.path.join(DS4_DIR, "ds4-server")
    model = _fast_model_path()
    if not os.path.exists(binpath):
        FAST["error"] = f"missing binary {binpath} (make ds4-server)"
        print(f"[serve] fast engine unavailable: {FAST['error']}",
              file=sys.stderr, flush=True)
        return
    if model is None:
        FAST["error"] = f"no GGUF matches {CFG.fast_model}"
        print(f"[serve] fast engine unavailable: {FAST['error']}",
              file=sys.stderr, flush=True)
        return
    logpath = os.path.join(ROOT, "logs", "ds4_server_child.log")
    logf = open(logpath, "ab", buffering=0)
    cmd = [binpath, "-m", model, "--metal", "--ssd-streaming",
           "--ssd-streaming-cache-experts", CFG.fast_cache,
           "--host", "127.0.0.1", "--port", str(CFG.fast_port),
           "-c", str(CFG.fast_ctx), "-n", str(CFG.max_tokens_cap)]
    logf.write(f"\n=== spawn {time.strftime('%F %T')}: {' '.join(cmd)}\n"
               .encode())
    proc = subprocess.Popen(cmd, cwd=DS4_DIR, stdout=logf, stderr=logf,
                            stdin=subprocess.DEVNULL)
    FAST.update(proc=proc, started_at=time.time())
    print(f"[serve] spawned ds4-server pid {proc.pid} (log: {logpath})",
          file=sys.stderr, flush=True)
    atexit.register(_fast_stop)
    threading.Thread(target=_fast_watchdog, args=(proc, logpath),
                     daemon=True, name="ds4-watchdog").start()


def _fast_stop():
    proc = FAST.get("proc")
    if proc is not None and proc.poll() is None:
        proc.terminate()
        with contextlib.suppress(Exception):
            proc.wait(timeout=10)
        if proc.poll() is None:
            proc.kill()


def _fast_require():
    if FAST["ready"]:
        return
    if FAST["error"]:
        raise HTTPException(status_code=503,
                            detail=f"fast engine unavailable: {FAST['error']}")
    raise HTTPException(status_code=503,
                        detail="fast engine still loading, try again shortly")


def _fast_events(payload: dict):
    """Proxy one streaming request to ds4-server.

    Yields ("delta", text) | ("done", {finish_reason, usage}) | ("error", msg)
    | ("stopped", None). Cancellation: CURRENT["cancel"] shuts the socket down
    from another thread; ds4-server sees the disconnect and aborts generation.
    """
    stop_flag = {"stopped": False, "sock": None}
    conn = http.client.HTTPConnection("127.0.0.1", CFG.fast_port, timeout=600)

    def cancel():
        stop_flag["stopped"] = True
        # ds4-server answers SSE with Connection: close, so http.client
        # detaches conn.sock right after getresponse(); the live socket is the
        # response's SocketIO. shutdown(2) unblocks the reader immediately and
        # ds4-server aborts generation on the disconnect.
        sock = stop_flag["sock"] if stop_flag["sock"] is not None else conn.sock
        with contextlib.suppress(Exception):
            sock.shutdown(2)
        with contextlib.suppress(Exception):
            conn.close()

    CURRENT["cancel"] = cancel
    finish, usage = None, None
    try:
        payload = dict(payload, stream=True,
                       stream_options={"include_usage": True})
        conn.request("POST", "/v1/chat/completions", body=json.dumps(payload),
                     headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        with contextlib.suppress(Exception):
            stop_flag["sock"] = resp.fp.raw._sock
        if resp.status != 200:
            body = resp.read(2000).decode("utf-8", "replace")
            yield ("error", f"ds4-server HTTP {resp.status}: {body}")
            return
        while True:
            line = resp.readline()
            if not line:
                if stop_flag["stopped"]:
                    yield ("stopped", None)
                    return
                break
            line = line.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            try:
                j = json.loads(data)
            except ValueError:
                continue
            if j.get("error"):
                yield ("error", str(j["error"].get("message", j["error"])))
                return
            if j.get("usage"):
                usage = j["usage"]
            for ch in j.get("choices", []):
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
                delta = (ch.get("delta") or {}).get("content")
                if delta:
                    yield ("delta", delta)
        yield ("done", {"finish_reason": finish or "stop", "usage": usage})
    except OSError as err:
        if stop_flag["stopped"]:
            yield ("stopped", None)
        else:
            yield ("error", f"ds4-server connection failed: {err}")
    finally:
        CURRENT["cancel"] = None
        with contextlib.suppress(Exception):
            conn.close()


def _fast_usage_stats(usage: Optional[dict], t0, t_first, t_end,
                      n_deltas: int, n_chars: int) -> dict:
    """Debug stats for a fast turn, from ds4 usage + proxy-side wall clocks."""
    tokens_in = tokens_out = cached = None
    if usage:
        tokens_in = usage.get("prompt_tokens")
        tokens_out = usage.get("completion_tokens")
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    if tokens_out is None:
        tokens_out = n_deltas  # ds4 streams ~one chunk per token
    ttft = round(t_first - t0, 2) if t_first else None
    decode_s = (t_end - t_first) if t_first else 0.0
    return {"engine": "fast", "model": MODEL_FAST,
            "ttft_s": ttft, "total_s": round(t_end - t0, 2),
            "prefill_s": ttft,  # ds4 first token lands right after prefill
            "decode_tok_s": round(tokens_out / decode_s, 2)
            if decode_s > 0.05 and tokens_out else None,
            "tokens_in": tokens_in, "tokens_out": tokens_out,
            "kv_reused_tokens": cached, "chars_out": n_chars}


# ------------------------------------------------------- quality engine (MLX)

def _slot_acquire(ids: List[int], max_new: int):
    """Return (cache, prefix_len) for this request. Worker thread only."""
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


class _Job:
    """One queued quality-engine generation request and its result channel."""

    __slots__ = ("ids", "sampler", "max_new", "stream", "out", "lock",
                 "started", "cancelled", "stop_event")

    def __init__(self, ids, sampler, max_new, stream: bool):
        self.ids, self.sampler, self.max_new = ids, sampler, max_new
        self.stream = stream
        self.out: queue.Queue = queue.Queue()
        self.lock = threading.Lock()  # arbitrates start vs queue-timeout cancel
        self.started = False
        self.cancelled = False
        self.stop_event = threading.Event()


def _worker_main():
    """The persistent MLX generation thread: loads the model, then serves jobs.

    Every MLX touch lives here — load, seeding, slot caches, generation — so no
    lazy array ever crosses threads. A per-request failure returns an error to
    that request and drops the KV slot; it never kills the worker or process.
    Holds GEN_LOCK for the duration of each job (both engines share it).
    """
    global MODEL, ARGS, STORE, TOK, STOPS, WORKER_LOAD_ERROR
    try:
        if CFG.seed is not None:
            mx.random.seed(CFG.seed)  # lazy key array -> must be made here
        t0 = time.time()
        MODEL, ARGS, STORE = load_streaming(CFG.repacked, CFG.cache_gb, CFG.store)
        TOK = load_tokenizer(CFG.repacked)
        STOPS = stop_ids(TOK)
        print(f"[serve] quality model loaded in {time.time()-t0:.1f}s | "
              f"cache {CFG.cache_gb} GB | {CFG.thinking_mode} mode | "
              f"max_tokens cap {CFG.max_tokens_cap} | "
              f"kv reuse {'on' if CFG.kv_reuse else 'off'}",
              file=sys.stderr, flush=True)
    except Exception as err:
        WORKER_LOAD_ERROR = f"{type(err).__name__}: {err}"
        WORKER_READY.set()
        return
    WORKER_READY.set()

    while True:
        job = JOBS.get()
        GEN_LOCK.acquire()  # released in finally; fast engine shares this lock
        try:
            with job.lock:
                if job.cancelled:  # requester gave up (queue timeout) — skip
                    continue
                job.started = True
            CURRENT["cancel"] = job.stop_event.set
            job.out.put(("start", None))
            try:
                cache, prefix_len = _slot_acquire(job.ids, job.max_new)
                if job.stream:
                    with contextlib.redirect_stdout(_DeltaWriter(job.out)):
                        text, hit_eos, stats = generate_turn(
                            MODEL, ARGS, TOK, job.ids, job.sampler, STOPS,
                            max_new=job.max_new, thinking_mode=CFG.thinking_mode,
                            echo=True, cache=cache, prefix_len=prefix_len,
                            stop_event=job.stop_event)
                else:
                    text, hit_eos, stats = generate_turn(
                        MODEL, ARGS, TOK, job.ids, job.sampler, STOPS,
                        max_new=job.max_new, thinking_mode=CFG.thinking_mode,
                        echo=False, cache=cache, prefix_len=prefix_len,
                        stop_event=job.stop_event)
                _slot_commit(job.ids, stats["out_ids"], cache)
                job.out.put(("done", (text, hit_eos, stats)))
            except Exception as err:
                import traceback
                traceback.print_exc(file=sys.stderr)
                _slot_drop()
                job.out.put(("error", f"{type(err).__name__}: {err}"))
        finally:
            CURRENT["cancel"] = None
            GEN_LOCK.release()


def _submit(ids, sampler, max_new, stream: bool) -> _Job:
    """Enqueue a job and wait for the worker to start it (429 on timeout)."""
    job = _Job(ids, sampler, max_new, stream)
    JOBS.put(job)
    try:
        job.out.get(timeout=CFG.queue_timeout)  # always ("start", None)
    except queue.Empty:
        with job.lock:
            if not job.started:
                job.cancelled = True
                raise HTTPException(
                    status_code=429,
                    detail="generation busy: queued past --queue-timeout")
        job.out.get()  # started at the same instant we timed out — proceed
    return job


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


def _quality_stats(stats: dict, ttft_s: float, total_s: float) -> dict:
    return {"engine": "quality", "model": MODEL_QUALITY,
            "ttft_s": round(ttft_s, 2), "total_s": round(total_s, 2),
            "prefill_s": stats["prefill_s"],
            "decode_tok_s": stats["decode_tok_s"],
            "tokens_in": stats["prompt_tokens"],
            "tokens_out": stats["new_tokens"],
            "kv_reused_tokens": stats["reused_tokens"],
            "processed_tokens": stats["processed_tokens"],
            "store": STORE.summary(),
            "peak_gb": round(mx.get_peak_memory() / 1e9, 1),
            "stopped": stats.get("stopped", False)}


def _prepare_quality(messages: List[dict], temperature, top_p, max_tokens):
    """Encode chat messages for the quality engine via chat.py's machinery."""
    try:
        prompt = E.encode_messages(messages, thinking_mode=CFG.thinking_mode,
                                   reasoning_effort=CFG.reasoning_effort)
    except Exception as err:
        raise HTTPException(status_code=400,
                            detail=f"could not encode messages: {err}")
    ids = TOK(prompt, add_special_tokens=False)["input_ids"]
    sampler = make_sampler(temperature or 0.0,
                           top_p if top_p is not None else 1.0)
    max_new = min(max_tokens or CFG.max_tokens_cap, CFG.max_tokens_cap)
    return ids, sampler, max(1, max_new)


# ------------------------------------------------------------------ REST: app

class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: Optional[str] = None
    messages: List[Dict[str, Any]]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: bool = False
    think: Optional[bool] = None    # dsv4-fast only; default False (see below)


class SessionCreate(BaseModel):
    model_config = ConfigDict(extra="ignore")
    model: Optional[str] = None


class SessionPatch(BaseModel):
    model_config = ConfigDict(extra="ignore")
    title: Optional[str] = None
    model: Optional[str] = None


class MessagePost(BaseModel):
    model_config = ConfigDict(extra="ignore")
    content: str
    model: Optional[str] = None       # override session model for this turn
    max_tokens: Optional[int] = None


@contextlib.asynccontextmanager
async def _lifespan(app):
    print(f"[serve] READY http://{CFG.host}:{CFG.port} "
          f"(models: {MODEL_FAST}, {MODEL_QUALITY})",
          file=sys.stderr, flush=True)
    yield
    _fast_stop()


app = FastAPI(title="dsv4-streaming", version="0.2", lifespan=_lifespan)


@app.get("/")
def index():
    return FileResponse(os.path.join(WEBUI_DIR, "index.html"),
                        headers={"Cache-Control": "no-cache"})


@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)


@app.get("/api/config")
def api_config():
    fast_state = ("ready" if FAST["ready"] else
                  ("error" if FAST["error"] else "loading"))
    return {
        "models": [
            {"id": MODEL_FAST, "label": "Fast", "sub": "11 tok/s · q2",
             "tip": "ds4 (antirez) compiled runtime, 2-bit experts: much "
                    "faster, slightly rougher answers.",
             "available": fast_state},
            {"id": MODEL_QUALITY, "label": "Quality", "sub": "1-2 tok/s · 4/8-bit",
             "tip": "This repo's MLX streaming stack, mixed 4/8-bit experts: "
                    "best output quality, slow.",
             "available": "ready"},
        ],
        "default_model": CFG.default_model,
        "thinking_mode": CFG.thinking_mode,
        "voice": False,
        "info": {
            "repo": "https://github.com/kk-r/dsv4-streaming",
            "machine": "MacBook Pro, M5 Pro, 64 GB unified memory",
            "model": "DeepSeek-V4-Flash-0731 — 284B-parameter MoE "
                     "(~13.8B active, 43 layers, 256 experts top-6)",
            "engines": [
                {"name": "Fast — ds4 (DwarfStar, antirez)",
                 "detail": "C/Metal runtime, q2 GGUF (IQ2_XXS/Q2_K experts, "
                           "81 GB file), SSD expert streaming, 8 GB expert "
                           "cache. Measured on this machine: ~11.4 tok/s "
                           "decode, ~5 GB RSS, no long-context collapse."},
                {"name": "Quality — dsv4-streaming (this repo)",
                 "detail": "Python/MLX SSD expert streaming, mixed 4/8-bit "
                           "checkpoint (156 GB experts on SSD, ~9.4 GB "
                           "resident), 8 GB LRU. ~1-2 tok/s decode, "
                           "~18 GB peak. Perplexity through the streaming "
                           "path: 6.1250 vs 6.1262 published (-0.02%) — "
                           "numerically transparent."},
            ],
            "headlines": [
                "Model is ~2.4-4x bigger than RAM; experts stream from SSD "
                "on demand in both engines.",
                "8 GB expert cache beats 32 GB on this machine — in both "
                "engines (small-cache law).",
                "Full-corpus wikitext-2 perplexity through the streaming "
                "path matches the published number to 0.02%.",
            ],
            "out_of_scope": "Images: the model is text-only. Voice input "
                            "needs whisper.cpp (not installed) and "
                            "text-to-speech is out of scope for a local "
                            "research rig.",
        },
    }


@app.get("/api/health")
def api_health():
    return {"quality": "ready", "fast": "ready" if FAST["ready"] else
            ("error: " + FAST["error"] if FAST["error"] else "loading"),
            "busy": GEN_LOCK.locked()}


# --------------------------------------------------------------- sessions CRUD

@app.get("/api/sessions")
def sessions_list():
    return _q("""SELECT s.*, COUNT(m.id) AS n_messages
                 FROM sessions s LEFT JOIN messages m ON m.session_id = s.id
                 GROUP BY s.id ORDER BY s.updated DESC""")


@app.post("/api/sessions")
def sessions_create(body: SessionCreate):
    model = body.model or CFG.default_model
    if model not in (MODEL_FAST, MODEL_QUALITY):
        raise HTTPException(status_code=400, detail=f"unknown model {model!r}")
    sid = uuid.uuid4().hex[:12]
    now = time.time()
    _q("INSERT INTO sessions(id, title, model, created, updated) "
       "VALUES(?,?,?,?,?)", (sid, "New chat", model, now, now), write=True)
    return _q("SELECT * FROM sessions WHERE id=?", (sid,), one=True)


@app.patch("/api/sessions/{sid}")
def sessions_patch(sid: str, body: SessionPatch):
    _session_or_404(sid)
    if body.model is not None and body.model not in (MODEL_FAST, MODEL_QUALITY):
        raise HTTPException(status_code=400,
                            detail=f"unknown model {body.model!r}")
    if body.title is not None:
        _q("UPDATE sessions SET title=?, updated=? WHERE id=?",
           (body.title.strip()[:120] or "New chat", time.time(), sid),
           write=True)
    if body.model is not None:
        _q("UPDATE sessions SET model=?, updated=? WHERE id=?",
           (body.model, time.time(), sid), write=True)
    return _q("SELECT * FROM sessions WHERE id=?", (sid,), one=True)


@app.delete("/api/sessions/{sid}")
def sessions_delete(sid: str):
    _session_or_404(sid)
    _q("DELETE FROM sessions WHERE id=?", (sid,), write=True)
    return {"deleted": sid}


@app.get("/api/sessions/{sid}/messages")
def messages_list(sid: str):
    _session_or_404(sid)
    rows = _q("SELECT * FROM messages WHERE session_id=? ORDER BY id", (sid,))
    for r in rows:
        r["stats"] = json.loads(r["stats"]) if r["stats"] else None
    return rows


@app.post("/api/stop")
def api_stop():
    cancel = CURRENT.get("cancel")
    if cancel is None:
        return {"stopped": False}
    with contextlib.suppress(Exception):
        cancel()
    return {"stopped": True}


# ------------------------------------------------- session message + SSE reply

def _sse(payload) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _autotitle(sid: str, content: str):
    n = _q("SELECT COUNT(*) AS n FROM messages WHERE session_id=? "
           "AND role='user'", (sid,), one=True)["n"]
    if n == 1:  # the message just inserted is the first user message
        title = " ".join(content.strip().split())
        if len(title) > 60:
            title = title[:60].rsplit(" ", 1)[0] + "…"
        _q("UPDATE sessions SET title=? WHERE id=?",
           (title or "New chat", sid), write=True)


def _persist_assistant(sid: str, content: str, reasoning: str, engine: str,
                       stats: dict) -> dict:
    mid = _q("INSERT INTO messages(session_id, role, content, reasoning, "
             "engine, stats, created) VALUES(?,?,?,?,?,?,?)",
             (sid, "assistant", content, reasoning, engine,
              json.dumps(stats), time.time()), write=True)
    _q("UPDATE sessions SET updated=? WHERE id=?", (time.time(), sid),
       write=True)
    return _msg_row(mid)


@app.post("/api/sessions/{sid}/messages")
def messages_post(sid: str, body: MessagePost):
    sess = _session_or_404(sid)
    content = body.content.strip()
    if not content:
        raise HTTPException(status_code=400, detail="empty message")
    model = body.model or sess["model"]
    engine = ENGINE_BY_MODEL.get(model)
    if engine is None:
        raise HTTPException(status_code=400, detail=f"unknown model {model!r}")
    if engine == "fast":
        _fast_require()

    history = _q("SELECT role, content FROM messages WHERE session_id=? "
                 "ORDER BY id", (sid,))
    uid = _q("INSERT INTO messages(session_id, role, content, engine, created)"
             " VALUES(?,?,?,?,?)", (sid, "user", content, engine, time.time()),
             write=True)
    _autotitle(sid, content)
    _q("UPDATE sessions SET updated=? WHERE id=?", (time.time(), sid),
       write=True)
    history.append({"role": "user", "content": content})
    max_new = min(body.max_tokens or CFG.max_tokens_cap, CFG.max_tokens_cap)

    gen = (_session_events_fast if engine == "fast"
           else _session_events_quality)(sid, uid, model, history, max_new)
    return StreamingResponse(gen, media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


def _session_events_fast(sid, uid, model, history, max_new):
    def events():
        if not GEN_LOCK.acquire(timeout=CFG.queue_timeout):
            yield _sse({"type": "error",
                        "message": "generation busy: queued past timeout"})
            yield "data: [DONE]\n\n"
            return
        try:
            yield _sse({"type": "start", "user_message_id": uid,
                        "model": model})
            t0 = time.time()
            t_first = None
            parts: List[str] = []
            # think:false — ds4-server defaults to hidden high-effort thinking,
            # which stalls streaming for tens of seconds and burns the token
            # budget; the fast engine should feel fast.
            payload = {"model": "deepseek-v4-flash", "messages": history,
                       "temperature": 0, "max_tokens": max_new, "think": False}
            for kind, data in _fast_events(payload):
                if kind == "delta":
                    if t_first is None:
                        t_first = time.time()
                    parts.append(data)
                    yield _sse({"type": "delta", "text": data})
                elif kind == "error":
                    yield _sse({"type": "error", "message": data})
                    yield "data: [DONE]\n\n"
                    return
                else:  # done | stopped
                    t_end = time.time()
                    stats = _fast_usage_stats(
                        data.get("usage") if kind == "done" else None,
                        t0, t_first, t_end, len(parts),
                        sum(len(p) for p in parts))
                    stats["stopped"] = kind == "stopped"
                    text = "".join(parts)
                    msg = _persist_assistant(sid, text, "", "fast", stats)
                    _log_request(f"session/{model}", stats)
                    yield _sse({"type": "done", "message": msg,
                                "stats": stats})
                    yield "data: [DONE]\n\n"
                    return
            # stream ended without a done event (child died mid-generation)
            yield _sse({"type": "error", "message": "fast engine stream "
                        "ended unexpectedly"})
            yield "data: [DONE]\n\n"
        finally:
            GEN_LOCK.release()
    return events()


def _session_events_quality(sid, uid, model, history, max_new):
    ids, sampler, max_new = _prepare_quality(history, 0.0, 1.0, max_new)

    def events():
        try:
            job = _submit(ids, sampler, max_new, stream=True)
        except HTTPException as err:
            yield _sse({"type": "error", "message": str(err.detail)})
            yield "data: [DONE]\n\n"
            return
        yield _sse({"type": "start", "user_message_id": uid, "model": model})
        t0 = time.time()
        t_first = None
        acc = ""            # full echoed stream (thinking + "\n---" + answer)
        emitted = 0         # chars of acc already sent to the client
        sep_at = None       # index of "\n---" once seen (thinking mode only)
        thinking = CFG.thinking_mode == "thinking"
        while True:
            kind, payload = job.out.get()
            if kind == "delta":
                if t_first is None:
                    t_first = time.time()
                acc += payload
                if thinking and sep_at is None:
                    sep_at = acc.find("\n---")
                    if sep_at == -1:
                        sep_at = None
                        # hold back a potential partial separator
                        safe = max(emitted, len(acc) - 4)
                        if safe > emitted:
                            yield _sse({"type": "thinking",
                                        "text": acc[emitted:safe]})
                            emitted = safe
                        continue
                    if sep_at > emitted:
                        yield _sse({"type": "thinking",
                                    "text": acc[emitted:sep_at]})
                    emitted = sep_at + 4
                    continue
                if len(acc) > emitted:
                    yield _sse({"type": "delta", "text": acc[emitted:]})
                    emitted = len(acc)
            elif kind == "done":
                text, hit_eos, stats = payload
                t_end = time.time()
                msg_parsed = parse_turn(text, hit_eos, CFG.thinking_mode)
                st = _quality_stats(stats, (t_first or t_end) - t0,
                                    t_end - t0)
                msg = _persist_assistant(
                    sid, msg_parsed["content"],
                    msg_parsed.get("reasoning_content") or "", "quality", st)
                _log_request(f"session/{model}", st)
                yield _sse({"type": "done", "message": msg, "stats": st})
                yield "data: [DONE]\n\n"
                return
            else:  # error
                yield _sse({"type": "error", "message": payload})
                yield "data: [DONE]\n\n"
                return
    return events()


def _log_request(kind: str, stats: dict):
    print(f"[serve] {kind}: {json.dumps(stats, ensure_ascii=False)}",
          file=sys.stderr, flush=True)


# ------------------------------------------------------ OpenAI-compatible API

@app.get("/v1/models")
def list_models():
    now = int(time.time())
    return {"object": "list",
            "data": [{"id": MODEL_FAST, "object": "model", "created": now,
                      "owned_by": "local"},
                     {"id": MODEL_QUALITY, "object": "model", "created": now,
                      "owned_by": "local"},
                     {"id": LEGACY_ID, "object": "model", "created": now,
                      "owned_by": "local"}]}


def _content_to_text(content) -> str:
    """Flatten OpenAI content (string or list of parts) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return "" if content is None else str(content)


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages must be non-empty")
    model = req.model or MODEL_QUALITY
    engine = ENGINE_BY_MODEL.get(model)
    if engine is None:
        raise HTTPException(status_code=404,
                            detail=f"unknown model {model!r}; use "
                                   f"{MODEL_FAST} or {MODEL_QUALITY}")
    messages = [{"role": m.get("role", "user"),
                 "content": _content_to_text(m.get("content"))}
                for m in req.messages]
    if engine == "fast":
        _fast_require()
        return _openai_fast(req, messages)
    ids, sampler, max_new = _prepare_quality(
        messages, req.temperature, req.top_p, req.max_tokens)
    if req.stream:
        return _openai_quality_stream(ids, sampler, max_new)
    return _openai_quality_blocking(ids, sampler, max_new)


def _finish_reason(stats: dict, max_new: int) -> str:
    return "length" if stats["new_tokens"] >= max_new else "stop"


def _usage(stats: dict) -> dict:
    return {"prompt_tokens": stats["prompt_tokens"],
            "completion_tokens": stats["new_tokens"],
            "total_tokens": stats["prompt_tokens"] + stats["new_tokens"],
            "prompt_tokens_details": {"cached_tokens": stats["reused_tokens"]}}


# ---- fast passthrough (model field rewritten to our public id)

def _openai_fast(req: ChatRequest, messages):
    payload = {"model": "deepseek-v4-flash", "messages": messages,
               "temperature": req.temperature or 0,
               "max_tokens": min(req.max_tokens or CFG.max_tokens_cap,
                                 CFG.max_tokens_cap),
               "think": bool(req.think)}  # ds4 default is hidden thinking
    if req.top_p is not None:
        payload["top_p"] = req.top_p

    if not req.stream:
        if not GEN_LOCK.acquire(timeout=CFG.queue_timeout):
            raise HTTPException(status_code=429, detail="generation busy")
        try:
            conn = http.client.HTTPConnection("127.0.0.1", CFG.fast_port,
                                              timeout=600)
            try:
                conn.request("POST", "/v1/chat/completions",
                             body=json.dumps(dict(payload, stream=False)),
                             headers={"Content-Type": "application/json"})
                resp = conn.getresponse()
                body = resp.read()
                if resp.status != 200:
                    raise HTTPException(status_code=502,
                                        detail=f"ds4-server HTTP {resp.status}: "
                                               f"{body[:500].decode('utf-8', 'replace')}")
                j = json.loads(body)
                j["model"] = MODEL_FAST
                return j
            finally:
                conn.close()
        finally:
            GEN_LOCK.release()

    rid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    def chunk(delta: dict, finish=None, usage=None) -> str:
        body = {"id": rid, "object": "chat.completion.chunk",
                "created": created, "model": MODEL_FAST,
                "choices": [{"index": 0, "delta": delta,
                             "finish_reason": finish}]}
        if usage is not None:
            body["usage"] = usage
        return _sse(body)

    def events():
        if not GEN_LOCK.acquire(timeout=CFG.queue_timeout):
            yield _sse({"error": {"message": "generation busy",
                                  "type": "rate_limit_error"}})
            yield "data: [DONE]\n\n"
            return
        try:
            yield chunk({"role": "assistant", "content": ""})
            for kind, data in _fast_events(payload):
                if kind == "delta":
                    yield chunk({"content": data})
                elif kind == "error":
                    yield _sse({"error": {"message": data,
                                          "type": "server_error"}})
                    break
                elif kind == "stopped":
                    yield chunk({}, finish="stop")
                    break
                else:  # done
                    yield chunk({}, finish=data["finish_reason"],
                                usage=data.get("usage"))
                    break
            yield "data: [DONE]\n\n"
        finally:
            GEN_LOCK.release()

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


# ---- quality (in-process)

def _openai_quality_blocking(ids, sampler, max_new):
    job = _submit(ids, sampler, max_new, stream=False)
    kind, payload = job.out.get()  # blocks for the whole generation
    if kind == "error":
        raise HTTPException(status_code=500,
                            detail=f"generation failed: {payload}")
    text, hit_eos, stats = payload

    msg = parse_turn(text, hit_eos, CFG.thinking_mode)
    _log_request("chat.completion", _quality_stats(stats, stats["prefill_s"],
                                                   0.0))
    message = {"role": "assistant", "content": msg["content"]}
    if msg.get("reasoning_content"):
        message["reasoning_content"] = msg["reasoning_content"]
    return {"id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": MODEL_QUALITY,
            "choices": [{"index": 0, "message": message,
                         "finish_reason": _finish_reason(stats, max_new)}],
            "usage": _usage(stats)}


def _openai_quality_stream(ids, sampler, max_new):
    job = _submit(ids, sampler, max_new, stream=True)  # 429s while queued
    q = job.out

    rid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    def chunk(delta: dict, finish=None, usage=None) -> str:
        body = {"id": rid, "object": "chat.completion.chunk",
                "created": created, "model": MODEL_QUALITY,
                "choices": [{"index": 0, "delta": delta,
                             "finish_reason": finish}]}
        if usage is not None:
            body["usage"] = usage
        return _sse(body)

    def events():
        yield chunk({"role": "assistant", "content": ""})
        pending = None  # one-delta lag so the echo path's final "\n" is dropped
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
                _log_request("chat.completion.chunk",
                             _quality_stats(stats, stats["prefill_s"], 0.0))
                yield chunk({}, finish=_finish_reason(stats, max_new),
                            usage=_usage(stats))
                yield "data: [DONE]\n\n"
                return
            else:  # error
                yield _sse({"error": {"message": payload,
                                      "type": "server_error"}})
                yield "data: [DONE]\n\n"
                return

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


# ----------------------------------------------------------------------- main

def main():
    global CFG

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repacked", default=os.path.join(ROOT, "repacked"))
    ap.add_argument("--cache-gb", type=float, default=8.0,
                    help="quality-engine expert LRU budget; 8 GB is the "
                         "measured optimum")
    ap.add_argument("--store", choices=["lru", "stacked"], default="lru")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8399)
    ap.add_argument("--db", default=os.path.join(ROOT, "data", "chat.db"))
    ap.add_argument("--default-model", choices=[MODEL_FAST, MODEL_QUALITY],
                    default=MODEL_FAST, help="model for new sessions")
    ap.add_argument("--thinking-mode", choices=["chat", "thinking"],
                    default="chat",
                    help="quality engine, server-wide; default chat (fast)")
    ap.add_argument("--reasoning-effort", choices=["low", "high", "max"],
                    default=None, help="thinking mode only")
    ap.add_argument("--max-tokens-cap", type=int, default=512,
                    help="default and upper clamp for max_tokens (both engines)")
    ap.add_argument("--queue-timeout", type=float, default=600.0,
                    help="seconds a request may wait for the GPU before 429")
    ap.add_argument("--no-kv-reuse", dest="kv_reuse", action="store_false",
                    help="disable the quality conversation KV slot")
    ap.add_argument("--kv-slot-len", type=int, default=8192,
                    help="minimum max_seq_len for the resident conversation "
                         "cache")
    ap.add_argument("--seed", type=int, default=None)
    # fast engine (ds4-server child)
    ap.add_argument("--no-fast", action="store_true",
                    help="do not spawn ds4-server (quality engine only)")
    ap.add_argument("--fast-model", default=os.path.expanduser(
        "~/.cache/huggingface/hub/models--antirez--deepseek-v4-gguf/"
        "snapshots/*/DeepSeek-V4-Flash-IQ2XXS*imatrix-0731.gguf"),
        help="glob for the ds4 q2 GGUF")
    ap.add_argument("--fast-port", type=int, default=8398,
                    help="loopback port for the ds4-server child")
    ap.add_argument("--fast-cache", default="8GB",
                    help="ds4 --ssd-streaming-cache-experts value "
                         "(8GB measured best on this machine)")
    ap.add_argument("--fast-ctx", type=int, default=32768,
                    help="ds4 context tokens")
    ap.add_argument("--fast-startup-timeout", type=float, default=300.0)
    CFG = ap.parse_args()

    _db_init()
    _fast_start()

    # All MLX work — including model load and seeding — happens on the one
    # persistent generation thread (see _worker_main).
    threading.Thread(target=_worker_main, name="generation-worker",
                     daemon=True).start()
    WORKER_READY.wait()
    if WORKER_LOAD_ERROR is not None:
        print(f"[serve] quality model load failed: {WORKER_LOAD_ERROR}",
              file=sys.stderr, flush=True)
        _fast_stop()
        sys.exit(1)

    app.mount("/static", StaticFiles(directory=WEBUI_DIR), name="static")

    import uvicorn
    uvicorn.run(app, host=CFG.host, port=CFG.port, log_level="info")


if __name__ == "__main__":
    main()
