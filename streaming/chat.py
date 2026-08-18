"""Chat with DeepSeek-V4-Flash over the SSD expert-streaming runtime.

Wraps run_streaming's model loader with DeepSeek's official chat encoding
(deepseek_v4_mlx/encoding_dsv4.py, verified against the bundled test vectors):
messages are rendered with encode_messages, the completion is cut at the model's
stop tokens and parsed with parse_message_from_completion_text, so the user sees
clean text instead of raw-completion scaffolding.

One-shot:
  python3 chat.py --prompt "Why is the sky blue?" --max-new 256

REPL (multi-turn; the persistent process keeps the expert cache warm across
turns, which is measured to help):
  python3 chat.py
  Commands: /exit  /reset  /stats

Thinking mode (--thinking-mode thinking) makes the model reason inside
<think>...</think> first; the reasoning is streamed dimmed and kept out of the
final answer. Default is chat mode, which answers directly and is faster.

Sampling is greedy unless --temp > 0 (then mx.random.categorical, with optional
--top-p nucleus truncation).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import mlx.core as mx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "deepseek-v4-mlx"))
sys.path.insert(0, os.path.dirname(__file__))

from deepseek_v4_mlx import encoding_dsv4 as E   # noqa: E402
from deepseek_v4_mlx.cache import make_cache     # noqa: E402
from deepseek_v4_mlx.generate import load_tokenizer  # noqa: E402
from kv_append import append_tokens, model_store  # noqa: E402
from run_streaming import load_streaming         # noqa: E402

EOS_ID = 1  # generation_config.json; == tokenizer id of E.eos_token

DIM, RESET = "\033[2m", "\033[0m"


def stop_ids(tok):
    """eos id 1 plus the end-of-assistant markers encoding_dsv4 defines.

    The assistant template ends every turn with eos_token; a model that instead
    starts a new turn emits the User/Assistant special tokens, so those stop
    generation too (as malformed, not clean, stops).
    """
    ids = {EOS_ID}
    for t in (E.eos_token, E.USER_SP_TOKEN, E.ASSISTANT_SP_TOKEN):
        i = tok.convert_tokens_to_ids(t)
        if i is not None and i >= 0:
            ids.add(i)
    return ids


def make_sampler(temp: float, top_p: float):
    if temp <= 0:
        return lambda logits: int(mx.argmax(logits).item())

    def sample(logits):
        probs = mx.softmax(logits.astype(mx.float32) / temp, axis=-1)
        if 0 < top_p < 1:
            order = mx.argsort(probs)            # ascending
            sp = probs[order]
            keep = mx.cumsum(sp, axis=-1) > 1 - top_p
            choice = mx.random.categorical(mx.log(mx.where(keep, sp, 0.0)))
            return int(order[choice].item())
        return int(mx.random.categorical(mx.log(probs)).item())

    return sample


def generate_turn(model, args, tok, prompt_ids, sampler, stops,
                  max_new: int = 512, thinking_mode: str = "chat",
                  echo: bool = True, cache=None, prefix_len: int = 0,
                  stop_event=None):
    """Generate one assistant turn. Returns (text, hit_eos, stats dict).

    Streams decoded text to stdout as it arrives (reasoning dimmed in thinking
    mode). Deltas come from re-decoding the full token list and diffing, since a
    single BPE token can be a partial UTF-8 sequence.

    ``cache``/``prefix_len`` support conversation KV reuse (serve.py): pass a
    cache whose first ``prefix_len`` positions already hold KV for
    ``prompt_ids[:prefix_len]`` and only the suffix is processed. Batched
    prefill cannot be used for the suffix: the prefill branch of attention
    assigns absolute positions 0..s-1 and reseeds the window ring +
    compressor/indexer state from scratch (``Attention._seed_cache``), so it
    cannot extend a nonzero offset. The suffix instead goes through
    ``kv_append.append_tokens`` — the spec verifier's teacher-forced exact
    forward (decode-shape kernels per token, bitwise-identical to sequential
    decode-append, but only 43 per-layer index syncs per chunk instead of 43
    per token, plus parallel get_many SSD reads). DSV4_SUFFIX_APPEND=decode
    falls back to the old one-token-at-a-time decode loop (A/B control);
    DSV4_APPEND_CHUNK sets the tokens-per-pass bound (default 32).
    """
    assert 0 <= prefix_len < len(prompt_ids)
    if cache is None:
        assert prefix_len == 0
        cache = make_cache(args, bsz=1, max_seq_len=len(prompt_ids) + max_new + 8)

    t0 = time.time()
    if prefix_len:
        suffix = prompt_ids[prefix_len:]
        batch_ok = (os.environ.get("DSV4_SUFFIX_APPEND", "batch") == "batch"
                    and model_store(model) is not None)
        if batch_ok:
            chunk = int(os.environ.get("DSV4_APPEND_CHUNK", "32"))
            logits = append_tokens(model, cache, suffix, chunk=chunk)
        else:
            for t in suffix:
                logits = model(mx.array([[t]]), last_logit_only=True, cache=cache)
                mx.eval(logits)  # bound the lazy graph per token, as decode does
    else:
        logits = model(mx.array([prompt_ids]), last_logit_only=True, cache=cache)
    nxt = sampler(logits[0, -1])
    prefill_s = time.time() - t0

    out, hit_eos, stopped = [], False, False
    in_reasoning = thinking_mode == "thinking"
    emitted = 0
    t1 = time.time()
    for _ in range(max_new):
        if stop_event is not None and stop_event.is_set():
            stopped = True
            break
        if nxt in stops:
            hit_eos = nxt == EOS_ID
            break
        out.append(nxt)
        if echo:
            text = tok.decode(out)
            if len(text) > emitted:
                delta, emitted = text[emitted:], len(text)
                if in_reasoning and E.thinking_end_token in text:
                    head, _, tail = delta.partition(E.thinking_end_token)
                    print(f"{DIM}{head}{RESET}\n---", end="", flush=True)
                    in_reasoning = False
                    delta = tail
                if delta:
                    print(f"{DIM}{delta}{RESET}" if in_reasoning else delta,
                          end="", flush=True)
        logits = model(mx.array([[nxt]]), last_logit_only=True, cache=cache)
        nxt = sampler(logits[0, -1])
    decode_s = time.time() - t1
    if echo:
        print(flush=True)

    stats = {"prompt_tokens": len(prompt_ids), "new_tokens": len(out),
             "prefill_s": round(prefill_s, 1),
             "decode_tok_s": round(len(out) / decode_s, 2) if decode_s > 0 else 0.0,
             "reused_tokens": prefix_len,
             "processed_tokens": len(prompt_ids) - prefix_len,
             "stopped": stopped,
             "out_ids": out}
    return tok.decode(out), hit_eos, stats


def parse_turn(text: str, hit_eos: bool, thinking_mode: str) -> dict:
    """Parse a completion into an assistant message dict, tolerating truncation."""
    if hit_eos:
        try:
            msg = E.parse_message_from_completion_text(text + E.eos_token,
                                                       thinking_mode)
            msg["truncated"] = False
            return msg
        except Exception as err:
            return {"role": "assistant", "content": text, "reasoning_content": "",
                    "tool_calls": [], "truncated": False, "parse_error": str(err)}
    reasoning = ""
    if thinking_mode == "thinking" and E.thinking_end_token in text:
        reasoning, _, text = text.partition(E.thinking_end_token)
    return {"role": "assistant", "content": text, "reasoning_content": reasoning,
            "tool_calls": [], "truncated": True}


def run_turn(model, args, tok, messages, cli, sampler, stops):
    prompt = E.encode_messages(messages, thinking_mode=cli.thinking_mode,
                               reasoning_effort=cli.reasoning_effort)
    ids = tok(prompt, add_special_tokens=False)["input_ids"]
    text, hit_eos, stats = generate_turn(
        model, args, tok, ids, sampler, stops, max_new=cli.max_new,
        thinking_mode=cli.thinking_mode)
    msg = parse_turn(text, hit_eos, cli.thinking_mode)
    if msg.get("parse_error"):
        print(f"[chat] parse error (raw text kept): {msg['parse_error']}",
              file=sys.stderr)
    if msg["truncated"]:
        print(f"[chat] hit --max-new {cli.max_new} before the turn finished",
              file=sys.stderr)
    return msg, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repacked", default=os.path.join(os.path.dirname(__file__),
                                                       "..", "repacked"))
    ap.add_argument("--cache-gb", type=float, default=8.0)
    ap.add_argument("--store", choices=["lru", "stacked"], default="lru")
    ap.add_argument("--prompt", default=None,
                    help="one-shot user message; omit for the interactive REPL")
    ap.add_argument("--system", default=None, help="optional system prompt")
    ap.add_argument("--thinking-mode", choices=["chat", "thinking"], default="chat")
    ap.add_argument("--reasoning-effort", choices=["low", "high", "max"],
                    default=None, help="thinking mode only")
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--temp", type=float, default=0.0,
                    help="0 = greedy (default); >0 samples via mx.random.categorical")
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=None)
    cli = ap.parse_args()

    if cli.seed is not None:
        mx.random.seed(cli.seed)

    model, args, store = load_streaming(cli.repacked, cli.cache_gb, cli.store)
    tok = load_tokenizer(cli.repacked)
    sampler = make_sampler(cli.temp, cli.top_p)
    stops = stop_ids(tok)

    messages = []
    if cli.system:
        messages.append({"role": "system", "content": cli.system})

    if cli.prompt is not None:
        messages.append({"role": "user", "content": cli.prompt})
        msg, stats = run_turn(model, args, tok, messages, cli, sampler, stops)
        print(f"[chat] prefill {stats['prefill_s']}s ({stats['prompt_tokens']} tok), "
              f"decode {stats['decode_tok_s']} tok/s ({stats['new_tokens']} tok) | "
              f"store {store.summary()}", file=sys.stderr)
        if getattr(store, "coordinator", None) is not None:
            print(f"[prefetch] {store.coordinator.report()}", file=sys.stderr)
        print(f"[ids] {stats['out_ids']}", file=sys.stderr)  # exact-output check
        print(f"[mem] peak {mx.get_peak_memory()/1e9:.1f} GB", file=sys.stderr)
        return

    print(f"[chat] REPL — {cli.thinking_mode} mode, cache {cli.cache_gb} GB. "
          "/exit to quit, /reset to clear history, /stats for store counters.")
    while True:
        try:
            user = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user == "/exit":
            break
        if user == "/reset":
            messages = [m for m in messages if m.get("role") == "system"]
            print("[chat] history cleared")
            continue
        if user == "/stats":
            print(f"[store] {store.summary()}")
            print(f"[mem] peak {mx.get_peak_memory()/1e9:.1f} GB")
            continue
        messages.append({"role": "user", "content": user})
        msg, stats = run_turn(model, args, tok, messages, cli, sampler, stops)
        messages.append({"role": "assistant", "content": msg["content"],
                         **({"reasoning_content": msg["reasoning_content"]}
                            if msg.get("reasoning_content") else {})})
        print(f"[chat] prefill {stats['prefill_s']}s, "
              f"decode {stats['decode_tok_s']} tok/s | store {store.summary()}",
              file=sys.stderr)


if __name__ == "__main__":
    main()
