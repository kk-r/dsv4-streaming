"""Speculative decoding for the streaming runtime: prompt-lookup drafting,
greedy batched verification, KV rewind on partial acceptance.

Why: decode cost here is dominated by per-forward-pass overhead (~128 ms model
compute + ~300 ms of 43 per-layer Python sync round-trips on routing indices),
not per-token weight reads. Verifying k drafted tokens in ONE forward amortizes
the per-pass cost across every accepted token.

Drafting is classic prompt-lookup (n-gram) decoding: find the most recent
longest match (n = n_max..n_min) of the current token suffix inside
prompt+generated ids and copy up to k continuation tokens. No draft model, so
it is tokenizer-safe, and under greedy verification the emitted stream is the
plain greedy stream (accepted drafts equal the greedy argmaxes by construction;
the bonus token is the model's own argmax after the accepted prefix).

The verify pass needs a multi-token forward at NONZERO cache offset, which the
port does not have: Attention.__call__ sends s>1 to the prefill branch, which
assigns positions 0..s-1 and reseeds the cache (_seed_cache). This module
therefore patches Attention.__call__: for s>1 with cache.offset>0 it unrolls
into s lazy step() calls inside one graph — numerically the exact op sequence
sequential decode performs (same ring writes, compressor steps, indexer
top-k; functional arrays make in-graph sequential ring writes correct at any
offset) — while the MoE runs batched over the s tokens through the existing
StreamingExperts >1-token gather_qmm path. The 43 per-layer Python syncs
happen once per PASS instead of once per token: that is the whole speedup.

KV rewind: after each unrolled step a value snapshot of the layer cache is
recorded (offset, kv, idx_kv, compressor kv_state/score_state, indexer
compressor state). NOTE: the optimistic.py claim that saved references are
snapshots is WRONG for these buffers — mx.array.__setitem__ mutates the same
Python object in place (verified empirically; optimistic.py never noticed
because its redo rewrites identical values). Snapshots here are new x[:]
node-wrapping objects, which the lazy graph isolates from later scatters. On
acceptance of a drafts, every layer restores snaps[a] — the state right
after the last accepted input token. This also rewinds the compressor
partial-span state, whose destructive shift() position-overwrite semantics
alone could never repair.

Usage (one cold process per measurement):
  caffeinate -i python3 streaming/spec_generate.py --cache-gb 8 \
      --prompt "The capital of France is" --max-new 96 --k 8 \
      --save-json logs/spec_runs/capitals_k8.json
  --k 0 is the plain greedy baseline (byte-identical to run_streaming.py's
  greedy_generate by construction: same ops in the same order, timed).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import mlx.core as mx
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "deepseek-v4-mlx"))
sys.path.insert(0, os.path.dirname(__file__))

from deepseek_v4_mlx.attention import Attention   # noqa: E402
from deepseek_v4_mlx.cache import make_cache      # noqa: E402
from deepseek_v4_mlx.generate import load_tokenizer  # noqa: E402
from deepseek_v4_mlx.hyper_connections import hc_head, hc_post, hc_pre  # noqa: E402
from run_streaming import load_streaming          # noqa: E402

EOS_ID = 1


# ---------------------------------------------------------------------------
# Batched decode-append: unrolled lazy steps + per-position snapshots
# ---------------------------------------------------------------------------

def _cp(x):
    """Snapshot one cache buffer: a new array object wrapping the CURRENT
    value node.

    NOT just a reference save. Measured here (and contrary to the lore in
    optimistic.py): mx.array.__setitem__ mutates the same Python object in
    place — `snap = c.kv; c.kv[:, i] = v` leaves snap showing the update, so
    reference-saving is no snapshot for the setitem-updated buffers
    (LayerCache.kv/idx_kv, CompressorCache.kv_state/score_state via write()
    and shift()). `x[:]` creates a distinct lazy node holding the current
    value; the subsequent scatter replaces only the original object's node.
    (optimistic.py's restore never caught this because its redo rewrites
    byte-identical values at the same positions — a rollback no-op.)"""
    return None if x is None else x[:]


def _snap_layer(c):
    """Value snapshot of one LayerCache at the current position boundary."""
    return (c.offset, c.dtype, _cp(c.kv), _cp(c.idx_kv),
            (_cp(c.comp.kv_state), _cp(c.comp.score_state)) if c.comp else None,
            (_cp(c.idx_comp.kv_state), _cp(c.idx_comp.score_state))
            if c.idx_comp else None)


def _restore_layer(c, s):
    c.offset, c.dtype, c.kv, c.idx_kv = s[0], s[1], s[2], s[3]
    if s[4]:
        c.comp.kv_state, c.comp.score_state = s[4]
    if s[5]:
        c.idx_comp.kv_state, c.idx_comp.score_state = s[5]


_orig_attn_call = Attention.__call__


def _attn_call_spec(self, x, cache=None):
    """s>1 at nonzero offset = decode-append: unroll into lazy per-position
    steps (exact decode numerics, no Python syncs) and record per-position
    snapshots for rewind. Everything else falls through untouched."""
    if cache is not None and cache.offset > 0 and x.shape[1] > 1:
        outs, snaps = [], []
        for j in range(x.shape[1]):
            outs.append(self.step(x[:, j:j + 1], cache))
            snaps.append(_snap_layer(cache))
        cache._spec_snaps = snaps
        return mx.concatenate(outs, axis=1)
    return _orig_attn_call(self, x, cache)


Attention.__call__ = _attn_call_spec


def _rewind(cache, n_accepted):
    """Restore every layer to the state after input position n_accepted
    (i.e. keep KV for the confirmed token + the accepted drafts)."""
    for c in cache:
        _restore_layer(c, c._spec_snaps[n_accepted])
        c._spec_snaps = None


# ---------------------------------------------------------------------------
# Bitwise-exact verify: per-token ops at decode shapes, batched sync + IO only
# ---------------------------------------------------------------------------

def _verify_exact(model, toks, cache, store):
    """Multi-token verify whose logits are BITWISE identical to sequential
    decode of the same tokens.

    The batched verify (model.__call__ over s>1) is not: MoE gate GEMM,
    gather_qmm, hc projections etc. run different-shaped kernels than the
    s=1 decode path, and the fp accumulation differences (measured: logit
    margin drift up to ~1.0) flip near-tied argmaxes — the correctness gate
    caught exactly that. Here every op runs per token at decode shapes
    ([1,1,...]), so every kernel is the one decode runs, on the same bits.
    Only two things are batched, and neither touches numerics:

    * ONE routing-index sync per layer (np.asarray on the s tokens'
      concatenated gate indices — probed with the same gate call the MoE
      will deterministically recompute lazily) instead of s syncs;
    * ONE store.get_many per layer prefetching the expert union with
      parallel preads; the untouched single-token expert path then runs on
      pure LRU hits.

    So the pass still costs 43 sync round-trips instead of 43*s, which is
    the overhead the hypothesis says dominates. Per-position cache
    snapshots are recorded exactly like the batched path for _rewind.
    Returns logits [1, s, vocab].
    """
    s = len(toks)
    hs = []
    for t in toks:
        e = model.embed(mx.array([[t]]))
        hs.append(mx.broadcast_to(e[:, :, None, :],
                                  (*e.shape[:2], model.hc_mult, e.shape[-1])))
    snaps = [[] for _ in model.layers]
    for L, layer in enumerate(model.layers):
        c = cache[L]
        for j in range(s):
            r = hs[j]
            h, post, comb = hc_pre(r, layer.hc_attn_fn, layer.hc_attn_scale,
                                   layer.hc_attn_base, layer.hc_mult,
                                   layer.hc_iters, layer.norm_eps, layer.hc_eps)
            h = layer.attn(layer.attn_norm(h), c)     # [1,1,d] -> decode step()
            hs[j] = hc_post(h, r, post, comb)
            snaps[L].append(_snap_layer(c))
        pre, idx_cat = [], []
        for j in range(s):
            r = hs[j]
            h, post, comb = hc_pre(r, layer.hc_ffn_fn, layer.hc_ffn_scale,
                                   layer.hc_ffn_base, layer.hc_mult,
                                   layer.hc_iters, layer.norm_eps, layer.hc_eps)
            h = layer.ffn_norm(h)
            _, ind = layer.ffn.gate(h.reshape(-1, layer.ffn.dim),
                                    mx.array([toks[j]]))
            pre.append((r, h, post, comb))
            idx_cat.append(ind.reshape(-1))
        idx = np.asarray(mx.concatenate(idx_cat))       # ONE sync per layer
        store.get_many(L, list(dict.fromkeys(idx.tolist())))  # parallel prefetch
        for j in range(s):
            r, h, post, comb = pre[j]
            y = layer.ffn(h, mx.array([[toks[j]]]))     # full MoE, all LRU hits
            hs[j] = hc_post(y, r, post, comb)
    logits = []
    for j in range(s):
        h = hc_head(hs[j], model.hc_head_fn, model.hc_head_scale,
                    model.hc_head_base, model.args.norm_eps, model.args.hc_eps)
        logits.append(model.head(model.norm(h).astype(mx.float32)))
    for L, c in enumerate(cache):
        c._spec_snaps = snaps[L]
    return mx.concatenate(logits, axis=1)


# ---------------------------------------------------------------------------
# Prompt-lookup drafting
# ---------------------------------------------------------------------------

def propose_draft(ids, k, n_max=6, n_min=2):
    """Copy the continuation of the most recent longest n-gram match.

    Tries n = n_max..n_min; for each n, scans backward for the most recent
    earlier occurrence of the last n tokens and returns up to k tokens that
    followed it. Overlapping matches are fine (periodic text: the draft is
    "the token one period back"). Returns [] when nothing matches.
    """
    L = len(ids)
    for n in range(min(n_max, L - 1), n_min - 1, -1):
        pat = ids[L - n:]
        for start in range(L - n - 1, -1, -1):
            if ids[start:start + n] == pat:
                return ids[start + n:start + n + k]
    return []


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def spec_generate(model, args, prompt_ids, max_new, k=8, n_max=6, n_min=2,
                  eos_id=EOS_ID, verbose=False, mode="exact", store=None):
    """Greedy generation with prompt-lookup speculation. k=0 = plain greedy
    (byte-identical op sequence to generate.greedy_generate, but timed).

    mode="exact" verifies through _verify_exact (bitwise-identical to plain
    greedy decode; requires store); mode="batched" verifies through the
    model's batched forward (faster kernels, but fp margin drift can flip
    near-tied argmaxes — NOT byte-identical, see _verify_exact docstring).

    Returns (out_ids, stats). out_ids may overshoot max_new by up to k
    (callers trim); every emitted token is a plain-greedy token (exact mode).
    """
    assert mode in ("exact", "batched")
    if mode == "exact" and k > 0:
        assert store is not None
    ids = list(prompt_ids)
    cache = make_cache(args, bsz=1, max_seq_len=len(ids) + max_new + k + 8)

    t0 = time.time()
    logits = model(mx.array([ids]), last_logit_only=True, cache=cache)
    nxt = int(mx.argmax(logits[0, -1]).item())
    prefill_s = time.time() - t0

    out = [nxt]
    all_ids = ids + [nxt]
    passes = []          # (n_drafted, n_accepted) per verify pass
    n_plain = 0          # single-token fallback steps (no draft found)
    t1 = time.time()
    while len(out) < max_new and out[-1] != eos_id:
        drafts = propose_draft(all_ids, k, n_max, n_min) if k > 0 else []
        if drafts:
            toks = [out[-1]] + drafts
            if mode == "exact":
                logits = _verify_exact(model, toks, cache, store)
            else:
                logits = model(mx.array([toks]), last_logit_only=False,
                               cache=cache)
            preds = np.asarray(mx.argmax(logits[0], axis=-1)).tolist()
            a = 0
            while a < len(drafts) and preds[a] == drafts[a]:
                a += 1
            emitted = drafts[:a] + [preds[a]]
            _rewind(cache, a)
            passes.append((len(drafts), a))
            if verbose:
                print(f"[pass {len(passes):3d}] drafted {len(drafts)} "
                      f"accepted {a} -> +{a + 1} tok", flush=True)
        else:
            logits = model(mx.array([[out[-1]]]), last_logit_only=True,
                           cache=cache)
            emitted = [int(mx.argmax(logits[0, -1]).item())]
            n_plain += 1
        if eos_id in emitted:
            emitted = emitted[:emitted.index(eos_id) + 1]
        out.extend(emitted)
        all_ids.extend(emitted)
    decode_s = time.time() - t1

    drafted = sum(d for d, _ in passes)
    accepted = sum(a for _, a in passes)
    n_decode = len(out) - 1          # tokens produced during the decode phase
    stats = {
        "k": k, "n_max": n_max, "n_min": n_min, "mode": mode if k else "plain",
        "prompt_tokens": len(ids), "new_tokens": len(out),
        "prefill_s": round(prefill_s, 2), "decode_s": round(decode_s, 2),
        "decode_tok_s": round(n_decode / decode_s, 3) if decode_s > 0 else 0.0,
        "forwards": 1 + len(passes) + n_plain,
        "verify_passes": len(passes), "plain_steps": n_plain,
        "drafted": drafted, "accepted": accepted,
        "acceptance_rate": round(accepted / drafted, 4) if drafted else None,
        "mean_accepted_per_pass": round(accepted / len(passes), 3) if passes else None,
        "decode_tokens_per_forward": round(
            n_decode / (len(passes) + n_plain), 3) if (passes or n_plain) else None,
        "accept_hist": {},
    }
    for _, a in passes:
        stats["accept_hist"][a] = stats["accept_hist"].get(a, 0) + 1
    return out, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repacked", default=os.path.join(os.path.dirname(__file__),
                                                       "..", "repacked"))
    ap.add_argument("--cache-gb", type=float, default=8.0)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--k", type=int, default=8,
                    help="max draft tokens per pass; 0 = plain greedy baseline")
    ap.add_argument("--n-max", type=int, default=6)
    ap.add_argument("--n-min", type=int, default=2)
    ap.add_argument("--mode", choices=["exact", "batched"], default="exact")
    ap.add_argument("--save-json", default=None)
    ap.add_argument("--verbose", action="store_true")
    cli = ap.parse_args()

    model, args, store = load_streaming(cli.repacked, cli.cache_gb, "lru")
    tok = load_tokenizer(cli.repacked)
    ids = tok.encode(cli.prompt)
    print(f"[spec] prompt {len(ids)} tok, k={cli.k}, n={cli.n_min}..{cli.n_max}, "
          f"mode={cli.mode}, cache {cli.cache_gb} GB, max_new {cli.max_new}")

    out, stats = spec_generate(model, args, ids, cli.max_new, k=cli.k,
                               n_max=cli.n_max, n_min=cli.n_min,
                               verbose=cli.verbose, mode=cli.mode, store=store)
    out_trim = out[:cli.max_new]
    text = tok.decode(ids + out_trim)
    print(f"\n=== output ===\n{text}\n==============")
    print(f"[spec] prefill {stats['prefill_s']}s | decode "
          f"{stats['decode_tok_s']} tok/s ({stats['new_tokens']} tok in "
          f"{stats['decode_s']}s) | forwards {stats['forwards']} "
          f"(verify {stats['verify_passes']}, plain {stats['plain_steps']})")
    if stats["drafted"]:
        print(f"[spec] drafted {stats['drafted']}, accepted {stats['accepted']} "
              f"({stats['acceptance_rate']}), mean accepted/pass "
              f"{stats['mean_accepted_per_pass']}, decode tokens/forward "
              f"{stats['decode_tokens_per_forward']}, hist {stats['accept_hist']}")
    print(f"[store] {store.summary()}")
    print(f"[mem] peak {mx.get_peak_memory()/1e9:.1f} GB")

    if cli.save_json:
        os.makedirs(os.path.dirname(cli.save_json), exist_ok=True)
        stats.update({"prompt": cli.prompt, "prompt_ids": ids,
                      "out_ids": out, "out_ids_trimmed": out_trim,
                      "text": text, "store": store.summary(),
                      "peak_gb": round(mx.get_peak_memory() / 1e9, 2)})
        json.dump(stats, open(cli.save_json, "w"), indent=1)
        print(f"[spec] saved {cli.save_json}")


if __name__ == "__main__":
    main()
