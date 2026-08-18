"""Batched teacher-forced KV append: process a known suffix in one pass.

Problem this solves: conversation KV reuse (serve.py / chat.generate_turn)
appends the new suffix (prior reply tail + new user turn, typically 10-100
tokens) through the decode path ONE token at a time. At ~1 tok/s that is most
of the ~15 s turn-2+ time-to-first-token. The port's prefill branch cannot
help: it assigns absolute positions 0..s-1 and reseeds the ring/compressor/
indexer state from position 0 (Attention._seed_cache), clobbering a warm cache.

This module is spec_generate._verify_exact minus the acceptance machinery:
teacher-forced consumption of KNOWN tokens is exactly the verifier's forward
with every token given, and no rewind — so no per-position snapshots, and the
compressor's destructive shift() is a non-issue (we only ever move forward).

Bitwise identity with sequential decode-append holds by the same construction
proven for the verifier: every numeric op runs per token at decode shapes
([1, 1, ...]) — same kernels, same bits (batched shape-changed forwards are
NOT bitwise on this stack; kernel fp-accumulation drift flips near-tied
argmaxes). Only two things are batched, and neither touches numerics:

* ONE routing-index sync per layer (np.asarray over the s tokens' gate
  indices, probed with the same gate call the MoE deterministically
  recomputes lazily) instead of s syncs — 43 syncs per chunk instead of 43*s;
* ONE store.get_many per layer prefetching the expert union with parallel
  preads; the untouched single-token expert path then runs on pure LRU hits.

get_many normally runs under the prefill fill-no-evict policy (misses are not
inserted once the LRU is full — right for speculative/prefill traffic, wrong
here: these are decode-bound tokens whose experts sequential decode-append
would have inserted via get()). The policy is disabled around the append so
the LRU ends up populated the way decode-append would have left it.

Threading: everything runs on the caller's thread. get_many's worker pool
only reads raw bytes; mx arrays are created on the caller's thread (the MLX
thread-affinity rule serve.py's generation worker depends on).

Note: model.token_hook (pre-gated prefetch, DSV4_PREFETCH=1) does not fire
here — it is a timing-only feature and DSV4_PREFETCH is default-off.
"""

from __future__ import annotations

import os
import sys

import mlx.core as mx
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "deepseek-v4-mlx"))

from deepseek_v4_mlx.hyper_connections import hc_head, hc_post, hc_pre  # noqa: E402


def model_store(model):
    """The ExpertStore behind the streaming experts, or None (e.g. noop)."""
    experts = getattr(model.layers[0].ffn, "experts", None)
    store = getattr(experts, "store", None)
    return store if store is not None and hasattr(store, "get_many") else None


def append_tokens(model, cache, token_ids, chunk: int = 64):
    """Teacher-forced append of ``token_ids`` to a warm ``cache``.

    Bitwise-identical cache state and logits to feeding the tokens through the
    decode path one at a time. Long suffixes are processed ``chunk`` tokens
    per pass to bound the lazy graph and per-layer working memory.

    Returns logits [1, 1, vocab] for the LAST appended token (what the caller
    samples the next token from). Requires cache offset > 0 (a seeded cache):
    at offset 0 the attention step branch is not taken by Attention.__call__.
    """
    assert token_ids, "append_tokens needs at least one token"
    assert cache[0].offset > 0, "append_tokens requires a warm (seeded) cache"
    store = model_store(model)
    assert store is not None, "append_tokens needs the streaming ExpertStore"
    logits = None
    for i in range(0, len(token_ids), chunk):
        logits = _append_chunk(model, cache, token_ids[i:i + chunk], store)
        mx.eval(logits)  # bound the lazy graph per chunk, as decode does per token
    return logits


def _append_chunk(model, cache, toks, store):
    s = len(toks)
    hs = []
    for t in toks:
        e = model.embed(mx.array([[t]]))
        hs.append(mx.broadcast_to(e[:, :, None, :],
                                  (*e.shape[:2], model.hc_mult, e.shape[-1])))
    no_evict = store.prefill_no_evict
    store.prefill_no_evict = False   # decode-bound tokens: insert like get() would
    try:
        for L, layer in enumerate(model.layers):
            c = cache[L]
            for j in range(s):
                r = hs[j]
                h, post, comb = hc_pre(r, layer.hc_attn_fn, layer.hc_attn_scale,
                                       layer.hc_attn_base, layer.hc_mult,
                                       layer.hc_iters, layer.norm_eps,
                                       layer.hc_eps)
                h = layer.attn(layer.attn_norm(h), c)   # [1,1,d] -> decode step()
                hs[j] = hc_post(h, r, post, comb)
            pre, idx_cat = [], []
            for j in range(s):
                r = hs[j]
                h, post, comb = hc_pre(r, layer.hc_ffn_fn, layer.hc_ffn_scale,
                                       layer.hc_ffn_base, layer.hc_mult,
                                       layer.hc_iters, layer.norm_eps,
                                       layer.hc_eps)
                h = layer.ffn_norm(h)
                _, ind = layer.ffn.gate(h.reshape(-1, layer.ffn.dim),
                                        mx.array([toks[j]]))
                pre.append((r, h, post, comb))
                idx_cat.append(ind.reshape(-1))
            idx = np.asarray(mx.concatenate(idx_cat))        # ONE sync per layer
            store.get_many(L, list(dict.fromkeys(idx.tolist())))  # parallel preads
            for j in range(s):
                r, h, post, comb = pre[j]
                y = layer.ffn(h, mx.array([[toks[j]]]))      # full MoE, LRU hits
                hs[j] = hc_post(y, r, post, comb)
    finally:
        store.prefill_no_evict = no_evict
    h = hc_head(hs[-1], model.hc_head_fn, model.hc_head_scale,
                model.hc_head_base, model.args.norm_eps, model.args.hc_eps)
    return model.head(model.norm(h).astype(mx.float32))
