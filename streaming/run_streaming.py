"""Run DeepSeek-V4-Flash with SSD-streamed experts on a 64 GB Mac.

Builds the pipenetwork model, quantizes the module tree to match the checkpoint,
then swaps every layer's SwitchGLU experts for a StreamingExperts backed by the
repacked per-layer blob files. Only resident weights (~9.4 GB) are loaded into
memory; the 149 GB of routed experts stream from SSD through the LRU store.

  python3 run_streaming.py --repacked ../repacked --cache-gb 8 \
      --prompt "The capital of France is" --max-new 16
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "deepseek-v4-mlx"))
sys.path.insert(0, os.path.dirname(__file__))

from deepseek_v4_mlx.config import ModelArgs          # noqa: E402
from deepseek_v4_mlx.generate import greedy_generate, load_tokenizer  # noqa: E402
from deepseek_v4_mlx.load import quant_predicate      # noqa: E402
from deepseek_v4_mlx.model import Model               # noqa: E402
from expert_store import (ExpertStore, PrefetchCoordinator,  # noqa: E402
                          StackedStreamingExperts, StreamingExperts)


def load_streaming(repacked: str, cache_gb: float, store_kind: str = "lru"):
    cfg = json.load(open(os.path.join(repacked, "config.json")))
    args = ModelArgs.from_dict(cfg)
    # Inference-time top-k override (DSV4_TOPK=4 serves top-4 of the trained
    # top-6): each expert dropped cuts decode reads ~17%; quality cost is
    # measured in logs/topk_experiment.txt before trusting it.
    tk = int(os.environ.get("DSV4_TOPK", "0"))
    if tk:
        args.n_activated_experts = tk
        print(f"[model] top-k override: {tk} of trained 6")
    q = cfg["quantization"]

    model = Model(args)
    nn.quantize(model, group_size=q["group_size"], bits=q["bits"],
                class_predicate=quant_predicate(q["group_size"], q["bits"],
                                                q.get("expert_bits")))

    class NoopExperts:
        """Diagnostic: zero expert output — isolates resident-path cost."""
        def __init__(self, store, layer, **kw):
            self.topk = args.n_activated_experts

        def __call__(self, x, indices):
            return mx.zeros((x.shape[0], self.topk, x.shape[1]), dtype=x.dtype)

    from optimistic import OptimisticExperts, OptimisticStore
    experts_cls = {"stacked": StackedStreamingExperts,
                   "noop": NoopExperts,
                   "optimistic": OptimisticExperts}.get(store_kind, StreamingExperts)
    if store_kind == "optimistic":
        plan_path = os.path.join(os.path.dirname(__file__), "..", "logs",
                                 "slot_plan.json")
        plan = json.load(open(plan_path)) if os.path.exists(plan_path) else None
        store = OptimisticStore(os.path.join(repacked, "experts"),
                                cache_bytes=int(cache_gb * 1e9), slot_plan=plan)
        print(f"[store] slot plan: {'measured profile' if plan else 'uniform'}")
    else:
        store = ExpertStore(os.path.join(repacked, "experts"),
                            cache_bytes=int(cache_gb * 1e9))
    # Pre-gated expert prefetch (DSV4_PREFETCH=1, lru store only): hash layers
    # 0-2 are prefetched exactly from the token id at token start; each scored
    # layer L predicts L+1's routing by applying L+1's resident gate to its own
    # MoE input, then fetches the predicted-missing blobs async while L's
    # experts and L+1's attention compute. Timing-only: routing, math and LRU
    # residency are untouched (mispredictions never enter the cache).
    prefetch_on = (os.environ.get("DSV4_PREFETCH", "0") == "1"
                   and experts_cls is StreamingExperts)
    coord = None
    if prefetch_on:
        workers = int(os.environ.get("DSV4_PREFETCH_WORKERS", "6"))
        coord = PrefetchCoordinator(store, workers=workers)
        print(f"[prefetch] pre-gated prefetch ON ({workers} workers)")
    for i, layer in enumerate(model.layers):
        kw = {}
        if coord is not None:
            nxt = i + 1
            kw["coordinator"] = coord
            if nxt < args.n_layers and not args.is_hash_layer(nxt):
                kw["next_gate"] = model.layers[nxt].ffn.gate
        layer.ffn.experts = experts_cls(
            store, i, group_size=q["group_size"], bits=q.get("expert_bits", 4),
            limit=args.swiglu_limit, **kw)

    expected = {k for k, _ in tree_flatten(model.parameters())}
    loaded = set()
    t0 = time.time()
    for shard in sorted(glob.glob(os.path.join(repacked, "resident-*.safetensors"))):
        w = mx.load(shard)
        model.load_weights(list(w.items()), strict=False)
        mx.eval(list(w.values()))
        loaded.update(w.keys())
        del w
    missing = expected - loaded
    if missing:
        raise ValueError(f"{len(missing)} resident params missing, "
                         f"e.g. {sorted(missing)[:3]}")
    model.eval()
    if coord is not None:
        # After weights load: snapshot the hash tables to CPU and install the
        # decode-time hook that prefetches layers 0-2 from the token id.
        coord.register_hash_layers(
            {i: model.layers[i].ffn.gate for i in range(args.n_hash_layers)})
        model.token_hook = coord.token_start
    print(f"[load] resident weights in {time.time()-t0:.1f}s "
          f"({mx.get_active_memory()/1e9:.1f} GB active)")
    return model, args, store


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repacked", default=os.path.join(os.path.dirname(__file__), "..", "repacked"))
    ap.add_argument("--cache-gb", type=float, default=8.0)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-new", type=int, default=16)
    ap.add_argument("--store", choices=["lru", "stacked", "noop", "optimistic"],
                    default="lru")
    ap.add_argument("--repeat", type=int, default=1)
    args_cli = ap.parse_args()

    model, args, store = load_streaming(args_cli.repacked, args_cli.cache_gb,
                                        args_cli.store)
    tok = load_tokenizer(args_cli.repacked)
    ids = tok.encode(args_cli.prompt)
    print(f"[gen] prompt: {len(ids)} tokens; cache budget {args_cli.cache_gb} GB")

    from optimistic import optimistic_generate
    for run in range(args_cli.repeat):
        t0 = time.time()
        if args_cli.store == "optimistic":
            out = optimistic_generate(model, args, store, ids,
                                      max_new_tokens=args_cli.max_new, verbose=True)
        else:
            out = greedy_generate(model, args, ids, max_new_tokens=args_cli.max_new,
                                  verbose=True)
        dt = time.time() - t0
        n_new = len(out) - len(ids)
        print(f"[gen] run {run+1}: {n_new} tokens in {dt:.1f}s "
              f"-> {n_new/dt:.2f} tok/s | store {store.summary()}")
    new = out[len(ids):]
    print(f"\n=== output ===\n{tok.decode(out)}\n==============")
    print(f"[gen] {len(new)} new tokens in {dt:.1f}s -> {len(new)/dt:.2f} tok/s "
          f"(prefill included)")
    print(f"[store] {store.summary()}")
    if store.coordinator is not None:
        print(f"[prefetch] {store.coordinator.report()}")
    per_layer = {k: dict(v) for k, v in sorted(store.stats.items())}
    json.dump(per_layer, open(os.path.join(os.path.dirname(__file__), "..",
                                           "logs", "hitrate_last_run.json"), "w"),
              indent=1)
    print(f"[mem] active {mx.get_active_memory()/1e9:.1f} GB, "
          f"peak {mx.get_peak_memory()/1e9:.1f} GB")


if __name__ == "__main__":
    main()
