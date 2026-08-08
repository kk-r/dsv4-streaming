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
from expert_store import ExpertStore, StreamingExperts  # noqa: E402
from slot_store import SlotExpertStore, SlotStreamingExperts  # noqa: E402


def load_streaming(repacked: str, cache_gb: float, store_kind: str = "lru"):
    cfg = json.load(open(os.path.join(repacked, "config.json")))
    args = ModelArgs.from_dict(cfg)
    q = cfg["quantization"]

    model = Model(args)
    nn.quantize(model, group_size=q["group_size"], bits=q["bits"],
                class_predicate=quant_predicate(q["group_size"], q["bits"],
                                                q.get("expert_bits")))

    store_cls, experts_cls = ((SlotExpertStore, SlotStreamingExperts)
                              if store_kind == "slot"
                              else (ExpertStore, StreamingExperts))
    store = store_cls(os.path.join(repacked, "experts"),
                      cache_bytes=int(cache_gb * 1e9))
    for i, layer in enumerate(model.layers):
        layer.ffn.experts = experts_cls(
            store, i, group_size=q["group_size"], bits=q.get("expert_bits", 4),
            limit=args.swiglu_limit)

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
    print(f"[load] resident weights in {time.time()-t0:.1f}s "
          f"({mx.get_active_memory()/1e9:.1f} GB active)")
    return model, args, store


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repacked", default=os.path.join(os.path.dirname(__file__), "..", "repacked"))
    ap.add_argument("--cache-gb", type=float, default=8.0)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-new", type=int, default=16)
    ap.add_argument("--store", choices=["lru", "slot"], default="lru")
    args_cli = ap.parse_args()

    model, args, store = load_streaming(args_cli.repacked, args_cli.cache_gb,
                                        args_cli.store)
    tok = load_tokenizer(args_cli.repacked)
    ids = tok.encode(args_cli.prompt)
    print(f"[gen] prompt: {len(ids)} tokens; cache budget {args_cli.cache_gb} GB")

    t0 = time.time()
    out = greedy_generate(model, args, ids, max_new_tokens=args_cli.max_new,
                          verbose=True)
    dt = time.time() - t0
    new = out[len(ids):]
    print(f"\n=== output ===\n{tok.decode(out)}\n==============")
    print(f"[gen] {len(new)} new tokens in {dt:.1f}s -> {len(new)/dt:.2f} tok/s "
          f"(prefill included)")
    print(f"[store] {store.summary()}")
    per_layer = {k: dict(v) for k, v in sorted(store.stats.items())}
    json.dump(per_layer, open(os.path.join(os.path.dirname(__file__), "..",
                                           "logs", "hitrate_last_run.json"), "w"),
              indent=1)
    print(f"[mem] active {mx.get_active_memory()/1e9:.1f} GB, "
          f"peak {mx.get_peak_memory()/1e9:.1f} GB")


if __name__ == "__main__":
    main()
