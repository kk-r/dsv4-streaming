"""Diverse-workload eval: 8 prompts through one persistent-cache process.

Loads the streaming model once, then runs 8 prompts from different domains
sequentially so the expert LRU cache warms ACROSS prompts. Records per-prompt
tok/s and store-stats deltas (hits / misses / GB read).

  python3 multi_prompt_eval.py --cache-gb 44
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import mlx.core as mx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "deepseek-v4-mlx"))

from run_streaming import load_streaming                       # noqa: E402
from deepseek_v4_mlx.generate import greedy_generate, load_tokenizer  # noqa: E402

PROMPTS = [
    ("factual", "What is the tallest mountain in the world, and how tall is it?"),
    ("coding", "Write a Python function that returns the n-th Fibonacci number "
               "using iteration:\n\ndef fib(n):"),
    ("math", "A train travels 60 miles per hour for 2.5 hours, then 40 miles "
             "per hour for 1.5 hours. How many miles did it travel in total? "
             "Let's work through it step by step."),
    ("creative", "The lighthouse keeper found the letter on the forty-first "
                 "morning of the fog. It began:"),
    ("translation", "Translate the following sentence into French: "
                    "\"The weather is beautiful today, so we will walk to the "
                    "market and buy fresh bread.\""),
    ("science", "Explain in simple terms why the sky is blue during the day "
                "but red at sunset."),
    ("history", "Who was the first emperor of Rome, and how did he come to "
                "power?"),
    ("casual", "Hey! I just got back from a week of camping in the rain. Any "
               "tips for drying out my gear?"),
]


def stats_snapshot(store):
    return {
        "hits": sum(s["hits"] for s in store.stats.values()),
        "misses": sum(s["misses"] for s in store.stats.values()),
        "bytes": sum(s["bytes"] for s in store.stats.values()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repacked", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "repacked"))
    ap.add_argument("--cache-gb", type=float, default=44.0)
    ap.add_argument("--max-new", type=int, default=48)
    args_cli = ap.parse_args()

    logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "logs")
    os.makedirs(logs_dir, exist_ok=True)

    model, args, store = load_streaming(args_cli.repacked, args_cli.cache_gb,
                                        "lru")
    tok = load_tokenizer(args_cli.repacked)

    results = []
    for i, (domain, prompt) in enumerate(PROMPTS, 1):
        ids = tok.encode(prompt)
        before = stats_snapshot(store)
        t0 = time.time()
        out = greedy_generate(model, args, ids,
                              max_new_tokens=args_cli.max_new, verbose=False)
        dt = time.time() - t0
        after = stats_snapshot(store)

        n_new = len(out) - len(ids)
        d_hits = after["hits"] - before["hits"]
        d_miss = after["misses"] - before["misses"]
        d_gb = (after["bytes"] - before["bytes"]) / 1e9
        rate = d_hits / (d_hits + d_miss) if d_hits + d_miss else 0.0
        text = tok.decode(out[len(ids):])

        rec = {
            "idx": i, "domain": domain, "prompt": prompt,
            "prompt_tokens": len(ids), "new_tokens": n_new,
            "seconds": round(dt, 2), "tok_s": round(n_new / dt, 3),
            "hits": d_hits, "misses": d_miss, "hit_rate": round(rate, 4),
            "gb_read": round(d_gb, 2),
            "cum_misses": after["misses"],
            "cached_gb": round(store.used / 1e9, 2),
            "output_head": text[:60],
        }
        results.append(rec)
        print(f"[{i}/8] {domain:11s} {n_new} tok in {dt:.1f}s "
              f"-> {rec['tok_s']:.2f} tok/s | hit {rate:.3f} "
              f"({d_hits}h/{d_miss}m, {d_gb:.1f} GB) | "
              f"cache {rec['cached_gb']:.1f} GB | {text[:60]!r}", flush=True)

    total = stats_snapshot(store)
    summary = {
        "cache_gb_budget": args_cli.cache_gb,
        "max_new_tokens": args_cli.max_new,
        "total_hits": total["hits"], "total_misses": total["misses"],
        "overall_hit_rate": round(
            total["hits"] / (total["hits"] + total["misses"]), 4),
        "total_gb_read": round(total["bytes"] / 1e9, 2),
        "cached_gb_final": round(store.used / 1e9, 2),
        "peak_mem_gb": round(mx.get_peak_memory() / 1e9, 2),
        "active_mem_gb": round(mx.get_active_memory() / 1e9, 2),
    }

    json.dump({"summary": summary, "prompts": results},
              open(os.path.join(logs_dir, "diverse_eval.json"), "w"), indent=1)

    lines = [
        f"Diverse-workload eval — 8 prompts, persistent cache "
        f"({args_cli.cache_gb:.0f} GB budget), max_new={args_cli.max_new}",
        "",
        f"{'#':>2} {'domain':11s} {'tok/s':>6} {'hit%':>6} {'GBread':>7} "
        f"{'miss':>5} {'cumMiss':>7}  output (first 60 chars)",
        "-" * 110,
    ]
    for r in results:
        lines.append(
            f"{r['idx']:>2} {r['domain']:11s} {r['tok_s']:>6.2f} "
            f"{100*r['hit_rate']:>5.1f}% {r['gb_read']:>7.2f} "
            f"{r['misses']:>5} {r['cum_misses']:>7}  {r['output_head']!r}")
    lines += [
        "-" * 110,
        f"overall hit rate {100*summary['overall_hit_rate']:.1f}%  "
        f"total read {summary['total_gb_read']:.1f} GB  "
        f"unique experts (cum misses) {summary['total_misses']}  "
        f"cache used {summary['cached_gb_final']:.1f} GB  "
        f"peak mem {summary['peak_mem_gb']:.1f} GB",
    ]
    txt = "\n".join(lines) + "\n"
    open(os.path.join(logs_dir, "diverse_eval.txt"), "w").write(txt)
    print("\n" + txt)


if __name__ == "__main__":
    main()
