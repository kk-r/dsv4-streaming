"""Teacher-forced perplexity through the SSD-streaming path.

Same window construction and NLL computation as
deepseek-v4-mlx/scripts/ppl_large.py (corpus .npy from ppl_corpus.py logic,
fixed windows of SEQ tokens, full-window forward, per-window NLL), but the
model is loaded through load_streaming() — resident weights + StreamingExperts
over the repacked blob store — instead of their load(). A matching aggregate
against pipenetwork's published number is the numerical-transparency check for
the streaming path.

    python3 ppl_streaming.py --windows 8 --cache-gb 8 \
        --corpus ../logs/ppl_corpus.npy --out ../logs/ppl_streaming.txt
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import mlx.core as mx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_streaming import load_streaming  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def log(f, msg):
    print(msg, flush=True)
    f.write(msg + "\n")
    f.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repacked", default=os.path.join(ROOT, "repacked"))
    ap.add_argument("--cache-gb", type=float, default=8.0)
    ap.add_argument("--corpus", default=os.path.join(ROOT, "logs", "ppl_corpus.npy"))
    ap.add_argument("--windows", type=int, default=8)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--out", default=os.path.join(ROOT, "logs", "ppl_streaming.txt"))
    a = ap.parse_args()

    ids_all = np.load(a.corpus)
    n_win = min(a.windows, len(ids_all) // a.seq)
    f = open(a.out, "a")
    log(f, f"=== ppl_streaming start {time.strftime('%F %T')} ===")
    log(f, f"corpus {a.corpus} ({len(ids_all):,} tokens, "
           f"{len(ids_all) // a.seq} windows available); running first {n_win} "
           f"windows x {a.seq} tokens; cache {a.cache_gb} GB, store lru")

    t0 = time.time()
    model, args, store = load_streaming(a.repacked, a.cache_gb, "lru")
    log(f, f"[load] done in {time.time()-t0:.1f}s")

    win_nll, win_tok = [], []
    t0 = time.time()
    for w in range(n_win):
        tw = time.time()
        ids = ids_all[w * a.seq:(w + 1) * a.seq].tolist()
        lg = model(mx.array([ids]))[0].astype(mx.float32)
        lg = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
        nll = -lg[mx.arange(len(ids) - 1), mx.array(ids[1:])]
        s = float(nll.sum().item())
        win_nll.append(s)
        win_tok.append(len(ids) - 1)
        mx.clear_cache()
        dt = time.time() - tw
        cum = math.exp(sum(win_nll) / sum(win_tok))
        log(f, f"[win {w+1}/{n_win}] nll_sum {s:.2f} over {len(ids)-1} tok  "
               f"ppl {math.exp(s/(len(ids)-1)):.4f}  cumulative ppl {cum:.4f}  "
               f"{dt:.0f}s ({len(ids)/dt:.2f} tok/s)  "
               f"store {store.summary()}  "
               f"mem peak {mx.get_peak_memory()/1e9:.1f} GB")

    ppl = math.exp(sum(win_nll) / sum(win_tok))
    el = time.time() - t0
    log(f, f"[done] aggregate ppl {ppl:.4f} over {sum(win_tok):,} scored tokens "
           f"({n_win} windows x {a.seq})  total {el:.0f}s "
           f"({n_win*a.seq/el:.2f} tok/s incl load-excluded)")
    res = {"windows": n_win, "seq": a.seq, "cache_gb": a.cache_gb,
           "window_nll": win_nll, "window_tok": win_tok,
           "ppl": ppl, "seconds": el}
    log(f, "[json] " + json.dumps(res))
    f.close()


if __name__ == "__main__":
    main()
