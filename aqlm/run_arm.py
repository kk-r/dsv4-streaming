"""Batch-quantize layers for one arm, writing replacement layer_NN.bin files.

  python3 run_arm.py --arm B --layers 18-23 --workers 10

Output: aqlm/out_<ARM>/layer_NN.bin (same blob layout as repacked/experts) plus
layer_NN.progress.json (resume checkpoint) and mse.jsonl (per-expert stats).
Workers are numpy + mx-CPU only — no GPU use, safe to run anytime.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time

# single-threaded BLAS in workers; parallelism comes from the pool
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aqlm_lib as L  # noqa: E402

_SPECS = None  # per-process layout cache


def _spec(layer):
    global _SPECS
    if _SPECS is None:
        _SPECS = L.load_layout()["layers"]
    return _SPECS[str(layer)]


def work(task):
    layer, expert, arm = task
    spec = _spec(layer)
    blob = L.read_blob(layer, expert, spec)
    t0 = time.time()
    tensors, stats = {}, {"layer": layer, "expert": expert, "arm": arm, "proj": {}}
    for proj in L.PROJS:
        W = L.dequant_proj(blob, spec, proj)
        pstat = {}
        if arm == "A":
            Wh = W
        elif arm == "B":
            Wh, cs = L.codebook2bit(W, seed=layer * 1000 + expert)
            pstat["codebook_util"] = cs["codebook_util"]
            pstat["rel_mse_quant"] = round(cs["rel_mse"], 6)
        elif arm == "C":
            Wh = L.affine2bit_roundtrip(W)
            pstat["rel_mse_quant"] = round(L.rel_mse(W, Wh), 6)
        else:
            raise ValueError(arm)
        wq, sc16, bi16 = L.quantize_container(Wh)
        Wfin = L.dequant_container(wq, sc16, bi16, W.shape[0])
        pstat["rel_mse_final"] = round(L.rel_mse(W, Wfin), 6)
        stats["proj"][proj] = pstat
        tensors[f"{proj}.weight"] = wq
        tensors[f"{proj}.scales"] = sc16
        tensors[f"{proj}.biases"] = bi16
    stats["seconds"] = round(time.time() - t0, 2)
    return layer, expert, L.build_blob(spec, tensors), stats


def parse_layers(s):
    out = set()
    for tok in s.split(","):
        if "-" in tok:
            a, b = tok.split("-")
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(tok))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["A", "B", "C"])
    ap.add_argument("--layers", default="18-23")
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()

    layers = parse_layers(a.layers)
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"out_{a.arm}")
    os.makedirs(out_dir, exist_ok=True)
    layout = L.load_layout()
    mse_path = os.path.join(out_dir, "mse.jsonl")

    for layer in layers:
        spec = layout["layers"][str(layer)]
        n = spec["n_experts"]
        path = os.path.join(out_dir, f"layer_{layer:02d}.bin")
        prog_path = path.replace(".bin", ".progress.json")
        done = set()
        if os.path.exists(prog_path):
            done = set(json.load(open(prog_path))["done"])
        if not os.path.exists(path) or os.path.getsize(path) != spec["stride"] * n:
            with open(path, "wb") as f:
                f.truncate(spec["stride"] * n)
            done = set()
        todo = [(layer, e, a.arm) for e in range(n) if e not in done]
        if not todo:
            print(f"[{a.arm}] layer {layer}: complete ({n} experts), skipping")
            continue
        print(f"[{a.arm}] layer {layer}: {len(todo)} experts to do "
              f"({len(done)} already)", flush=True)
        t0 = time.time()
        fout = open(path, "r+b")
        mse_f = open(mse_path, "a")
        with mp.get_context("spawn").Pool(a.workers) as pool:
            for i, (lyr, e, blob, stats) in enumerate(
                    pool.imap_unordered(work, todo, chunksize=1)):
                os.pwrite(fout.fileno(), blob, e * spec["stride"])
                mse_f.write(json.dumps(stats) + "\n")
                done.add(e)
                if (i + 1) % 16 == 0 or i + 1 == len(todo):
                    os.fsync(fout.fileno())
                    mse_f.flush()
                    json.dump({"done": sorted(done)}, open(prog_path, "w"))
                    el = time.time() - t0
                    print(f"[{a.arm}] layer {lyr}: {i+1}/{len(todo)} "
                          f"({el:.0f}s, {el/(i+1):.1f}s/expert)", flush=True)
        fout.close()
        mse_f.close()
        json.dump({"done": sorted(done)}, open(prog_path, "w"))
        print(f"[{a.arm}] layer {layer}: DONE in {time.time()-t0:.0f}s", flush=True)
    print(f"[{a.arm}] all layers done")


if __name__ == "__main__":
    main()
