"""Offline unit tests for pre-gated prefetch (ExpertPrefetcher/PrefetchCoordinator).

Builds the same synthetic 2-layer, 4-expert repack as test_roundtrip, then checks:
  1. a prefetched entry is bit-identical to a demand-decoded one
  2. store.get serves misses from the buffer, with correct counters, and the
     LRU residency/order stays identical to a prefetch-free store
  3. request dedupe, purge/wasted accounting
  4. StreamingExperts with a perfect next-layer prediction: outputs bitwise
     equal to the no-prefetch path, prediction hit rate 1.0, misses served
  5. misprediction: outputs still bitwise equal, hit rate 0, wasted counted,
     mispredicted blobs never enter the LRU
  6. coordinator token_start: hash-table prefetch issues the right keys

No model, no repacked/ access — safe to run while the GPU is busy.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import mlx.core as mx
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from expert_store import (ExpertPrefetcher, ExpertStore,  # noqa: E402
                          PrefetchCoordinator, StreamingExperts, _wrap_np)

DIM, INTER, N_EXP, TOPK, GS, BITS = 128, 64, 4, 2, 64, 4
rng = np.random.default_rng(0)


def make_repack(tmp):
    snap = os.path.join(tmp, "snap"); os.makedirs(snap)
    out = os.path.join(tmp, "out")
    tensors = {}
    for lyr in range(2):
        for proj, (out_d, in_d) in {"gate_proj": (INTER, DIM), "up_proj": (INTER, DIM),
                                    "down_proj": (DIM, INTER)}.items():
            w = mx.array(rng.standard_normal((N_EXP, out_d, in_d), dtype=np.float32) * 0.05)
            qw, scales, biases = mx.quantize(w, group_size=GS, bits=BITS)
            base = f"layers.{lyr}.ffn.experts.{proj}"
            tensors[f"{base}.weight"] = qw
            tensors[f"{base}.scales"] = scales.astype(mx.bfloat16)
            tensors[f"{base}.biases"] = biases.astype(mx.bfloat16)
        tensors[f"layers.{lyr}.attn.norm.weight"] = mx.ones((DIM,))
    mx.save_safetensors(os.path.join(snap, "model-00001-of-00001.safetensors"), tensors)
    json.dump({"quantization": {"group_size": GS, "bits": BITS}},
              open(os.path.join(snap, "config.json"), "w"))
    r = subprocess.run([sys.executable,
                        os.path.join(os.path.dirname(__file__), "repack.py"),
                        "--snapshot", snap, "--out", out],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return os.path.join(out, "experts")


def entries_equal(a, b):
    if set(a) != set(b):
        return False
    return all(mx.array_equal(a[k].view(mx.uint16) if a[k].dtype == mx.bfloat16
                              else a[k],
                              b[k].view(mx.uint16) if b[k].dtype == mx.bfloat16
                              else b[k]) for k in a)


class StubGate:
    """next_gate stand-in: returns fixed indices regardless of input."""
    def __init__(self, ids):
        self.ids = ids

    def __call__(self, x):
        return None, mx.array(np.array([self.ids], dtype=np.int32))


def wait_idle(pf, timeout=5.0):
    t0 = time.time()
    while pf.futures and time.time() - t0 < timeout:
        time.sleep(0.005)


def main():
    tmp = tempfile.mkdtemp()
    try:
        exp_dir = make_repack(tmp)
        cap = 10 * 1024 * 1024

        # --- 1. prefetched entry bit-identical to demand decode ---
        store = ExpertStore(exp_dir, cache_bytes=cap)
        pf = ExpertPrefetcher(store, workers=2)
        pf.request(0, [1])
        got = pf.take(0, 1)
        assert got is not None
        want = store._decode(0, store.layers[0].read_blob(1))
        assert entries_equal(_wrap_np(got), want), "prefetch decode mismatch"
        assert pf.issued == 1 and pf.served_ready + pf.served_wait == 1
        print("1. prefetch decode bit-identical: OK")

        # --- 2. store.get serves from buffer; counters; LRU state identical ---
        store_t = ExpertStore(exp_dir, cache_bytes=cap)      # treatment
        store_c = ExpertStore(exp_dir, cache_bytes=cap)      # control
        pf_t = ExpertPrefetcher(store_t, workers=2)
        store_t.prefetcher = pf_t
        seq = [(0, 2), (0, 3), (1, 0), (0, 2), (1, 3)]
        pf_t.request(0, [2])         # predict one of the upcoming misses
        for l, e in seq:
            a = store_t.get(l, e)
            b = store_c.get(l, e)
            assert entries_equal(a, b)
        assert store_t.prefetch_served == 1 and pf_t.served_ready + pf_t.served_wait == 1
        assert store_t.summary()["hits"] == store_c.summary()["hits"]
        assert store_t.summary()["misses"] == store_c.summary()["misses"]
        assert list(store_t.cache.keys()) == list(store_c.cache.keys()), \
            "LRU residency/order diverged"
        assert "pf_issued" in store_t.summary() and "pf_issued" not in store_c.summary()
        print("2. store.get buffer serve + LRU parity: OK")

        # --- 3. dedupe + purge/wasted ---
        pf_t.request(1, [1, 1, 1])
        assert pf_t.issued == 2, "dedupe failed"       # 1 from test 2, 1 new
        wait_idle(pf_t)
        pf_t.purge(1)
        assert pf_t.wasted == 1 and not pf_t.buf and not pf_t.futures
        pf_t.request(1, [1])                            # re-request after purge works
        assert pf_t.take(1, 1) is not None
        print("3. dedupe + purge accounting: OK")

        # --- 4. perfect prediction through StreamingExperts ---
        x = mx.array(rng.standard_normal((1, DIM), dtype=np.float32))
        idx0 = mx.array(np.array([[0, 2]], dtype=np.int32))
        idx1 = mx.array(np.array([[1, 3]], dtype=np.int32))

        store_p = ExpertStore(exp_dir, cache_bytes=cap)
        coord = PrefetchCoordinator(store_p, workers=2)
        se0 = StreamingExperts(store_p, 0, group_size=GS, bits=BITS,
                               coordinator=coord, next_gate=StubGate([1, 3]))
        se1 = StreamingExperts(store_p, 1, group_size=GS, bits=BITS,
                               coordinator=coord)
        store_r = ExpertStore(exp_dir, cache_bytes=cap)
        r0 = StreamingExperts(store_r, 0, group_size=GS, bits=BITS)
        r1 = StreamingExperts(store_r, 1, group_size=GS, bits=BITS)

        y0, w0 = se0(x, idx0), r0(x, idx0)
        y1, w1 = se1(x, idx1), r1(x, idx1)
        assert np.array_equal(np.asarray(y0), np.asarray(w0))
        assert np.array_equal(np.asarray(y1), np.asarray(w1))
        s = coord.pred_stats[1]
        assert s["pred_hit"] == 2 and s["pred_total"] == 2
        assert store_p.prefetch_served == 2, store_p.prefetch_served
        assert list(store_p.cache.keys()) == list(store_r.cache.keys())
        print("4. perfect pre-gate prediction, bitwise outputs: OK")

        # --- 5. misprediction: correct output, nothing cached, wasted counted ---
        store_m = ExpertStore(exp_dir, cache_bytes=cap)
        coordm = PrefetchCoordinator(store_m, workers=2)
        m0 = StreamingExperts(store_m, 0, group_size=GS, bits=BITS,
                              coordinator=coordm, next_gate=StubGate([0, 2]))
        m1 = StreamingExperts(store_m, 1, group_size=GS, bits=BITS,
                              coordinator=coordm)
        z0 = m0(x, idx0)
        z1 = m1(x, idx1)                     # actual 1,3 vs predicted 0,2
        assert np.array_equal(np.asarray(z0), np.asarray(w0))
        assert np.array_equal(np.asarray(z1), np.asarray(w1))
        s = coordm.pred_stats[1]
        assert s["pred_hit"] == 0 and s["pred_total"] == 2
        assert coordm.prefetcher.wasted == 2
        assert (1, 0) not in store_m.cache and (1, 2) not in store_m.cache
        assert list(store_m.cache.keys()) == list(store_r.cache.keys())
        print("5. misprediction purged, outputs bitwise equal: OK")

        # --- 6. token_start hash prefetch ---
        store_h = ExpertStore(exp_dir, cache_bytes=cap)
        coordh = PrefetchCoordinator(store_h, workers=2)

        class HashGate:
            tid2eid = mx.array(np.array([[0, 1], [2, 3]], dtype=np.int32))
        coordh.register_hash_layers({0: HashGate(), 1: HashGate()})
        coordh.token_start(mx.array([[1]]))
        assert coordh.predicted[0] == {2, 3} and coordh.predicted[1] == {2, 3}
        wait_idle(coordh.prefetcher)
        assert set(coordh.prefetcher.buf) == {(0, 2), (0, 3), (1, 2), (1, 3)}
        # a second token_start purges the stale ones
        coordh.token_start(mx.array([[0]]))
        assert coordh.prefetcher.wasted == 4
        print("6. token_start hash prefetch + stale purge: OK")

        print("PREFETCH TESTS OK")
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
