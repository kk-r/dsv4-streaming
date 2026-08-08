"""Slot-pool expert store: batched gather_qmm compute + parallel miss reads.

v1 of the streaming store. Where expert_store.py held each cached expert as
its own set of arrays and looped per expert (kernel-launch bound, ~480 ms/token),
this keeps one preallocated stacked pool per layer per projection:

    weight [S, out, in_packed]  scales/biases [S, out, groups]

Cached experts occupy slots; an LRU per layer decides eviction. A visit maps
expert ids -> slot ids and runs one mx.gather_qmm per projection — the same
kernel SwitchGLU uses — so compute per token collapses to 3 calls/layer.
Misses are pread in parallel (thread pool; SSD does ~13 GB/s only when asked
concurrently), decoded to numpy, then written into slots with in-place
setitem; MLX buffer donation keeps that from copying the pool.
"""

from __future__ import annotations

import json
import os
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor

import mlx.core as mx
import numpy as np

NP_DTYPE = {"U32": np.uint32, "BF16": np.uint16, "F16": np.float16, "F32": np.float32}
PROJS = ("gate_proj", "up_proj", "down_proj")


class SlotLayer:
    def __init__(self, path, spec, n_slots):
        self.f = open(path, "rb", buffering=0)
        self.stride = spec["stride"]
        self.slices = spec["slices"]
        self.n_slots = n_slots
        self.slot_of = OrderedDict()  # expert_id -> slot (LRU: last = most recent)
        self.pool = {}
        for proj in PROJS:
            for part, key in (("weight", f"{proj}.weight"), ("scales", f"{proj}.scales"),
                              ("biases", f"{proj}.biases")):
                s = self.slices[key]
                dt = mx.uint32 if s["dtype"] == "U32" else mx.bfloat16
                self.pool[key] = mx.zeros((n_slots, *s["shape"]), dtype=dt)
        mx.eval(list(self.pool.values()))

    def read_blob(self, expert_id):
        return os.pread(self.f.fileno(), self.stride, expert_id * self.stride)

    def decode(self, blob, key):
        s = self.slices[key]
        dt = NP_DTYPE[s["dtype"]]
        arr = np.frombuffer(blob, dtype=dt, count=s["nbytes"] // np.dtype(dt).itemsize,
                            offset=s["offset"]).reshape(s["shape"])
        return mx.array(arr).view(mx.bfloat16) if s["dtype"] == "BF16" else mx.array(arr)


class SlotExpertStore:
    def __init__(self, expert_dir, cache_bytes, io_threads=8):
        layout = json.load(open(os.path.join(expert_dir, "layout.json")))
        n_layers = len(layout["layers"])
        stride = next(iter(layout["layers"].values()))["stride"]
        n_slots = max(8, int(cache_bytes / n_layers / stride))
        self.layers = {
            int(k): SlotLayer(os.path.join(expert_dir, f"layer_{int(k):02d}.bin"),
                              v, n_slots)
            for k, v in layout["layers"].items()
        }
        self.pool = ThreadPoolExecutor(max_workers=io_threads)
        self.stats = defaultdict(lambda: {"hits": 0, "misses": 0, "bytes": 0})
        self.n_slots = n_slots

    def slots_for(self, layer, expert_ids):
        """Map expert ids -> slot ids, streaming in any misses (parallel reads)."""
        lf = self.layers[layer]
        st = self.stats[layer]
        need = []
        for e in expert_ids:
            if e in lf.slot_of:
                st["hits"] += 1
                lf.slot_of.move_to_end(e)
            else:
                st["misses"] += 1
                need.append(e)

        if need:
            st["bytes"] += len(need) * lf.stride
            blobs = list(self.pool.map(lf.read_blob, need))
            protect = set(expert_ids)
            for e, blob in zip(need, blobs):
                if len(lf.slot_of) < lf.n_slots:
                    slot = len(lf.slot_of)
                else:  # evict LRU not part of this visit
                    victim = next(k for k in lf.slot_of if k not in protect)
                    slot = lf.slot_of.pop(victim)
                lf.slot_of[e] = slot
                for proj in PROJS:
                    for part in ("weight", "scales", "biases"):
                        key = f"{proj}.{part}"
                        lf.pool[key][slot] = lf.decode(blob, key)
            mx.eval(list(lf.pool.values()))
        return [lf.slot_of[e] for e in expert_ids]

    def summary(self):
        h = sum(s["hits"] for s in self.stats.values())
        m = sum(s["misses"] for s in self.stats.values())
        gb = sum(s["bytes"] for s in self.stats.values()) / 1e9
        return {"hits": h, "misses": m,
                "hit_rate": round(h / (h + m), 4) if h + m else 0.0,
                "read_gb": round(gb, 2), "slots_per_layer": self.n_slots}


class SlotStreamingExperts:
    """SwitchGLU-compatible expert call over the slot pools."""

    def __init__(self, store: SlotExpertStore, layer: int, group_size=64, bits=4,
                 limit: float = 0.0):
        self.store, self.layer = store, layer
        self.group_size, self.bits = group_size, bits
        self.limit = limit

    def _qmm(self, x, key_base, slot_idx):
        lf = self.store.layers[self.layer]
        return mx.gather_qmm(
            x, lf.pool[f"{key_base}.weight"], lf.pool[f"{key_base}.scales"],
            lf.pool[f"{key_base}.biases"], rhs_indices=slot_idx,
            transpose=True, group_size=self.group_size, bits=self.bits)

    def __call__(self, x, indices):
        # x [tokens, dim], indices [tokens, topk] -> [tokens, topk, dim]
        idx = np.asarray(indices)
        toks, topk = idx.shape
        flat = idx.reshape(-1)
        uniq = list(dict.fromkeys(flat.tolist()))
        slot_of = dict(zip(uniq, self.store.slots_for(self.layer, uniq)))
        slot_idx = mx.array(np.array([slot_of[e] for e in flat], dtype=np.int32)
                            .reshape(toks, topk))

        xe = mx.expand_dims(x, (-2, -3))          # [tokens, 1, 1, dim]
        gate = self._qmm(xe, "gate_proj", slot_idx).astype(mx.float32)
        up = self._qmm(xe, "up_proj", slot_idx).astype(mx.float32)
        if self.limit > 0:
            up = mx.clip(up, -self.limit, self.limit)
            gate = mx.minimum(gate, self.limit)
        h = (mx.sigmoid(gate) * gate * up).astype(x.dtype)
        return self._qmm(h, "down_proj", slot_idx).squeeze(-2)
