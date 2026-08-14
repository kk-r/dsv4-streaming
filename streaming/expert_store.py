"""SSD-backed expert store with an LRU slot cache and hit-rate instrumentation.

Explicit pread into reusable slots — never mmap page faults — per the
TurboFieldfare finding that fault-driven streaming is too slow and per this
project's llama.cpp baseline, where fault storms SIGBUSed under a raised wired
limit. A blob (one expert's nine quantized slices, ~14.2 MB) is the unit of
transfer and of caching.

v0 scope: correctness + instrumentation. Single-threaded reads, no prefetch;
decode batch of 1. Prefetch and chunked prefill come after the hit-rate curve
says the cache actually works.
"""

from __future__ import annotations

import fcntl
import json
import os
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor

import mlx.core as mx
import numpy as np

NP_DTYPE = {"U32": np.uint32, "BF16": np.uint16, "F16": np.float16, "F32": np.float32}

F_NOCACHE = 48  # macOS fcntl(2): bypass the unified buffer cache on this fd


class LayerFile:
    """One per-layer blob file, held open on two descriptors.

    self.f goes through the page cache — the decode path depends on the OS
    caching recently read blobs (see README, cache-size sweep). self.fd_nocache
    has F_NOCACHE set: prefill streams >200 GB of misses through get_many, and
    routing those reads here keeps them from evicting the file pages decode
    needs. Blobs are 16 KB-aligned fixed stride, so nocache preads stay aligned.
    """

    def __init__(self, path, spec):
        self.f = open(path, "rb", buffering=0)
        self.fd_nocache = os.open(path, os.O_RDONLY)
        fcntl.fcntl(self.fd_nocache, F_NOCACHE, 1)
        self.stride = spec["stride"]
        self.n_experts = spec["n_experts"]
        self.slices = spec["slices"]

    def read_blob(self, expert_id):
        return os.pread(self.f.fileno(), self.stride, expert_id * self.stride)

    def read_blob_nocache(self, expert_id):
        return os.pread(self.fd_nocache, self.stride, expert_id * self.stride)


class ExpertStore:
    """cache_bytes of expert blobs across all layers, global LRU."""

    def __init__(self, expert_dir, cache_bytes, io_threads=8,
                 prefill_nocache: bool | None = None):
        layout = json.load(open(os.path.join(expert_dir, "layout.json")))
        self.layers = {
            int(k): LayerFile(os.path.join(expert_dir, f"layer_{int(k):02d}.bin"), v)
            for k, v in layout["layers"].items()
        }
        self.capacity = cache_bytes
        self.cache = OrderedDict()  # (layer, expert) -> dict of mx arrays
        self.used = 0
        self.stats = defaultdict(lambda: {"hits": 0, "misses": 0, "bytes": 0})
        self.io = ThreadPoolExecutor(max_workers=io_threads)
        # Route get_many (the prefill path) through the F_NOCACHE descriptor so
        # a long prefill cannot evict the page-cache blobs decode relies on.
        # get (the decode path) always uses the cached descriptor. Blobs fetched
        # by get_many still land in the MLX-side LRU either way, so prefill's
        # own intra-prompt reuse is unaffected; only OS-level caching of
        # prefill-read file pages is suppressed. Default on; disable with
        # DSV4_PREFILL_NOCACHE=0 or prefill_nocache=False.
        if prefill_nocache is None:
            # Default OFF: the A/B (logs/fnocache_ab.txt) measured no decode
            # recovery and up to +75% prefill cost. Kept behind the flag for
            # re-testing on a quiet machine.
            prefill_nocache = os.environ.get("DSV4_PREFILL_NOCACHE", "0") == "1"
        self.prefill_nocache = prefill_nocache
        # ds4-style prefill insert policy (docs/ds4-recon.md §2.4): prefill may
        # FILL the LRU but never EVICT for it. Under DSV4_PREFILL_NO_EVICT=1 a
        # get_many miss is decoded and returned for the pass, but inserted into
        # the LRU only while free capacity remains — an existing entry is never
        # evicted on behalf of a prefill batch. get (the decode path) is
        # unchanged and still evicts normally. Default OFF for a clean A/B.
        # Default ON: the A/B (logs/prefill_lru.txt) showed decode unchanged
        # but prefill 2x FASTER — skipping insert/evict churn for blobs the
        # LRU cannot hold anyway. Disable with DSV4_PREFILL_NO_EVICT=0.
        self.prefill_no_evict = os.environ.get("DSV4_PREFILL_NO_EVICT", "1") != "0"
        self.noevict_skipped = 0  # misses returned for the pass but not cached
        # Per-path counters: get() is only called by single-token decode,
        # get_many() only by batched prefill, so this splits the aggregate
        # stats by phase without touching the callers.
        self.decode_stats = {"hits": 0, "misses": 0, "bytes": 0}
        print(f"[store] prefill (get_many) reads: "
              f"{'F_NOCACHE' if self.prefill_nocache else 'page-cached'}; "
              f"insert policy: "
              f"{'fill-no-evict' if self.prefill_no_evict else 'lru-evict'}")

    def get_many(self, layer, expert_ids):
        """Fetch several experts, reading misses from SSD in parallel."""
        st = self.stats[layer]
        lf = self.layers[layer]
        # Mark this batch's already-cached experts most-recent BEFORE inserting
        # misses: the eviction loop below pops from the LRU front, and on a warm
        # full cache it could otherwise evict a blob this same batch is about to
        # use (out{} would then silently omit it -> KeyError downstream). First
        # hit in practice: a server prefilling request N+1 over request N's cache.
        for e in expert_ids:
            key = (layer, e)
            if key in self.cache:
                self.cache.move_to_end(key)
        missing = [e for e in expert_ids if (layer, e) not in self.cache]
        fresh = {}  # this pass's decoded misses; returned regardless of caching
        if missing:
            read = lf.read_blob_nocache if self.prefill_nocache else lf.read_blob
            blobs = list(self.io.map(read, missing))
            for e, blob in zip(missing, blobs):
                st["misses"] += 1
                st["bytes"] += lf.stride
                fresh[e] = self._decode(layer, blob)
            if self.prefill_no_evict:
                # Fill-but-never-evict: insert only while free capacity remains.
                # Entries not inserted still live in `fresh`/`out` for the
                # duration of the pass (the gather stack needs them), then drop.
                for e, entry in fresh.items():
                    if self.used + lf.stride <= self.capacity:
                        self.cache[(layer, e)] = entry
                        self.used += lf.stride
                    else:
                        self.noevict_skipped += 1
            else:
                for e, entry in fresh.items():
                    self.cache[(layer, e)] = entry
                    self.used += lf.stride
                while self.used > self.capacity and len(self.cache) > len(expert_ids):
                    old_key, _ = self.cache.popitem(last=False)
                    self.used -= self.layers[old_key[0]].stride
        out = {}
        for e in expert_ids:
            key = (layer, e)
            if e in out:
                continue
            if key in self.cache:
                if e not in fresh:
                    st["hits"] += 1
                self.cache.move_to_end(key)
                out[e] = self.cache[key]
            elif e in fresh:
                out[e] = fresh[e]
        return out

    def _decode(self, layer, blob):
        lf = self.layers[layer]
        out = {}
        for key, s in lf.slices.items():
            dt = NP_DTYPE[s["dtype"]]
            arr = np.frombuffer(blob, dtype=dt,
                                count=s["nbytes"] // np.dtype(dt).itemsize,
                                offset=s["offset"]).reshape(s["shape"])
            if s["dtype"] == "BF16":
                out[key] = mx.array(arr).view(mx.bfloat16)
            else:
                out[key] = mx.array(arr)
        return out

    def get(self, layer, expert_id):
        key = (layer, expert_id)
        st = self.stats[layer]
        if key in self.cache:
            st["hits"] += 1
            self.decode_stats["hits"] += 1
            self.cache.move_to_end(key)
            return self.cache[key]
        st["misses"] += 1
        self.decode_stats["misses"] += 1
        lf = self.layers[layer]
        st["bytes"] += lf.stride
        self.decode_stats["bytes"] += lf.stride
        entry = self._decode(layer, lf.read_blob(expert_id))
        self.cache[key] = entry
        self.used += lf.stride
        while self.used > self.capacity and len(self.cache) > 1:
            old_key, _ = self.cache.popitem(last=False)
            self.used -= self.layers[old_key[0]].stride
        return entry

    def summary(self):
        h = sum(s["hits"] for s in self.stats.values())
        m = sum(s["misses"] for s in self.stats.values())
        gb = sum(s["bytes"] for s in self.stats.values()) / 1e9
        rate = h / (h + m) if h + m else 0.0
        return {"hits": h, "misses": m, "hit_rate": round(rate, 4),
                "read_gb": round(gb, 2), "cached_gb": round(self.used / 1e9, 2),
                "decode_hits": self.decode_stats["hits"],
                "decode_misses": self.decode_stats["misses"],
                "decode_read_gb": round(self.decode_stats["bytes"] / 1e9, 2),
                "noevict_skipped": self.noevict_skipped}


class StreamingExperts:
    """Drop-in for the SwitchGLU expert call: (x, indices) -> [tokens, topk, dim].

    Dispatches on batch size: a single token (decode) runs the per-expert
    quantized_matmul loop below, which measures faster at batch 1 because it
    avoids the stacking copies; more than one token (prefill) delegates to
    StackedStreamingExperts, whose gather_qmm path batches all of a layer's
    visits into 3 kernels and reads misses from SSD in parallel via get_many.
    """

    def __init__(self, store: ExpertStore, layer: int, group_size=64, bits=4,
                 limit: float = 0.0):
        self.store, self.layer = store, layer
        self.group_size, self.bits = group_size, bits
        self.limit = limit
        self._stacked = StackedStreamingExperts(store, layer, group_size=group_size,
                                                bits=bits, limit=limit)

    def _proj(self, x, e, p):
        return mx.quantized_matmul(
            x, e[f"{p}.weight"], scales=e[f"{p}.scales"], biases=e[f"{p}.biases"],
            transpose=True, group_size=self.group_size, bits=self.bits)

    def __call__(self, x, indices):
        # x [tokens, dim], indices [tokens, topk] -> [tokens, topk, dim]
        if x.shape[0] > 1:          # prefill: batch the visits through gather_qmm
            return self._stacked(x, indices)
        idx = np.asarray(indices)
        outs = []
        for t in range(idx.shape[0]):
            row = []
            for j in range(idx.shape[1]):
                e = self.store.get(self.layer, int(idx[t, j]))
                xi = x[t][None]
                gate = self._proj(xi, e, "gate_proj").astype(mx.float32)
                up = self._proj(xi, e, "up_proj").astype(mx.float32)
                if self.limit > 0:
                    up = mx.clip(up, -self.limit, self.limit)
                    gate = mx.minimum(gate, self.limit)
                h = (mx.sigmoid(gate) * gate * up).astype(x.dtype)
                row.append(self._proj(h, e, "down_proj"))
            outs.append(mx.concatenate(row, axis=0))
        return mx.stack(outs)


class StackedStreamingExperts:
    """Batched expert call: per visit, stack the selected experts' weights and
    run one gather_qmm per projection (3 kernels/layer instead of 18 matmuls).

    The stack is a GPU-side copy of ~85 MB per layer visit — cheap next to the
    kernel-dispatch overhead it removes. Misses are read from SSD in parallel
    via ExpertStore.get_many. The per-layer sync (np.asarray on the routing
    indices) is inherent to streaming: expert ids must reach the CPU before the
    disk can be asked for them.
    """

    def __init__(self, store: ExpertStore, layer: int, group_size=64, bits=4,
                 limit: float = 0.0):
        self.store, self.layer = store, layer
        self.group_size, self.bits = group_size, bits
        self.limit = limit

    def _stack(self, entries, uniq, key):
        return mx.stack([entries[e][key] for e in uniq])

    def _qmm(self, x, w, s, b, pos):
        return mx.gather_qmm(x, w, s, b, rhs_indices=pos, transpose=True,
                             group_size=self.group_size, bits=self.bits)

    def __call__(self, x, indices):
        # x [tokens, dim], indices [tokens, topk] -> [tokens, topk, dim]
        idx = np.asarray(indices)
        toks, topk = idx.shape
        flat = idx.reshape(-1).tolist()
        uniq = list(dict.fromkeys(flat))
        entries = self.store.get_many(self.layer, uniq)
        slot = {e: i for i, e in enumerate(uniq)}
        pos = mx.array(np.array([slot[e] for e in flat], dtype=np.int32)
                       .reshape(toks, topk))

        xe = mx.expand_dims(x, (-2, -3))  # [tokens, 1, 1, dim]
        proj = {}
        for p in ("gate_proj", "up_proj", "down_proj"):
            proj[p] = tuple(self._stack(entries, uniq, f"{p}.{part}")
                            for part in ("weight", "scales", "biases"))
        gate = self._qmm(xe, *proj["gate_proj"], pos).astype(mx.float32)
        up = self._qmm(xe, *proj["up_proj"], pos).astype(mx.float32)
        if self.limit > 0:
            up = mx.clip(up, -self.limit, self.limit)
            gate = mx.minimum(gate, self.limit)
        h = (mx.sigmoid(gate) * gate * up).astype(x.dtype)
        return self._qmm(h, *proj["down_proj"], pos).squeeze(-2)
