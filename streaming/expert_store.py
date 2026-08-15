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
import threading
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


def _decode_np(lf: LayerFile, blob: bytes) -> dict:
    """Decode a blob into numpy views — NO MLX, safe off the main thread.

    Returns {slice_name: (np_array, is_bf16)}. BF16 stays a uint16 view; the
    mx.bfloat16 reinterpret happens in _wrap_np on the main thread.
    """
    out = {}
    for key, s in lf.slices.items():
        dt = NP_DTYPE[s["dtype"]]
        arr = np.frombuffer(blob, dtype=dt,
                            count=s["nbytes"] // np.dtype(dt).itemsize,
                            offset=s["offset"]).reshape(s["shape"])
        out[key] = (arr, s["dtype"] == "BF16")
    return out


def _wrap_np(np_entry: dict) -> dict:
    """np entry -> dict of mx arrays. MAIN THREAD ONLY (MLX streams are
    thread-bound; creating mx arrays off-thread is the known crash class)."""
    out = {}
    for key, (arr, is_bf16) in np_entry.items():
        a = mx.array(arr)
        out[key] = a.view(mx.bfloat16) if is_bf16 else a
    return out


class ExpertPrefetcher:
    """Async SSD prefetch of predicted experts.

    Worker threads do os.pread (page-cached fd, positionless — thread-safe)
    plus the numpy view decode, and park the result in a buffer dict keyed
    (layer, expert). They never touch MLX. The main thread drains entries via
    take() at the point a miss actually needs them; unpredicted-but-fetched
    entries are purged (counted wasted) and never enter the LRU, so cache
    residency and eviction order stay byte-identical to a prefetch-free run.
    """

    def __init__(self, store: "ExpertStore", workers: int = 6):
        self.store = store
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.lock = threading.Lock()
        self.buf = {}       # (layer, expert) -> np entry, decoded and ready
        self.futures = {}   # (layer, expert) -> Future, read in flight
        self.issued = 0         # reads submitted
        self.served_ready = 0   # miss served from a completed prefetch
        self.served_wait = 0    # miss waited on an in-flight prefetch read
        self.wasted = 0         # fetched (or in flight) but purged unused
        self.bytes_read = 0     # bytes submitted for prefetch reads

    def request(self, layer: int, expert_ids):
        """Issue async reads for the given experts, skipping LRU-cached and
        already-requested ones. Main thread only."""
        lf = self.store.layers[layer]
        for e in expert_ids:
            key = (layer, int(e))
            if key in self.store.cache:
                continue
            with self.lock:
                if key in self.buf or key in self.futures:
                    continue
                self.issued += 1
                self.bytes_read += lf.stride
                self.futures[key] = self.pool.submit(self._fetch, key, lf)

    def _fetch(self, key, lf):
        # WORKER THREAD: pread + numpy only.
        blob = os.pread(lf.f.fileno(), lf.stride, key[1] * lf.stride)
        entry = _decode_np(lf, blob)
        with self.lock:
            if key in self.futures:     # not taken/purged mid-flight
                self.buf[key] = entry
                del self.futures[key]   # a key lives in buf XOR futures
        return entry

    def take(self, layer: int, expert_id: int):
        """Return the np entry for a key, or None if never prefetched.
        Waits on an in-flight read rather than restarting it. Main thread."""
        key = (layer, expert_id)
        with self.lock:
            fut = self.futures.pop(key, None)   # popping stops _fetch's buf insert
            entry = self.buf.pop(key, None)
        if entry is not None:
            self.served_ready += 1
            return entry
        if fut is not None:
            entry = fut.result()                # already reading; wait, don't redo
            with self.lock:
                self.buf.pop(key, None)         # in case _fetch won the race
            self.served_wait += 1
            return entry
        return None

    def purge(self, layer: int | None = None):
        """Drop unused entries (all layers, or one). Mispredictions die here."""
        with self.lock:
            doomed = [k for k in self.buf if layer is None or k[0] == layer]
            for k in doomed:
                del self.buf[k]
            inflight = [k for k in self.futures if layer is None or k[0] == layer]
            for k in inflight:
                self.futures.pop(k)             # _fetch will drop its result
            self.wasted += len(doomed) + len(inflight)

    def counters(self) -> dict:
        return {"pf_issued": self.issued, "pf_served_ready": self.served_ready,
                "pf_served_wait": self.served_wait, "pf_wasted": self.wasted,
                "pf_read_gb": round(self.bytes_read / 1e9, 2)}


class PrefetchCoordinator:
    """Pre-gated prefetch prediction (Pre-gated MoE / Mixtral-offloading style).

    Two prediction sources, both exact-cost-free on the decode critical path:

    * hash layers 0..n_hash-1: EXACT — routing is tid2eid[token_id], known at
      token start. token_start() fires from Model.token_hook before the
      forward begins, so these reads overlap embedding + early attention.
    * scored layer L+1: APPROXIMATE — L+1's resident gate applied to layer L's
      MoE input xf (papers: hidden states change slowly across layers). The
      gate matmul is folded into layer L's existing per-layer sync
      (mx.eval(indices, pred) — still ONE sync), then reads are issued async
      and overlap layer L's expert FFN + L+1's attention.

    Prediction accuracy is scored per layer (pred_hit/pred_total) against the
    routing the model actually performs.
    """

    def __init__(self, store: "ExpertStore", workers: int = 6):
        self.store = store
        self.prefetcher = ExpertPrefetcher(store, workers=workers)
        store.prefetcher = self.prefetcher
        store.coordinator = self
        self.predicted = {}     # layer -> set of predicted expert ids (this token)
        self.pred_stats = defaultdict(lambda: {"pred_hit": 0, "pred_total": 0})
        self.hash_tables = {}   # layer -> np tid2eid [vocab, topk]

    def register_hash_layers(self, gates: dict):
        """gates: {layer_id: Gate module with .tid2eid}. Call after weights load;
        keeps a CPU copy (~3 MB/layer) so token_start needs no MLX work."""
        for layer, gate in gates.items():
            self.hash_tables[layer] = np.asarray(gate.tid2eid)

    def token_start(self, input_ids):
        """Model.token_hook: fires at the top of a single-token forward."""
        tid = int(input_ids[0, 0].item())
        self.prefetcher.purge()             # stale leftovers from the last token
        self.predicted.clear()
        for layer, table in self.hash_tables.items():
            self.predict(layer, [int(v) for v in table[tid]])

    def predict(self, layer: int, expert_ids):
        self.predicted[layer] = set(expert_ids)
        self.prefetcher.request(layer, expert_ids)

    def observe_actual(self, layer: int, actual_ids):
        """Score the prediction for this layer against actual routing."""
        pred = self.predicted.pop(layer, None)
        if pred is None:
            return
        st = self.pred_stats[layer]
        st["pred_hit"] += len(pred & set(actual_ids))
        st["pred_total"] += len(actual_ids)

    def layer_done(self, layer: int):
        """After a layer's expert accesses: drop its unused prefetches."""
        self.prefetcher.purge(layer)

    def report(self) -> str:
        lines = []
        bands = {"hash 0-2": range(0, 3), "scored 3-15": range(3, 16),
                 "scored 16-29": range(16, 30), "scored 30-42": range(30, 43)}
        for name, rng in bands.items():
            h = sum(self.pred_stats[l]["pred_hit"] for l in rng if l in self.pred_stats)
            t = sum(self.pred_stats[l]["pred_total"] for l in rng if l in self.pred_stats)
            if t:
                lines.append(f"{name}: {h}/{t} = {h/t:.3f}")
        per_layer = {l: round(s["pred_hit"] / s["pred_total"], 4)
                     for l, s in sorted(self.pred_stats.items()) if s["pred_total"]}
        return (f"pred hit rate by band: {'; '.join(lines)} | "
                f"prefetcher {self.prefetcher.counters()} | per-layer {per_layer}")


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
        # Optional pre-gated prefetcher (DSV4_PREFETCH=1): set by
        # PrefetchCoordinator. When present, get() consults its buffer before
        # touching the disk; a served miss is still a miss in every counter
        # (the read happened, just earlier and off-thread).
        self.prefetcher = None
        self.coordinator = None
        self.prefetch_served = 0
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
        return _wrap_np(_decode_np(self.layers[layer], blob))

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
        entry = None
        if self.prefetcher is not None:
            np_entry = self.prefetcher.take(layer, expert_id)
            if np_entry is not None:
                self.prefetch_served += 1
                entry = _wrap_np(np_entry)   # main thread: np -> mx here only
        if entry is None:
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
        out = {"hits": h, "misses": m, "hit_rate": round(rate, 4),
               "read_gb": round(gb, 2), "cached_gb": round(self.used / 1e9, 2),
               "decode_hits": self.decode_stats["hits"],
               "decode_misses": self.decode_stats["misses"],
               "decode_read_gb": round(self.decode_stats["bytes"] / 1e9, 2),
               "noevict_skipped": self.noevict_skipped}
        if self.prefetcher is not None:
            out["pf_served"] = self.prefetch_served
            out.update(self.prefetcher.counters())
        return out


class StreamingExperts:
    """Drop-in for the SwitchGLU expert call: (x, indices) -> [tokens, topk, dim].

    Dispatches on batch size: a single token (decode) runs the per-expert
    quantized_matmul loop below, which measures faster at batch 1 because it
    avoids the stacking copies; more than one token (prefill) delegates to
    StackedStreamingExperts, whose gather_qmm path batches all of a layer's
    visits into 3 kernels and reads misses from SSD in parallel via get_many.
    """

    def __init__(self, store: ExpertStore, layer: int, group_size=64, bits=4,
                 limit: float = 0.0, coordinator: PrefetchCoordinator | None = None,
                 next_gate=None):
        self.store, self.layer = store, layer
        self.group_size, self.bits = group_size, bits
        self.limit = limit
        # Pre-gated prefetch (DSV4_PREFETCH=1): next_gate is layer L+1's
        # resident Gate module (scored layers only); applying it to this
        # layer's MoE input xf approximates L+1's routing one layer early.
        self.coordinator = coordinator
        self.next_gate = next_gate
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
        coord = self.coordinator
        pred = None
        if coord is not None and self.next_gate is not None:
            # L+1's gate on L's MoE input — tiny matmul, folded into the
            # per-layer sync that HEAD already pays (ONE eval, not two).
            _, pred = self.next_gate(x)
            mx.eval(indices, pred)
        idx = np.asarray(indices)
        if coord is not None:
            if pred is not None:
                coord.predict(self.layer + 1,
                              [int(v) for v in np.asarray(pred).reshape(-1)])
            coord.observe_actual(self.layer, [int(v) for v in idx.reshape(-1)])
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
        if coord is not None:
            coord.layer_done(self.layer)   # purge this layer's mispredictions
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
