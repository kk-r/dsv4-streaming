"""Optimistic expert streaming: one sync per token, repair-and-redo on miss.

The v0/v1 stores pay 43 Python round-trips per token (sync on every layer's
routing indices) — measured at ~284 ms/token against ~128 ms of actual model
compute. Here the whole token runs lazily:

* experts live in per-layer stacked slot pools; an ``id2slot`` int table maps
  expert id -> slot **inside the graph** (stale entries map somewhere valid but
  wrong — harmless, see below);
* every layer's routing indices are captured (lazily) during the forward;
* ONE ``mx.eval`` materializes logits + all captured indices;
* then Python checks the captured ids against the pools. All cached (the common
  warm case): accept the token. Any miss: fetch the blobs (parallel pread),
  batch-scatter them into the pools, roll the KV cache back, and rerun the
  token — now correct. Retries are bounded; each repair strictly grows the
  cached set, and routing prefixes stabilize layer by layer.

Rollback note — CORRECTED 2026-08-15 (see spec_generate.py): the original
claim here was that setitem rebinds and saved references are free snapshots.
That is FALSE: ``mx.array.__setitem__`` mutates the same Python object (6-line
repro in logs/spec_decode.txt). This file's redo path only worked because the
redo rewrites identical values over the mutated state. A real snapshot needs
an explicit copy (``x[:]``), as spec_generate.py's exact-mode rewind does.

Correctness: an accepted token only ever used pool slots whose expert ids were
verified after evaluation; wrong-slot outputs occur only in passes that get
discarded and redone.
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
PARTS = ("weight", "scales", "biases")


class OptimisticLayer:
    def __init__(self, path, spec, n_slots, n_experts=256):
        self.f = open(path, "rb", buffering=0)
        self.stride = spec["stride"]
        self.slices = spec["slices"]
        self.n_slots = n_slots
        self.slot_of = OrderedDict()          # expert -> slot, LRU order
        self.id2slot = mx.zeros((n_experts,), dtype=mx.int32)
        self.pool = {}
        for proj in PROJS:
            for part in PARTS:
                key = f"{proj}.{part}"
                s = self.slices[key]
                dt = mx.uint32 if s["dtype"] == "U32" else mx.bfloat16
                self.pool[key] = mx.zeros((n_slots, *s["shape"]), dtype=dt)
        mx.eval(list(self.pool.values()), self.id2slot)

    def read_blob(self, e):
        return os.pread(self.f.fileno(), self.stride, e * self.stride)

    def decode(self, blob, key):
        s = self.slices[key]
        dt = NP_DTYPE[s["dtype"]]
        arr = np.frombuffer(blob, dtype=dt, count=s["nbytes"] // np.dtype(dt).itemsize,
                            offset=s["offset"]).reshape(s["shape"])
        return mx.array(arr).view(mx.bfloat16) if s["dtype"] == "BF16" else mx.array(arr)


class OptimisticStore:
    def __init__(self, expert_dir, cache_bytes, io_threads=8, slot_plan=None):
        layout = json.load(open(os.path.join(expert_dir, "layout.json")))
        n_layers = len(layout["layers"])
        stride = next(iter(layout["layers"].values()))["stride"]
        if slot_plan:
            # per-layer sizing from a measured usage profile, scaled to budget.
            # Uniform allocation thrashes: per-layer working sets are skewed,
            # and a layer over its pool size re-inserts forever.
            budget = cache_bytes / stride
            want = {int(k): v for k, v in slot_plan.items()}
            scale = min(1.0, budget / max(1, sum(want.values())))
            plan = {k: min(256, max(8, int(v * scale))) for k, v in want.items()}
        else:
            plan = {int(k): max(8, int(cache_bytes / n_layers / stride))
                    for k in layout["layers"]}
        self.n_slots = plan
        self.layers = {
            int(k): OptimisticLayer(os.path.join(expert_dir, f"layer_{int(k):02d}.bin"),
                                    v, plan[int(k)])
            for k, v in layout["layers"].items()
        }
        self.io = ThreadPoolExecutor(max_workers=io_threads)
        self.stats = defaultdict(lambda: {"hits": 0, "misses": 0, "bytes": 0, "redos": 0})
        self.captured = {}
        # sync_mode trades the one-sync-per-token property for miss handling
        # inline at each layer — one pass, no redo cascade. Used for cold
        # prefill; decode runs optimistically.
        self.sync_mode = False
        self.clean_streak = 0

    def begin(self):
        self.captured = {}
        self.inserted = False

    def ensure(self, lyr, experts):
        """Synchronously make `experts` resident in layer `lyr`'s pools."""
        lf = self.layers[lyr]
        missing = [e for e in experts if e not in lf.slot_of]
        if missing:
            self.inserted = True
            self.repair({lyr: (missing, set(experts))})
            self.stats[lyr]["redos"] -= 1  # not a redo, an inline fill
        for e in experts:
            if e not in missing:
                self.stats[lyr]["hits"] += 1
            lf.slot_of.move_to_end(e)

    def check_and_touch(self):
        """After eval: which captured experts are missing? Touch LRU for hits."""
        needs = {}
        for lyr, idx in self.captured.items():
            lf = self.layers[lyr]
            st = self.stats[lyr]
            used = list(dict.fromkeys(np.asarray(idx).reshape(-1).tolist()))
            missing = [e for e in used if e not in lf.slot_of]
            for e in used:
                if e in lf.slot_of:
                    st["hits"] += 1
                    lf.slot_of.move_to_end(e)
            if missing:
                needs[lyr] = (missing, set(used))
        return needs

    def repair(self, needs):
        """Fetch missing experts and batch-scatter them into the pools."""
        reads = [(lyr, e) for lyr, (missing, _) in needs.items() for e in missing]
        blobs = list(self.io.map(lambda t: self.layers[t[0]].read_blob(t[1]), reads))
        blob_of = dict(zip(reads, blobs))
        for lyr, (missing, protect) in needs.items():
            lf = self.layers[lyr]
            st = self.stats[lyr]
            st["misses"] += len(missing)
            st["bytes"] += len(missing) * lf.stride
            st["redos"] += 1
            slots = []
            for e in missing:
                if len(lf.slot_of) < lf.n_slots:
                    slot = len(lf.slot_of)
                else:
                    victim = next(k for k in lf.slot_of if k not in protect)
                    slot = lf.slot_of.pop(victim)
                lf.slot_of[e] = slot
                slots.append(slot)
            slot_arr = mx.array(np.array(slots, dtype=np.int32))
            for key in lf.pool:
                stacked = mx.stack([lf.decode(blob_of[(lyr, e)], key) for e in missing])
                lf.pool[key][slot_arr] = stacked
            lf.id2slot[mx.array(np.array(missing, dtype=np.int32))] = slot_arr
            mx.eval(list(lf.pool.values()), lf.id2slot)

    def summary(self):
        h = sum(s["hits"] for s in self.stats.values())
        m = sum(s["misses"] for s in self.stats.values())
        r = sum(s["redos"] for s in self.stats.values())
        gb = sum(s["bytes"] for s in self.stats.values()) / 1e9
        return {"hits": h, "misses": m,
                "hit_rate": round(h / (h + m), 4) if h + m else 0.0,
                "read_gb": round(gb, 2), "layer_redos": r,
                "slots_total": sum(self.n_slots.values())}


class OptimisticExperts:
    """Fully-lazy expert call over the slot pools; captures routing indices."""

    def __init__(self, store: OptimisticStore, layer: int, group_size=64, bits=4,
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
        if self.store.sync_mode:
            uniq = list(dict.fromkeys(np.asarray(indices).reshape(-1).tolist()))
            self.store.ensure(self.layer, uniq)
        else:
            self.store.captured[self.layer] = indices
        slot_idx = mx.take(self.store.layers[self.layer].id2slot, indices)
        xe = mx.expand_dims(x, (-2, -3))
        gate = self._qmm(xe, "gate_proj", slot_idx).astype(mx.float32)
        up = self._qmm(xe, "up_proj", slot_idx).astype(mx.float32)
        if self.limit > 0:
            up = mx.clip(up, -self.limit, self.limit)
            gate = mx.minimum(gate, self.limit)
        h = (mx.sigmoid(gate) * gate * up).astype(x.dtype)
        return self._qmm(h, "down_proj", slot_idx).squeeze(-2)


def _snapshot(cache):
    out = []
    for c in cache:
        out.append((c.offset, c.kv, c.idx_kv,
                    (c.comp.kv_state, c.comp.score_state) if c.comp else None,
                    (c.idx_comp.kv_state, c.idx_comp.score_state) if c.idx_comp else None))
    return out


def _restore(cache, snap):
    for c, (off, kv, idx_kv, comp, idx_comp) in zip(cache, snap):
        c.offset, c.kv, c.idx_kv = off, kv, idx_kv
        if comp:
            c.comp.kv_state, c.comp.score_state = comp
        if idx_comp:
            c.idx_comp.kv_state, c.idx_comp.score_state = idx_comp


def optimistic_generate(model, args, store, prompt_ids, max_new_tokens=32,
                        eos_id=1, max_retries=None, verbose=False):
    # A fully cold token can cascade repairs one layer-prefix per attempt, so
    # the bound is the layer count, not a small constant.
    max_retries = max_retries or (args.n_layers + 8)
    from deepseek_v4_mlx.cache import make_cache
    ids = list(prompt_ids)
    cache = make_cache(args, bsz=1, max_seq_len=len(ids) + max_new_tokens + 8)

    import time as _time

    def step(tokens, tok_no=-1):
        t0 = _time.time()
        snap = _snapshot(cache)
        for attempt in range(max_retries):
            store.begin()
            logits = model(mx.array([tokens]), last_logit_only=True, cache=cache)
            mx.eval(logits, *store.captured.values())
            needs = store.check_and_touch()
            if not needs:
                # only a streak of clean tokens earns optimistic mode — a single
                # clean token in cold territory caused expensive ping-ponging
                if attempt > 0 or store.inserted:
                    store.sync_mode = True
                    store.clean_streak = 0
                else:
                    store.clean_streak += 1
                    if store.clean_streak >= 3:
                        store.sync_mode = False
                if verbose:
                    mode = "sync" if store.sync_mode else "opt"
                    print(f"[tok {tok_no}] {_time.time()-t0:6.2f}s "
                          f"attempts={attempt+1} next={mode}", flush=True)
                return logits
            store.repair(needs)
            _restore(cache, snap)
        raise RuntimeError("token did not stabilize after repairs")

    # Cold prefill: inline sync fill (one pass) instead of the redo cascade.
    store.sync_mode = True
    logits = step(ids)
    store.sync_mode = False
    nxt = int(mx.argmax(logits[0, -1]).item())
    ids.append(nxt)
    if verbose:
        print(f"[gen] prefilled {len(ids)-1} tokens", flush=True)
    for i in range(max_new_tokens - 1):
        if eos_id is not None and nxt == eos_id:
            break
        logits = step([nxt], tok_no=i)
        nxt = int(mx.argmax(logits[0, -1]).item())
        ids.append(nxt)
    return ids
