"""Micro-benchmark: real layer-0 expert blobs — SSD read, decode, matmul.

Reads use F_NOCACHE to bypass the unified buffer cache, so 'cold' numbers are
true SSD reads, not page-cache hits. Compute is the v0 per-expert quantized
FFN at decode shape (1 token, dim 4096).

Derived estimate: per-token cost = 258 expert visits (6 x 43 layers), where a
miss pays read + decode + matmul and a hit pays decode-free matmul only.
"""

import fcntl
import os
import sys
import time

import mlx.core as mx
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from expert_store import ExpertStore, StreamingExperts  # noqa: E402

F_NOCACHE = 48  # macOS fcntl
EXPERT_DIR = os.path.join(os.path.dirname(__file__), "..", "bench", "experts")
N = 16
VISITS_PER_TOKEN = 6 * 43  # routed visits; shared expert is resident


def timed(fn, reps):
    out = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t) * 1e3)
    return out


def main():
    store = ExpertStore(EXPERT_DIR, cache_bytes=1 << 30)
    lf = store.layers[0]
    fcntl.fcntl(lf.f.fileno(), F_NOCACHE, 1)

    # --- raw SSD blob reads, cache bypassed ---
    cold = timed(lambda: lf.read_blob(int(np.random.randint(N))), 32)
    cold_ms = float(np.median(cold))
    bw = lf.stride / (cold_ms / 1e3) / 1e9

    # --- decode bytes -> mx arrays ---
    blob = lf.read_blob(0)
    dec = timed(lambda: store._decode(0, blob), 16)
    dec_ms = float(np.median(dec))

    # --- per-expert FFN at decode shape ---
    fcntl.fcntl(lf.f.fileno(), F_NOCACHE, 0)
    x = mx.random.normal((1, 4096)).astype(mx.bfloat16)
    se = StreamingExperts(store, 0, group_size=64, bits=4, limit=10.0)
    idx = mx.array([[0]])
    mx.eval(se(x, idx))  # warm kernels + cache expert 0
    comp = timed(lambda: mx.eval(se(x, idx)), 32)
    comp_ms = float(np.median(comp))

    # --- one full cached-expert visit including store lookup ---
    idx6 = mx.array(np.random.choice(N, 6, replace=False).astype(np.int32))[None]
    mx.eval(se(x, idx6))
    visit6 = timed(lambda: mx.eval(se(x, idx6)), 16)
    visit6_ms = float(np.median(visit6))

    print(f"blob stride            : {lf.stride/1e6:.2f} MB")
    print(f"SSD read (F_NOCACHE)   : {cold_ms:.2f} ms/blob  ({bw:.2f} GB/s)")
    print(f"decode bytes->mx       : {dec_ms:.2f} ms/blob")
    print(f"FFN 1 expert, 1 token  : {comp_ms:.2f} ms (cached)")
    print(f"6-expert visit (cached): {visit6_ms:.2f} ms")
    print()
    for hit in (0.0, 0.3, 0.5, 0.7, 0.9):
        miss = 1 - hit
        ms = VISITS_PER_TOKEN * (miss * (cold_ms + dec_ms) + comp_ms)
        print(f"hit {hit:.0%}: ~{ms:7.0f} ms/token -> {1e3/ms:5.2f} tok/s "
              f"(+ resident attn, sequential-read model)")


if __name__ == "__main__":
    main()
