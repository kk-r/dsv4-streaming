"""AQLM week-1 quality gate: blob IO, dequant, additive codebook 2-bit, container requant.

All heavy math is numpy; mx is used only on the CPU device for quantize/dequantize
parity with the serving path (mx.quantized_matmul semantics). Nothing here touches
the GPU, so quantization can run in parallel workers while no eval job is active.

Container: the repacked 4-bit affine blob format (group 64, bf16 scales/biases).
All three arms reconstruct fp32 weights and re-enter this container, so the eval
serving path is byte-for-byte the normal one; arm A (dequant->requant, no codebook)
measures the container roundtrip error in isolation.
"""

from __future__ import annotations

import json
import os

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPACKED = os.path.join(REPO, "repacked")
EXPERTS = os.path.join(REPACKED, "experts")

PROJS = ("gate_proj", "up_proj", "down_proj")
GROUP, BITS = 64, 4


def load_layout():
    return json.load(open(os.path.join(EXPERTS, "layout.json")))


def bf16_to_f32(u16: np.ndarray) -> np.ndarray:
    return (u16.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16_u16(f32: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even f32 -> bf16, kept as uint16 bit pattern."""
    u = f32.astype(np.float32).view(np.uint32)
    rounded = u + 0x7FFF + ((u >> 16) & 1)
    return (rounded >> 16).astype(np.uint16)


def read_blob(layer: int, expert: int, spec: dict, expert_dir: str = EXPERTS) -> bytes:
    path = os.path.join(expert_dir, f"layer_{layer:02d}.bin")
    with open(path, "rb") as f:
        return os.pread(f.fileno(), spec["stride"], expert * spec["stride"])


def slice_arr(blob: bytes, m: dict, np_dt) -> np.ndarray:
    return np.frombuffer(blob, dtype=np_dt,
                         count=m["nbytes"] // np.dtype(np_dt).itemsize,
                         offset=m["offset"]).reshape(m["shape"])


def dequant_proj(blob: bytes, spec: dict, proj: str) -> np.ndarray:
    """Manual fp32 dequant of one projection: w = q * scale + bias, group 64.

    MLX 4-bit packing: 8 nibbles per uint32, element i of the group of 8 in bits
    [4i, 4i+4) (low bits first). Verified bit-exact against mx.dequantize in
    validate_one.py before being trusted.
    """
    s = spec["slices"]
    wq = slice_arr(blob, s[f"{proj}.weight"], np.uint32)          # [out, in/8]
    sc = bf16_to_f32(slice_arr(blob, s[f"{proj}.scales"], np.uint16))
    bi = bf16_to_f32(slice_arr(blob, s[f"{proj}.biases"], np.uint16))
    out_dim = wq.shape[0]
    shifts = (4 * np.arange(8, dtype=np.uint32))[None, None, :]
    q = ((wq[:, :, None] >> shifts) & 0xF).astype(np.float32)     # [out, in/8, 8]
    q = q.reshape(out_dim, -1)                                    # [out, in]
    n_groups = sc.shape[1]
    w = q.reshape(out_dim, n_groups, GROUP) * sc[:, :, None] + bi[:, :, None]
    return w.reshape(out_dim, -1)


def quantize_container(w: np.ndarray):
    """fp32 -> 4-bit affine container tensors (mx.quantize on CPU for serving
    parity), scales/biases cast to bf16 as stored. Returns (wq_u32, sc_u16, bi_u16)."""
    import mlx.core as mx
    with mx.stream(mx.cpu):
        wq, sc, bi = mx.quantize(mx.array(w.astype(np.float32)),
                                 group_size=GROUP, bits=BITS)
        sc16 = sc.astype(mx.bfloat16).view(mx.uint16)
        bi16 = bi.astype(mx.bfloat16).view(mx.uint16)
        mx.eval(wq, sc16, bi16)
    return np.array(wq), np.array(sc16), np.array(bi16)


def dequant_container(wq, sc_u16, bi_u16, out_dim):
    """fp32 reconstruction of container tensors (same math as dequant_proj)."""
    sc = bf16_to_f32(np.asarray(sc_u16))
    bi = bf16_to_f32(np.asarray(bi_u16))
    shifts = (4 * np.arange(8, dtype=np.uint32))[None, None, :]
    q = ((np.asarray(wq)[:, :, None] >> shifts) & 0xF).astype(np.float32)
    q = q.reshape(out_dim, -1)
    n_groups = sc.shape[1]
    w = q.reshape(out_dim, n_groups, GROUP) * sc[:, :, None] + bi[:, :, None]
    return w.reshape(out_dim, -1)


def affine2bit_roundtrip(w: np.ndarray) -> np.ndarray:
    """Arm C: mx affine 2-bit (group 64) quantize->dequantize, fp32 out."""
    import mlx.core as mx
    with mx.stream(mx.cpu):
        wq, sc, bi = mx.quantize(mx.array(w.astype(np.float32)),
                                 group_size=GROUP, bits=2)
        wh = mx.dequantize(wq, sc, bi, group_size=GROUP, bits=2)
        mx.eval(wh)
    return np.array(wh)


def build_blob(spec: dict, tensors: dict) -> bytes:
    """tensors: {slice_key: np array with the container dtype}; -> stride bytes."""
    buf = bytearray(spec["stride"])
    for key, m in spec["slices"].items():
        raw = np.ascontiguousarray(tensors[key]).tobytes()
        assert len(raw) == m["nbytes"], (key, len(raw), m["nbytes"])
        buf[m["offset"]:m["offset"] + m["nbytes"]] = raw
    return bytes(buf)


# ---------------- additive codebook 2-bit (AQLM-class, weight-only) ----------------

VDIM = 8          # vector = 8 consecutive weights along the input axis
NCODE = 256       # entries per codebook; 2 codebooks -> 16 bits / 8 weights = 2 b/w


def _assign(X: np.ndarray, C: np.ndarray, chunk: int = 1 << 18) -> np.ndarray:
    """argmin_k ||x - C_k||^2 for each row of X. fp32, chunked."""
    c2 = (C * C).sum(1)
    out = np.empty(len(X), np.int32)
    for i in range(0, len(X), chunk):
        xb = X[i:i + chunk]
        d = xb @ C.T
        d *= -2.0
        d += c2[None, :]
        out[i:i + chunk] = np.argmin(d, 1)
    return out


def _update(X: np.ndarray, a: np.ndarray, rng) -> np.ndarray:
    """Cluster means (least-squares codebook update given assignments)."""
    counts = np.bincount(a, minlength=NCODE).astype(np.float32)
    C = np.empty((NCODE, VDIM), np.float32)
    for j in range(VDIM):
        C[:, j] = np.bincount(a, weights=X[:, j], minlength=NCODE)
    empty = counts == 0
    counts[empty] = 1.0
    C /= counts[:, None]
    if empty.any():   # reseed dead entries to random data rows
        C[empty] = X[rng.integers(0, len(X), int(empty.sum()))]
    return C


def _kmeans_init(X: np.ndarray, rng, sub_n: int = 1 << 18, iters: int = 8):
    sub = X[rng.choice(len(X), min(sub_n, len(X)), replace=False)]
    C = sub[rng.choice(len(sub), NCODE, replace=False)].copy()
    for _ in range(iters):
        C = _update(sub, _assign(sub, C), rng)
    return C


def codebook2bit(W: np.ndarray, seed: int, refine: int = 6) -> tuple[np.ndarray, dict]:
    """Additive 2-codebook VQ of one projection. Returns (W_hat fp32, stats).

    - per-output-channel scale (init RMS, final per-row least squares)
    - C1: k-means on normalized vectors; C2: k-means on residuals
    - `refine` alternating passes: reassign+update C1 against (V - C2[a2]),
      then C2 against (V - C1[a1]) — block coordinate descent on ||V - C1 - C2||^2.
    Bit cost: 16 bits per 8 weights = 2.0 b/w; row scales add 16/in_dim b/w
    (0.004 at in=4096) and the two codebooks are 16 KB per projection, shareable.
    """
    out_dim, in_dim = W.shape
    W = W.astype(np.float32)
    rng = np.random.default_rng(seed)
    s = np.sqrt((W * W).mean(1)) + 1e-12
    V = (W / s[:, None]).reshape(-1, VDIM)

    C1 = _kmeans_init(V, rng)
    a1 = _assign(V, C1)
    C1 = _update(V, a1, rng)
    R = V - C1[a1]
    C2 = _kmeans_init(R, rng)
    a2 = _assign(R, C2)
    C2 = _update(R, a2, rng)
    for _ in range(refine):
        T = V - C2[a2]
        a1 = _assign(T, C1)
        C1 = _update(T, a1, rng)
        T = V - C1[a1]
        a2 = _assign(T, C2)
        C2 = _update(T, a2, rng)

    Vh = (C1[a1] + C2[a2]).reshape(out_dim, in_dim)
    denom = (Vh * Vh).sum(1)
    denom[denom == 0] = 1.0
    s_ls = (W * Vh).sum(1) / denom          # per-row LS scale (replaces RMS init)
    Wh = s_ls[:, None] * Vh
    err = W - Wh
    stats = {
        "rel_mse": float((err * err).sum() / max((W * W).sum(), 1e-30)),
        "codebook_util": [int(len(np.unique(a1))), int(len(np.unique(a2)))],
    }
    return Wh, stats


def rel_mse(W: np.ndarray, Wh: np.ndarray) -> float:
    e = (W - Wh).astype(np.float64)
    return float((e * e).sum() / max((W.astype(np.float64) ** 2).sum(), 1e-30))
