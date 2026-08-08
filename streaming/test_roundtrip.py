"""Round-trip test: synthetic quantized checkpoint -> repack -> ExpertStore.

Builds a fake 2-layer, 4-expert stacked-expert shard with real MLX affine
quantization, repacks it, then checks that StreamingExperts reproduces the
reference SwitchGLU-style computation exactly (same quantized weights, so
outputs must match to float tolerance).
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile

import mlx.core as mx
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from expert_store import ExpertStore, StreamingExperts  # noqa: E402

DIM, INTER, N_EXP, TOPK, GS, BITS = 128, 64, 4, 2, 64, 4
rng = np.random.default_rng(0)


def make_shard(path):
    tensors = {}
    ref = {}
    for lyr in range(2):
        for proj, (out_d, in_d) in {"gate_proj": (INTER, DIM), "up_proj": (INTER, DIM),
                                    "down_proj": (DIM, INTER)}.items():
            w = mx.array(rng.standard_normal((N_EXP, out_d, in_d), dtype=np.float32) * 0.05)
            qw, scales, biases = mx.quantize(w, group_size=GS, bits=BITS)
            base = f"layers.{lyr}.ffn.experts.{proj}"
            tensors[f"{base}.weight"] = qw
            tensors[f"{base}.scales"] = scales.astype(mx.bfloat16)
            tensors[f"{base}.biases"] = biases.astype(mx.bfloat16)
            ref[(lyr, proj)] = (qw, scales.astype(mx.bfloat16), biases.astype(mx.bfloat16))
        # one resident tensor so write_resident has work
        tensors[f"layers.{lyr}.attn.norm.weight"] = mx.ones((DIM,))
    mx.save_safetensors(os.path.join(path, "model-00001-of-00001.safetensors"), tensors)
    json.dump({"quantization": {"group_size": GS, "bits": BITS}},
              open(os.path.join(path, "config.json"), "w"))
    return ref


def reference_out(ref, lyr, x, indices):
    outs = []
    for t in range(indices.shape[0]):
        row = []
        for j in range(indices.shape[1]):
            e = int(indices[t, j])
            xi = x[t][None]
            p = {}
            for proj in ("gate_proj", "up_proj", "down_proj"):
                qw, s, b = ref[(lyr, proj)]
                p[proj] = lambda v, qw=qw, s=s, b=b: mx.quantized_matmul(
                    v, qw[e], scales=s[e], biases=b[e], transpose=True,
                    group_size=GS, bits=BITS)
            gate = p["gate_proj"](xi).astype(mx.float32)
            up = p["up_proj"](xi).astype(mx.float32)
            h = (mx.sigmoid(gate) * gate * up).astype(x.dtype)
            row.append(p["down_proj"](h))
        outs.append(mx.concatenate(row, axis=0))
    return mx.stack(outs)


def main():
    tmp = tempfile.mkdtemp()
    try:
        snap = os.path.join(tmp, "snap"); os.makedirs(snap)
        out = os.path.join(tmp, "out")
        ref = make_shard(snap)

        r = subprocess.run([sys.executable,
                            os.path.join(os.path.dirname(__file__), "repack.py"),
                            "--snapshot", snap, "--out", out], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        print(r.stdout.strip())

        store = ExpertStore(os.path.join(out, "experts"), cache_bytes=10 * 1024 * 1024)
        x = mx.array(rng.standard_normal((3, DIM), dtype=np.float32))
        indices = mx.array(rng.integers(0, N_EXP, size=(3, TOPK)))

        for lyr in range(2):
            got = StreamingExperts(store, lyr, group_size=GS, bits=BITS)(x, indices)
            want = reference_out(ref, lyr, x, indices)
            err = float(mx.abs(got - want).max())
            print(f"layer {lyr}: max abs err {err:.2e}")
            assert err < 1e-4, "mismatch vs reference"

        # second pass must be all hits
        StreamingExperts(store, 0, group_size=GS, bits=BITS)(x, indices)
        s = store.summary()
        print("store:", s)
        assert s["hits"] > 0
        print("ROUNDTRIP OK")
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
