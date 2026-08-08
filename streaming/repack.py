"""Repack a pipenetwork DeepSeek-V4 MLX checkpoint for SSD expert streaming.

Splits the snapshot into:

  out/resident-XXXX.safetensors   everything except routed experts (~9.4 GB total)
  out/experts/layer_NN.bin        256 fixed-stride expert blobs per scored layer
  out/experts/layout.json         blob stride + per-slice offsets/shapes/dtypes
  out/config.json (+ tokenizer)   copied through unchanged

Expert tensors are stored stacked [n_experts, ...] with the expert axis first, so
each expert's slice of each tensor is one contiguous byte range in the shard. A
blob concatenates an expert's nine slices (3 projections x weight/scales/biases)
in a fixed order; the stride is page-aligned so a blob read never splits a page
with its neighbor.

Shards are processed one at a time through numpy memmaps — peak memory is one
resident tensor, never a full shard. Safe to run while the snapshot is still
downloading: pass --layers to repack only layers whose shards are complete, or
rerun later; finished layer files are skipped by size check.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import sys
from collections import defaultdict

import numpy as np

PAGE = 16384  # Apple Silicon page size
SLICE_ORDER = [
    f"{proj}.{part}"
    for proj in ("gate_proj", "up_proj", "down_proj")
    for part in ("weight", "scales", "biases")
]
DTYPE_SIZE = {"U32": 4, "BF16": 2, "F32": 4, "F16": 2, "U8": 1}


def read_header(path):
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(hlen))
    hdr.pop("__metadata__", None)
    return hdr, 8 + hlen


def expert_layer(name):
    # layers.N.ffn.experts.<proj>.<part>
    if ".ffn.experts." not in name:
        return None, None
    parts = name.split(".")
    return int(parts[1]), f"{parts[4]}.{parts[5]}"


def plan_layout(snapshot):
    """Walk every shard header once; return per-layer slice plan + resident list."""
    layers = defaultdict(dict)   # layer -> slice_key -> meta
    resident = defaultdict(list)  # shard -> [(name, meta)]
    for shard in sorted(f for f in os.listdir(snapshot) if f.endswith(".safetensors")):
        hdr, data_start = read_header(os.path.join(snapshot, shard))
        for name, meta in hdr.items():
            lyr, key = expert_layer(name)
            m = dict(meta, shard=shard, data_start=data_start)
            if lyr is None:
                resident[shard].append((name, m))
            else:
                layers[lyr][key] = m
    return layers, resident


def blob_layout(slices):
    """Fixed slice order -> per-slice offset within a blob, and the padded stride."""
    off, out = 0, {}
    for key in SLICE_ORDER:
        meta = slices[key]
        n_exp = meta["shape"][0]
        total = meta["data_offsets"][1] - meta["data_offsets"][0]
        per_exp = total // n_exp
        out[key] = {
            "offset": off,
            "nbytes": per_exp,
            "shape": meta["shape"][1:],  # per-expert shape
            "dtype": meta["dtype"],
        }
        off += per_exp
    stride = (off + PAGE - 1) // PAGE * PAGE
    return out, stride, n_exp


def repack_layer(snapshot, layers, lyr, out_dir):
    slices = layers[lyr]
    if set(slices) != set(SLICE_ORDER):
        raise ValueError(f"layer {lyr}: incomplete slice set {sorted(slices)}")
    slot_map, stride, n_exp = blob_layout(slices)
    dst_path = os.path.join(out_dir, f"layer_{lyr:02d}.bin")
    want = stride * n_exp
    if os.path.exists(dst_path) and os.path.getsize(dst_path) == want:
        return slot_map, stride, n_exp, "skipped"

    with open(dst_path + ".tmp", "wb") as dst:
        dst.truncate(want)
        for key in SLICE_ORDER:
            meta = slices[key]
            src = os.path.join(snapshot, meta["shard"])
            a, _ = meta["data_offsets"]
            per = slot_map[key]["nbytes"]
            mm = np.memmap(src, dtype=np.uint8, mode="r")
            base = meta["data_start"] + a
            for e in range(n_exp):
                dst.seek(e * stride + slot_map[key]["offset"])
                dst.write(mm[base + e * per: base + (e + 1) * per].tobytes())
            del mm
    os.replace(dst_path + ".tmp", dst_path)
    return slot_map, stride, n_exp, "written"


def write_resident(snapshot, resident, out):
    """Copy non-expert tensors through, one output file per source shard."""
    import mlx.core as mx
    for i, (shard, tensors) in enumerate(sorted(resident.items())):
        dst = os.path.join(out, f"resident-{i:04d}.safetensors")
        if os.path.exists(dst):
            continue
        names = [n for n, _ in tensors]
        w = mx.load(os.path.join(snapshot, shard))
        keep = {n: w[n] for n in names}
        # mx.save_safetensors appends the extension itself, so the temp name
        # must already end in .safetensors for os.replace to find it
        tmp = dst.replace(".safetensors", ".tmp.safetensors")
        mx.save_safetensors(tmp, keep)
        os.replace(tmp, dst)
        del w, keep
        print(f"  resident {i}: {len(names)} tensors from {shard}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", help="comma list / ranges, e.g. 0-5,40; default all")
    ap.add_argument("--skip-resident", action="store_true")
    args = ap.parse_args()

    exp_dir = os.path.join(args.out, "experts")
    os.makedirs(exp_dir, exist_ok=True)
    layers, resident = plan_layout(args.snapshot)

    todo = sorted(layers)
    if args.layers:
        sel = set()
        for tok in args.layers.split(","):
            if "-" in tok:
                a, b = tok.split("-")
                sel.update(range(int(a), int(b) + 1))
            else:
                sel.add(int(tok))
        todo = sorted(sel & set(layers))

    layout = {"page": PAGE, "slice_order": SLICE_ORDER, "layers": {}}
    lp = os.path.join(exp_dir, "layout.json")
    if os.path.exists(lp):
        layout = json.load(open(lp))
    for lyr in todo:
        try:
            slot_map, stride, n_exp, state = repack_layer(args.snapshot, layers, lyr, exp_dir)
        except (FileNotFoundError, ValueError) as e:
            print(f"  layer {lyr}: not ready ({e})")
            continue
        layout["layers"][str(lyr)] = {"stride": stride, "n_experts": n_exp, "slices": slot_map}
        print(f"  layer {lyr}: {state} ({n_exp} x {stride/1e6:.2f} MB)")
        json.dump(layout, open(lp, "w"), indent=1)

    if not args.skip_resident:
        write_resident(args.snapshot, resident, args.out)
        for aux in ("config.json", "tokenizer.json", "tokenizer_config.json",
                    "generation_config.json"):
            src = os.path.join(args.snapshot, aux)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(args.out, aux))
    print("done")


if __name__ == "__main__":
    sys.exit(main())
