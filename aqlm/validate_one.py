"""Pipeline validation on one expert (layer 18, expert 0) before any batch work.

Checks, in order:
1. manual numpy dequant == mx.dequantize (bit-exact) -> trust dequant_proj
2. arm A container roundtrip rel-MSE (should be tiny; this is the plumbing error)
3. arm C affine-2bit rel-MSE (known-bad comparator, should be large)
4. arm B codebook-2bit rel-MSE + wall time (should land well under arm C)
5. blob rebuild: arm-A blob decoded back == arm-A tensors (container IO sane)
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aqlm_lib as L  # noqa: E402


def main():
    layer, expert = 18, 0
    layout = L.load_layout()
    spec = layout["layers"][str(layer)]
    blob = L.read_blob(layer, expert, spec)

    import mlx.core as mx
    print("== 1. manual dequant vs mx.dequantize ==")
    for proj in L.PROJS:
        s = spec["slices"]
        wq = L.slice_arr(blob, s[f"{proj}.weight"], np.uint32)
        sc = L.bf16_to_f32(L.slice_arr(blob, s[f"{proj}.scales"], np.uint16))
        bi = L.bf16_to_f32(L.slice_arr(blob, s[f"{proj}.biases"], np.uint16))
        with mx.stream(mx.cpu):
            ref = mx.dequantize(mx.array(wq), mx.array(sc), mx.array(bi),
                                group_size=L.GROUP, bits=L.BITS)
            mx.eval(ref)
        ref = np.array(ref)
        mine = L.dequant_proj(blob, spec, proj)
        exact = np.array_equal(ref, mine)
        print(f"  {proj}: shapes {mine.shape} bit-exact={exact} "
              f"maxabsdiff={np.abs(ref-mine).max():.3e}")
        assert exact or np.abs(ref - mine).max() < 1e-6, proj

    print("== 2/3/4. per-arm rel MSE on all three projections ==")
    for proj in L.PROJS:
        W = L.dequant_proj(blob, spec, proj)
        # arm A: container roundtrip
        wq, sc16, bi16 = L.quantize_container(W)
        Wa = L.dequant_container(wq, sc16, bi16, W.shape[0])
        # arm C: affine 2-bit
        Wc = L.affine2bit_roundtrip(W)
        # arm B: codebook 2-bit
        t0 = time.time()
        Wb, stats = L.codebook2bit(W, seed=layer * 1000 + expert)
        tb = time.time() - t0
        print(f"  {proj}: A(container)={L.rel_mse(W, Wa):.3e}  "
              f"B(codebook2b)={L.rel_mse(W, Wb):.3e} ({tb:.1f}s, util {stats['codebook_util']})  "
              f"C(affine2b)={L.rel_mse(W, Wc):.3e}")

    print("== 5. blob rebuild roundtrip ==")
    tensors = {}
    for proj in L.PROJS:
        W = L.dequant_proj(blob, spec, proj)
        wq, sc16, bi16 = L.quantize_container(W)
        tensors[f"{proj}.weight"] = wq
        tensors[f"{proj}.scales"] = sc16
        tensors[f"{proj}.biases"] = bi16
    blob2 = L.build_blob(spec, tensors)
    assert len(blob2) == spec["stride"]
    for proj in L.PROJS:
        s = spec["slices"]
        back = L.slice_arr(blob2, s[f"{proj}.weight"], np.uint32)
        assert np.array_equal(back, tensors[f"{proj}.weight"]), proj
        back_sc = L.slice_arr(blob2, s[f"{proj}.scales"], np.uint16)
        assert np.array_equal(back_sc, tensors[f"{proj}.scales"]), proj
    print("  blob rebuild: OK (weights+scales roundtrip bit-exact)")


if __name__ == "__main__":
    main()
