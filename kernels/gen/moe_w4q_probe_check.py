#!/usr/bin/env python3
"""HW validation of the radix-5 split-FP4 decode probe (moe_w4q_probe).

32 lanes x 32 elems of e2m1 nibbles (first 16 forced = all nibble values,
rest random) -> base 2-bit codes (proximity map, byte w = elems 4w..4w+3,
2-bit LE) + 80-bit radix-5 lane records (8 words x 10 bits; word =
d0+5d1+25d2+125d3, digit = mag_idx for small / mag_idx-5 for big).
Kernel must reproduce the e2m1 e4m3 byte EXACTLY for all 16 nibbles
(zero included) -- the bit-exactness criterion at the op level.

Env: CUBIN (default /tmp/moe_w4q_probe.cubin), SEED, RUNS.
"""
import ctypes
import os
import sys

import numpy as np

sys.path.insert(0, "/workspace/cubit/tools")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from culaunch import Cuda  # noqa: E402

CUBIN = os.environ.get("CUBIN", "/tmp/moe_w4q_probe.cubin")
RUNS = int(os.environ.get("RUNS", "4"))
rng = np.random.default_rng(int(os.environ.get("SEED", "5")))

NIBBLE_TO_CODE = np.array([2] * 5 + [3] * 3 + [1] * 5 + [0] * 3, dtype=np.uint8)
E4M3_MAG = np.array([0x00, 0x30, 0x38, 0x3C, 0x40, 0x44, 0x48, 0x4C],
                    dtype=np.uint8)

LANES, ELEMS = 32, 32
nib = rng.integers(0, 16, (LANES, ELEMS), dtype=np.uint8)
nib[:, :16] = np.arange(16, dtype=np.uint8)      # every nibble, every lane

mag = nib & 7
code = NIBBLE_TO_CODE[nib]
big = (code == 0) | (code == 3)
digit = np.where(big, mag - 5, mag).astype(np.uint16)

# base plane: byte w = elems 4w..4w+3, 2-bit fields LE
c = code.reshape(LANES, 8, 4).astype(np.uint16)
base_bytes = (c[..., 0] | (c[..., 1] << 2) | (c[..., 2] << 4)
              | (c[..., 3] << 6)).astype(np.uint8)          # [LANES, 8]

# radix-5 records: word w = digits of elems 4w..4w+3
d = digit.reshape(LANES, 8, 4)
words = (d[..., 0] + 5 * d[..., 1] + 25 * d[..., 2]
         + 125 * d[..., 3]).astype(np.uint64)               # [LANES, 8]
rec = np.zeros(LANES, dtype=object)
p8 = np.zeros((LANES, 8), dtype=np.uint8)
p2 = np.zeros((LANES, 2), dtype=np.uint8)
for ln in range(LANES):
    v = 0
    for w in range(8):
        v |= int(words[ln, w]) << (10 * w)
    p8[ln] = np.frombuffer((v & (2**64 - 1)).to_bytes(8, "little"), np.uint8)
    p2[ln] = np.frombuffer((v >> 64).to_bytes(2, "little"), np.uint8)
ref_plane = np.concatenate([p8.reshape(-1), p2.reshape(-1)])  # 256B + 64B

expect = (E4M3_MAG[mag] | np.where(nib >= 8, 0x80, 0)).astype(np.uint8)

cu = Cuda()
fn = cu.load_kernel(CUBIN, "moe_w4q_probe")
d_ref = cu.to_device(ref_plane)
d_base = cu.to_device(base_bytes.reshape(-1))
d_out = cu.alloc(LANES * ELEMS)

outs = []
for r in range(RUNS):
    cu.memset32(d_out, 0, LANES * ELEMS // 4)
    cu.launch(fn, (1, 1, 1), (32, 1, 1),
              [d_ref, d_base, d_out])
    cu.synchronize()
    got = cu.from_device(d_out, LANES * ELEMS, dtype=np.uint8).copy()
    outs.append(got.tobytes())
got = np.frombuffer(outs[-1], dtype=np.uint8).reshape(LANES, ELEMS)

bad = np.nonzero(got != expect)
n_bad = len(bad[0])
det = len(set(outs)) == 1
print(f"elems={LANES*ELEMS} mismatches={n_bad} deterministic={det}")
if n_bad:
    for i in range(min(8, n_bad)):
        ln, e = bad[0][i], bad[1][i]
        print(f"  lane {ln} elem {e}: nib={nib[ln,e]:#x} "
              f"got={got[ln,e]:#04x} want={expect[ln,e]:#04x}")
ok = n_bad == 0 and det
print("RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
