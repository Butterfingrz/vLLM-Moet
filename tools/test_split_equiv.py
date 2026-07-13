#!/usr/bin/env python3
"""E2E split==non-split equivalence (split-FP4 fix acceptance criterion 3).

Same GPU-resident layer state, same inputs, same PROMOTED EXPERT SET at
equal SLOT COUNT (not GiB): one forward through moe_w2_cubit with the FP4
tier in NON-SPLIT mode (moe_w4_mm, full nibble slots), one with the tier
in SPLIT mode (moe_w4q_mm, quintal slots read against the resident base).
The quintal decode reconstructs the SAME e4m3 bytes as the nibble decode
(bit-exact e2m1 for all 16 nibbles), the QMMA K-march and epilogue are
identical — so the outputs must be BIT-IDENTICAL. Any divergence means a
decode defect (the legacy 2-bit refinement fails this on every expert
with zeros: mag-0 -> 0.5 merge).

Run (inside the vllm image, 1 GPU): python3 tools/test_split_equiv.py
"""
import os
import sys

import torch

os.environ.setdefault("VLLM_MOE_W2", "1")

from vllm.model_executor.layers.quantization.utils import moe_w2_cubit  # noqa: E402
from vllm.model_executor.layers.quantization.utils import moe_w2_delta  # noqa: E402
from vllm.model_executor.layers.quantization.utils.moe_w2_planes import (  # noqa: E402
    mxfp4_to_codes, mxfp4_to_nibbles, pack_fp4_fragment_major,
    pack_fragment_major, pack_quintal_fragment_major, pack_scales,
)

assert moe_w2_cubit._ensure_ready(), "cubins not found"
dev = torch.device("cuda")
torch.manual_seed(29)

E = int(os.environ.get("E", "32"))
H = int(os.environ.get("H", "4096"))
I = int(os.environ.get("I", "2048"))
T = int(os.environ.get("T", "12"))
TOPK = 6

w13 = torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, device=dev)
s13 = torch.randint(118, 124, (E, 2 * I, H // 32), dtype=torch.uint8, device=dev)
w2 = torch.randint(0, 256, (E, H, I // 2), dtype=torch.uint8, device=dev)
s2 = torch.randint(118, 124, (E, H, I // 32), dtype=torch.uint8, device=dev)

moe_w2_cubit._LAYERS[0] = dict(
    N13=2 * I, K13=H, N2=H, K2=I, E=E, base=False,
    planes13=torch.stack([pack_fragment_major(mxfp4_to_codes(w13[e]))
                          for e in range(E)]),
    sc13=torch.stack([pack_scales(s13[e]) for e in range(E)]),
    planes2=torch.stack([pack_fragment_major(mxfp4_to_codes(w2[e]))
                         for e in range(E)]),
    sc2=torch.stack([pack_scales(s2[e]) for e in range(E)]),
)

x = (torch.randn(T, H, device=dev) * 0.3).to(torch.bfloat16)
topk_ids = torch.stack([torch.randperm(E, device=dev)[:TOPK]
                        for _ in range(T)]).to(torch.int32)
topk_w = torch.rand(T, TOPK, device=dev) * 0.5
promoted = list(range(0, E, 2))          # identical set in both modes
SLOTS = len(promoted)


def run_mode(split: bool) -> torch.Tensor:
    moe_w2_delta._SPLIT = split
    if split:
        w13b, w2b = 2 * I * H * 5 // 16, H * I * 5 // 16
        pk13, pk2 = pack_quintal_fragment_major, pack_quintal_fragment_major
    else:
        w13b, w2b = 2 * I * H // 2, H * I // 2
        pk13, pk2 = pack_fp4_fragment_major, pack_fp4_fragment_major
    pool_gb = SLOTS * (w13b + w2b) / 2**30
    tier = moe_w2_delta.DeltaTier(1, E, dev, w13_bytes=w13b, w2_bytes=w2b,
                                  pool_gb=pool_gb, policy="freq", tag="fp4",
                                  host_pinned=True)
    moe_w2_delta._TIER = tier
    fp13 = torch.stack([pk13(mxfp4_to_nibbles(w13[e])) for e in range(E)])
    fp2 = torch.stack([pk2(mxfp4_to_nibbles(w2[e])) for e in range(E)])
    tier.add_layer_host_planes(0, fp13, fp2)
    with tier._lock:
        for e in promoted:                       # pre-promote IDENTICAL list
            slot = tier._take_slots_batch(1)[0]
            tier._promote(0, e, slot)
    torch.cuda.synchronize()
    assert all(int(tier._mirror[0, e]) >= 0 for e in promoted)
    out = moe_w2_cubit._moe_w2_forward(x, topk_w, topk_ids, 0)
    torch.cuda.synchronize()
    moe_w2_delta._TIER = None
    del tier
    return out.clone()


out_ns = run_mode(split=False)
out_sp = run_mode(split=True)
biteq = torch.equal(out_ns.view(torch.int16), out_sp.view(torch.int16))
md = (out_ns.float() - out_sp.float()).abs().max().item()
print(f"split(w4q) vs non-split(w4), {SLOTS}/{E} experts FP4-mapped, "
      f"T={T} topk={TOPK}: bit-identical={biteq} max_abs_diff={md:.3e}")
print("RESULT:", "PASS" if biteq else "FAIL")
sys.exit(0 if biteq else 1)
