"""Group-scale INT4 (Q4_0 style) — bf16 scale per 32-K block.

Per-channel scale (v0.6) handles magnitude variation across output channels
but not along K. Real MoE weights show significant amplitude drift along K,
so one scale per 32-param K-block cuts the L1 error from ~22 % to ~5 %.

Layout
------
W  [K, N] bf16 input
→ groups = K / GROUP_K (= K / 32)
→ packed [K/2, N] uint8  (2 nibbles per byte, same as v0.6)
→ scales [groups, N] bf16  (one scale per K-group per output channel)

Effective bits per param:
  4 (int4) + 16 / 32 (scale per 32-group) = 4.5 bits/param
  Matches GGUF Q4_0 density.
"""

from __future__ import annotations

import torch

GROUP_K = 32


def quantize_int4_groups(w: torch.Tensor, group_k: int = GROUP_K) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize [K, N] bf16 → packed [K, N/2] uint8 + scales [K/group_k, N] bf16."""
    K, N = w.shape
    assert K % group_k == 0, f"K={K} must be multiple of group_k={group_k}"
    assert N % 2 == 0
    G = K // group_k

    w_f32 = w.to(torch.float32)

    # Per-group amax → scale = amax / 7 (symmetric)
    w_grp = w_f32.view(G, group_k, N)
    amax = w_grp.abs().amax(dim=1).clamp(min=1e-6)  # [G, N]
    scales = (amax / 7.0).to(torch.bfloat16)

    # Quantize: divide each K-group by its scale
    scales_expanded = scales.to(torch.float32).unsqueeze(1).expand(G, group_k, N).reshape(K, N)
    w_q = (w_f32 / scales_expanded).round().clamp(-8, 7).to(torch.int32) + 8
    w_q = w_q.to(torch.uint8)

    # Pack nibbles (same as v0.6)
    low = w_q[:, 0::2] & 0xF
    high = (w_q[:, 1::2] & 0xF) << 4
    packed = (low | high).to(torch.uint8)
    return packed, scales


def dequantize_int4_groups(packed: torch.Tensor, scales: torch.Tensor,
                            K: int, N: int, group_k: int = GROUP_K) -> torch.Tensor:
    """Reference dequant for correctness checks."""
    G = K // group_k
    low = packed & 0xF
    high = (packed >> 4) & 0xF
    w_q = torch.empty((K, N), device=packed.device, dtype=torch.int32)
    w_q[:, 0::2] = low.to(torch.int32)
    w_q[:, 1::2] = high.to(torch.int32)
    w_s = (w_q - 8).to(torch.float32)
    scales_f32 = scales.to(torch.float32)                               # [G, N]
    scales_exp = scales_f32.unsqueeze(1).expand(G, group_k, N).reshape(K, N)
    return (w_s * scales_exp).to(torch.bfloat16)
