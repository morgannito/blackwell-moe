"""Shared FP8 E4M3 quantization helpers.

Single source of truth for per-tensor FP8 quantize/dequant, used by all
kernels and runtime code. Removes duplicated `_quant_fp8` copies that had
drifted across `fp8_moe_torch`, `fp8_moe_v2`, `fp8_moe_v3`, `int4_group`,
`shared_expert_fp8`, `deepseek_patch`, and `loader`.
"""

from __future__ import annotations

import torch

__all__ = [
    "FP8_MAX_E4M3",
    "dequant_fp8_e4m3",
    "quant_fp8_block",
    "quant_fp8_e4m3",
    "quant_fp8_per_row",
]

FP8_MAX_E4M3: float = 448.0


def quant_fp8_e4m3(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-tensor symmetric FP8 E4M3 quantize.

    Returns (x_fp8, scale) where ``(x_fp8.to(fp32) / scale) ≈ x``.
    """
    amax = x.abs().amax().clamp(min=1e-4).to(torch.float32)
    scale = (FP8_MAX_E4M3 / amax).to(torch.float32)
    q = (x.to(torch.float32) * scale).clamp(-FP8_MAX_E4M3, FP8_MAX_E4M3).to(torch.float8_e4m3fn)
    return q, scale


def dequant_fp8_e4m3(x_fp8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (x_fp8.to(torch.float32) / scale).to(torch.bfloat16)


def quant_fp8_per_row(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row FP8 E4M3 quantization for [M, D] activations.

    Returns (x_fp8 [M, D], scales [M]). Per-row scaling reduces error vs
    per-tensor when row magnitudes vary widely (typical for hidden states
    after norm layers).
    """
    assert x.dim() == 2
    amax = x.abs().amax(dim=-1).clamp(min=1e-4).to(torch.float32)
    scales = (FP8_MAX_E4M3 / amax).to(torch.float32)                # [M]
    q = (x.to(torch.float32) * scales.unsqueeze(-1)).clamp(-FP8_MAX_E4M3, FP8_MAX_E4M3)
    return q.to(torch.float8_e4m3fn), scales


def quant_fp8_block(
    x: torch.Tensor, block_k: int = 32
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-K-block FP8 quantization for [K, N] weights.

    Returns (x_fp8, scales [K/block_k, N]). Mirrors the INT4 group-scale
    layout — useful for FP8 weights with strong K-axis amplitude drift
    (rare in practice, but available for ablations).
    """
    K, N = x.shape
    assert K % block_k == 0
    G = K // block_k
    x_grp = x.view(G, block_k, N).to(torch.float32)
    amax = x_grp.abs().amax(dim=1).clamp(min=1e-4)                  # [G, N]
    scales = (FP8_MAX_E4M3 / amax).to(torch.float32)
    q = (x_grp * scales.unsqueeze(1)).clamp(-FP8_MAX_E4M3, FP8_MAX_E4M3)
    return q.reshape(K, N).to(torch.float8_e4m3fn), scales
