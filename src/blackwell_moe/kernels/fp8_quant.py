"""Shared FP8 E4M3 quantization helpers.

Single source of truth for per-tensor FP8 quantize/dequant, used by all
kernels and runtime code. Removes duplicated `_quant_fp8` copies that had
drifted across `fp8_moe_torch`, `fp8_moe_v2`, `fp8_moe_v3`, `int4_group`,
`shared_expert_fp8`, `deepseek_patch`, and `loader`.
"""

from __future__ import annotations

import torch

__all__ = ["FP8_MAX_E4M3", "quant_fp8_e4m3", "dequant_fp8_e4m3"]

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
