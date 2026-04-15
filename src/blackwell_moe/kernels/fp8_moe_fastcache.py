"""MoE forward using FastExpertCache — expects remap tensor from batch fetch."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from blackwell_moe.kernels.grouped_fp8 import grouped_fp8_gemm
from blackwell_moe.kernels.segment_ops import (
    segment_fp8_scales,
    segment_quant_fp8,
    segment_quant_fp8_fused,
)


def fp8_moe_forward_fastcache(
    x: torch.Tensor,                    # [T, D] bf16
    router_weights: torch.Tensor,       # [T, K] fp32
    expert_indices: torch.Tensor,       # [T, K] int32 (original expert ids)
    remap: torch.Tensor,                # [E] expert_id -> slot (or -1)
    cache_g: torch.Tensor,              # [N, D, H] fp8
    cache_u: torch.Tensor,
    cache_d: torch.Tensor,              # [N, H, D] fp8
    cache_sg: torch.Tensor,             # [N] fp32
    cache_su: torch.Tensor,
    cache_sd: torch.Tensor,
    n_slots: int,
    top_k: int = 8,
) -> torch.Tensor:
    T, D = x.shape

    slot_ids = remap[expert_indices.to(torch.long)]  # [T, K] int32
    flat_ids = slot_ids.reshape(-1).to(torch.int32)
    sorted_ids, sort_perm = torch.sort(flat_ids, stable=True)
    TK = flat_ids.numel()
    src_idx = torch.arange(TK, device=x.device, dtype=torch.long) // top_k
    inverse_idx = src_idx[sort_perm]
    x_perm = x[inverse_idx]

    counts = torch.bincount(sorted_ids, minlength=n_slots)
    offsets = torch.zeros(n_slots + 1, dtype=torch.int32, device=x.device)
    offsets[1:] = counts.cumsum(0).to(torch.int32)

    x_perm_fp8, scales_x = segment_quant_fp8_fused(x_perm, offsets)
    gate = grouped_fp8_gemm(x_perm_fp8, cache_g, offsets, scales_x, cache_sg)
    up = grouped_fp8_gemm(x_perm_fp8, cache_u, offsets, scales_x, cache_su)
    h = F.silu(gate) * up
    scales_h = segment_fp8_scales(h, offsets)
    h_fp8 = segment_quant_fp8(h, offsets, scales_h)
    y_perm = grouped_fp8_gemm(h_fp8, cache_d, offsets, scales_h, cache_sd)

    flat_w = router_weights.reshape(-1)[sort_perm].to(y_perm.dtype).unsqueeze(-1)
    out = torch.zeros((T, D), device=x.device, dtype=x.dtype)
    out.index_add_(0, inverse_idx, (y_perm * flat_w).to(x.dtype))
    return out
