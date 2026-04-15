"""v0.3 MoE forward: segment-reduce amax + fused gate+up + online SwiGLU+quant."""

from __future__ import annotations

import torch

import torch.nn.functional as F

from blackwell_moe.kernels.grouped_fp8 import grouped_fp8_gemm
from blackwell_moe.kernels.grouped_fp8_gateup import fused_gate_up_gemm
from blackwell_moe.kernels.segment_ops import (
    segment_fp8_scales,
    segment_quant_fp8,
    segment_quant_fp8_fused,
)


def fp8_moe_forward_v3(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    experts_w_gate_fp8: torch.Tensor,
    experts_w_up_fp8: torch.Tensor,
    experts_w_down_fp8: torch.Tensor,
    scales_gate_w: torch.Tensor,
    scales_up_w: torch.Tensor,
    scales_down_w: torch.Tensor,
    top_k: int = 8,
) -> torch.Tensor:
    T, D = x.shape
    E = experts_w_gate_fp8.shape[0]

    # Routing (torch native; negligible cost vs MoE)
    logits = x.to(torch.float32) @ w_gate.to(torch.float32)
    probs = torch.softmax(logits, dim=-1)
    weights, indices = torch.topk(probs, k=top_k, dim=-1)
    weights = weights / weights.sum(dim=-1, keepdim=True)

    flat_ids = indices.reshape(-1).to(torch.int32)
    sorted_ids, sort_perm = torch.sort(flat_ids, stable=True)
    TK = flat_ids.numel()
    src_idx = torch.arange(TK, device=x.device, dtype=torch.long) // top_k
    inverse_idx = src_idx[sort_perm]
    x_perm = x[inverse_idx]

    counts = torch.bincount(sorted_ids, minlength=E)
    offsets = torch.zeros(E + 1, dtype=torch.int32, device=x.device)
    offsets[1:] = counts.cumsum(0).to(torch.int32)

    # Triton segment amax + quant — single launch, no Python loop
    x_perm_fp8, scales_x = segment_quant_fp8_fused(x_perm, offsets)

    # Fused gate+up — one kernel, shared X load
    gate, up = fused_gate_up_gemm(
        x_perm_fp8, experts_w_gate_fp8, experts_w_up_fp8,
        offsets, scales_x, scales_gate_w, scales_up_w,
    )

    # SwiGLU + per-segment quantize (reuses Triton segment ops)
    h = F.silu(gate) * up
    scales_h = segment_fp8_scales(h, offsets)
    h_fp8 = segment_quant_fp8(h, offsets, scales_h)

    # Grouped down projection
    y_perm = grouped_fp8_gemm(
        h_fp8, experts_w_down_fp8, offsets, scales_h, scales_down_w,
    )

    # Unpermute + weighted combine
    flat_w = weights.reshape(-1)[sort_perm].to(y_perm.dtype).unsqueeze(-1)
    out = torch.zeros((T, D), device=x.device, dtype=x.dtype)
    out.index_add_(0, inverse_idx, (y_perm * flat_w).to(x.dtype))
    return out
