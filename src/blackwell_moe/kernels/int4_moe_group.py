"""INT4 group-scale MoE forward (Q4_0 style, bf16 activations)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from blackwell_moe.kernels.grouped_int4_group import grouped_int4_group_gemm


def int4_group_moe_forward(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    experts_w_gate_q: torch.Tensor,     # [E, D, H/2] uint8
    experts_w_up_q: torch.Tensor,
    experts_w_down_q: torch.Tensor,     # [E, H, D/2] uint8
    scales_gate: torch.Tensor,          # [E, D/32, H] bf16
    scales_up: torch.Tensor,
    scales_down: torch.Tensor,          # [E, H/32, D] bf16
    inter_dim: int,
    hidden_dim: int,
    top_k: int = 8,
) -> torch.Tensor:
    T, D = x.shape
    E = experts_w_gate_q.shape[0]

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

    gate = grouped_int4_group_gemm(x_perm, experts_w_gate_q, scales_gate, offsets, N=inter_dim)
    up = grouped_int4_group_gemm(x_perm, experts_w_up_q, scales_up, offsets, N=inter_dim)
    h = F.silu(gate) * up

    y_perm = grouped_int4_group_gemm(h, experts_w_down_q, scales_down, offsets, N=hidden_dim)

    flat_w = weights.reshape(-1)[sort_perm].to(y_perm.dtype).unsqueeze(-1)
    out = torch.zeros((T, D), device=x.device, dtype=x.dtype)
    out.index_add_(0, inverse_idx, (y_perm * flat_w).to(x.dtype))
    return out
