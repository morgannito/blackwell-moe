"""INT4 MoE forward — bf16 activations, int4 weights.

Half VRAM of FP8 for expert weights (4 bits vs 8 bits per param).
SwiGLU in bf16 (no quant pass needed; activations stay bf16 throughout).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from blackwell_moe.kernels.grouped_int4 import grouped_int4_gemm


def int4_moe_forward(
    x: torch.Tensor,                   # [T, D] bf16
    w_gate: torch.Tensor,              # [D, E] bf16
    experts_w_gate_q: torch.Tensor,    # [E, D, H/2] uint8 packed int4
    experts_w_up_q: torch.Tensor,      # [E, D, H/2] uint8
    experts_w_down_q: torch.Tensor,    # [E, H, D/2] uint8
    scales_gate: torch.Tensor,         # [E, H] bf16
    scales_up: torch.Tensor,           # [E, H] bf16
    scales_down: torch.Tensor,         # [E, D] bf16
    inter_dim: int,                    # H
    hidden_dim: int,                   # D
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

    # Gate & up projections (int4 weights)
    gate = grouped_int4_gemm(x_perm, experts_w_gate_q, scales_gate, offsets, N=inter_dim)
    up = grouped_int4_gemm(x_perm, experts_w_up_q, scales_up, offsets, N=inter_dim)

    h = F.silu(gate) * up  # bf16

    # Down projection (int4)
    y_perm = grouped_int4_gemm(h, experts_w_down_q, scales_down, offsets, N=hidden_dim)

    flat_w = weights.reshape(-1)[sort_perm].to(y_perm.dtype).unsqueeze(-1)
    out = torch.zeros((T, D), device=x.device, dtype=x.dtype)
    out.index_add_(0, inverse_idx, (y_perm * flat_w).to(x.dtype))
    return out
