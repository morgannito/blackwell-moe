"""v0.2 MoE forward: grouped FP8 GEMM + fused SwiGLU+quant.

Replaces the per-expert dispatch loop of v0.1 with:
  1. Single permute of tokens by assigned expert
  2. One grouped FP8 GEMM for gate
  3. One grouped FP8 GEMM for up
  4. Fused SwiGLU + quantize → FP8 hidden
  5. One grouped FP8 GEMM for down
  6. Unpermute + weighted combine
"""

from __future__ import annotations

import torch

from blackwell_moe.kernels.fused_swiglu import compute_segment_scales, fused_swiglu_quant
from blackwell_moe.kernels.grouped_fp8 import grouped_fp8_gemm
from blackwell_moe.kernels.permute import permute_tokens, unpermute_and_combine


def _quant_fp8(x: torch.Tensor) -> tuple[torch.Tensor, float]:
    amax = x.abs().amax().clamp(min=1e-4).to(torch.float32)
    scale = (448.0 / amax).to(torch.float32)
    return (x.to(torch.float32) * scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn), scale


def _quant_fp8_per_segment(x: torch.Tensor, offsets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    E = offsets.numel() - 1
    scales = compute_segment_scales(x, offsets)
    x_fp8 = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    for e in range(E):
        s, en = int(offsets[e].item()), int(offsets[e + 1].item())
        if en > s:
            q = (x[s:en].to(torch.float32) * scales[e]).clamp(-448.0, 448.0)
            x_fp8[s:en] = q.to(torch.float8_e4m3fn)
    return x_fp8, scales


def fp8_moe_forward_v2(
    x: torch.Tensor,                    # [T, D] bf16
    w_gate: torch.Tensor,               # [D, E] bf16
    experts_w_gate_fp8: torch.Tensor,   # [E, D, H] fp8
    experts_w_up_fp8: torch.Tensor,     # [E, D, H] fp8
    experts_w_down_fp8: torch.Tensor,   # [E, H, D] fp8
    scales_gate_w: torch.Tensor,        # [E] fp32
    scales_up_w: torch.Tensor,          # [E] fp32
    scales_down_w: torch.Tensor,        # [E] fp32
    top_k: int = 8,
) -> torch.Tensor:
    T, D = x.shape
    E = experts_w_gate_fp8.shape[0]

    # Routing
    logits = x.to(torch.float32) @ w_gate.to(torch.float32)
    probs = torch.softmax(logits, dim=-1)
    weights, indices = torch.topk(probs, k=top_k, dim=-1)
    weights = weights / weights.sum(dim=-1, keepdim=True)  # [T,K] fp32

    # Permute tokens: [T,K] slot → sorted by expert id
    flat_ids = indices.reshape(-1).to(torch.int32)
    sorted_ids, sort_perm = torch.sort(flat_ids, stable=True)
    TK = flat_ids.numel()
    src_idx = torch.arange(TK, device=x.device, dtype=torch.long) // top_k
    inverse_idx = src_idx[sort_perm]
    x_perm = x[inverse_idx]  # [TK, D] bf16

    counts = torch.bincount(sorted_ids, minlength=E)
    offsets = torch.zeros(E + 1, dtype=torch.int32, device=x.device)
    offsets[1:] = counts.cumsum(0).to(torch.int32)

    # Quantize activations per expert segment (tight scales)
    x_perm_fp8, scales_x = _quant_fp8_per_segment(x_perm, offsets)

    # Grouped GEMMs: gate & up
    gate = grouped_fp8_gemm(x_perm_fp8, experts_w_gate_fp8, offsets,
                            scales_x, scales_gate_w)
    up = grouped_fp8_gemm(x_perm_fp8, experts_w_up_fp8, offsets,
                          scales_x, scales_up_w)

    # Fused SwiGLU + quant — scales computed inline (pre-pass on h)
    h_tmp = torch.nn.functional.silu(gate) * up  # bf16
    scales_h = compute_segment_scales(h_tmp, offsets)
    h_fp8 = fused_swiglu_quant(gate, up, offsets, scales_h)

    # Down projection (grouped)
    y_perm = grouped_fp8_gemm(h_fp8, experts_w_down_fp8, offsets,
                              scales_h, scales_down_w)

    # Unpermute + weighted combine
    flat_w = weights.reshape(-1)[sort_perm].to(y_perm.dtype).unsqueeze(-1)
    weighted = y_perm * flat_w
    out = torch.zeros((T, D), device=x.device, dtype=x.dtype)
    out.index_add_(0, inverse_idx, weighted.to(x.dtype))
    return out
