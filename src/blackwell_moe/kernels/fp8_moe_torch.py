"""FP8 MoE using native torch._scaled_mm (no Triton, Blackwell sm_120+).

torch._scaled_mm is PyTorch's native FP8 matmul kernel using cuBLAS/cuBLASLt.
On Blackwell, it dispatches to FP8 tensor cores directly. No custom kernel needed
for the GEMM — we focus value on routing, dispatch, and fused activation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


from blackwell_moe.kernels.fp8_quant import quant_fp8_e4m3 as _quant_fp8  # re-export


def _scaled_mm(a_fp8, b_fp8, sa, sb, out_dtype=torch.bfloat16):
    # a: [M,K] fp8 row-major. b must be column-major for _scaled_mm → transpose T
    # API: _scaled_mm(a, b, scale_a, scale_b, out_dtype, use_fast_accum)
    return torch._scaled_mm(
        a_fp8, b_fp8,
        scale_a=(1.0 / sa).to(torch.float32),
        scale_b=(1.0 / sb).to(torch.float32),
        out_dtype=out_dtype,
        use_fast_accum=True,
    )


def fp8_moe_forward_torch(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    experts_w_gate_fp8: torch.Tensor,   # [E, D, H] fp8, col-major friendly
    experts_w_up_fp8: torch.Tensor,
    experts_w_down_fp8: torch.Tensor,
    scales_gate: torch.Tensor,
    scales_up: torch.Tensor,
    scales_down: torch.Tensor,
    top_k: int = 8,
) -> torch.Tensor:
    T, D = x.shape
    E = experts_w_gate_fp8.shape[0]

    # Routing (torch-native, no triton)
    logits = x.to(torch.float32) @ w_gate.to(torch.float32)
    probs = torch.softmax(logits, dim=-1)
    weights, indices = torch.topk(probs, k=top_k, dim=-1)
    weights = (weights / weights.sum(dim=-1, keepdim=True)).to(x.dtype)

    out = torch.zeros_like(x)

    for e in range(E):
        mask = (indices == e).any(dim=-1)
        if not mask.any():
            continue
        tok = mask.nonzero(as_tuple=True)[0]
        x_e = x[tok]

        x_fp8, sx = _quant_fp8(x_e)
        # _scaled_mm wants b as column-major; .t().contiguous().t() ensures that
        w_g = experts_w_gate_fp8[e].t().contiguous().t()
        w_u = experts_w_up_fp8[e].t().contiguous().t()
        w_d = experts_w_down_fp8[e].t().contiguous().t()

        gate = _scaled_mm(x_fp8, w_g, sx, scales_gate[e])
        up = _scaled_mm(x_fp8, w_u, sx, scales_up[e])
        hidden = F.silu(gate) * up

        h_fp8, sh = _quant_fp8(hidden)
        y = _scaled_mm(h_fp8, w_d, sh, scales_down[e])

        w_slot = (indices[tok] == e).to(x.dtype)
        w_e = (weights[tok] * w_slot).sum(dim=-1, keepdim=True)
        out.index_add_(0, tok, y * w_e)

    return out
