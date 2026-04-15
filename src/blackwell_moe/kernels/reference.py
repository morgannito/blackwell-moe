"""Reference bf16 implementation for correctness testing."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def moe_forward_bf16(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    experts_w_gate: torch.Tensor,  # [E, D, H] bf16
    experts_w_up: torch.Tensor,    # [E, D, H] bf16
    experts_w_down: torch.Tensor,  # [E, H, D] bf16
    top_k: int = 8,
) -> torch.Tensor:
    logits = x.to(torch.float32) @ w_gate.to(torch.float32)
    probs = F.softmax(logits, dim=-1)
    w, idx = torch.topk(probs, k=top_k, dim=-1)
    w = (w / w.sum(dim=-1, keepdim=True)).to(x.dtype)

    T, D = x.shape
    out = torch.zeros_like(x)
    E = experts_w_gate.shape[0]

    for e in range(E):
        mask = (idx == e).any(dim=-1)
        if not mask.any():
            continue
        tok = mask.nonzero(as_tuple=True)[0]
        x_e = x[tok]
        gate = x_e @ experts_w_gate[e]
        up = x_e @ experts_w_up[e]
        h = F.silu(gate) * up
        y = h @ experts_w_down[e]
        w_slot = (idx[tok] == e).to(x.dtype)
        w_e = (w[tok] * w_slot).sum(dim=-1, keepdim=True)
        out.index_add_(0, tok, y * w_e)
    return out
