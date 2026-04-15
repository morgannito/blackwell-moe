"""Top-k MoE router (gate softmax + expert selection).

Fused forward: logits -> softmax -> top-k -> normalized weights + indices.
Produces dense indices suitable for grouped GEMM dispatch.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _top_k_gate_kernel(
    hidden_ptr,           # [T, D]
    gate_w_ptr,           # [D, E]
    topk_w_ptr,           # [T, K] out
    topk_idx_ptr,         # [T, K] out
    T: int, D: tl.constexpr, E: tl.constexpr, K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= T:
        return

    # Load hidden[pid, :]
    off_d = tl.arange(0, BLOCK_D)
    mask_d = off_d < D
    h = tl.load(hidden_ptr + pid * D + off_d, mask=mask_d, other=0.0).to(tl.float32)

    # Compute logits = h @ gate_w  -> [E]
    logits = tl.zeros([E], dtype=tl.float32)
    for e in range(E):
        w = tl.load(gate_w_ptr + off_d * E + e, mask=mask_d, other=0.0).to(tl.float32)
        logits_e = tl.sum(h * w)
        logits = tl.where(tl.arange(0, E) == e, logits_e, logits)

    # Softmax
    m = tl.max(logits)
    p = tl.exp(logits - m)
    p = p / tl.sum(p)

    # Top-k (simple selection sort, K small)
    for k in range(K):
        best_val = tl.max(p)
        # find index of best
        is_best = p == best_val
        best_idx = tl.sum(tl.where(is_best, tl.arange(0, E), 0))
        tl.store(topk_w_ptr + pid * K + k, best_val)
        tl.store(topk_idx_ptr + pid * K + k, best_idx)
        p = tl.where(is_best, -1.0, p)

    # Renormalize top-k weights
    offs_k = tl.arange(0, K)
    w = tl.load(topk_w_ptr + pid * K + offs_k)
    w = w / tl.sum(w)
    tl.store(topk_w_ptr + pid * K + offs_k, w)


def top_k_router(
    hidden: torch.Tensor,   # [T, D] bf16
    gate_w: torch.Tensor,   # [D, E] bf16
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (weights[T,K] fp32, indices[T,K] int32)."""
    T, D = hidden.shape
    _, E = gate_w.shape
    w_out = torch.empty((T, k), device=hidden.device, dtype=torch.float32)
    i_out = torch.empty((T, k), device=hidden.device, dtype=torch.int32)

    BLOCK_D = triton.next_power_of_2(D)
    grid = (T,)
    _top_k_gate_kernel[grid](
        hidden, gate_w, w_out, i_out,
        T, D, E, k, BLOCK_D=BLOCK_D,
        num_warps=4,
    )
    return w_out, i_out


def top_k_router_ref(
    hidden: torch.Tensor, gate_w: torch.Tensor, k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference PyTorch implementation for correctness checking."""
    logits = hidden.to(torch.float32) @ gate_w.to(torch.float32)
    probs = torch.softmax(logits, dim=-1)
    w, idx = torch.topk(probs, k=k, dim=-1)
    w = w / w.sum(dim=-1, keepdim=True)
    return w, idx.to(torch.int32)
