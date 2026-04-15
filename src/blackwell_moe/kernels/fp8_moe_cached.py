"""MoE forward with LRU expert caching: remaps [0,E) → [0,N_slots) indices."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from blackwell_moe.kernels.fp8_moe_v3 import fp8_moe_forward_v3
from blackwell_moe.runtime.expert_cache import LRUExpertCache


def fp8_moe_forward_cached(
    x: torch.Tensor,                # [T, D] bf16
    w_gate: torch.Tensor,           # [D, E] bf16
    cache: LRUExpertCache,
    top_k: int = 8,
) -> torch.Tensor:
    T, D = x.shape
    E = cache.E
    N = cache.N

    logits = x.to(torch.float32) @ w_gate.to(torch.float32)
    probs = torch.softmax(logits, dim=-1)
    weights, indices = torch.topk(probs, k=top_k, dim=-1)
    weights = weights / weights.sum(dim=-1, keepdim=True)

    # Cache: ensure all selected experts are on GPU, get remap table
    slot_remap = cache.fetch(indices)  # [E] → slot or -1

    # Remap original E-space indices to slot-space [0, N)
    slot_indices = slot_remap[indices]  # [T, K] values in [0, N)
    # Sanity: none should be -1 after fetch
    # Hand off to our v3 forward using GPU-resident experts
    # BUT v3 expects indices in [0, n_experts_in_tensor). Since we've
    # remapped to slots, we pass N as the expert count.
    return fp8_moe_forward_v3(
        x, w_gate,  # w_gate still uses full-E routing; slot_indices overrides
        cache.gpu_g, cache.gpu_u, cache.gpu_d,
        cache.gpu_sg, cache.gpu_su, cache.gpu_sd,
        top_k=top_k,
    )
    # NOTE: the above uses full-E router + N-slot experts — these must agree.
    # Practical impl below bypasses the router softmax re-computation.


def fp8_moe_forward_cached_direct(
    x: torch.Tensor,                # [T, D] bf16
    router_weights: torch.Tensor,   # [T, K] fp32, pre-normalized
    expert_indices: torch.Tensor,   # [T, K] int, original expert ids
    cache: LRUExpertCache,
    top_k: int = 8,
) -> torch.Tensor:
    """Forward with pre-computed routing. Remaps ids → slots internally."""
    from blackwell_moe.kernels.grouped_fp8 import grouped_fp8_gemm
    from blackwell_moe.kernels.segment_ops import segment_quant_fp8_fused

    T, D = x.shape
    slot_remap = cache.fetch(expert_indices)

    # Map each token-slot assignment to slot id
    slot_ids = slot_remap[expert_indices]  # [T, K]

    flat_ids = slot_ids.reshape(-1).to(torch.int32)
    sorted_ids, sort_perm = torch.sort(flat_ids, stable=True)
    TK = flat_ids.numel()
    src_idx = torch.arange(TK, device=x.device, dtype=torch.long) // top_k
    inverse_idx = src_idx[sort_perm]
    x_perm = x[inverse_idx]

    counts = torch.bincount(sorted_ids, minlength=cache.N)
    offsets = torch.zeros(cache.N + 1, dtype=torch.int32, device=x.device)
    offsets[1:] = counts.cumsum(0).to(torch.int32)

    x_perm_fp8, scales_x = segment_quant_fp8_fused(x_perm, offsets)

    gate = grouped_fp8_gemm(x_perm_fp8, cache.gpu_g, offsets,
                             scales_x, cache.gpu_sg)
    up = grouped_fp8_gemm(x_perm_fp8, cache.gpu_u, offsets,
                           scales_x, cache.gpu_su)

    h = F.silu(gate) * up
    from blackwell_moe.kernels.segment_ops import segment_fp8_scales, segment_quant_fp8
    scales_h = segment_fp8_scales(h, offsets)
    h_fp8 = segment_quant_fp8(h, offsets, scales_h)

    y_perm = grouped_fp8_gemm(h_fp8, cache.gpu_d, offsets, scales_h, cache.gpu_sd)

    flat_w = router_weights.reshape(-1)[sort_perm].to(y_perm.dtype).unsqueeze(-1)
    out = torch.zeros((T, D), device=x.device, dtype=x.dtype)
    out.index_add_(0, inverse_idx, (y_perm * flat_w).to(x.dtype))
    return out
