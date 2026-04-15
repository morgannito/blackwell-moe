"""Streaming MoE forward — fetches active experts on-demand from 3-tier cache.

For models too big to fit in VRAM (Mixtral-8x22B etc.), the per-layer MoE
forward calls into `ThreeTierExpertCache.fetch()` to ensure the top-k chosen
experts for the current batch are GPU-resident, then runs the standard
grouped-FP8 GEMM kernel against them.

Strategy
--------
* Routing happens on GPU as usual (small bf16 matmul)
* Top-k expert ids are sent to the cache as a single fetch
* Cache returns a [N_slots] remap so we can dispatch with v3-style kernels
* We process the layer, then optionally pre-fetch the next layer's experts
  on a separate stream while we compute attention
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from blackwell_moe.kernels.grouped_fp8 import grouped_fp8_gemm
from blackwell_moe.kernels.scatter import scatter_add
from blackwell_moe.kernels.segment_ops import (
    segment_fp8_scales,
    segment_quant_fp8,
    segment_quant_fp8_fused,
)
from blackwell_moe.runtime.disk_expert_pool import ThreeTierExpertCache


def streaming_moe_forward(
    x: torch.Tensor,                   # [T, D] bf16
    w_gate: torch.Tensor,              # [D, E_total] bf16
    cache: ThreeTierExpertCache,
    layer_idx: int,
    top_k: int,
) -> torch.Tensor:
    T, D = x.shape

    # Routing — we need original [0, E_total) indices to query the cache
    logits = x.to(torch.float32) @ w_gate.to(torch.float32)
    probs = torch.softmax(logits, dim=-1)
    weights, indices = torch.topk(probs, k=top_k, dim=-1)
    weights = weights / weights.sum(dim=-1, keepdim=True)

    # Unique expert ids needed for this batch
    unique_eids = torch.unique(indices.flatten()).tolist()

    # Fetch into GPU slots; returns slot id per requested expert id (in order)
    slots_for_eids = cache.fetch(layer_idx, unique_eids)

    # Build remap: original_eid → slot_idx
    remap = torch.full((cache.gpu_slots,), -1, dtype=torch.int32, device=x.device)
    eid_to_slot: dict[int, int] = {}
    for eid, slot in zip(unique_eids, slots_for_eids.tolist()):
        eid_to_slot[eid] = slot

    # Map every (token, k) routing slot to its GPU slot index
    slot_indices = torch.empty_like(indices, dtype=torch.int32)
    flat_idx = indices.reshape(-1)
    for eid, slot in eid_to_slot.items():
        slot_indices.view(-1)[flat_idx == eid] = slot

    # Now standard v3 forward but using cache.gpu tensors as the expert pool
    flat_ids = slot_indices.reshape(-1).to(torch.int32)
    sorted_ids, sort_perm = torch.sort(flat_ids, stable=True)
    TK = flat_ids.numel()
    src_idx = torch.arange(TK, device=x.device, dtype=torch.long) // top_k
    inverse_idx = src_idx[sort_perm]
    x_perm = x[inverse_idx]

    counts = torch.bincount(sorted_ids, minlength=cache.gpu_slots)
    offsets = torch.zeros(cache.gpu_slots + 1, dtype=torch.int32, device=x.device)
    offsets[1:] = counts.cumsum(0).to(torch.int32)

    x_perm_fp8, scales_x = segment_quant_fp8_fused(x_perm, offsets)

    gpu = cache.gpu
    gate = grouped_fp8_gemm(x_perm_fp8, gpu["gate_q"], offsets,
                             scales_x, gpu["scale_g"])
    up = grouped_fp8_gemm(x_perm_fp8, gpu["up_q"], offsets,
                           scales_x, gpu["scale_u"])
    h = F.silu(gate) * up
    scales_h = segment_fp8_scales(h, offsets)
    h_fp8 = segment_quant_fp8(h, offsets, scales_h)
    y_perm = grouped_fp8_gemm(h_fp8, gpu["down_q"], offsets,
                               scales_h, gpu["scale_d"])

    flat_w = weights.reshape(-1)[sort_perm].to(y_perm.dtype).unsqueeze(-1)
    weighted = (y_perm * flat_w).to(x.dtype).contiguous()
    out = torch.zeros((T, D), device=x.device, dtype=x.dtype)
    scatter_add(out, inverse_idx, weighted)
    return out
