"""Token permutation utilities for MoE dispatch.

Given token→expert assignments, produce a permuted token layout where
tokens routed to the same expert are contiguous, with segment offsets.
"""

from __future__ import annotations

import torch


def permute_tokens(
    x: torch.Tensor,            # [T, D]
    expert_ids: torch.Tensor,   # [T * K]  (top-k flattened)
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (x_perm [M,D], offsets [E+1], inverse_idx [M]).

    M = T*K (each token duplicated once per selected expert).
    inverse_idx[i] = source token of row i in x_perm (for scatter-add later).
    """
    T, D = x.shape
    TK = expert_ids.numel()

    # Source token index for each of the TK slots
    src_idx = torch.arange(TK, device=x.device) // (TK // T)

    # Sort by expert id
    sorted_eids, sort_perm = torch.sort(expert_ids.view(-1), stable=True)
    inverse_idx = src_idx[sort_perm]

    x_perm = x[inverse_idx]

    # Offsets via bincount
    counts = torch.bincount(sorted_eids, minlength=num_experts)
    offsets = torch.zeros(num_experts + 1, dtype=torch.int32, device=x.device)
    offsets[1:] = counts.cumsum(0).to(torch.int32)

    return x_perm, offsets, inverse_idx


def unpermute_and_combine(
    y_perm: torch.Tensor,       # [M, D]
    inverse_idx: torch.Tensor,  # [M]
    routing_weights: torch.Tensor,  # [T, K]
    top_k_ids: torch.Tensor,    # [T, K] (in original order)
    expert_ids_sorted_perm: torch.Tensor,  # perm that sorted the flat ids
    T: int,
) -> torch.Tensor:
    """Scatter-add weighted expert outputs back to token positions."""
    K = routing_weights.shape[1]
    # routing_weights is [T,K]; in flat order matching expert_ids.view(-1),
    # slot (t,k) carries weight routing_weights[t,k]. After sort, row i of
    # y_perm corresponds to slot sort_perm[i].
    flat_w = routing_weights.reshape(-1)[expert_ids_sorted_perm].to(y_perm.dtype)
    weighted = y_perm * flat_w.unsqueeze(-1)
    out = torch.zeros((T, y_perm.shape[1]), device=y_perm.device, dtype=y_perm.dtype)
    out.index_add_(0, inverse_idx, weighted)
    return out
