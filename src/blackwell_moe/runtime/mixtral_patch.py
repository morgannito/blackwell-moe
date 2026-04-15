"""Patch a Mixtral model to use streaming MoE with our 3-tier expert cache.

Mixtral-8x22B / 8x7B layout:
  model.layers.<L>.block_sparse_moe.gate                -> [E, D]
  model.layers.<L>.block_sparse_moe.experts.<E>.w1      -> gate_proj  [H, D]
  model.layers.<L>.block_sparse_moe.experts.<E>.w3      -> up_proj    [H, D]
  model.layers.<L>.block_sparse_moe.experts.<E>.w2      -> down_proj  [D, H]

The native `MixtralSparseMoeBlock` keeps all 8 experts as full nn.Linear.
We replace it with `StreamingMixtralMoE` that holds only a router weight
and delegates expert compute to a shared `ThreeTierExpertCache`.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from blackwell_moe.runtime.disk_expert_pool import ThreeTierExpertCache
from blackwell_moe.runtime.streaming_moe import streaming_moe_forward


class StreamingMixtralMoE(nn.Module):
    """Drop-in replacement for `MixtralSparseMoeBlock` backed by 3-tier cache."""

    def __init__(self, original: nn.Module, layer_idx: int,
                  cache: ThreeTierExpertCache):
        super().__init__()
        self.layer_idx = layer_idx
        self.cache = cache
        self.top_k = original.top_k
        self.hidden_dim = original.hidden_dim
        # Router stays bf16 on GPU (it's tiny: D × E)
        gate_w = original.gate.weight.detach().clone()
        self.register_buffer("w_router", gate_w.t().contiguous())  # [D, E]
        # Forward jitter / norm if any — preserve original behavior
        self.gate = original.gate

    def forward(self, hidden_states: torch.Tensor):
        shape = hidden_states.shape
        x = hidden_states.reshape(-1, self.hidden_dim).to(torch.bfloat16)
        out = streaming_moe_forward(
            x, self.w_router, self.cache, self.layer_idx, top_k=self.top_k,
        )
        # Mixtral returns (out, router_logits) tuple
        return out.reshape(shape), torch.zeros(1, device=x.device)


def patch_mixtral_streaming(model: nn.Module, cache: ThreeTierExpertCache) -> int:
    patched = 0
    for name, module in list(model.named_modules()):
        if module.__class__.__name__ == "MixtralSparseMoeBlock":
            layer_idx = int(name.split(".")[2])
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            parent = model.get_submodule(parent_name)
            setattr(parent, child_name,
                    StreamingMixtralMoE(module, layer_idx, cache))
            del module
            patched += 1
    return patched
