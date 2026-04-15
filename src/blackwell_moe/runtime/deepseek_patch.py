"""Patch DeepSeek-V2 MoE layer to use our FP8 kernels.

Weights are pre-quantized by the streaming loader and passed in as a dict.
The patch installs an FP8MoELayer that references those stacked tensors.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from blackwell_moe.kernels.fp8_moe_v3 import fp8_moe_forward_v3


class FP8MoELayer(nn.Module):
    def __init__(self, deepseek_moe: nn.Module, fp8_weights: dict):
        super().__init__()
        cfg = deepseek_moe.config
        self.n_experts = cfg.n_routed_experts
        self.top_k = cfg.num_experts_per_tok
        self.hidden_dim = cfg.hidden_size
        self.routed_scale = cfg.routed_scaling_factor

        # Router weight from original (it was loaded bf16 by the loader)
        gate_w = deepseek_moe.gate.weight.detach().clone()  # [E, D]
        self.register_buffer("w_router", gate_w.t().contiguous())  # [D, E]

        # Stacked FP8 expert weights (pre-quantized by loader)
        self.register_buffer("experts_w_gate_fp8", fp8_weights["gate"])
        self.register_buffer("experts_w_up_fp8", fp8_weights["up"])
        self.register_buffer("experts_w_down_fp8", fp8_weights["down"])
        self.register_buffer("scales_gate", fp8_weights["s_g"])
        self.register_buffer("scales_up", fp8_weights["s_u"])
        self.register_buffer("scales_down", fp8_weights["s_d"])

        self.shared_experts = deepseek_moe.shared_experts
        self.gate = deepseek_moe.gate

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shape = hidden_states.shape
        x = hidden_states.reshape(-1, self.hidden_dim).to(torch.bfloat16)

        shared_out = self.shared_experts(hidden_states).reshape(-1, self.hidden_dim)

        routed = fp8_moe_forward_v3(
            x, self.w_router,
            self.experts_w_gate_fp8, self.experts_w_up_fp8, self.experts_w_down_fp8,
            self.scales_gate, self.scales_up, self.scales_down,
            top_k=self.top_k,
        )
        if self.routed_scale != 1.0:
            routed = routed * self.routed_scale
        return (routed + shared_out.to(routed.dtype)).reshape(shape)


def patch_deepseek_moe_with_store(model: nn.Module, fp8_store: dict[int, dict]) -> int:
    patched = 0
    for name, module in list(model.named_modules()):
        if module.__class__.__name__ != "DeepseekV2MoE":
            continue
        # name like "model.layers.1.mlp"; extract layer idx
        layer_idx = int(name.split(".")[2])
        if layer_idx not in fp8_store:
            continue
        parent_name = ".".join(name.split(".")[:-1])
        child_name = name.split(".")[-1]
        parent = model.get_submodule(parent_name) if parent_name else model
        fp8_layer = FP8MoELayer(module, fp8_store[layer_idx])
        setattr(parent, child_name, fp8_layer)
        # Free original experts from model graph to release any residual refs
        del module
        patched += 1
    return patched
