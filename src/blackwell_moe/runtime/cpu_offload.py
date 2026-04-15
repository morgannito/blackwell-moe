"""CPU offload wrappers for embed_tokens and lm_head.

Both layers are hit only twice per generated token (embed lookup at the start,
final logits at the end). Their weights together occupy ~0.8 GB of bf16 VRAM
on DeepSeek-V2-Lite (vocab 102 400 × hidden 2048 × 2 bytes × 2 layers).

Keeping the weights on CPU and shuttling activations across PCIe is faster
than wrestling with `accelerate.AlignDevicesHook` (which broke generation in
v0.5) because:
  - input_ids → small (T int64 tensor) — H2D / D2H is sub-millisecond
  - hidden_states for lm_head → small (T × hidden bf16)
  - matmul on CPU at this size: ~3 ms with mkl, vs 50 µs on GPU + 0.8 GB VRAM
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CPUEmbedding(nn.Module):
    """Drop-in replacement for an `nn.Embedding` whose weight lives on CPU."""

    def __init__(self, original: nn.Embedding, gpu_device: str = "cuda"):
        super().__init__()
        self.gpu_device = gpu_device
        self.num_embeddings = original.num_embeddings
        self.embedding_dim = original.embedding_dim
        self.padding_idx = original.padding_idx
        weight_cpu = original.weight.detach().cpu().contiguous()
        self.register_buffer("weight", weight_cpu)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        ids_cpu = input_ids.cpu() if input_ids.device.type != "cpu" else input_ids
        out = F.embedding(ids_cpu, self.weight, self.padding_idx)
        return out.to(self.gpu_device, non_blocking=True)


class CPULinear(nn.Module):
    """Drop-in replacement for `nn.Linear` (no bias) whose weight stays on CPU."""

    def __init__(self, original: nn.Linear, gpu_device: str = "cuda"):
        super().__init__()
        self.gpu_device = gpu_device
        self.in_features = original.in_features
        self.out_features = original.out_features
        weight_cpu = original.weight.detach().cpu().contiguous()
        self.register_buffer("weight", weight_cpu)
        if original.bias is not None:
            bias_cpu = original.bias.detach().cpu().contiguous()
            self.register_buffer("bias", bias_cpu)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_cpu = x.detach().cpu() if x.device.type != "cpu" else x
        out = F.linear(x_cpu, self.weight, self.bias)
        return out.to(self.gpu_device, non_blocking=True)


def offload_embed_and_lm_head(model: nn.Module, gpu_device: str = "cuda") -> int:
    """Replace `model.model.embed_tokens` and `model.lm_head` with CPU versions.

    Returns the number of MB freed on GPU.
    """
    freed_mb = 0
    embed = model.model.embed_tokens
    if isinstance(embed, nn.Embedding):
        size = embed.weight.numel() * embed.weight.element_size()
        freed_mb += size / 1e6
        model.model.embed_tokens = CPUEmbedding(embed, gpu_device)
        del embed

    lm_head = model.lm_head
    if isinstance(lm_head, nn.Linear):
        size = lm_head.weight.numel() * lm_head.weight.element_size()
        freed_mb += size / 1e6
        model.lm_head = CPULinear(lm_head, gpu_device)
        del lm_head

    torch.cuda.empty_cache()
    return int(freed_mb)
