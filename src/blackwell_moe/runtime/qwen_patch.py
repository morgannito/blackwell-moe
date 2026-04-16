"""Patch Qwen3.6-35B-A3B MoE blocks to stream experts from disk on demand.

Design:
  - Non-expert weights (router, shared_expert, attention, norms) stay on GPU
  - Expert weights live in per-layer safetensors shards (one file per layer,
    all 256 experts inside)
  - A per-layer `LayerShardReader` opens the shard once with mmap and serves
    tensors by (expert_id, proj_kind) via `get_tensor(key)`
  - A tiny GPU LRU cache keyed by (layer_idx, expert_id) holds recently-used
    experts across forward passes

Block-scale FP8 (128 × 128 blocks): each weight has a companion
`weight_scale_inv` tensor we dequantize on the fly.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open


_PROJ_NAMES = ("gate_proj", "up_proj", "down_proj")


def _dequant_block_fp8(w_fp8: torch.Tensor, scale_inv: torch.Tensor,
                       block: int = 128) -> torch.Tensor:
    """Dequantize a [M,N] FP8 weight with [M/b, N/b] bf16 block scales."""
    M, N = w_fp8.shape
    Mb, Nb = scale_inv.shape
    assert Mb * block >= M and Nb * block >= N, (w_fp8.shape, scale_inv.shape)
    w = w_fp8.to(torch.bfloat16)
    scale = scale_inv.to(torch.bfloat16)
    scale_full = scale.repeat_interleave(block, dim=0)[:M]
    scale_full = scale_full.repeat_interleave(block, dim=1)[:, :N]
    return w * scale_full


class LayerShardReader:
    """Open a single layers-{L}.safetensors file and serve expert weights."""

    def __init__(self, path: Path, layer_key_prefix: str):
        self.path = Path(path)
        self.prefix = layer_key_prefix  # e.g. "model.language_model.layers.0.mlp.experts"
        self._handle = safe_open(str(self.path), framework="pt", device="cpu")

    def load_expert_fp8(self, expert_id: int, device: str = "cuda"):
        """Return {gate, up, down} raw FP8 + scales on GPU (no dequant)."""
        out: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for proj in _PROJ_NAMES:
            w_key = f"{self.prefix}.{expert_id}.{proj}.weight"
            s_key = f"{self.prefix}.{expert_id}.{proj}.weight_scale_inv"
            w_fp8 = self._handle.get_tensor(w_key).to(device, non_blocking=True)
            scale = self._handle.get_tensor(s_key).to(device, non_blocking=True)
            out[proj] = (w_fp8, scale)
        return out


class StreamingQwenMoE(nn.Module):
    """Drop-in replacement for `Qwen3_5MoeSparseMoeBlock`.

    Keeps router, shared_expert, shared_expert_gate on GPU (from original).
    Reads routed experts lazily from a per-layer shard reader, caches the
    last `cache_size` experts in GPU bf16.
    """

    def __init__(self, original: nn.Module, layer_idx: int,
                 reader: LayerShardReader, cache_size: int = 4):
        super().__init__()
        self.layer_idx = layer_idx
        self.reader = reader
        self.cache_size = cache_size
        self.gate = original.gate
        self.shared_expert = original.shared_expert
        self.shared_expert_gate = original.shared_expert_gate
        self.num_experts = original.experts.num_experts
        self.hidden_dim = original.experts.hidden_dim
        # Cache raw FP8 weights (3 MB each) — dequant on use
        self._cache: OrderedDict[int, dict[str, tuple]] = OrderedDict()

    def _get_expert_fp8(self, eid: int):
        if eid in self._cache:
            self._cache.move_to_end(eid)
            return self._cache[eid]
        weights = self.reader.load_expert_fp8(eid, device="cuda")
        self._cache[eid] = weights
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return weights

    def forward(self, hidden_states: torch.Tensor):
        bsz, seq_len, hidden_dim = hidden_states.shape
        x = hidden_states.view(-1, hidden_dim)
        shared = self.shared_expert(x)
        _, routing_weights, selected = self.gate(x)

        out = torch.zeros_like(x)
        unique_eids = torch.unique(selected).tolist()
        for eid in unique_eids:
            if eid == self.num_experts:
                continue
            wfp8 = self._get_expert_fp8(int(eid))
            pos = (selected == eid).nonzero(as_tuple=False)
            token_idx, k_idx = pos[:, 0], pos[:, 1]
            cur = x[token_idx]
            # Dequant on the fly, immediately free intermediates
            w_g = _dequant_block_fp8(*wfp8["gate_proj"])
            gate = F.linear(cur, w_g)
            del w_g
            w_u = _dequant_block_fp8(*wfp8["up_proj"])
            up = F.linear(cur, w_u)
            del w_u
            h = F.silu(gate) * up
            del gate, up
            w_d = _dequant_block_fp8(*wfp8["down_proj"])
            y = F.linear(h, w_d)
            del w_d, h
            weight = routing_weights[token_idx, k_idx].unsqueeze(-1).to(y.dtype)
            out.index_add_(0, token_idx, (y * weight).to(out.dtype))

        shared_gate = torch.sigmoid(self.shared_expert_gate(x))
        out = out + shared_gate * shared
        return out.view(bsz, seq_len, hidden_dim)


def patch_qwen_streaming(model: nn.Module, model_dir: str) -> int:
    """Replace every Qwen3_5MoeSparseMoeBlock with StreamingQwenMoE."""
    model_path = Path(model_dir)
    readers: dict[int, LayerShardReader] = {}

    # Discover layer -> shard mapping from the weight map
    import json
    idx = json.loads((model_path / "model.safetensors.index.json").read_text())
    weight_map = idx["weight_map"]

    def shard_for_layer(layer_idx: int) -> Path:
        for key, fname in weight_map.items():
            if f"layers.{layer_idx}.mlp.experts" in key:
                return model_path / fname
        raise FileNotFoundError(f"No shard for layer {layer_idx}")

    patched = 0
    for name, module in list(model.named_modules()):
        if module.__class__.__name__ == "Qwen3_5MoeSparseMoeBlock":
            layer_idx = int(name.split(".layers.")[1].split(".")[0])
            if layer_idx not in readers:
                path = shard_for_layer(layer_idx)
                prefix = f"model.language_model.layers.{layer_idx}.mlp.experts"
                readers[layer_idx] = LayerShardReader(path, prefix)
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            parent = model.get_submodule(parent_name)
            setattr(parent, child_name,
                    StreamingQwenMoE(module, layer_idx, readers[layer_idx]))
            patched += 1
    return patched
