"""DeepSeek-V2-Lite end-to-end with streaming experts (3-tier cache).

Validates the full streaming pipeline on a model we already have. Once
proven on DeepSeek-V2-Lite, the same code paths drive Mixtral-8x22B.

Differences vs `loader.py` + `deepseek_patch.py`:
  - Routed experts are NOT loaded to VRAM — they live as per-expert files
  - At runtime, `streaming_moe_forward` fetches via `ThreeTierExpertCache`
  - Shared experts stay on GPU as FP8 (still small)
  - Non-MoE weights stay on GPU bf16
"""

from __future__ import annotations

import gc
import json
import re
from pathlib import Path

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM

from blackwell_moe.runtime.disk_expert_pool import ThreeTierExpertCache


_EXPERT_RE = re.compile(
    r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight"
)


def _set_param(model: nn.Module, key: str, tensor: torch.Tensor) -> None:
    parts = key.split(".")
    mod = model
    for p in parts[:-1]:
        mod = getattr(mod, p)
    leaf = parts[-1]
    cur = getattr(mod, leaf, None)
    if isinstance(cur, nn.Parameter):
        mod._parameters[leaf] = nn.Parameter(tensor, requires_grad=False)
    else:
        mod._buffers[leaf] = tensor


def load_deepseek_streaming(
    model_dir: str,
    expert_root: str,
    gpu_slots: int = 16,
    ram_slots: int = 32,
    device: str = "cuda",
):
    """Load DeepSeek-V2-Lite with routed experts on disk (3-tier cache)."""
    print("Building empty DeepSeek skeleton")
    cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)

    print("Streaming non-routed-expert weights to GPU bf16...")
    idx = json.loads((Path(model_dir) / "model.safetensors.index.json").read_text())
    weight_map = idx["weight_map"]
    shards = sorted({Path(model_dir) / s for s in set(weight_map.values())})

    n_loaded = 0
    n_skipped = 0
    for shard_path in shards:
        with safe_open(str(shard_path), framework="pt", device=device) as f:
            for k in f.keys():
                if _EXPERT_RE.match(k):
                    n_skipped += 1
                    continue
                t = f.get_tensor(k).to(torch.bfloat16)
                _set_param(model, k, t)
                n_loaded += 1
        gc.collect()
        torch.cuda.empty_cache()
        mem_gb = torch.cuda.memory_allocated() / 1e9
        print(f"  {shard_path.name}: GPU {mem_gb:.2f} GB")

    print(f"Loaded {n_loaded} non-expert tensors, skipped {n_skipped} routed-expert tensors")

    for p in model.parameters():
        p.requires_grad_(False)

    cache = ThreeTierExpertCache(
        expert_root=expert_root,
        n_layers=cfg.num_hidden_layers,
        n_experts_per_layer=cfg.n_routed_experts,
        gpu_slots=gpu_slots,
        ram_slots=ram_slots,
        gpu_buffer_specs={
            "gate_q": (cfg.hidden_size, cfg.moe_intermediate_size),
            "up_q":   (cfg.hidden_size, cfg.moe_intermediate_size),
            "down_q": (cfg.moe_intermediate_size, cfg.hidden_size),
        },
        device=device,
    )

    n_patched = patch_deepseek_streaming(model, cache)
    print(f"Patched {n_patched} DeepseekV2MoE layers with streaming cache")
    print(f"VRAM after patching: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    return model, cache


# --- Patch -------------------------------------------------------------------

from blackwell_moe.runtime.streaming_moe import streaming_moe_forward


class StreamingDeepseekMoE(nn.Module):
    def __init__(self, original: nn.Module, layer_idx: int,
                  cache: ThreeTierExpertCache):
        super().__init__()
        cfg = original.config
        self.layer_idx = layer_idx
        self.cache = cache
        self.top_k = cfg.num_experts_per_tok
        self.hidden_dim = cfg.hidden_size
        self.routed_scale = cfg.routed_scaling_factor
        gate_w = original.gate.weight.detach().clone()
        self.register_buffer("w_router", gate_w.t().contiguous())
        self.shared_experts = original.shared_experts
        self.gate = original.gate

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shape = hidden_states.shape
        x = hidden_states.reshape(-1, self.hidden_dim).to(torch.bfloat16)
        shared = self.shared_experts(hidden_states).reshape(-1, self.hidden_dim)
        next_layer = self.layer_idx + 1
        prefetch = next_layer if next_layer < self.cache.n_layers else None
        routed = streaming_moe_forward(
            x, self.w_router, self.cache, self.layer_idx, top_k=self.top_k,
            prefetch_next_layer=prefetch,
        )
        if self.routed_scale != 1.0:
            routed = routed * self.routed_scale
        return (routed + shared.to(routed.dtype)).reshape(shape)


def patch_deepseek_streaming(model: nn.Module, cache: ThreeTierExpertCache) -> int:
    patched = 0
    for name, module in list(model.named_modules()):
        if module.__class__.__name__ != "DeepseekV2MoE":
            continue
        layer_idx = int(name.split(".")[2])
        parent_name = ".".join(name.split(".")[:-1])
        child_name = name.split(".")[-1]
        parent = model.get_submodule(parent_name)
        setattr(parent, child_name,
                StreamingDeepseekMoE(module, layer_idx, cache))
        del module
        patched += 1
    return patched
