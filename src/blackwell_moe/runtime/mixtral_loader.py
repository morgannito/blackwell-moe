"""Stream-load Mixtral with non-MoE weights on GPU and experts on disk.

Pre-condition: experts have been extracted with `scripts/extract_experts_to_disk.py`
into `expert_root`. Then this loader:
  1. Builds the model skeleton on the meta device
  2. Streams non-expert weights to GPU bf16 from the original safetensors shards
  3. Sets up the `ThreeTierExpertCache` pointed at `expert_root`
  4. Patches MoE layers to delegate expert compute to the cache
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
from blackwell_moe.runtime.mixtral_patch import patch_mixtral_streaming


_EXPERT_RE = re.compile(
    r"model\.layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.(w[123])\.weight"
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


def load_mixtral_streaming(
    model_dir: str,
    expert_root: str,
    gpu_slots: int = 8,        # how many experts to keep in VRAM
    ram_slots: int = 32,       # RAM tier capacity
    device: str = "cuda",
):
    print("Building empty Mixtral skeleton")
    cfg = AutoConfig.from_pretrained(model_dir)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(cfg)

    print("Streaming non-expert weights to GPU bf16 (skipping MoE experts)...")
    idx_path = Path(model_dir) / "model.safetensors.index.json"
    weight_map = json.loads(idx_path.read_text())["weight_map"]
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

    print(f"Loaded {n_loaded} non-expert tensors, skipped {n_skipped} expert tensors")

    for p in model.parameters():
        p.requires_grad_(False)

    # Build expert cache
    H = cfg.intermediate_size
    D = cfg.hidden_size
    cache = ThreeTierExpertCache(
        expert_root=expert_root,
        n_layers=cfg.num_hidden_layers,
        n_experts_per_layer=cfg.num_local_experts,
        gpu_slots=gpu_slots,
        ram_slots=ram_slots,
        gpu_buffer_specs={
            "gate_q": (D, H),
            "up_q":   (D, H),
            "down_q": (H, D),
        },
        device=device,
    )

    n_patched = patch_mixtral_streaming(model, cache)
    print(f"Patched {n_patched} Mixtral MoE blocks with streaming cache")
    print(f"GPU after patching: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    return model, cache
