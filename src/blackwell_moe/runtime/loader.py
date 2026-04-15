"""Streaming FP8 loader: quantize MoE experts on-the-fly to fit in 12.5 GB VRAM.

Strategy
--------
DeepSeek-V2-Lite bf16 = 31 GB. Neither RAM (16 GB free) nor VRAM (12.5 GB free)
fit it. Solution: stream tensors one at a time, FP8-quantize routed-expert
weights as they land on GPU, keep everything else bf16.

Layout after load:
  - Embedding + lm_head:   GPU bf16  (~0.8 GB)
  - Attention (MLA):       GPU bf16  (~1.5 GB)
  - Norms, router gates:   GPU bf16  (small)
  - Shared experts:        GPU bf16  (~0.8 GB)
  - Routed experts:        GPU FP8   (~14 GB → fits)
  - Scales:                GPU fp32  (~8 KB per expert)

Peak transient: 1 shard (~8 GB on disk) loaded incrementally — each tensor
moves to GPU and frees immediately.
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


_EXPERT_RE = re.compile(r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight")


def _is_routed_expert_weight(key: str) -> tuple[int, int, str] | None:
    m = _EXPERT_RE.match(key)
    if m:
        return int(m.group(1)), int(m.group(2)), m.group(3)
    return None


def _find_shard_map(model_dir: str) -> dict[str, str]:
    with open(Path(model_dir) / "model.safetensors.index.json") as f:
        return json.load(f)["weight_map"]


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


def _quant_fp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    amax = x.abs().amax().clamp(min=1e-4).to(torch.float32)
    scale = (448.0 / amax).to(torch.float32)
    q = (x.to(torch.float32) * scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return q, scale


def load_deepseek_fp8_streaming(model_dir: str, device: str = "cuda"):
    """Load DeepSeek-V2-Lite with routed experts pre-quantized to FP8 on GPU."""
    print("Building empty model skeleton (meta device)")
    cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)

    # Pre-allocate stacked FP8 expert tensors per MoE layer
    n_layers = cfg.num_hidden_layers
    first_moe = cfg.first_k_dense_replace
    n_experts = cfg.n_routed_experts
    D, H = cfg.hidden_size, cfg.moe_intermediate_size

    fp8_store: dict[int, dict[str, torch.Tensor]] = {}
    for li in range(first_moe, n_layers):
        fp8_store[li] = {
            "gate": torch.empty((n_experts, D, H), device=device, dtype=torch.float8_e4m3fn),
            "up":   torch.empty((n_experts, D, H), device=device, dtype=torch.float8_e4m3fn),
            "down": torch.empty((n_experts, H, D), device=device, dtype=torch.float8_e4m3fn),
            "s_g":  torch.empty(n_experts, device=device, dtype=torch.float32),
            "s_u":  torch.empty(n_experts, device=device, dtype=torch.float32),
            "s_d":  torch.empty(n_experts, device=device, dtype=torch.float32),
        }

    shard_map = _find_shard_map(model_dir)
    shards = sorted({Path(model_dir) / s for s in set(shard_map.values())})
    print(f"Streaming {len(shards)} shards, quantizing routed experts on-the-fly")

    n_total = 0
    n_experts_q = 0
    for shard_path in shards:
        with safe_open(str(shard_path), framework="pt", device=device) as f:
            for k in f.keys():
                t = f.get_tensor(k)
                ex = _is_routed_expert_weight(k)
                if ex is not None:
                    layer_idx, expert_idx, which = ex
                    store = fp8_store[layer_idx]
                    # DeepSeek expert proj is [out, in], we want [in, out] for our kernel
                    if which == "gate":
                        q, s = _quant_fp8(t.t().contiguous().to(torch.bfloat16))
                        store["gate"][expert_idx] = q
                        store["s_g"][expert_idx] = s
                    elif which == "up":
                        q, s = _quant_fp8(t.t().contiguous().to(torch.bfloat16))
                        store["up"][expert_idx] = q
                        store["s_u"][expert_idx] = s
                    else:  # down
                        q, s = _quant_fp8(t.t().contiguous().to(torch.bfloat16))
                        store["down"][expert_idx] = q
                        store["s_d"][expert_idx] = s
                    del t
                    n_experts_q += 1
                else:
                    _set_param(model, k, t.to(torch.bfloat16))
                n_total += 1
        gc.collect()
        torch.cuda.empty_cache()
        mem = torch.cuda.memory_allocated() / 1e9
        print(f"  {shard_path.name}: {n_total} tensors loaded, VRAM {mem:.1f} GB")

    print(f"Loaded {n_total} tensors total, quantized {n_experts_q} expert weights")
    for p in model.parameters():
        p.requires_grad_(False)
    return model, fp8_store


# Keep old name for backward compat in cli
load_deepseek_fp8 = load_deepseek_fp8_streaming
