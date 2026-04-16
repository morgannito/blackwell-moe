"""Stream-load Qwen3.6-35B-A3B-FP8 non-expert weights to GPU.

Layout per shard `layers-{L}.safetensors`:
    model.language_model.layers.{L}.input_layernorm.weight
    model.language_model.layers.{L}.post_attention_layernorm.weight
    model.language_model.layers.{L}.linear_attn.{...}            (hybrid DeltaNet)
    model.language_model.layers.{L}.self_attn.{...}              (periodic full-attn)
    model.language_model.layers.{L}.mlp.gate.weight              (router)
    model.language_model.layers.{L}.mlp.shared_expert.{...}
    model.language_model.layers.{L}.mlp.experts.{E}.{gate|up|down}_proj.weight
    model.language_model.layers.{L}.mlp.experts.{E}.*.weight_scale_inv  (FP8)

`outside.safetensors` holds embed / final norm / lm_head (and the visual
tower we skip for text-only).
"""

from __future__ import annotations

import gc
import re
from pathlib import Path

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM


_EXPERT_RE = re.compile(r"\.mlp\.experts\.\d+\.")
_VISUAL_RE = re.compile(r"\.visual\.")


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


def _remap_key(k: str) -> str:
    return k.replace("model.language_model.", "model.")


def _dequant_block_fp8(w: torch.Tensor, scale_inv: torch.Tensor,
                       block: int = 128) -> torch.Tensor:
    """Block-scale FP8 dequant: w_bf16 = w_fp8 * scale[i//b, j//b]"""
    M, N = w.shape
    Mb, Nb = scale_inv.shape
    s = scale_inv.to(torch.bfloat16)
    s = s.repeat_interleave(block, dim=0)[:M]
    s = s.repeat_interleave(block, dim=1)[:, :N]
    return w.to(torch.bfloat16) * s


def _stream_shard(model: nn.Module, path: Path, device: str,
                    skip_experts: bool, skip_visual: bool) -> tuple[int, int]:
    loaded = skipped = 0
    with safe_open(str(path), framework="pt", device=device) as f:
        all_keys = list(f.keys())
        kept_keys = set()
        for k in all_keys:
            if skip_experts and _EXPERT_RE.search(k):
                continue
            if skip_visual and _VISUAL_RE.search(k):
                continue
            kept_keys.add(k)
        # Pair weight with its weight_scale_inv
        scale_keys = {k for k in kept_keys if k.endswith(".weight_scale_inv")}
        weight_keys = kept_keys - scale_keys

        for k in weight_keys:
            remapped = _remap_key(k)
            scale_k = k.replace(".weight", ".weight_scale_inv")
            try:
                t = f.get_tensor(k)
            except Exception as e:
                print(f"  WARN: {k}: {e}")
                continue
            if scale_k in scale_keys:
                try:
                    scale = f.get_tensor(scale_k)
                    t = _dequant_block_fp8(t, scale)
                except Exception as e:
                    print(f"  WARN dequant {k}: {e}")
                    continue
            try:
                _set_param(model, remapped, t)
                loaded += 1
            except AttributeError:
                skipped += 1
        skipped += len(all_keys) - len(weight_keys)
    return loaded, skipped


def load_qwen_streaming(
    model_dir: str,
    device: str = "cuda",
    skip_experts: bool = True,
    skip_visual: bool = True,
):
    """Load Qwen3.6-35B-A3B skeleton + non-expert weights streamed to GPU.

    Returns (model, config). Experts still on meta/disk — patch the MoE
    layers to fetch them on-demand via `ThreeTierExpertCache`.
    """
    print("Building empty Qwen3.6 skeleton")
    cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    # transformers 5.5 needs text_config attrs hoisted to root for init
    if hasattr(cfg, "text_config"):
        for k, v in vars(cfg.text_config).items():
            if not k.startswith("_") and not hasattr(cfg, k):
                setattr(cfg, k, v)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)

    model_path = Path(model_dir)
    shards = sorted(model_path.glob("*.safetensors"))
    print(f"Streaming {len(shards)} shards to {device} "
          f"(skip_experts={skip_experts}, skip_visual={skip_visual})...")

    total_loaded = total_skipped = 0
    for shard_path in shards:
        loaded, skipped = _stream_shard(model, shard_path, device,
                                          skip_experts, skip_visual)
        total_loaded += loaded
        total_skipped += skipped
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            mem_gb = torch.cuda.memory_allocated() / 1e9
            print(f"  {shard_path.name}: +{loaded} kept, -{skipped} skip, "
                  f"GPU {mem_gb:.2f} GB")

    print(f"Loaded {total_loaded} tensors, skipped {total_skipped}")

    _materialize_rotary(model, device)

    for p in model.parameters():
        p.requires_grad_(False)

    return model, cfg


def _materialize_rotary(model: nn.Module, device: str) -> None:
    """Ensure rotary buffers (inv_freq, etc.) live on the target device."""
    n = 0
    for name, module in model.named_modules():
        cls = module.__class__.__name__
        if "Rotary" not in cls and "rope" not in cls.lower():
            continue
        # Move any buffer on meta / cpu to device
        for bname, buf in list(module.named_buffers(recurse=False)):
            if buf.device.type != device:
                module._buffers[bname] = buf.to(device)
                n += 1
    if n:
        print(f"Moved {n} rotary buffers to {device}")
