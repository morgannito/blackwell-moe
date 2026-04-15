"""Convert a model's MoE experts to per-expert FP8 safetensors files on disk.

For Mixtral-8x22B / DeepSeek-V2 etc., we don't want to load the whole 100+ GB
checkpoint to RAM at runtime. Instead, we pre-process once: each expert
becomes its own small file that can be mmap'd in isolation.

Usage:
    python scripts/extract_experts_to_disk.py \\
        --src J:/models/Mixtral-8x22B-Instruct \\
        --dst J:/models/Mixtral-8x22B-Instruct-fp8-experts
"""

from __future__ import annotations

import argparse
import gc
import json
import re
from pathlib import Path

import torch
from safetensors import safe_open

from blackwell_moe.kernels.fp8_quant import quant_fp8_e4m3
from blackwell_moe.runtime.disk_expert_pool import save_expert_to_disk


# Mixtral pattern: model.layers.<L>.block_sparse_moe.experts.<E>.{w1,w2,w3}.weight
MIXTRAL_RE = re.compile(
    r"model\.layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.(w1|w2|w3)\.weight"
)
# DeepSeek pattern (already supported in loader.py)
DEEPSEEK_RE = re.compile(
    r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight"
)


def detect_format(weight_keys: list[str]) -> str:
    for k in weight_keys[:50]:
        if MIXTRAL_RE.match(k):
            return "mixtral"
        if DEEPSEEK_RE.match(k):
            return "deepseek"
    raise RuntimeError("Could not detect MoE layout from weight keys")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="HF model directory")
    p.add_argument("--dst", required=True, help="Output dir for per-expert files")
    args = p.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    # Read safetensors index
    idx = json.loads((src / "model.safetensors.index.json").read_text())
    weight_map = idx["weight_map"]
    fmt = detect_format(list(weight_map.keys()))
    pattern = MIXTRAL_RE if fmt == "mixtral" else DEEPSEEK_RE
    triplet_keys = (("w1", "w3", "w2") if fmt == "mixtral"
                    else ("gate", "up", "down"))
    print(f"Detected {fmt} layout, projection naming = {triplet_keys}")

    # Walk per-shard, per-expert
    shards = sorted({src / s for s in set(weight_map.values())})
    expert_buf: dict[tuple[int, int], dict[str, torch.Tensor]] = {}
    n_done = 0

    for sh in shards:
        with safe_open(str(sh), framework="pt", device="cpu") as f:
            for k in f.keys():
                m = pattern.match(k)
                if not m:
                    continue
                layer = int(m.group(1))
                expert = int(m.group(2))
                proj = m.group(3)
                t = f.get_tensor(k).to(torch.bfloat16)
                # Mixtral: w1=gate, w3=up, w2=down. Need [in, out] layout.
                t = t.t().contiguous() if t.dim() == 2 else t
                key = (layer, expert)
                expert_buf.setdefault(key, {})[proj] = t

                # Once an expert has all three projections, quantize and save
                if all(p in expert_buf[key] for p in triplet_keys):
                    g = expert_buf[key][triplet_keys[0]]
                    u = expert_buf[key][triplet_keys[1]]
                    d = expert_buf[key][triplet_keys[2]]
                    g_q, sg = quant_fp8_e4m3(g)
                    u_q, su = quant_fp8_e4m3(u)
                    d_q, sd = quant_fp8_e4m3(d)
                    save_expert_to_disk(dst, layer, expert,
                                          g_q, u_q, d_q, sg, su, sd)
                    del expert_buf[key]
                    n_done += 1
                    if n_done % 10 == 0:
                        print(f"  saved {n_done} experts")
                        gc.collect()
        gc.collect()

    print(f"Done: {n_done} experts written to {dst}")


if __name__ == "__main__":
    main()
