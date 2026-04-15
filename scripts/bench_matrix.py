"""Bench matrix: all kernels × representative MoE shapes → CSV.

Output: bench_results/matrix.csv with columns
  shape_name, kernel, ms_per_iter, tokens_per_sec, peak_mb
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import torch

SHAPES = {
    "toy_E16":               dict(T=256,  D=1024, E=16,  K=4, H=512),
    "olmoe_1B_7B":           dict(T=1024, D=2048, E=64,  K=8, H=1024),
    "qwen3_30B_A3B":         dict(T=1024, D=2048, E=64,  K=8, H=1536),
    "qwen3_moe_57B_14B":     dict(T=1024, D=3584, E=64,  K=8, H=2560),
    "mixtral_8x7B":          dict(T=4096, D=4096, E=8,   K=2, H=14336),
    "mixtral_8x22B_prefill": dict(T=512,  D=6144, E=8,   K=2, H=16384),
    "mixtral_8x22B_decode":  dict(T=16,   D=6144, E=8,   K=2, H=16384),
    "deepseek_v2_lite":      dict(T=1024, D=2048, E=64,  K=6, H=1408),
    "deepseek_E128":         dict(T=2048, D=4096, E=128, K=8, H=2048),
}


def main():
    out_dir = Path("bench_results")
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "matrix.csv"

    from blackwell_moe.bench.cli import run

    rows = []
    for name, cfg in SHAPES.items():
        print(f"\n=== {name}: {cfg} ===")
        results = run(cfg["T"], cfg["D"], cfg["E"], cfg["K"], cfg["H"])
        for r in results:
            rows.append({
                "shape": name,
                "kernel": r.name,
                "ms_per_iter": round(r.mean_ms, 3),
                "tokens_per_sec": round(r.tok_per_s, 1),
                "peak_mb": round(r.peak_mem_mb, 1),
            })

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {csv_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
