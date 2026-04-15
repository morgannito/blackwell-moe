"""Benchmark harness: our FP8 kernel vs bf16 reference.

Usage:
    bwmoe-bench --tokens 1024 --dim 2048 --experts 128 --topk 8 --hidden 1536
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import torch


@dataclass
class BenchResult:
    name: str
    mean_ms: float
    tok_per_s: float
    peak_mem_mb: float


def _bench(fn, warmup: int = 5, iters: int = 20, tokens: int = 1) -> tuple[float, float]:
    torch.cuda.reset_peak_memory_stats()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    peak = torch.cuda.max_memory_allocated() / 1e6
    return dt * 1000, peak


def run(T: int, D: int, E: int, K: int, H: int, device: str = "cuda"):
    torch.manual_seed(0)
    dtype = torch.bfloat16

    x = torch.randn(T, D, device=device, dtype=dtype) * 0.1
    w_gate = torch.randn(D, E, device=device, dtype=dtype) * 0.02
    e_g = torch.randn(E, D, H, device=device, dtype=dtype) * 0.02
    e_u = torch.randn(E, D, H, device=device, dtype=dtype) * 0.02
    e_d = torch.randn(E, H, D, device=device, dtype=dtype) * 0.02

    from blackwell_moe.kernels.reference import moe_forward_bf16
    from blackwell_moe.kernels.fp8_moe_torch import (
        fp8_moe_forward_torch,
        _quant_fp8 as to_fp8_e4m3_,
    )
    from blackwell_moe.kernels.fp8_moe_v2 import fp8_moe_forward_v2
    to_fp8_e4m3 = to_fp8_e4m3_

    # Pre-quantize expert weights (done once at load time in production)
    e_g_fp8, s_g = zip(*[to_fp8_e4m3(e_g[i]) for i in range(E)])
    e_u_fp8, s_u = zip(*[to_fp8_e4m3(e_u[i]) for i in range(E)])
    e_d_fp8, s_d = zip(*[to_fp8_e4m3(e_d[i]) for i in range(E)])
    e_g_fp8 = torch.stack(list(e_g_fp8))
    e_u_fp8 = torch.stack(list(e_u_fp8))
    e_d_fp8 = torch.stack(list(e_d_fp8))
    s_g = torch.tensor([float(s) for s in s_g], device=device)
    s_u = torch.tensor([float(s) for s in s_u], device=device)
    s_d = torch.tensor([float(s) for s in s_d], device=device)

    fns = {
        "bf16_ref": lambda: moe_forward_bf16(x, w_gate, e_g, e_u, e_d, K),
        "fp8_v1_torch": lambda: fp8_moe_forward_torch(
            x, w_gate, e_g_fp8, e_u_fp8, e_d_fp8, s_g, s_u, s_d, K
        ),
        "fp8_v2_grouped": lambda: fp8_moe_forward_v2(
            x, w_gate, e_g_fp8, e_u_fp8, e_d_fp8, s_g, s_u, s_d, K
        ),
    }

    results: list[BenchResult] = []
    for name, fn in fns.items():
        ms, mem = _bench(fn, tokens=T)
        results.append(BenchResult(name, ms, T / (ms / 1000), mem))

    print(f"\n{'impl':<14} {'ms/iter':>10} {'tok/s':>12} {'peak MB':>10}")
    print("-" * 50)
    for r in results:
        print(f"{r.name:<14} {r.mean_ms:>10.3f} {r.tok_per_s:>12.1f} {r.peak_mem_mb:>10.1f}")
    return results


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, default=512)
    p.add_argument("--dim", type=int, default=2048)
    p.add_argument("--experts", type=int, default=64)
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--hidden", type=int, default=1536)
    args = p.parse_args()
    run(args.tokens, args.dim, args.experts, args.topk, args.hidden)


if __name__ == "__main__":
    main()
