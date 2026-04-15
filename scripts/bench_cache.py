"""Benchmark LRU expert cache: simulate Zipf expert selection, measure hit rate & tok/s.

Scenario: E=128 experts total, gpu_slots=64 (50 % on GPU), routing follows
Zipf distribution (typical of trained MoE models — power-law usage).
"""

import time

import torch

from blackwell_moe.kernels.fp8_moe_cached import fp8_moe_forward_cached_direct
from blackwell_moe.kernels.fp8_moe_torch import _quant_fp8
from blackwell_moe.runtime.expert_cache import LRUExpertCache


def make_zipf_choices(n_items: int, n_choices: int, zipf_s: float = 1.1) -> torch.Tensor:
    weights = 1.0 / (torch.arange(1, n_items + 1, dtype=torch.float32) ** zipf_s)
    weights = weights / weights.sum()
    return torch.multinomial(weights, n_choices, replacement=True)


def main():
    torch.manual_seed(0)
    # Realistic MoE shape, moderate batch so unique experts per step stays
    # under the GPU slot budget
    T, D, E, K, H = 128, 2048, 64, 6, 1408
    N_slots = 48  # 75 % cache, enough for most batches under zipf

    # Build fake experts in bf16 CPU, quantize to FP8 pinned
    print(f"Creating {E} experts, quantizing to FP8 pinned CPU...")
    cpu_g = torch.empty((E, D, H), dtype=torch.float8_e4m3fn, pin_memory=True)
    cpu_u = torch.empty((E, D, H), dtype=torch.float8_e4m3fn, pin_memory=True)
    cpu_d = torch.empty((E, H, D), dtype=torch.float8_e4m3fn, pin_memory=True)
    cpu_sg = torch.empty(E, dtype=torch.float32, pin_memory=True)
    cpu_su = torch.empty(E, dtype=torch.float32, pin_memory=True)
    cpu_sd = torch.empty(E, dtype=torch.float32, pin_memory=True)
    for e in range(E):
        wg = torch.randn(D, H, dtype=torch.bfloat16) * 0.02
        wu = torch.randn(D, H, dtype=torch.bfloat16) * 0.02
        wd = torch.randn(H, D, dtype=torch.bfloat16) * 0.02
        qg, sg = _quant_fp8(wg)
        qu, su = _quant_fp8(wu)
        qd, sd = _quant_fp8(wd)
        cpu_g[e] = qg
        cpu_u[e] = qu
        cpu_d[e] = qd
        cpu_sg[e] = sg
        cpu_su[e] = su
        cpu_sd[e] = sd

    cache = LRUExpertCache(cpu_g, cpu_u, cpu_d, cpu_sg, cpu_su, cpu_sd, N_slots)

    # Warm with most-frequent experts under Zipf
    zipf_hot = list(range(N_slots))  # 0..N-1 are assumed hot
    cache.warmup(zipf_hot)
    print(f"Warmed {N_slots} experts into GPU slots")

    x = torch.randn(T, D, device="cuda", dtype=torch.bfloat16) * 0.1

    # Simulate 50 forwards with Zipf routing
    n_forwards = 50
    # Pre-build fake router outputs (skip softmax for simplicity)
    print(f"Running {n_forwards} forwards with Zipf expert selection...")
    t0 = time.perf_counter()
    for step in range(n_forwards):
        choices = make_zipf_choices(E, T * K, zipf_s=2.0).view(T, K)
        choices = choices.to("cuda")
        w = torch.rand(T, K, device="cuda", dtype=torch.float32)
        w = w / w.sum(dim=-1, keepdim=True)
        _ = fp8_moe_forward_cached_direct(x, w, choices, cache, top_k=K)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    stats = cache.stats()
    print(f"\nResults over {n_forwards} forwards:")
    print(f"  Total time: {dt:.2f}s ({dt / n_forwards * 1000:.1f} ms/forward)")
    print(f"  Hit rate: {stats['hit_rate']:.1%}")
    print(f"  Hits: {stats['hits']}  Misses: {stats['misses']}")
    print(f"  Effective cache size: {stats['slots_used']}/{N_slots}")
    vram = torch.cuda.max_memory_allocated() / 1e9
    print(f"  Peak VRAM: {vram:.2f} GB")
    cpu_mem_mb = (cpu_g.numel() + cpu_u.numel() + cpu_d.numel()) / 1e6
    print(f"  CPU pinned reservoir: {cpu_mem_mb:.1f} MB")

    # Baseline: all experts on GPU, same kernels, no cache overhead
    print("\n=== Baseline: all experts on GPU (no cache) ===")
    gpu_g = cpu_g.to("cuda")
    gpu_u = cpu_u.to("cuda")
    gpu_d = cpu_d.to("cuda")
    gpu_sg = cpu_sg.to("cuda")
    gpu_su = cpu_su.to("cuda")
    gpu_sd = cpu_sd.to("cuda")
    # Use the same kernels directly
    from blackwell_moe.kernels.grouped_fp8 import grouped_fp8_gemm
    from blackwell_moe.kernels.segment_ops import (
        segment_fp8_scales,
        segment_quant_fp8,
        segment_quant_fp8_fused,
    )
    import torch.nn.functional as F

    def forward_baseline(x, w, choices):
        flat = choices.reshape(-1).to(torch.int32)
        s_ids, perm = torch.sort(flat, stable=True)
        TK = flat.numel()
        src = torch.arange(TK, device="cuda", dtype=torch.long) // K
        inv = src[perm]
        xp = x[inv]
        counts = torch.bincount(s_ids, minlength=E)
        off = torch.zeros(E + 1, dtype=torch.int32, device="cuda")
        off[1:] = counts.cumsum(0).to(torch.int32)
        xp_q, sx = segment_quant_fp8_fused(xp, off)
        g = grouped_fp8_gemm(xp_q, gpu_g, off, sx, gpu_sg)
        u = grouped_fp8_gemm(xp_q, gpu_u, off, sx, gpu_su)
        h = F.silu(g) * u
        sh = segment_fp8_scales(h, off)
        hq = segment_quant_fp8(h, off, sh)
        y = grouped_fp8_gemm(hq, gpu_d, off, sh, gpu_sd)
        fw = w.reshape(-1)[perm].to(y.dtype).unsqueeze(-1)
        out = torch.zeros_like(x)
        out.index_add_(0, inv, (y * fw).to(x.dtype))
        return out

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for step in range(n_forwards):
        choices = make_zipf_choices(E, T * K, zipf_s=2.0).view(T, K).to("cuda")
        w = torch.rand(T, K, device="cuda", dtype=torch.float32)
        w = w / w.sum(dim=-1, keepdim=True)
        _ = forward_baseline(x, w, choices)
    torch.cuda.synchronize()
    dt2 = time.perf_counter() - t0
    vram2 = torch.cuda.max_memory_allocated() / 1e9
    print(f"  Total time: {dt2:.2f}s ({dt2 / n_forwards * 1000:.1f} ms/forward)")
    print(f"  Peak VRAM: {vram2:.2f} GB (all experts resident)")
    print(f"\n  Cache overhead: {(dt / dt2 - 1) * 100:+.1f}%")
    print(f"  VRAM saved: {vram2 - vram:.2f} GB ({(1 - vram/vram2) * 100:.0f}%)")


if __name__ == "__main__":
    main()
