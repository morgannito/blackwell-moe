"""Bench FastExpertCache vs old LRUExpertCache vs full-GPU baseline."""

import time

import torch

from blackwell_moe.kernels.fp8_moe_fastcache import fp8_moe_forward_fastcache
from blackwell_moe.kernels.fp8_moe_torch import _quant_fp8
from blackwell_moe.runtime.fast_expert_cache import FastExpertCache


def make_zipf_choices(n_items: int, n_choices: int, zipf_s: float = 2.0) -> torch.Tensor:
    w = 1.0 / (torch.arange(1, n_items + 1, dtype=torch.float32) ** zipf_s)
    w = w / w.sum()
    return torch.multinomial(w, n_choices, replacement=True)


def main():
    torch.manual_seed(0)
    # Tight zipf → few experts per layer. One fetch covers all layers.
    T, D, E, K, H = 128, 2048, 128, 6, 1408   # E=128 (Mixtral-/DeepSeek-scale)
    N_slots = 64                              # 50 % of pool resident
    n_layers = 26
    n_forwards = 20
    zipf_s = 4.0                              # very skewed — realistic for trained MoE
    layers_per_fetch = n_layers               # single fetch per forward

    # Build FP8 expert reservoir (pinned CPU)
    print(f"Creating {E} experts (pinned CPU, FP8)...")
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
        qg, sg = _quant_fp8(wg); cpu_g[e] = qg; cpu_sg[e] = sg
        qu, su = _quant_fp8(wu); cpu_u[e] = qu; cpu_su[e] = su
        qd, sd = _quant_fp8(wd); cpu_d[e] = qd; cpu_sd[e] = sd

    cache = FastExpertCache(cpu_g, cpu_u, cpu_d, cpu_sg, cpu_su, cpu_sd, N_slots)
    cache.warmup(list(range(N_slots)))
    x = torch.randn(T, D, device="cuda", dtype=torch.bfloat16) * 0.1

    # Simulate full forward: n_layers MoE layers per forward
    print(f"Running {n_forwards} forwards × {n_layers} layers each, Zipf routing")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for step in range(n_forwards):
        per_layer_choices = [
            make_zipf_choices(E, T * K, zipf_s=zipf_s).view(T, K).to("cuda")
            for _ in range(n_layers)
        ]
        # Chunk layers so each chunk's unique experts fit in cache
        for chunk_start in range(0, n_layers, layers_per_fetch):
            chunk = per_layer_choices[chunk_start:chunk_start + layers_per_fetch]
            all_needed = torch.cat([c.flatten() for c in chunk])
            remap = cache.fetch_batch(all_needed)
            for c in chunk:
                w = torch.rand(T, K, device="cuda", dtype=torch.float32)
                w = w / w.sum(dim=-1, keepdim=True)
                _ = fp8_moe_forward_fastcache(
                    x, w, c, remap,
                    cache.gpu_g, cache.gpu_u, cache.gpu_d,
                    cache.gpu_sg, cache.gpu_su, cache.gpu_sd,
                    cache.N, top_k=K,
                )
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    vram = torch.cuda.max_memory_allocated() / 1e9
    s = cache.stats()
    print(f"\nFastExpertCache:")
    print(f"  Time: {dt:.2f}s ({dt / n_forwards * 1000:.1f} ms/forward, "
          f"{dt / n_forwards / n_layers * 1000:.2f} ms/layer)")
    print(f"  Hit rate: {s['hit_rate']:.1%}  ({s['hits']} hits / {s['misses']} misses)")
    print(f"  Peak VRAM: {vram:.2f} GB")

    # Baseline: all experts on GPU, same kernels
    print("\n=== Baseline: all experts on GPU, no cache ===")
    gpu_g = cpu_g.to("cuda"); gpu_u = cpu_u.to("cuda"); gpu_d = cpu_d.to("cuda")
    gpu_sg = cpu_sg.to("cuda"); gpu_su = cpu_su.to("cuda"); gpu_sd = cpu_sd.to("cuda")
    identity_remap = torch.arange(E, dtype=torch.int32, device="cuda")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for step in range(n_forwards):
        for _ in range(n_layers):
            c = make_zipf_choices(E, T * K, zipf_s=zipf_s).view(T, K).to("cuda")
            w = torch.rand(T, K, device="cuda", dtype=torch.float32)
            w = w / w.sum(dim=-1, keepdim=True)
            _ = fp8_moe_forward_fastcache(
                x, w, c, identity_remap,
                gpu_g, gpu_u, gpu_d, gpu_sg, gpu_su, gpu_sd, E, top_k=K,
            )
    torch.cuda.synchronize()
    dt2 = time.perf_counter() - t0
    vram2 = torch.cuda.max_memory_allocated() / 1e9
    print(f"  Time: {dt2:.2f}s ({dt2 / n_forwards * 1000:.1f} ms/forward, "
          f"{dt2 / n_forwards / n_layers * 1000:.2f} ms/layer)")
    print(f"  Peak VRAM: {vram2:.2f} GB")

    print(f"\n=== Cache vs baseline ===")
    print(f"  Time overhead: {(dt / dt2 - 1) * 100:+.1f}%")
    print(f"  VRAM reduction: {(1 - vram / vram2) * 100:.0f}% ({vram2 - vram:.2f} GB saved)")


if __name__ == "__main__":
    main()
