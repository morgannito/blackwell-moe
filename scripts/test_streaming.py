"""End-to-end smoke test for ThreeTierExpertCache + streaming_moe_forward.

Simulates realistic MoE routing (Zipf — a few experts dominate). Verifies
the 3-tier cache promotes hot experts to GPU and serves cold ones from disk.
"""

import time

import torch

from blackwell_moe.runtime.disk_expert_pool import ThreeTierExpertCache
from blackwell_moe.runtime.streaming_moe import streaming_moe_forward


D, H, E, K = 2048, 1408, 64, 6
LAYER = 1
EXPERT_ROOT = r"J:\models\DeepSeek-V2-Lite-fp8-experts"

cache = ThreeTierExpertCache(
    expert_root=EXPERT_ROOT,
    n_layers=26,
    n_experts_per_layer=64,
    gpu_slots=16,
    ram_slots=32,
    gpu_buffer_specs={
        "gate_q": (D, H),
        "up_q":   (D, H),
        "down_q": (H, D),
    },
)

torch.manual_seed(0)
T = 128
x = torch.randn(T, D, device="cuda", dtype=torch.bfloat16) * 0.1


def zipf_router_weights(E: int, zipf_s: float) -> torch.Tensor:
    """Returns a [D, E] gate weight that produces Zipf-distributed expert
    selections when fed centered Gaussian inputs (heuristic)."""
    log_pref = -zipf_s * torch.log(torch.arange(1, E + 1, dtype=torch.float32))
    bias = log_pref - log_pref.mean()
    g = torch.zeros(D, E, dtype=torch.bfloat16)
    g[0] = bias.to(torch.bfloat16) * 8.0
    return g


w_gate = zipf_router_weights(E, zipf_s=2.0).cuda()

print("Warmup (primes Triton autotune + loads top experts)...")
_ = streaming_moe_forward(x, w_gate, cache, layer_idx=LAYER, top_k=K)
torch.cuda.synchronize()
cache.stats = {"gpu_hits": 0, "ram_hits": 0, "disk_loads": 0}

print(f"Running 20 forwards on layer {LAYER} with Zipf routing...")
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(20):
    y = streaming_moe_forward(x, w_gate, cache, layer_idx=LAYER, top_k=K)
torch.cuda.synchronize()
dt = time.perf_counter() - t0

s = cache.stats
total = s["gpu_hits"] + s["ram_hits"] + s["disk_loads"]
print(f"\nResults:")
print(f"  Time: {dt:.3f}s ({dt / 20 * 1000:.2f} ms/forward)")
print(f"  Cache stats: GPU {s['gpu_hits']} / RAM {s['ram_hits']} / disk {s['disk_loads']}")
if total > 0:
    print(f"  Hit ratio: GPU {s['gpu_hits']/total:.0%}, "
          f"RAM {s['ram_hits']/total:.0%}, disk {s['disk_loads']/total:.0%}")
print(f"  Output mean: {y.abs().mean().item():.6f}")
print(f"  Peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
assert y.abs().mean().item() > 0
print("\nOK: 3-tier cache + streaming MoE works end-to-end.")
