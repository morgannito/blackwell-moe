"""Profile fp8_moe_forward_v3 with torch.profiler — find next bottleneck."""

import torch
from torch.profiler import ProfilerActivity, profile

from blackwell_moe import fp8_moe_forward_v3, quant_fp8_e4m3

torch.manual_seed(0)
T, D, E, K, H = 1024, 2048, 64, 8, 1536
device = "cuda"
dt = torch.bfloat16

x = torch.randn(T, D, device=device, dtype=dt) * 0.1
wg = torch.randn(D, E, device=device, dtype=dt) * 0.02
e_g = torch.randn(E, D, H, device=device, dtype=dt) * 0.02
e_u = torch.randn(E, D, H, device=device, dtype=dt) * 0.02
e_d = torch.randn(E, H, D, device=device, dtype=dt) * 0.02

eg8 = torch.stack([quant_fp8_e4m3(e_g[i])[0] for i in range(E)])
eu8 = torch.stack([quant_fp8_e4m3(e_u[i])[0] for i in range(E)])
ed8 = torch.stack([quant_fp8_e4m3(e_d[i])[0] for i in range(E)])
sg = torch.tensor([float(quant_fp8_e4m3(e_g[i])[1]) for i in range(E)], device=device)
su = torch.tensor([float(quant_fp8_e4m3(e_u[i])[1]) for i in range(E)], device=device)
sd = torch.tensor([float(quant_fp8_e4m3(e_d[i])[1]) for i in range(E)], device=device)

# Warmup
for _ in range(5):
    fp8_moe_forward_v3(x, wg, eg8, eu8, ed8, sg, su, sd, K)
torch.cuda.synchronize()

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
              record_shapes=False) as prof:
    for _ in range(10):
        fp8_moe_forward_v3(x, wg, eg8, eu8, ed8, sg, su, sd, K)
    torch.cuda.synchronize()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
