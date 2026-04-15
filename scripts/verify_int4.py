"""INT4 kernel correctness check vs bf16 reference."""

import torch

from blackwell_moe.kernels.int4_moe import int4_moe_forward
from blackwell_moe.kernels.int4_quant import quantize_int4_per_channel
from blackwell_moe.kernels.reference import moe_forward_bf16

torch.manual_seed(0)
T, D, E, K, H = 64, 256, 8, 2, 128
x = torch.randn(T, D, device="cuda", dtype=torch.bfloat16) * 0.1
wg = torch.randn(D, E, device="cuda", dtype=torch.bfloat16) * 0.02
eg = torch.randn(E, D, H, device="cuda", dtype=torch.bfloat16) * 0.02
eu = torch.randn(E, D, H, device="cuda", dtype=torch.bfloat16) * 0.02
ed = torch.randn(E, H, D, device="cuda", dtype=torch.bfloat16) * 0.02

y_ref = moe_forward_bf16(x, wg, eg, eu, ed, K)

eg_q = torch.empty((E, D, H // 2), device="cuda", dtype=torch.uint8)
eu_q = torch.empty((E, D, H // 2), device="cuda", dtype=torch.uint8)
ed_q = torch.empty((E, H, D // 2), device="cuda", dtype=torch.uint8)
sg = torch.empty((E, H), device="cuda", dtype=torch.bfloat16)
su = torch.empty((E, H), device="cuda", dtype=torch.bfloat16)
sd = torch.empty((E, D), device="cuda", dtype=torch.bfloat16)
for i in range(E):
    eg_q[i], sg[i] = quantize_int4_per_channel(eg[i])
    eu_q[i], su[i] = quantize_int4_per_channel(eu[i])
    ed_q[i], sd[i] = quantize_int4_per_channel(ed[i])

y = int4_moe_forward(x, wg, eg_q, eu_q, ed_q, sg, su, sd, H, D, K)
torch.cuda.synchronize()

print("ref mean:", y_ref.abs().mean().item())
print("int4 mean:", y.abs().mean().item())
rel = ((y - y_ref).abs().mean() / y_ref.abs().mean()).item()
print(f"relative L1 error: {rel:.4%}")
print(f"max abs err: {(y - y_ref).abs().max().item():.6f}")
print(f"y zero fraction: {(y == 0).float().mean().item():.3%}")
