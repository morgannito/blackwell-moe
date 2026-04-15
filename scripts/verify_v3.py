"""Correctness sanity check for v3 — compare against bf16 reference."""

import torch

from blackwell_moe.kernels.fp8_moe_torch import _quant_fp8 as qf8
from blackwell_moe.kernels.fp8_moe_v3 import fp8_moe_forward_v3
from blackwell_moe.kernels.reference import moe_forward_bf16

torch.manual_seed(0)
T, D, E, K, H = 64, 256, 8, 2, 128
x = torch.randn(T, D, device="cuda", dtype=torch.bfloat16) * 0.1
wg = torch.randn(D, E, device="cuda", dtype=torch.bfloat16) * 0.02
eg = torch.randn(E, D, H, device="cuda", dtype=torch.bfloat16) * 0.02
eu = torch.randn(E, D, H, device="cuda", dtype=torch.bfloat16) * 0.02
ed = torch.randn(E, H, D, device="cuda", dtype=torch.bfloat16) * 0.02

y_ref = moe_forward_bf16(x, wg, eg, eu, ed, K)

eg8 = torch.stack([qf8(eg[i])[0] for i in range(E)])
eu8 = torch.stack([qf8(eu[i])[0] for i in range(E)])
ed8 = torch.stack([qf8(ed[i])[0] for i in range(E)])
sg = torch.tensor([float(qf8(eg[i])[1]) for i in range(E)], device="cuda")
su = torch.tensor([float(qf8(eu[i])[1]) for i in range(E)], device="cuda")
sd = torch.tensor([float(qf8(ed[i])[1]) for i in range(E)], device="cuda")

y = fp8_moe_forward_v3(x, wg, eg8, eu8, ed8, sg, su, sd, K)
torch.cuda.synchronize()

print("ref mean/std:", y_ref.abs().mean().item(), y_ref.std().item())
print("v3  mean/std:", y.abs().mean().item(), y.std().item())
rel = ((y - y_ref).abs().mean() / y_ref.abs().mean()).item()
print(f"relative L1 error: {rel:.4%}")
print(f"max abs err: {(y - y_ref).abs().max().item():.6f}")
print(f"y zero fraction: {(y == 0).float().mean().item():.3%}")
