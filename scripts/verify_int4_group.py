"""Correctness: group-scale INT4 vs per-channel INT4 vs bf16 reference."""

import torch

from blackwell_moe.kernels.int4_moe import int4_moe_forward
from blackwell_moe.kernels.int4_moe_group import int4_group_moe_forward
from blackwell_moe.kernels.int4_quant import quantize_int4_per_channel
from blackwell_moe.kernels.int4_group import quantize_int4_groups
from blackwell_moe.kernels.reference import moe_forward_bf16

torch.manual_seed(0)
T, D, E, K, H = 64, 256, 8, 2, 128  # D=256 divisible by 32
x = torch.randn(T, D, device="cuda", dtype=torch.bfloat16) * 0.1
wg = torch.randn(D, E, device="cuda", dtype=torch.bfloat16) * 0.02
eg = torch.randn(E, D, H, device="cuda", dtype=torch.bfloat16) * 0.02
eu = torch.randn(E, D, H, device="cuda", dtype=torch.bfloat16) * 0.02
ed = torch.randn(E, H, D, device="cuda", dtype=torch.bfloat16) * 0.02

y_ref = moe_forward_bf16(x, wg, eg, eu, ed, K)

# --- per-channel (v0.6)
eg_q = torch.empty((E, D, H // 2), device="cuda", dtype=torch.uint8)
eu_q = torch.empty((E, D, H // 2), device="cuda", dtype=torch.uint8)
ed_q = torch.empty((E, H, D // 2), device="cuda", dtype=torch.uint8)
sg_pc = torch.empty((E, H), device="cuda", dtype=torch.bfloat16)
su_pc = torch.empty((E, H), device="cuda", dtype=torch.bfloat16)
sd_pc = torch.empty((E, D), device="cuda", dtype=torch.bfloat16)
for i in range(E):
    eg_q[i], sg_pc[i] = quantize_int4_per_channel(eg[i])
    eu_q[i], su_pc[i] = quantize_int4_per_channel(eu[i])
    ed_q[i], sd_pc[i] = quantize_int4_per_channel(ed[i])

y_pc = int4_moe_forward(x, wg, eg_q, eu_q, ed_q, sg_pc, su_pc, sd_pc, H, D, K)
torch.cuda.synchronize()

# --- group-scale (v0.9)
GROUP_K = 32
eg_qg = torch.empty_like(eg_q)
eu_qg = torch.empty_like(eu_q)
ed_qg = torch.empty_like(ed_q)
sg_g = torch.empty((E, D // GROUP_K, H), device="cuda", dtype=torch.bfloat16)
su_g = torch.empty((E, D // GROUP_K, H), device="cuda", dtype=torch.bfloat16)
sd_g = torch.empty((E, H // GROUP_K, D), device="cuda", dtype=torch.bfloat16)
for i in range(E):
    eg_qg[i], sg_g[i] = quantize_int4_groups(eg[i])
    eu_qg[i], su_g[i] = quantize_int4_groups(eu[i])
    ed_qg[i], sd_g[i] = quantize_int4_groups(ed[i])

y_g = int4_group_moe_forward(x, wg, eg_qg, eu_qg, ed_qg, sg_g, su_g, sd_g, H, D, K)
torch.cuda.synchronize()

def rel(a, b):
    return ((a - b).abs().mean() / b.abs().mean()).item()

print(f"y_ref mean : {y_ref.abs().mean().item():.6f}")
print(f"y_pc  mean : {y_pc.abs().mean().item():.6f}  (per-channel)")
print(f"y_g   mean : {y_g.abs().mean().item():.6f}  (group-scale)")
print()
print(f"per-channel relative L1 err : {rel(y_pc, y_ref):.3%}")
print(f"group-scale  relative L1 err : {rel(y_g, y_ref):.3%}")
print(f"max abs err (group) : {(y_g - y_ref).abs().max().item():.6f}")
