"""Fused grouped FP8 GEMM for gate + up projections.

Single kernel launch computes BOTH gate = X @ W_gate and up = X @ W_up,
sharing the X load. Halves HBM traffic on activations.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


_CFG = [
    triton.Config({"BLOCK_M": 32,  "BLOCK_N": 64,  "BLOCK_K": 64},  num_stages=2, num_warps=4),
    triton.Config({"BLOCK_M": 32,  "BLOCK_N": 128, "BLOCK_K": 64},  num_stages=2, num_warps=4),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 128}, num_stages=4, num_warps=4),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 64},  num_stages=3, num_warps=4),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 128}, num_stages=3, num_warps=4),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 256, "BLOCK_K": 64},  num_stages=3, num_warps=8),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64},  num_stages=3, num_warps=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64},  num_stages=3, num_warps=8),
]


@triton.autotune(configs=_CFG, key=["K", "N"])
@triton.jit
def _fused_gate_up_kernel(
    x_ptr, wg_ptr, wu_ptr,
    gate_ptr, up_ptr,
    scales_x, scales_wg, scales_wu,
    offsets_ptr,
    K, N,
    stride_xm, stride_xk,
    stride_we, stride_wk, stride_wn,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_e = tl.program_id(0)
    pid_mb = tl.program_id(1)
    pid_n = tl.program_id(2)

    m_start = tl.load(offsets_ptr + pid_e)
    m_end = tl.load(offsets_ptr + pid_e + 1)
    if m_end - m_start <= 0:
        return
    m_block_start = m_start + pid_mb * BLOCK_M
    if m_block_start >= m_end:
        return

    offs_m = m_block_start + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_block = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    wg_block = wg_ptr + pid_e * stride_we + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn
    wu_block = wu_ptr + pid_e * stride_we + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    acc_g = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    acc_u = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        m_mask = offs_m[:, None] < m_end
        kx = (k + offs_k)[None, :] < K
        kw = (k + offs_k)[:, None] < K
        x = tl.load(x_block, mask=m_mask & kx, other=0.0)
        wg = tl.load(wg_block, mask=kw, other=0.0)
        wu = tl.load(wu_block, mask=kw, other=0.0)
        acc_g += tl.dot(x, wg, out_dtype=tl.float32)
        acc_u += tl.dot(x, wu, out_dtype=tl.float32)
        x_block += BLOCK_K * stride_xk
        wg_block += BLOCK_K * stride_wk
        wu_block += BLOCK_K * stride_wk

    sx = tl.load(scales_x + pid_e).to(tl.float32)
    sg = tl.load(scales_wg + pid_e).to(tl.float32)
    su = tl.load(scales_wu + pid_e).to(tl.float32)
    acc_g = acc_g / (sx * sg)
    acc_u = acc_u / (sx * su)

    mask = (offs_m[:, None] < m_end) & (offs_n[None, :] < N)
    tl.store(gate_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
             acc_g.to(gate_ptr.dtype.element_ty), mask=mask)
    tl.store(up_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
             acc_u.to(up_ptr.dtype.element_ty), mask=mask)


def fused_gate_up_gemm(
    x_perm: torch.Tensor,        # [M, K] fp8
    w_gate: torch.Tensor,        # [E, K, N] fp8
    w_up: torch.Tensor,          # [E, K, N] fp8
    offsets: torch.Tensor,       # [E+1]
    scales_x: torch.Tensor,      # [E]
    scales_wg: torch.Tensor,     # [E]
    scales_wu: torch.Tensor,     # [E]
    out_dtype: torch.dtype = torch.bfloat16,
    max_m_per_expert: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    M, K = x_perm.shape
    E, K2, N = w_gate.shape
    assert K == K2
    gate = torch.empty((M, N), device=x_perm.device, dtype=out_dtype)
    up = torch.empty((M, N), device=x_perm.device, dtype=out_dtype)

    if max_m_per_expert is None:
        max_m_per_expert = M

    def grid(META):
        return (E,
                triton.cdiv(max_m_per_expert, META["BLOCK_M"]),
                triton.cdiv(N, META["BLOCK_N"]))

    _fused_gate_up_kernel[grid](
        x_perm, w_gate, w_up, gate, up,
        scales_x, scales_wg, scales_wu,
        offsets,
        K, N,
        x_perm.stride(0), x_perm.stride(1),
        w_gate.stride(0), w_gate.stride(1), w_gate.stride(2),
        gate.stride(0), gate.stride(1),
    )
    return gate, up
