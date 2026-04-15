"""Specialized grouped FP8 GEMM for small-E MoE (Mixtral-style).

When E ≤ 8 and top_k ≤ 2 (Mixtral-8x7B / 8x22B), the per-expert dispatch
inside the generic `grouped_fp8_gemm` becomes wasteful — we launch 8 program
groups but 6 of them do trivial work during decode (T=1, only 2 active).

This variant:
  * uses larger BLOCK_M (128, 256) because each active expert sees many tokens
  * fewer BLOCK_N/K shapes in the autotune space (fast search)
  * same API as `grouped_fp8_gemm`, drop-in replacement when `E <= 8`
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


_CFG_SMALL_E = [
    # Mixtral-8x22B has H=16384 → big N, prefer wide BLOCK_N
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 256, "BLOCK_K": 64}, num_stages=3, num_warps=8),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 256, "BLOCK_K": 128}, num_stages=3, num_warps=8),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64}, num_stages=3, num_warps=8),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128}, num_stages=3, num_warps=8),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64}, num_stages=3, num_warps=8),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 128}, num_stages=4, num_warps=8),
    # Smaller when M per expert is small (decode case)
    triton.Config({"BLOCK_M": 16,  "BLOCK_N": 128, "BLOCK_K": 64}, num_stages=2, num_warps=4),
    triton.Config({"BLOCK_M": 16,  "BLOCK_N": 256, "BLOCK_K": 64}, num_stages=2, num_warps=8),
    triton.Config({"BLOCK_M": 32,  "BLOCK_N": 256, "BLOCK_K": 64}, num_stages=3, num_warps=8),
]


@triton.autotune(configs=_CFG_SMALL_E, key=["K", "N"])
@triton.jit
def _grouped_fp8_gemm_small_e_kernel(
    x_ptr, w_ptr, y_ptr,
    scales_x, scales_w,
    offsets_ptr,
    K, N, E,
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
    w_block = w_ptr + pid_e * stride_we + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        m_mask = offs_m[:, None] < m_end
        kx = (k + offs_k)[None, :] < K
        kw = (k + offs_k)[:, None] < K
        x = tl.load(x_block, mask=m_mask & kx, other=0.0)
        w = tl.load(w_block, mask=kw, other=0.0)
        acc += tl.dot(x, w, out_dtype=tl.float32)
        x_block += BLOCK_K * stride_xk
        w_block += BLOCK_K * stride_wk

    sx = tl.load(scales_x + pid_e).to(tl.float32)
    sw = tl.load(scales_w + pid_e).to(tl.float32)
    acc = acc / (sx * sw)

    mask = (offs_m[:, None] < m_end) & (offs_n[None, :] < N)
    tl.store(y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
             acc.to(y_ptr.dtype.element_ty), mask=mask)


def grouped_fp8_gemm_small_e(
    x_perm: torch.Tensor, w_all: torch.Tensor, offsets: torch.Tensor,
    scales_x: torch.Tensor, scales_w: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
    max_m_per_expert: int | None = None,
) -> torch.Tensor:
    M, K = x_perm.shape
    E, K2, N = w_all.shape
    y = torch.empty((M, N), device=x_perm.device, dtype=out_dtype)
    if max_m_per_expert is None:
        max_m_per_expert = M

    def grid(META):
        return (E,
                triton.cdiv(max_m_per_expert, META["BLOCK_M"]),
                triton.cdiv(N, META["BLOCK_N"]))

    _grouped_fp8_gemm_small_e_kernel[grid](
        x_perm, w_all, y,
        scales_x, scales_w,
        offsets,
        K, N, E,
        x_perm.stride(0), x_perm.stride(1),
        w_all.stride(0), w_all.stride(1), w_all.stride(2),
        y.stride(0), y.stride(1),
    )
    return y
