"""Grouped INT4 GEMM with K-group scales (Q4_0 style).

Differences vs v0.6 `grouped_int4`:
  * Scales are [E, G, N] instead of [E, N]
  * Inner K loop reloads the correct scale at every GROUP_K boundary

GROUP_K is fixed at 32 to keep the kernel simple. BLOCK_K must be a multiple.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

GROUP_K = 32


_CFG = [
    triton.Config({"BLOCK_M": 32,  "BLOCK_N": 128, "BLOCK_K": 64}, num_stages=3, num_warps=4),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 64}, num_stages=3, num_warps=4),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 128}, num_stages=3, num_warps=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_stages=3, num_warps=8),
]


@triton.autotune(configs=_CFG, key=["K", "N"])
@triton.jit
def _grouped_int4_group_gemm_kernel(
    x_ptr,            # [M, K] bf16
    w_ptr,            # [E, K, N/2] uint8 packed
    scales_ptr,       # [E, G, N] bf16 (G = K / GROUP_K)
    y_ptr,            # [M, N] bf16
    offsets_ptr,
    K, N, G,
    stride_xm, stride_xk,
    stride_we, stride_wk, stride_wn,
    stride_sc_e, stride_sc_g, stride_sc_n,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_K_C: tl.constexpr,
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

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    half_idx = offs_n // 2
    nibble_sel = (offs_n % 2).to(tl.int32)

    for k in range(0, K, BLOCK_K):
        m_mask = offs_m[:, None] < m_end
        kx = (k + offs_k)[None, :] < K
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + (k + offs_k)[None, :] * stride_xk,
            mask=m_mask & kx, other=0.0,
        ).to(tl.float32)

        kw = (k + offs_k)[:, None] < K
        w_byte = tl.load(
            w_ptr + pid_e * stride_we
            + (k + offs_k)[:, None] * stride_wk
            + half_idx[None, :] * stride_wn,
            mask=kw, other=0,
        )
        shift = nibble_sel * 4
        nibble = (w_byte >> shift[None, :]) & 0xF
        w_int = (nibble.to(tl.float32) - 8.0)                                  # [BLOCK_K, BLOCK_N]

        # Per-K-row group index → load scale[pid_e, g, offs_n]
        # Assumes BLOCK_K is a multiple of GROUP_K_C: scale is constant per row
        # within a GROUP_K stride. Use integer division of absolute K position.
        k_idx = k + offs_k                                                     # [BLOCK_K]
        g_idx = k_idx // GROUP_K_C                                             # [BLOCK_K]
        sc = tl.load(
            scales_ptr + pid_e * stride_sc_e
            + g_idx[:, None] * stride_sc_g
            + offs_n[None, :] * stride_sc_n,
            mask=(kw) & (offs_n[None, :] < N), other=0.0,
        ).to(tl.float32)                                                       # [BLOCK_K, BLOCK_N]

        w_val = w_int * sc                                                     # dequant fp32
        acc += tl.dot(x, w_val, out_dtype=tl.float32)

    mask = (offs_m[:, None] < m_end) & (offs_n[None, :] < N)
    tl.store(
        y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
        acc.to(y_ptr.dtype.element_ty),
        mask=mask,
    )


def grouped_int4_group_gemm(
    x_perm: torch.Tensor,            # [M, K] bf16
    w_packed: torch.Tensor,          # [E, K, N/2] uint8
    scales: torch.Tensor,            # [E, K/GROUP_K, N] bf16
    offsets: torch.Tensor,           # [E+1] int32
    N: int,
    out_dtype: torch.dtype = torch.bfloat16,
    max_m_per_expert: int | None = None,
) -> torch.Tensor:
    M, K = x_perm.shape
    E = w_packed.shape[0]
    G = K // GROUP_K
    y = torch.empty((M, N), device=x_perm.device, dtype=out_dtype)

    if max_m_per_expert is None:
        max_m_per_expert = M

    def grid(META):
        return (E,
                triton.cdiv(max_m_per_expert, META["BLOCK_M"]),
                triton.cdiv(N, META["BLOCK_N"]))

    _grouped_int4_group_gemm_kernel[grid](
        x_perm, w_packed, scales, y,
        offsets,
        K, N, G,
        x_perm.stride(0), x_perm.stride(1),
        w_packed.stride(0), w_packed.stride(1), w_packed.stride(2),
        scales.stride(0), scales.stride(1), scales.stride(2),
        y.stride(0), y.stride(1),
        GROUP_K_C=GROUP_K,
    )
    return y
