"""Grouped INT4 GEMM for MoE — bf16 activations × int4 weights.

Each program handles one (expert, M-block, N-block). Int4 weights are loaded
as packed uint8, unpacked to bf16 using per-channel scale, then fed to tl.dot
with bf16 activations. Accumulation in fp32.

Memory footprint for weights: N/2 bytes per [K,N] matrix → 4× smaller than bf16.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


_CFG = [
    triton.Config({"BLOCK_M": 32,  "BLOCK_N": 128, "BLOCK_K": 64}, num_stages=3, num_warps=4),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 64}, num_stages=3, num_warps=4),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 128}, num_stages=4, num_warps=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_stages=3, num_warps=8),
]


@triton.autotune(configs=_CFG, key=["K", "N"])
@triton.jit
def _grouped_int4_gemm_kernel(
    x_ptr,          # [M, K] bf16
    w_ptr,          # [E, K, N/2] uint8 packed
    scales_ptr,     # [E, N] bf16
    y_ptr,          # [M, N] bf16
    offsets_ptr,
    K, N,
    stride_xm, stride_xk,
    stride_we, stride_wk, stride_wn,
    stride_se, stride_sn,
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

    # Per-channel scales for this N-block
    sc = tl.load(scales_ptr + pid_e * stride_se + offs_n * stride_sn,
                 mask=offs_n < N, other=0.0).to(tl.float32)  # [BLOCK_N]

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # Packed int4 layout: each byte holds 2 nibbles (even N idx = low, odd = high)
    # Group BLOCK_N must be even; we unpack pairs at a time.
    # For simplicity assume BLOCK_N even and handle via gather of N//2 bytes.
    N_half = N // 2

    for k in range(0, K, BLOCK_K):
        # Load X [BLOCK_M, BLOCK_K] bf16
        m_mask = offs_m[:, None] < m_end
        kx = (k + offs_k)[None, :] < K
        x = tl.load(x_ptr + offs_m[:, None] * stride_xm + (k + offs_k)[None, :] * stride_xk,
                    mask=m_mask & kx, other=0.0).to(tl.float32)

        # Load packed int4 W [BLOCK_K, BLOCK_N/2] uint8
        # We treat each N column of X separately: unpack nibble at position offs_n[j]
        # Byte index = offs_n[j] // 2, nibble index = offs_n[j] % 2
        half_idx = offs_n // 2       # [BLOCK_N]
        nibble_sel = (offs_n % 2).to(tl.int32)  # 0 or 1

        kw = (k + offs_k)[:, None] < K
        w_byte = tl.load(
            w_ptr + pid_e * stride_we
            + (k + offs_k)[:, None] * stride_wk
            + half_idx[None, :] * stride_wn,
            mask=kw, other=0,
        )  # [BLOCK_K, BLOCK_N] uint8

        # Extract nibble
        shift = nibble_sel * 4  # 0 or 4
        nibble = (w_byte >> shift[None, :]) & 0xF  # [BLOCK_K, BLOCK_N] int in [0,15]
        w_val = (nibble.to(tl.float32) - 8.0) * sc[None, :]  # dequant to fp32

        # fp32 matmul (we already dequantized). Use tl.dot with fp32 operands.
        # Blackwell tl.dot supports fp32 via TF32 or fallback.
        acc += tl.dot(x, w_val, out_dtype=tl.float32)

    # Store output
    mask = (offs_m[:, None] < m_end) & (offs_n[None, :] < N)
    tl.store(
        y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
        acc.to(y_ptr.dtype.element_ty),
        mask=mask,
    )


def grouped_int4_gemm(
    x_perm: torch.Tensor,            # [M, K] bf16
    w_packed: torch.Tensor,          # [E, K, N/2] uint8
    scales: torch.Tensor,            # [E, N] bf16
    offsets: torch.Tensor,           # [E+1] int32
    N: int,
    out_dtype: torch.dtype = torch.bfloat16,
    max_m_per_expert: int | None = None,
) -> torch.Tensor:
    M, K = x_perm.shape
    E = w_packed.shape[0]
    y = torch.empty((M, N), device=x_perm.device, dtype=out_dtype)

    if max_m_per_expert is None:
        max_m_per_expert = M

    def grid(META):
        return (E,
                triton.cdiv(max_m_per_expert, META["BLOCK_M"]),
                triton.cdiv(N, META["BLOCK_N"]))

    _grouped_int4_gemm_kernel[grid](
        x_perm, w_packed, scales, y,
        offsets,
        K, N,
        x_perm.stride(0), x_perm.stride(1),
        w_packed.stride(0), w_packed.stride(1), w_packed.stride(2),
        scales.stride(0), scales.stride(1),
        y.stride(0), y.stride(1),
    )
    return y
