"""Segment-reduce operations in Triton.

Replaces Python for-loops over expert segments with a single Triton launch.
Each program handles one expert segment, computes amax → scale via block-level
reductions, stores scale[e] = 448 / amax.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _segment_amax_scale_kernel(
    x_ptr, scales_ptr, offsets_ptr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    pid_e = tl.program_id(0)
    m_start = tl.load(offsets_ptr + pid_e)
    m_end = tl.load(offsets_ptr + pid_e + 1)
    count = m_end - m_start
    if count <= 0:
        tl.store(scales_ptr + pid_e, 1.0)
        return

    acc_amax: tl.float32 = 0.0
    for m_off in range(0, count, BLOCK_M):
        offs_m = m_start + m_off + tl.arange(0, BLOCK_M)
        for d_off in range(0, D, BLOCK_D):
            offs_d = d_off + tl.arange(0, BLOCK_D)
            mask = (offs_m[:, None] < m_end) & (offs_d[None, :] < D)
            x = tl.load(x_ptr + offs_m[:, None] * D + offs_d[None, :],
                        mask=mask, other=0.0).to(tl.float32)
            blk_amax = tl.max(tl.abs(x).reshape(BLOCK_M * BLOCK_D), axis=0)
            acc_amax = tl.maximum(acc_amax, blk_amax)

    amax = tl.maximum(acc_amax, 1e-4)
    scale: tl.float32 = FP8_MAX / amax
    tl.store(scales_ptr + pid_e, scale)


def segment_fp8_scales(
    x: torch.Tensor,             # [M, D]
    offsets: torch.Tensor,       # [E+1] int32
    fp8_max: float = 448.0,
) -> torch.Tensor:
    """Per-segment FP8 scale (448/amax) in one Triton launch."""
    M, D = x.shape
    E = offsets.numel() - 1
    scales = torch.empty(E, device=x.device, dtype=torch.float32)
    grid = (E,)
    _segment_amax_scale_kernel[grid](
        x, scales, offsets,
        D=D,
        BLOCK_D=min(128, triton.next_power_of_2(D)),
        BLOCK_M=64,
        FP8_MAX=fp8_max,
        num_warps=4,
    )
    return scales


@triton.jit
def _segment_quant_fp8_kernel(
    x_ptr, out_ptr, scales_ptr, offsets_ptr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    pid_e = tl.program_id(0)
    pid_m = tl.program_id(1)

    m_start = tl.load(offsets_ptr + pid_e)
    m_end = tl.load(offsets_ptr + pid_e + 1)
    m_block_start = m_start + pid_m * BLOCK_M
    if m_block_start >= m_end:
        return

    scale = tl.load(scales_ptr + pid_e).to(tl.float32)
    offs_m = m_block_start + tl.arange(0, BLOCK_M)
    for d_off in range(0, D, BLOCK_D):
        offs_d = d_off + tl.arange(0, BLOCK_D)
        mask = (offs_m[:, None] < m_end) & (offs_d[None, :] < D)
        x = tl.load(x_ptr + offs_m[:, None] * D + offs_d[None, :],
                    mask=mask, other=0.0).to(tl.float32)
        q = tl.minimum(tl.maximum(x * scale, -FP8_MAX), FP8_MAX)
        tl.store(out_ptr + offs_m[:, None] * D + offs_d[None, :],
                 q.to(out_ptr.dtype.element_ty), mask=mask)


def segment_quant_fp8(
    x: torch.Tensor,             # [M, D] bf16/fp32
    offsets: torch.Tensor,
    scales: torch.Tensor,        # [E] from segment_fp8_scales
    fp8_max: float = 448.0,
) -> torch.Tensor:
    M, D = x.shape
    E = offsets.numel() - 1
    out = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    if M == 0:
        return out
    max_m = int((offsets[1:] - offsets[:-1]).max().item())
    if max_m == 0:
        return out
    BLOCK_M = 32
    grid = (E, triton.cdiv(max_m, BLOCK_M))
    _segment_quant_fp8_kernel[grid](
        x, out, scales, offsets,
        D=D,
        BLOCK_M=BLOCK_M,
        BLOCK_D=min(128, triton.next_power_of_2(D)),
        FP8_MAX=fp8_max,
        num_warps=4,
    )
    return out


def segment_quant_fp8_fused(
    x: torch.Tensor, offsets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One-shot: compute scales + quantize, no Python loop."""
    scales = segment_fp8_scales(x, offsets)
    x_fp8 = segment_quant_fp8(x, offsets, scales)
    return x_fp8, scales
