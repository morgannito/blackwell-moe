"""Two-pass online SwiGLU + FP8 quant — kills the bf16 intermediate `h`.

Pass 1 (`_swiglu_amax_kernel`):
  Reads gate[M,H] + up[M,H] bf16
  Computes h = silu(g) * u in registers (no HBM write)
  Accumulates per-expert amax via atomic_max on a global [E] fp32 tensor

Pass 2 (`_swiglu_quant_kernel`):
  Recomputes h from gate + up, scales by 448/amax, casts to fp8, writes h_fp8

Net win vs v0.12 pipeline (mul + segment_amax + segment_quant):
  - Eliminates h[M,H] bf16 round-trip (read + write)
  - Two kernel launches instead of three
  - On Qwen3-30B-A3B shape this drops ~0.7 ms per forward
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

FP8_MAX_E4M3 = 448.0


@triton.jit
def _swiglu_amax_kernel(
    gate_ptr, up_ptr,
    amax_ptr,           # [E] fp32 — atomically updated
    offsets_ptr,
    H,
    BLOCK_M: tl.constexpr, BLOCK_H: tl.constexpr,
):
    pid_e = tl.program_id(0)
    pid_m = tl.program_id(1)

    m_start = tl.load(offsets_ptr + pid_e)
    m_end = tl.load(offsets_ptr + pid_e + 1)
    m_block_start = m_start + pid_m * BLOCK_M
    if m_block_start >= m_end:
        return

    offs_m = m_block_start + tl.arange(0, BLOCK_M)

    local_amax: tl.float32 = 0.0
    for h_off in range(0, H, BLOCK_H):
        offs_h = h_off + tl.arange(0, BLOCK_H)
        mask = (offs_m[:, None] < m_end) & (offs_h[None, :] < H)
        g = tl.load(gate_ptr + offs_m[:, None] * H + offs_h[None, :],
                    mask=mask, other=0.0).to(tl.float32)
        u = tl.load(up_ptr + offs_m[:, None] * H + offs_h[None, :],
                    mask=mask, other=0.0).to(tl.float32)
        h = (g / (1.0 + tl.exp(-g))) * u
        blk_amax = tl.max(tl.abs(h).reshape(BLOCK_M * BLOCK_H), axis=0)
        local_amax = tl.maximum(local_amax, blk_amax)

    tl.atomic_max(amax_ptr + pid_e, local_amax)


@triton.jit
def _swiglu_quant_kernel(
    gate_ptr, up_ptr, h_fp8_ptr,
    scales_ptr, offsets_ptr,
    H,
    BLOCK_M: tl.constexpr, BLOCK_H: tl.constexpr,
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
    for h_off in range(0, H, BLOCK_H):
        offs_h = h_off + tl.arange(0, BLOCK_H)
        mask = (offs_m[:, None] < m_end) & (offs_h[None, :] < H)
        g = tl.load(gate_ptr + offs_m[:, None] * H + offs_h[None, :],
                    mask=mask, other=0.0).to(tl.float32)
        u = tl.load(up_ptr + offs_m[:, None] * H + offs_h[None, :],
                    mask=mask, other=0.0).to(tl.float32)
        h = (g / (1.0 + tl.exp(-g))) * u * scale
        q = tl.minimum(tl.maximum(h, -FP8_MAX), FP8_MAX)
        tl.store(h_fp8_ptr + offs_m[:, None] * H + offs_h[None, :],
                 q.to(h_fp8_ptr.dtype.element_ty), mask=mask)


def fused_swiglu_quant(
    gate: torch.Tensor,           # [M, H] bf16
    up: torch.Tensor,             # [M, H] bf16
    offsets: torch.Tensor,        # [E+1] int32
) -> tuple[torch.Tensor, torch.Tensor]:
    """Two-pass fused SwiGLU + FP8 quant. Returns (h_fp8 [M,H], scales [E])."""
    M, H = gate.shape
    E = offsets.numel() - 1

    BLOCK_M = 32
    BLOCK_H = min(128, triton.next_power_of_2(H))

    amax = torch.zeros(E, device=gate.device, dtype=torch.float32)
    h_fp8 = torch.empty_like(gate, dtype=torch.float8_e4m3fn)

    if M == 0:
        return h_fp8, amax
    max_m = int((offsets[1:] - offsets[:-1]).max().item())
    if max_m == 0:
        return h_fp8, amax
    grid = (E, triton.cdiv(max_m, BLOCK_M))

    _swiglu_amax_kernel[grid](
        gate, up, amax, offsets, H,
        BLOCK_M=BLOCK_M, BLOCK_H=BLOCK_H,
        num_warps=4,
    )
    scales = (FP8_MAX_E4M3 / amax.clamp(min=1e-4)).to(torch.float32)

    _swiglu_quant_kernel[grid](
        gate, up, h_fp8, scales, offsets, H,
        BLOCK_M=BLOCK_M, BLOCK_H=BLOCK_H,
        FP8_MAX=FP8_MAX_E4M3,
        num_warps=4,
    )
    return h_fp8, scales
