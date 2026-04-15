"""Online SwiGLU + FP8 quantization — two-pass, no bf16 intermediate.

Pass 1 (amax):  scan gate & up, compute silu(g)*u element-wise, track amax per expert
Pass 2 (quant): re-scan, emit fp8 directly using the computed scale

Total HBM: gate[M,H] bf16 read ×2, up[M,H] bf16 read ×2, out h_fp8[M,H] write.
vs previous v0.2: gate+up read once, silu*up write bf16, read back, quantize, write fp8.
Savings: kills the bf16 h_tmp materialization (M*H bytes × 2 × sizeof(bf16)).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _swiglu_amax_kernel(
    gate_ptr, up_ptr, amax_ptr, offsets_ptr,
    H: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_H: tl.constexpr,
):
    pid_e = tl.program_id(0)
    m_start = tl.load(offsets_ptr + pid_e)
    m_end = tl.load(offsets_ptr + pid_e + 1)
    if m_end - m_start <= 0:
        tl.store(amax_ptr + pid_e, 1e-4)
        return

    local_amax: tl.float32 = 0.0
    for m_off in range(0, m_end - m_start, BLOCK_M):
        offs_m = m_start + m_off + tl.arange(0, BLOCK_M)
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
    out_val: tl.float32 = tl.maximum(local_amax, 1e-4)
    tl.store(amax_ptr + pid_e, out_val)


@triton.jit
def _swiglu_quant_kernel(
    gate_ptr, up_ptr, out_ptr,
    scales_ptr, offsets_ptr,
    H: tl.constexpr,
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
        tl.store(out_ptr + offs_m[:, None] * H + offs_h[None, :],
                 q.to(out_ptr.dtype.element_ty), mask=mask)


def swiglu_fp8(
    gate: torch.Tensor,          # [M, H] bf16
    up: torch.Tensor,            # [M, H] bf16
    offsets: torch.Tensor,
    fp8_max: float = 448.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    M, H = gate.shape
    E = offsets.numel() - 1
    amax = torch.empty(E, device=gate.device, dtype=torch.float32)

    BLOCK_M = 32
    BLOCK_H = min(128, triton.next_power_of_2(H))

    _swiglu_amax_kernel[(E,)](
        gate, up, amax, offsets,
        H=H, BLOCK_M=BLOCK_M, BLOCK_H=BLOCK_H,
        num_warps=4,
    )
    scales = (fp8_max / amax).to(torch.float32)

    h_fp8 = torch.empty_like(gate, dtype=torch.float8_e4m3fn)
    if M == 0:
        return h_fp8, scales
    max_m = int((offsets[1:] - offsets[:-1]).max().item())
    if max_m == 0:
        return h_fp8, scales
    grid = (E, triton.cdiv(max_m, BLOCK_M))
    _swiglu_quant_kernel[grid](
        gate, up, h_fp8,
        scales, offsets,
        H=H, BLOCK_M=BLOCK_M, BLOCK_H=BLOCK_H,
        FP8_MAX=fp8_max,
        num_warps=4,
    )
    return h_fp8, scales
