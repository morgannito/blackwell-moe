"""Custom Triton scatter-add kernel.

Replaces `out.index_add_(0, inverse_idx, weighted_y)` from the MoE forward
combine step. The native `aten::index_add_` accounts for ~8 % of CUDA time
in the v0.10 profile.

Layout:
  src      [M, D]  — already weighted expert outputs (one row per token-expert
                     pair after permutation)
  index    [M]      — destination row index in `out` (`inverse_idx`)
  out      [T, D]  — accumulator, T tokens

We launch one program per (block of M rows, block of D cols). Each program
loads a tile from `src`, atomically adds it into `out[index[m], d]`. atomic
adds on global memory are well-supported on Blackwell for bf16/fp16/fp32.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _scatter_add_kernel(
    src_ptr,         # [M, D]
    index_ptr,       # [M] int64
    out_ptr,         # [T, D]
    M, D,
    stride_sm, stride_sd,
    stride_om, stride_od,
    BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_d = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    m_mask = offs_m < M
    dst_idx = tl.load(index_ptr + offs_m, mask=m_mask, other=0).to(tl.int64)

    src = tl.load(
        src_ptr + offs_m[:, None] * stride_sm + offs_d[None, :] * stride_sd,
        mask=m_mask[:, None] & (offs_d[None, :] < D),
        other=0.0,
    )

    out_ptrs = out_ptr + dst_idx[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.atomic_add(out_ptrs, src, mask=m_mask[:, None] & (offs_d[None, :] < D))


def scatter_add(
    out: torch.Tensor,        # [T, D]  pre-zeroed accumulator
    index: torch.Tensor,      # [M]     int64
    src: torch.Tensor,        # [M, D]
) -> torch.Tensor:
    """Equivalent to `out.index_add_(0, index, src)` but as a single Triton launch."""
    M, D = src.shape
    BLOCK_M = 32
    BLOCK_D = min(128, triton.next_power_of_2(D))
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(D, BLOCK_D))
    _scatter_add_kernel[grid](
        src, index.to(torch.int64), out,
        M, D,
        src.stride(0), src.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_D=BLOCK_D,
        num_warps=4,
    )
    return out
