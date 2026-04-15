"""Grouped FP8 GEMM for MoE — single Triton launch, all experts.

Layout:
  X_perm  [M_total, K]       fp8  (tokens permuted by assigned expert)
  W_all   [E, K, N]          fp8  (stacked expert weights, contiguous)
  offsets [E + 1]            int32 (prefix sum of tokens per expert)
  scales_x[E]                fp32 (per-expert activation scale, or per-token)
  scales_w[E]                fp32 (weight scale per expert)
  Y       [M_total, N]       bf16

One program = one (expert_block, N_block) pair. Each expert block handles
its own [m_start:m_end) segment. Empty experts skip via early return.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_M": 32,  "BLOCK_N": 64,  "BLOCK_K": 64},  num_stages=2, num_warps=4),
    triton.Config({"BLOCK_M": 32,  "BLOCK_N": 128, "BLOCK_K": 64},  num_stages=2, num_warps=4),
    triton.Config({"BLOCK_M": 32,  "BLOCK_N": 128, "BLOCK_K": 128}, num_stages=3, num_warps=4),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 128}, num_stages=4, num_warps=4),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 64},  num_stages=3, num_warps=4),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 128}, num_stages=3, num_warps=4),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 256, "BLOCK_K": 64},  num_stages=3, num_warps=8),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64},  num_stages=3, num_warps=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128}, num_stages=4, num_warps=8),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64},  num_stages=3, num_warps=8),
]


@triton.autotune(configs=_AUTOTUNE_CONFIGS, key=["K", "N"])
@triton.jit
def _grouped_fp8_gemm_kernel(
    x_ptr, w_ptr, y_ptr,
    scales_x_ptr, scales_w_ptr,
    offsets_ptr,
    K, N, E,
    stride_xm, stride_xk,
    stride_we, stride_wk, stride_wn,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_e = tl.program_id(0)     # expert index
    pid_mb = tl.program_id(1)    # M-block within expert
    pid_n = tl.program_id(2)     # N-block

    m_start = tl.load(offsets_ptr + pid_e)
    m_end = tl.load(offsets_ptr + pid_e + 1)
    m_count = m_end - m_start
    if m_count <= 0:
        return

    m_block_start = m_start + pid_mb * BLOCK_M
    if m_block_start >= m_end:
        return

    offs_m = m_block_start + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_block = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_block = (w_ptr + pid_e * stride_we
               + offs_k[:, None] * stride_wk
               + offs_n[None, :] * stride_wn)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        m_mask = offs_m[:, None] < m_end
        k_mask_x = (k + offs_k)[None, :] < K
        k_mask_w = (k + offs_k)[:, None] < K
        x = tl.load(x_block, mask=m_mask & k_mask_x, other=0.0)
        w = tl.load(w_block, mask=k_mask_w, other=0.0)
        acc += tl.dot(x, w, out_dtype=tl.float32)
        x_block += BLOCK_K * stride_xk
        w_block += BLOCK_K * stride_wk

    sx = tl.load(scales_x_ptr + pid_e).to(tl.float32)
    sw = tl.load(scales_w_ptr + pid_e).to(tl.float32)
    acc = acc / (sx * sw)

    y_block = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    mask = (offs_m[:, None] < m_end) & (offs_n[None, :] < N)
    tl.store(y_block, acc.to(y_ptr.dtype.element_ty), mask=mask)


def grouped_fp8_gemm(
    x_perm: torch.Tensor,          # [M_total, K] fp8_e4m3fn
    w_all: torch.Tensor,            # [E, K, N]    fp8_e4m3fn
    offsets: torch.Tensor,          # [E + 1]      int32
    scales_x: torch.Tensor,         # [E]          fp32
    scales_w: torch.Tensor,         # [E]          fp32
    out_dtype: torch.dtype = torch.bfloat16,
    max_m_per_expert: int | None = None,
) -> torch.Tensor:
    """Compute Y[m] = (X_perm[m] @ W_all[expert(m)]) / (sx * sw), bf16 out."""
    M, K = x_perm.shape
    E, K2, N = w_all.shape
    assert K == K2, f"{K=} != {K2=}"

    y = torch.empty((M, N), device=x_perm.device, dtype=out_dtype)

    if max_m_per_expert is None:
        # Upper bound = total tokens (cheap, oversubscribes for skewed routing)
        max_m_per_expert = M

    def grid(META):
        return (
            E,
            triton.cdiv(max_m_per_expert, META["BLOCK_M"]),
            triton.cdiv(N, META["BLOCK_N"]),
        )

    _grouped_fp8_gemm_kernel[grid](
        x_perm, w_all, y,
        scales_x, scales_w,
        offsets,
        K, N, E,
        x_perm.stride(0), x_perm.stride(1),
        w_all.stride(0), w_all.stride(1), w_all.stride(2),
        y.stride(0), y.stride(1),
    )
    return y
