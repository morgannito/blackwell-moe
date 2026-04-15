"""FP8 grouped GEMM for MoE experts on NVIDIA Blackwell (sm_120).

Core idea: permute tokens by expert assignment, launch one grouped GEMM
per expert using FP8 E4M3 inputs with per-tensor scaling, then scatter
results back weighted by router scores.

Targets SwiGLU experts (Qwen3, Mixtral, DeepSeek family):
    gate   = x @ W_gate   (fp8)
    up     = x @ W_up     (fp8)
    hidden = silu(gate) * up
    out    = hidden @ W_down (fp8)

References:
- NVIDIA Blackwell FP8 tensor cores (sm_120)
- Triton 3.2+ tl.dot with fp8 operands
- DeepSeek-V3 FP8 training recipe
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# ----- FP8 casting helpers -----------------------------------------------

def to_fp8_e4m3(x: torch.Tensor, scale: torch.Tensor | None = None):
    """Cast bf16/fp16 tensor to FP8 E4M3 with per-tensor scaling.

    Returns (x_fp8, scale) where x_fp8 = (x * scale).clamp(fp8_max) as e4m3fn.
    """
    if scale is None:
        amax = x.abs().max().to(torch.float32)
        scale = (448.0 / amax.clamp(min=1e-4)).to(torch.float32)
    x_scaled = (x.to(torch.float32) * scale).clamp(-448.0, 448.0)
    return x_scaled.to(torch.float8_e4m3fn), scale


# ----- Triton kernel: grouped FP8 GEMM for one expert --------------------

@triton.jit
def _expert_gemm_fp8(
    x_ptr, w_ptr, out_ptr,
    x_scale, w_scale,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Compute OUT = X @ W where X is [M,K] fp8, W is [K,N] fp8.

    Accumulation is fp32 on Blackwell tensor cores. Final dequant:
        out_bf16 = acc / (x_scale * w_scale)
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_block = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_block = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        m_mask = offs_m[:, None] < M
        k_mask = (k + offs_k)[None, :] < K
        x = tl.load(x_block, mask=m_mask & k_mask, other=0.0)
        w = tl.load(w_block, mask=(k + offs_k)[:, None] < K, other=0.0)
        # Blackwell: tl.dot with fp8 operands, fp32 accum
        acc += tl.dot(x, w, out_dtype=tl.float32)
        x_block += BLOCK_K * stride_xk
        w_block += BLOCK_K * stride_wk

    # Dequantize
    inv_s = 1.0 / (x_scale * w_scale)
    acc = acc * inv_s

    out_block = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_block, acc.to(out_ptr.dtype.element_ty), mask=mask)


def _gemm_fp8(x_fp8, w_fp8, x_scale, w_scale, out_dtype=torch.bfloat16):
    M, K = x_fp8.shape
    K2, N = w_fp8.shape
    assert K == K2
    out = torch.empty((M, N), device=x_fp8.device, dtype=out_dtype)

    BLOCK_M, BLOCK_N, BLOCK_K = 64, 128, 64
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _expert_gemm_fp8[grid](
        x_fp8, w_fp8, out,
        float(x_scale.item()), float(w_scale.item()),
        M, N, K,
        x_fp8.stride(0), x_fp8.stride(1),
        w_fp8.stride(0), w_fp8.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=3,
    )
    return out


# ----- Top-level MoE forward ---------------------------------------------

def fp8_moe_forward(
    x: torch.Tensor,                    # [T, D] bf16
    w_gate: torch.Tensor,               # [D, E] bf16
    experts_w_gate: torch.Tensor,       # [E, D, H] fp8_e4m3
    experts_w_up: torch.Tensor,         # [E, D, H] fp8_e4m3
    experts_w_down: torch.Tensor,       # [E, H, D] fp8_e4m3
    scales_gate: torch.Tensor,          # [E] fp32
    scales_up: torch.Tensor,            # [E] fp32
    scales_down: torch.Tensor,          # [E] fp32
    top_k: int = 8,
) -> torch.Tensor:
    """Forward pass of an FP8 SwiGLU MoE block.

    Strategy:
      1. Route tokens -> weights[T,K], indices[T,K]
      2. For each expert, gather its tokens, run SwiGLU MLP in FP8
      3. Scatter-add weighted outputs back to token positions
    """
    from blackwell_moe.kernels.routing import top_k_router

    T, D = x.shape
    E = experts_w_gate.shape[0]
    H = experts_w_gate.shape[2]

    weights, indices = top_k_router(x, w_gate, top_k)  # [T,K] fp32, int32
    out = torch.zeros_like(x)

    # Dynamic FP8 quantization of activations per expert batch
    # (per-tensor scale, recomputed each forward for best accuracy)
    for e in range(E):
        # Find tokens routed to expert e
        mask = (indices == e).any(dim=-1)
        if not mask.any():
            continue
        tok_idx = mask.nonzero(as_tuple=True)[0]
        x_e = x[tok_idx]  # [Me, D]

        x_e_fp8, sx = to_fp8_e4m3(x_e)

        gate = _gemm_fp8(x_e_fp8, experts_w_gate[e], sx, scales_gate[e])
        up = _gemm_fp8(x_e_fp8, experts_w_up[e], sx, scales_up[e])
        hidden = torch.nn.functional.silu(gate) * up

        hidden_fp8, sh = to_fp8_e4m3(hidden)
        y_e = _gemm_fp8(hidden_fp8, experts_w_down[e], sh, scales_down[e])

        # Gather routing weight for expert e per token
        # indices[tok_idx] : [Me, K]   weights[tok_idx] : [Me, K]
        w_slot = (indices[tok_idx] == e).float()
        w_e = (weights[tok_idx] * w_slot).sum(dim=-1, keepdim=True).to(y_e.dtype)

        out.index_add_(0, tok_idx, y_e * w_e)

    return out
