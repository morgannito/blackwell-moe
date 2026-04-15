"""FP8 quant helpers — round-trip and shape tests."""

import pytest
import torch

from blackwell_moe.kernels.fp8_quant import (
    FP8_MAX_E4M3,
    dequant_fp8_e4m3,
    quant_fp8_block,
    quant_fp8_e4m3,
    quant_fp8_per_row,
)


@pytest.mark.cuda
def test_per_tensor_roundtrip():
    torch.manual_seed(0)
    x = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16) * 0.1
    q, s = quant_fp8_e4m3(x)
    back = dequant_fp8_e4m3(q, s)
    rel = ((back.float() - x.float()).abs().mean() / x.float().abs().mean()).item()
    assert rel < 0.1


@pytest.mark.cuda
def test_per_row_better_than_per_tensor_with_drift():
    torch.manual_seed(0)
    M, D = 64, 128
    # Simulate per-row magnitude drift
    x = torch.randn(M, D, device="cuda", dtype=torch.bfloat16) * 0.05
    x[:M // 2] *= 10.0
    x[M // 2:] *= 0.01

    q_pt, s_pt = quant_fp8_e4m3(x)
    back_pt = dequant_fp8_e4m3(q_pt, s_pt)
    rel_pt = ((back_pt.float() - x.float()).abs().mean() / x.float().abs().mean()).item()

    q_pr, s_pr = quant_fp8_per_row(x)
    # Manual dequant with per-row scale
    back_pr = (q_pr.to(torch.float32) / s_pr.unsqueeze(-1)).to(torch.bfloat16)
    rel_pr = ((back_pr.float() - x.float()).abs().mean() / x.float().abs().mean()).item()

    assert rel_pr < rel_pt, f"per-row {rel_pr:.3f} should beat per-tensor {rel_pt:.3f}"


@pytest.mark.cuda
def test_block_quant_shapes():
    x = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16)
    q, s = quant_fp8_block(x, block_k=32)
    assert q.shape == (256, 128)
    assert s.shape == (8, 128)
    assert q.dtype == torch.float8_e4m3fn


@pytest.mark.cuda
def test_clamp_to_fp8_max():
    """Verify clamping never exceeds the FP8 E4M3 representable range."""
    x = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16) * 100.0
    q, s = quant_fp8_e4m3(x)
    assert q.float().abs().max().item() <= FP8_MAX_E4M3
