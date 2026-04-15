"""Segment amax / quantize correctness."""

import pytest
import torch


@pytest.mark.cuda
def test_segment_fp8_scales_matches_python_loop():
    from blackwell_moe.kernels.segment_ops import segment_fp8_scales

    torch.manual_seed(0)
    M, D = 128, 256
    x = torch.randn(M, D, device="cuda", dtype=torch.bfloat16) * 0.2

    # Build offsets for 4 equal segments
    offsets = torch.tensor([0, 32, 64, 96, 128], dtype=torch.int32, device="cuda")

    s_gpu = segment_fp8_scales(x, offsets)

    # Reference: Python per-segment amax
    s_ref = torch.zeros_like(s_gpu)
    for i in range(4):
        seg = x[offsets[i]:offsets[i + 1]]
        amax = seg.abs().amax().clamp(min=1e-4).to(torch.float32)
        s_ref[i] = 448.0 / amax
    torch.testing.assert_close(s_gpu, s_ref, rtol=1e-3, atol=1e-4)


@pytest.mark.cuda
def test_segment_quant_fused_dequant_close():
    from blackwell_moe.kernels.segment_ops import segment_quant_fp8_fused

    torch.manual_seed(0)
    M, D = 64, 512
    x = torch.randn(M, D, device="cuda", dtype=torch.bfloat16) * 0.2
    offsets = torch.tensor([0, 32, 64], dtype=torch.int32, device="cuda")

    x_fp8, scales = segment_quant_fp8_fused(x, offsets)
    assert x_fp8.dtype == torch.float8_e4m3fn
    assert scales.shape == (2,)

    # Dequant and compare per segment
    for i in range(2):
        s, e = offsets[i].item(), offsets[i + 1].item()
        x_seg = x[s:e].to(torch.float32)
        back = x_fp8[s:e].to(torch.float32) / scales[i]
        rel = ((back - x_seg).abs().mean() / x_seg.abs().mean()).item()
        assert rel < 0.1, f"segment {i} fp8 roundtrip err {rel:.1%}"
