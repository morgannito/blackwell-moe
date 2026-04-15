"""INT4 quantize / dequantize round-trip tests."""

import pytest
import torch

from blackwell_moe.kernels.int4_quant import (
    dequantize_int4_per_channel,
    quantize_int4_per_channel,
)
from blackwell_moe.kernels.int4_group import (
    dequantize_int4_groups,
    quantize_int4_groups,
)


@pytest.mark.cuda
@pytest.mark.parametrize("K,N", [(128, 64), (256, 128), (512, 256)])
def test_roundtrip_per_channel(K, N):
    torch.manual_seed(0)
    w = torch.randn(K, N, device="cuda", dtype=torch.bfloat16) * 0.1
    packed, scales = quantize_int4_per_channel(w)
    w_back = dequantize_int4_per_channel(packed, scales, N)
    rel = ((w_back.float() - w.float()).abs().mean() / w.float().abs().mean()).item()
    assert rel < 0.35, f"per-channel roundtrip err {rel:.1%} too high"


@pytest.mark.cuda
@pytest.mark.parametrize("K,N", [(128, 64), (256, 128), (512, 256)])
def test_roundtrip_groups(K, N):
    torch.manual_seed(0)
    w = torch.randn(K, N, device="cuda", dtype=torch.bfloat16) * 0.1
    packed, scales = quantize_int4_groups(w)
    w_back = dequantize_int4_groups(packed, scales, K, N)
    rel = ((w_back.float() - w.float()).abs().mean() / w.float().abs().mean()).item()
    assert rel < 0.30, f"group roundtrip err {rel:.1%} higher than expected"


@pytest.mark.cuda
def test_scales_shapes():
    w = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16)
    packed_pc, scales_pc = quantize_int4_per_channel(w)
    packed_g, scales_g = quantize_int4_groups(w)
    assert packed_pc.shape == (256, 64)
    assert scales_pc.shape == (128,)
    assert packed_g.shape == (256, 64)
    assert scales_g.shape == (256 // 32, 128)
