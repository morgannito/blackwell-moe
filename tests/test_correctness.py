"""Correctness: FP8 kernel output vs bf16 reference within tolerance."""

import pytest
import torch

pytest.importorskip("triton")


@pytest.mark.cuda
def test_routing_matches_reference():
    from blackwell_moe.kernels.routing import top_k_router, top_k_router_ref

    torch.manual_seed(0)
    x = torch.randn(64, 256, device="cuda", dtype=torch.bfloat16)
    g = torch.randn(256, 16, device="cuda", dtype=torch.bfloat16) * 0.05

    w1, i1 = top_k_router(x, g, k=4)
    w2, i2 = top_k_router_ref(x, g, k=4)

    assert torch.equal(i1, i2), "expert indices must match exactly"
    torch.testing.assert_close(w1, w2, rtol=1e-3, atol=1e-4)


@pytest.mark.cuda
def test_fp8_moe_close_to_bf16():
    from blackwell_moe.kernels.fp8_moe import fp8_moe_forward, to_fp8_e4m3
    from blackwell_moe.kernels.reference import moe_forward_bf16

    torch.manual_seed(0)
    T, D, E, K, H = 32, 512, 8, 2, 256
    x = torch.randn(T, D, device="cuda", dtype=torch.bfloat16) * 0.1
    wg = torch.randn(D, E, device="cuda", dtype=torch.bfloat16) * 0.02
    eg = torch.randn(E, D, H, device="cuda", dtype=torch.bfloat16) * 0.02
    eu = torch.randn(E, D, H, device="cuda", dtype=torch.bfloat16) * 0.02
    ed = torch.randn(E, H, D, device="cuda", dtype=torch.bfloat16) * 0.02

    y_ref = moe_forward_bf16(x, wg, eg, eu, ed, K)

    eg8 = torch.stack([to_fp8_e4m3(eg[i])[0] for i in range(E)])
    eu8 = torch.stack([to_fp8_e4m3(eu[i])[0] for i in range(E)])
    ed8 = torch.stack([to_fp8_e4m3(ed[i])[0] for i in range(E)])
    sg = torch.tensor([float(to_fp8_e4m3(eg[i])[1]) for i in range(E)], device="cuda")
    su = torch.tensor([float(to_fp8_e4m3(eu[i])[1]) for i in range(E)], device="cuda")
    sd = torch.tensor([float(to_fp8_e4m3(ed[i])[1]) for i in range(E)], device="cuda")

    y = fp8_moe_forward(x, wg, eg8, eu8, ed8, sg, su, sd, K)
    # FP8 tolerance: ~3% relative
    torch.testing.assert_close(y, y_ref, rtol=3e-2, atol=1e-2)
