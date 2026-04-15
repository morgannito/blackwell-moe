"""Top-k router correctness vs reference softmax+topk."""

import pytest
import torch


@pytest.mark.cuda
@pytest.mark.parametrize("T,D,E,K", [(32, 256, 8, 2), (128, 512, 16, 4), (256, 1024, 32, 8)])
def test_top_k_router_matches_reference(T, D, E, K):
    from blackwell_moe.kernels.routing import top_k_router, top_k_router_ref

    torch.manual_seed(0)
    x = torch.randn(T, D, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(D, E, device="cuda", dtype=torch.bfloat16) * 0.05

    w1, i1 = top_k_router(x, w, k=K)
    w2, i2 = top_k_router_ref(x, w, k=K)

    assert torch.equal(i1, i2), "expert indices must match exactly"
    torch.testing.assert_close(w1, w2, rtol=1e-3, atol=1e-4)


@pytest.mark.cuda
def test_router_weights_sum_to_one():
    from blackwell_moe.kernels.routing import top_k_router

    torch.manual_seed(0)
    T, D, E, K = 64, 512, 16, 4
    x = torch.randn(T, D, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(D, E, device="cuda", dtype=torch.bfloat16) * 0.05
    weights, _ = top_k_router(x, w, k=K)
    sums = weights.sum(dim=-1)
    torch.testing.assert_close(sums, torch.ones_like(sums), rtol=1e-3, atol=1e-4)
