"""Triton scatter_add vs torch.index_add_ correctness."""

import pytest
import torch


@pytest.mark.cuda
@pytest.mark.parametrize("M,T,D", [(64, 16, 128), (256, 32, 512), (1024, 128, 2048)])
def test_scatter_matches_index_add(M, T, D):
    from blackwell_moe.kernels.scatter import scatter_add

    torch.manual_seed(0)
    src = torch.randn(M, D, device="cuda", dtype=torch.bfloat16) * 0.05
    index = torch.randint(0, T, (M,), device="cuda", dtype=torch.int64)

    out_ref = torch.zeros(T, D, device="cuda", dtype=torch.bfloat16)
    out_ref.index_add_(0, index, src)

    out_triton = torch.zeros(T, D, device="cuda", dtype=torch.bfloat16)
    scatter_add(out_triton, index, src)

    torch.testing.assert_close(out_triton, out_ref, rtol=5e-2, atol=1e-2)


@pytest.mark.cuda
def test_scatter_handles_empty_targets():
    from blackwell_moe.kernels.scatter import scatter_add

    M, T, D = 32, 64, 128
    src = torch.ones(M, D, device="cuda", dtype=torch.bfloat16)
    # All indices map to first 8 rows — most of `out` should stay zero
    index = torch.randint(0, 8, (M,), device="cuda", dtype=torch.int64)
    out = torch.zeros(T, D, device="cuda", dtype=torch.bfloat16)
    scatter_add(out, index, src)
    assert (out[8:] == 0).all()
    assert (out[:8] != 0).any()
