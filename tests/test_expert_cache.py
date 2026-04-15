"""FastExpertCache LRU and hit-rate behavior."""

import pytest
import torch


@pytest.mark.cuda
def test_fast_cache_basic_hit_miss():
    from blackwell_moe.runtime.fast_expert_cache import FastExpertCache

    E, D, H, N = 8, 64, 32, 4
    cpu_g = torch.empty((E, D, H), dtype=torch.float8_e4m3fn, pin_memory=True)
    cpu_u = torch.empty_like(cpu_g)
    cpu_d = torch.empty((E, H, D), dtype=torch.float8_e4m3fn, pin_memory=True)
    cpu_sg = torch.empty(E, dtype=torch.float32, pin_memory=True)
    cpu_su = torch.empty(E, dtype=torch.float32, pin_memory=True)
    cpu_sd = torch.empty(E, dtype=torch.float32, pin_memory=True)
    cpu_sg.fill_(1.0); cpu_su.fill_(1.0); cpu_sd.fill_(1.0)

    cache = FastExpertCache(cpu_g, cpu_u, cpu_d, cpu_sg, cpu_su, cpu_sd, N)
    cache.warmup([0, 1, 2, 3])

    # Request 0,1 — all hits
    req = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    cache.fetch_batch(req)
    stats = cache.stats()
    assert stats["misses"] == 0
    assert stats["hits"] >= 2

    # Request 4 (cold) — 1 miss
    req2 = torch.tensor([4], dtype=torch.int32, device="cuda")
    cache.fetch_batch(req2)
    stats2 = cache.stats()
    assert stats2["misses"] == 1


@pytest.mark.cuda
def test_fast_cache_lru_eviction():
    from blackwell_moe.runtime.fast_expert_cache import FastExpertCache

    E, D, H, N = 8, 64, 32, 2
    cpu_g = torch.empty((E, D, H), dtype=torch.float8_e4m3fn, pin_memory=True)
    cpu_u = torch.empty_like(cpu_g)
    cpu_d = torch.empty((E, H, D), dtype=torch.float8_e4m3fn, pin_memory=True)
    cpu_sg = torch.empty(E, dtype=torch.float32, pin_memory=True); cpu_sg.fill_(1.0)
    cpu_su = torch.empty(E, dtype=torch.float32, pin_memory=True); cpu_su.fill_(1.0)
    cpu_sd = torch.empty(E, dtype=torch.float32, pin_memory=True); cpu_sd.fill_(1.0)

    cache = FastExpertCache(cpu_g, cpu_u, cpu_d, cpu_sg, cpu_su, cpu_sd, N)
    cache.fetch_batch(torch.tensor([0, 1], dtype=torch.int32, device="cuda"))  # slots full
    cache.fetch_batch(torch.tensor([0], dtype=torch.int32, device="cuda"))      # touch 0
    cache.fetch_batch(torch.tensor([2], dtype=torch.int32, device="cuda"))      # evicts 1 (LRU)

    # 0 should still be on GPU, 1 should be evicted
    assert cache.expert_to_slot[0].item() >= 0
    assert cache.expert_to_slot[1].item() == -1
    assert cache.expert_to_slot[2].item() >= 0


@pytest.mark.cuda
def test_fast_cache_raises_on_overflow():
    from blackwell_moe.runtime.fast_expert_cache import FastExpertCache

    E, D, H, N = 8, 64, 32, 2
    cpu_g = torch.empty((E, D, H), dtype=torch.float8_e4m3fn, pin_memory=True)
    cpu_u = torch.empty_like(cpu_g)
    cpu_d = torch.empty((E, H, D), dtype=torch.float8_e4m3fn, pin_memory=True)
    cpu_sg = torch.empty(E, dtype=torch.float32, pin_memory=True); cpu_sg.fill_(1.0)
    cpu_su = torch.empty(E, dtype=torch.float32, pin_memory=True); cpu_su.fill_(1.0)
    cpu_sd = torch.empty(E, dtype=torch.float32, pin_memory=True); cpu_sd.fill_(1.0)

    cache = FastExpertCache(cpu_g, cpu_u, cpu_d, cpu_sg, cpu_su, cpu_sd, N)
    with pytest.raises(RuntimeError, match="Batch needs"):
        cache.fetch_batch(torch.tensor([0, 1, 2], dtype=torch.int32, device="cuda"))
