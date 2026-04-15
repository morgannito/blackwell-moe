# LRU Expert Cache — findings (v0.7 experimental)

Goal: fit MoE models whose expert pool exceeds VRAM by keeping hot experts
on GPU and streaming cold ones from pinned CPU on demand.

## Setup

```
T=128  D=2048  E=64  top_k=6  H=1408
GPU slots: 48 (75 % of 64)
Zipf routing distribution, s=2.0
50 forwards
```

## Raw numbers

```
                      time/fwd     peak VRAM    notes
cached (this PR)      33.7 ms      0.70 GB     88.4 % hit rate
baseline (all GPU)     1.2 ms      0.99 GB     no cache
                      ────────     ────────
overhead             +2786 %       -29 % VRAM
```

## Honest finding

**Cache mechanism is correct but dominated by Python-side overhead**,
not by the raw PCIe transfer cost. The 3.6 expected misses per forward
(at 375 µs each) account for ~1.35 ms — yet forwards cost 33 ms.

Sources of overhead (in order of likely magnitude):

1. `torch.unique()` on the routing tensor — CPU roundtrip
2. Python `dict`/`deque`/`list` operations per forward
3. `torch.cuda.synchronize()` after every fetch (blocks on all queued GPU work)
4. Re-building `remap` tensor as CPU-to-GPU element-wise assignment

## When the cache actually wins

For models whose expert pool **physically cannot fit** in VRAM:

| Model | Expert pool FP8 | VRAM needed | 5080 (16 GB) |
|---|---|---|---|
| DeepSeek-V2-Lite (16B) | 14 GB | fits | cache = pure cost |
| Mixtral-8x7B (47B) | 40 GB | overflow | cache required |
| Mixtral-8x22B (141B) | 120 GB | overflow | cache + disk required |
| DeepSeek-V2 (236B) | 200 GB | overflow | cache + disk required |

For the first row: don't use this. For rows 2–4: the 27× overhead is
still much better than "doesn't run at all".

## Optimizations that would recover perf

- **Prefetch next layer's experts** on a separate CUDA stream while the
  current layer's kernel runs (overlap compute & H2D)
- **Fused fetch op** in C++ that runs the unique + dispatch on GPU
- **Keep LRU state on-device** — no CPU dict traffic per forward
- **Batch the full model's expert needs** up front (one sync total)
- **Persistent pinned buffers** for scales — currently one `to(device)` per fetch

With these, expected overhead drops to ~1.3 ms × #cold-layers per forward
(pure PCIe cost), which is negligible for generation-phase workloads.

## Reproducibility

```powershell
C:\Users\morga\blackwell-moe\.venv312\Scripts\python scripts\bench_cache.py
```

## Status

- `LRUExpertCache` class + FP8 forward path: **works, correctness OK**
- Performance: **experimental** — do not enable on VRAM-resident models
- Next step: move the fetch path off Python onto CUDA streams
