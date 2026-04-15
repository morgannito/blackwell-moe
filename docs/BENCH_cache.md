# LRU Expert Cache — v0.7 vs v0.8

## v0.7 (baseline — Python dict LRU)

```
T=128 D=2048 E=64 K=6 H=1408  N_slots=48  zipf_s=2.0  50 forwards
              time/fwd   VRAM    notes
cached        33.7 ms    0.70 GB  88 % hit rate
baseline all   1.2 ms    0.99 GB
overhead     +2786 %    -29 % VRAM
```

Root cause: Python `dict`, `OrderedDict`, `.item()` per call, per-expert
H2D copies, `torch.cuda.synchronize()` on every fetch.

## v0.8 (optimized — GPU-resident state, batched fetch, async stream)

```
T=128 D=2048 E=128 K=6 H=1408  N_slots=64  zipf_s=4.0  20 forwards × 26 layers
              time/fwd   time/layer  VRAM    notes
cached        105.8 ms   4.07 ms     0.84 GB  99 % hit rate
baseline all   30.3 ms   1.17 ms     1.68 GB
overhead      +249 %    +248 %      -50 % VRAM
```

### Changes that landed the 11× improvement

| Optimization | Effect |
|---|---|
| **GPU-resident `expert_to_slot` / `slot_to_expert` / `slot_last_used` tensors** | No Python dict → no serialization bottleneck |
| **Batched H2D** — `cpu_pool[miss_ids]` then single `.to(device, non_blocking=True)` | 1 copy instead of 3 × N_misses |
| **Dedicated `h2d_stream`** | Overlaps weight transfers with main-stream compute |
| **LRU-victim via `torch.topk(-slot_last_used, n)`** | O(N) GPU op vs Python `deque.popleft` |
| **Single `fetch_batch()` per forward** (all layers' needs at once) | Kills per-layer sync from `torch.unique()` and `.cpu()` |
| **Non-blocking stats** (GPU tensor accumulators) | No `.item()` in hot path |

### Residual overhead (+249 %)

Two irreducible sources remain:

1. `torch.unique(all_layer_ids)` returns a dynamic-shape tensor → sync
2. `miss_ids.cpu()` for CPU-pool indexing → sync

On hit-heavy workloads (99 % here), these cost ~5 ms per fetch. One fetch
per forward → ~100 ms per 20 forwards → the bulk of the remaining overhead.

Killing those would need a CUDA graph capture of the whole fetch path,
or a custom op that returns fixed-shape miss-buffer + count (still needs a
sync at some point before CPU indexing).

### When the cache is a net win

| Pool vs VRAM | Use cache? |
|---|---|
| Pool fits in VRAM | No — 2.5× slower for nothing |
| Pool = 1.5–3× VRAM | Yes — model runs that otherwise wouldn't |
| Pool > 10× VRAM | Yes, with disk-backed CPU pool as well |

### v0.9 roadmap

- CUDA graph capture of fetch+forward (static shapes)
- Streamed prefetch: launch fetch for forward N+1 on a separate stream during forward N
- Move LRU victim selection to a Triton kernel (eliminates `torch.topk` sync)
- Custom C++ op for fetch_batch (single sync point, native-speed dict)
