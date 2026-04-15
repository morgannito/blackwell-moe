# Design notes

## Target hardware

NVIDIA Blackwell consumer (sm_120):
- RTX 5080: 16 GB GDDR7, 10 752 CUDA cores, FP8 tensor cores, no TMA, no FP4 consumer
- Different SM partition vs Hopper → block sizes 64/128/64 beat the 128/256/64 Hopper defaults in our sweeps

## FP8 scaling strategy

Per-tensor E4M3 with dynamic activation scales, static weight scales computed at load.

```
x_fp8  = clip(x * sx, ±448)       sx = 448 / amax(x)
w_fp8  = clip(w * sw, ±448)       sw = 448 / amax(w), computed once
acc    = fp32 tensor-core accumulate
out    = acc / (sx * sw)  →  bf16
```

Per-channel scales for `W_down` are a documented TODO — DeepSeek-V3 showed the down projection has heavier tails.

## Dispatch strategy

Token-permutation (not expert-parallel): gather tokens per expert, one grouped GEMM per expert. Works well for top-k ≤ 8 and E ≤ 128 on a single GPU. For E > 256, switch to block-wise expert-parallel with NCCL all-to-all (future work, not single-GPU).

## Why not CUTLASS?

CUTLASS grouped GEMM has no upstream sm_120 tuning yet. Triton lets us iterate BLOCK sizes and async pipeline stages in a Python loop — shipping a first kernel in hours, not weeks. CUTLASS port is a post-v0.2 item once autotune lands.

## Non-goals (v0.x)

- Training (backward pass)
- Multi-GPU / NCCL
- FP4 — not available on consumer Blackwell
- Windows — Linux-only (Triton on Windows is a mess)
