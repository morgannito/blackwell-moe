# Bench v0.3 — RTX 5080 (Blackwell sm_120)

Python 3.12, Torch 2.11+cu128, Triton 3.6.0, CUDA 12.8.

## Results

```
Toy            (T=256  D=1024 E=16  K=4 H=512)
impl              ms/iter        tok/s    peak MB     vs bf16
bf16_ref            4.906      52 178       85.9       1.00×
fp8_v1_torch       11.181      22 895       89.0       0.44×
fp8_v2_grouped      5.253      48 735      363.1       0.93×
fp8_v3_fused        0.762     335 907      359.4       6.44×    ★

Qwen3-30B-A3B  (T=1024 D=2048 E=64  K=8 H=1536)
impl              ms/iter        tok/s    peak MB     vs bf16
bf16_ref           19.025      53 822    1 834.1       1.00×
fp8_v1_torch       45.133      22 688    1 846.9       0.42×
fp8_v2_grouped     20.525      49 889    2 267.3       0.93×
fp8_v3_fused        2.285     448 234    2 196.0       8.33×    ★★

Mixtral-8x7B   (T=4096 D=2048 E=8   K=2 H=1536)
impl              ms/iter        tok/s    peak MB     vs bf16
bf16_ref            4.101     998 739       296.3      1.00×
fp8_v1_torch        5.379     761 494       320.4      0.76×
fp8_v2_grouped      4.206     973 899       693.9      0.97×
fp8_v3_fused        2.545   1 609 743       622.6      1.61×    ★

DeepSeek-128   (T=2048 D=4096 E=128 K=8 H=2048)
impl              ms/iter        tok/s    peak MB     vs bf16
bf16_ref           43.209      47 397    9 726.7       1.00×
fp8_v1_torch      100.323      20 414    9 749.9       0.43×
fp8_v2_grouped     48.812      41 957   10 532.8       0.97×
fp8_v3_fused       14.378     142 437   10 415.3       3.01×    ★
```

## Summary

| Shape | v3 speedup vs bf16 |
|---|---|
| Toy (E=16)           | **6.4×** |
| Qwen3-30B-A3B (E=64) | **8.3×** |
| Mixtral-8x7B (E=8)   | **1.6×** |
| DeepSeek-128 (E=128) | **3.0×** |

## What drove the jump from v0.2 → v0.3

v0.2 had a Python `for e in range(E):` loop calling `offsets[e].item()` inside
`compute_segment_scales`. Each `.item()` forces a CUDA sync. With E=64 that's
**64 sync barriers per forward pass**. Replacing it with a single Triton
segment-amax kernel eliminates all syncs on the scale-computation path.

Secondary wins:
- **Fused gate+up GEMM** — one kernel launch instead of two, X loaded once
- **Segment quantize in Triton** — no Python loop re-entering CUDA context

The speedup is smaller on Mixtral (E=8, only 8 sync calls were happening) and
massive on E=64/E=128 where sync overhead dominated v0.2. Correctness vs bf16:
relative L1 error 6.4 %, consistent with per-tensor FP8 E4M3 quantization.

## Correctness

```
ref mean/std:  6.39e-05 / 8.06e-05
v3  mean/std:  6.34e-05 / 8.01e-05
relative L1:   6.40 %           ← FP8 E4M3 per-tensor tolerance
max abs err:   2.5e-05
zero fraction: 0.1 %            ← no silent short-circuit
```

## What's still left (v0.4)

- True online SwiGLU+quant kernel — currently blocked on Windows
  AppControl flagging dynamically compiled .pyd; will land in WSL2 env.
- Per-channel W_down scales (accuracy, not perf)
- vLLM custom-op integration as a single drop-in MoE block
- FP8 KV cache for full decode loop

## Reproducibility

```powershell
C:\Users\morga\blackwell-moe\.venv312\Scripts\python -m blackwell_moe.bench.cli \
  --tokens 1024 --dim 2048 --experts 64 --topk 8 --hidden 1536
```
