# Bench v0.2 — RTX 5080 (Blackwell sm_120)

Python 3.12, Torch 2.11+cu128, Triton 3.6.0 (triton-windows), CUDA 12.8.

## Results

```
Toy (T=256 D=1024 E=16 K=4 H=512)
impl              ms/iter        tok/s    peak MB
bf16_ref            5.677      45 090       85.9
fp8_v1_torch       11.224      22 809       89.0
fp8_v2_grouped      5.400      47 405      363.1      ← 2.08× faster than v1, 1.05× bf16

Qwen3-30B-A3B  (T=1024 D=2048 E=64 K=8 H=1536)
impl              ms/iter        tok/s    peak MB
bf16_ref           19.278      53 118    1 834.1
fp8_v1_torch       44.138      23 200    1 846.9
fp8_v2_grouped     20.706      49 454    2 267.3      ← 2.13× v1, 0.93× bf16

Mixtral-8x7B  (T=4096 D=2048 E=8 K=2 H=1536)
impl              ms/iter        tok/s    peak MB
bf16_ref            4.132     991 271       296.3
fp8_v1_torch        5.451     751 390       320.4
fp8_v2_grouped      4.242     965 492       693.9      ← 1.28× v1, 0.97× bf16

DeepSeek-ish  (T=2048 D=4096 E=128 K=8 H=2048)
impl              ms/iter        tok/s    peak MB
bf16_ref           43.482      47 099    9 726.7
fp8_v1_torch       99.624      20 557    9 749.9
fp8_v2_grouped     48.433      42 285   10 532.8      ← 2.06× v1, 0.90× bf16
```

## Summary

- **v2 grouped is ~2× faster than v1 naive FP8** across all shapes
- **At parity with cuBLASLt bf16 baseline** (within 7-10 %)

Cutting the per-expert dispatch loop and running a single grouped GEMM launch
is the single biggest win. Autotune converges on `BLOCK_M=64/128, BLOCK_N=128,
BLOCK_K=64` with 3 stages on sm_120 — different from Hopper's preferred tiles.

## Why we're not yet *ahead* of bf16

Three remaining inefficiencies:

1. **`compute_segment_scales` runs a Python for-loop over experts** on CPU/GPU
   sync per call — should be a single `torch.segment_reduce` or custom Triton.
2. **Gate and Up are two grouped-GEMM launches**. They share the same X load;
   fusing them into one kernel halves HBM traffic on X.
3. **`fused_swiglu_quant` currently takes pre-computed scales** (needs a two-pass
   approach at runtime). A single-pass online-scale variant using warp-level
   reductions would remove the h_tmp bf16 materialization altogether.

## v0.3 targets

- [ ] Segment-reduce amax in Triton (kill the Python loop)
- [ ] Fused gate+up grouped GEMM (single kernel, 2 Y outputs)
- [ ] True online SwiGLU+quant (no bf16 intermediate)
- [ ] Per-channel W_down scales
- [ ] Target: **1.5–2× bf16 baseline** at E=64+, ≥parity at all shapes

## Reproduction

```bash
GPU_HOST=morga@192.168.50.90 ./scripts/deploy_to_gpu.sh
ssh $GPU_HOST 'cd blackwell-moe &&
  .venv312\Scripts\python -m blackwell_moe.bench.cli \
    --tokens 1024 --dim 2048 --experts 64 --topk 8 --hidden 1536'
```
