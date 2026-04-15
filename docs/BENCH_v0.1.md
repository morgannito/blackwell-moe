# Bench v0.1 — RTX 5080 (Blackwell sm_120, driver 595.97)

Setup: Python 3.14, Torch 2.11+cu128, CUDA 12.8. FP8 path via `torch._scaled_mm`.

## Raw results

```
Shape: T=256 D=1024 E=16 K=4 H=512           (toy)
impl              ms/iter        tok/s    peak MB
bf16_ref            4.852      52 756       85.9
fp8_bwmoe           9.646      26 538       86.3      → FP8 1.99× slower

Shape: T=1024 D=2048 E=64 K=8 H=1536          (Qwen3-30B-A3B-ish)
impl              ms/iter        tok/s    peak MB
bf16_ref           18.641      54 933     1 834
fp8_bwmoe          47.256      21 669     1 835      → FP8 2.53× slower

Shape: T=4096 D=2048 E=8 K=2 H=1536           (Mixtral-8x7B)
impl              ms/iter        tok/s    peak MB
bf16_ref            4.219     970 822       296
fp8_bwmoe           6.135     667 692       311      → FP8 1.45× slower
```

## Honest findings

**Our v0.1 FP8 path is slower than bf16 across all shapes.** Root cause:

1. **Per-expert dispatch** — we do E separate `_scaled_mm` calls on small M
   (tokens/expert). Kernel launch overhead + FP8 startup cost dominates.
2. **No grouped GEMM** — `torch._scaled_mm` has no batched variant. Each expert
   fires its own kernel, serializing on small work.
3. **Dynamic activation quantization** — per-forward amax + scale + cast happens
   per expert per forward. Not amortized.
4. **No fusion** — gate/up matmuls run separately; SwiGLU activation is a
   separate pass. Two HBM round-trips instead of one.

## Why this is the right problem

This is **exactly the gap** consumer Blackwell needs filled:

- CUTLASS grouped GEMM: Hopper-tuned, no sm_120 tiles upstream
- vLLM: bf16 default on Blackwell, FP8 path is CUTLASS Hopper
- cuBLASLt: no FP8 grouped variant
- SGLang / DeepSeek kernels: Hopper-targeted

A Triton FP8 **grouped** GEMM for sm_120 would be genuinely new.

## v0.2 targets

- [ ] Fix Triton toolchain (pin Python 3.12 + triton-windows, or bring WSL2)
- [ ] Triton grouped FP8 GEMM (one launch → all experts)
- [ ] Fused gate+up matmul (share token load)
- [ ] Fused SwiGLU+quantize for down-projection input
- [ ] Autotune BLOCK_M/N/K sweep for sm_120 (64/128/64 candidates vs 128/256/64)
- [ ] Target: ≥ parity with bf16 at E=64 / ≥ 1.5× at E=128

## Environment notes

- Triton 3.6.0 installed (`triton-windows` package) but fails to JIT on Py 3.14
  (TCC compilation error, `NTSTATUS 3236495362`). Working Triton will need
  Python 3.12 venv or WSL2.
- `torch._scaled_mm` requires `b` operand in column-major; current code does
  `.t().contiguous().t()` to force it — has its own overhead to remove.
