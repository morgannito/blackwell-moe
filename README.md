# blackwell-moe

FP8 Mixture-of-Experts inference kernels hand-tuned for NVIDIA Blackwell consumer GPUs (sm_120 — RTX 5080/5090).

> Target niche: **consumer Blackwell** is brand new and under-served. Hopper (sm_90) has CUTLASS/cuBLAS FP8 MoE, but the sm_120 tuning space is wide open — the terrain solo contributors can still move.

## Why

- Most FP8 MoE kernels ship Hopper tiles; consumer Blackwell needs different block sizes, warp layouts, and FP8 scaling recipes.
- vLLM and SGLang remain bf16 on sm_120 by default; FP8 path is unoptimized.
- Consumer Blackwell = FP8 tensor cores, no FP4, no TMA — kernels need to be rewritten, not ported.

## What

- `fp8_moe_forward` — SwiGLU MoE block in FP8 E4M3, per-tensor scales, bf16 accumulate-out.
- `top_k_router` — fused gate → softmax → top-k routing in one Triton launch.
- Reference bf16 implementation for correctness.
- Bench harness comparing bf16 eager vs our FP8 kernel on the shapes of Qwen3-30B-A3B, Mixtral-8x7B, DeepSeek-V2-Lite.

## Quick start

```bash
git clone <repo> && cd blackwell-moe
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[bench,dev]"

# Bench on a Blackwell machine
bwmoe-bench --tokens 1024 --experts 64 --topk 8 --hidden 1536

# Correctness tests
pytest tests/
```

## Remote GPU workflow

Dev on Mac, run on Blackwell box:

```bash
GPU_HOST=user@5080-rig ./scripts/deploy_to_gpu.sh bench
```

## Current results (v0.2, RTX 5080)

| Shape | bf16 | fp8_v1 naive | fp8_v2 grouped |
|---|---|---|---|
| Qwen3-30B-A3B | 53k tok/s | 23k tok/s | **49k tok/s** (2.1× v1) |
| Mixtral-8x7B | 991k tok/s | 751k tok/s | **965k tok/s** (1.3× v1) |
| DeepSeek 128e | 47k tok/s | 20k tok/s | **42k tok/s** (2.1× v1) |

→ ~parity with cuBLASLt bf16, 2× faster than naive FP8 dispatch. Full table: `docs/BENCH_v0.2.md`.

## Roadmap

- [x] Triton FP8 E4M3 GEMM, per-tensor scales
- [x] Fused top-k router
- [x] Bench harness + correctness tests
- [x] **Grouped FP8 GEMM** — single launch, all experts (v0.2)
- [x] **Autotune BLOCK_M/N/K** for sm_120 (v0.2)
- [ ] Segment-reduce amax in Triton (kill Python loop)
- [ ] Fused gate+up grouped GEMM (share X load)
- [ ] True online SwiGLU+quant (no bf16 intermediate)
- [ ] Per-channel W_down scales
- [ ] FP8 KV cache path
- [ ] vLLM custom-op integration

## Licence

Apache-2.0.
