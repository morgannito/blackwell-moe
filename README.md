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

## Current results (v0.3, RTX 5080)

| Shape | bf16 cuBLAS | fp8_v3 fused | **speedup** |
|---|---|---|---|
| Qwen3-30B-A3B (E=64)  | 53k tok/s | **448k tok/s** | **8.3×** |
| Toy (E=16)            | 52k tok/s | **336k tok/s** | 6.4× |
| DeepSeek (E=128)      | 47k tok/s | **142k tok/s** | 3.0× |
| Mixtral-8x7B (E=8)    | 999k tok/s | **1.61M tok/s** | 1.6× |

FP8 correctness: 6.4 % relative L1 error vs bf16 ref (per-tensor E4M3 tolerance).
Full table: `docs/BENCH_v0.3.md`.

## Roadmap

- [x] Triton FP8 E4M3 GEMM, per-tensor scales
- [x] Fused top-k router
- [x] Bench harness + correctness tests
- [x] **Grouped FP8 GEMM** — single launch, all experts (v0.2)
- [x] **Autotune BLOCK_M/N/K** for sm_120 (v0.2)
- [x] **Segment-reduce amax in Triton** — kill Python loop, -64 syncs (v0.3)
- [x] **Fused gate+up grouped GEMM** — share X load (v0.3)
- [ ] True online SwiGLU+quant kernel (Windows AppControl blocking, TBD in WSL2)
- [ ] Per-channel W_down scales
- [ ] FP8 KV cache path
- [ ] vLLM custom-op integration

## Licence

Apache-2.0.
