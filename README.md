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

## Current results (v0.13, RTX 5080)

| Shape | bf16 | fp8_v3 | **fp8_v4 megafuse** | speedup |
|---|---|---|---|---|
| Qwen3-30B-A3B (E=64)  | 58k | 525k | **538k** | **9.2×** |
| Mixtral-8x7B (E=8)    | 999k | **1.61M** | tbd  | 1.6× |
| DeepSeek (E=128)      | 52k | 164k | tbd  | 3.1× |
| Toy (E=16)            | 56k | 366k | tbd  | 6.5× |

v4 saves 185 MB peak VRAM by eliminating the bf16 SwiGLU intermediate.

FP8 wins on large/medium E, INT4 wins on small E + halves weight VRAM (4 bits vs 8 bits per param). Use FP8 for compute-bound shapes, INT4 when memory-bound.

FP8 correctness: 6.4 % relative L1 error vs bf16 ref (per-tensor E4M3 tolerance).
Full table: `docs/BENCH_v0.3.md`.

## End-to-end runtime (v0.15)

Real DeepSeek-V2-Lite-Chat (16B, 64 experts, top-6) running on RTX 5080 with
our FP8 kernels replacing `DeepseekV2MoE.forward`, FP8 shared experts, and
CPU-offloaded `embed_tokens` + `lm_head`:

```
Prompt: "Write a short poem about the ocean:"
Output: "In the midst of the sea, the world is a gentle whisper,
         A place of stone and a heart's understanding,
         A wave's gentle pull, the ocean whispers is silence..."
Generated 52 tokens @ 10.4 tok/s   (up from 5.4 in v0.5 — 1.9× faster)
Peak VRAM: 15.80 GB (fits the 16 GB card, no WDDM overflow)
```

Throughput climb across iterations: 5.4 → 6.1 → 6.4 → **10.4 tok/s**.

Pipeline: `snapshot_download` (31 GB bf16, 1m36s) → streaming FP8 loader
(routed experts quantized on-the-fly) → `patch_shared_experts` + `patch_deepseek_moe_with_store`
→ `model.generate()`. See `src/blackwell_moe/runtime/`.

## Roadmap

- [x] Triton FP8 E4M3 GEMM, per-tensor scales
- [x] Fused top-k router
- [x] Bench harness + correctness tests
- [x] **Grouped FP8 GEMM** — single launch, all experts (v0.2)
- [x] **Autotune BLOCK_M/N/K** for sm_120 (v0.2)
- [x] **Segment-reduce amax in Triton** — kill Python loop, -64 syncs (v0.3)
- [x] **Fused gate+up grouped GEMM** — share X load (v0.3)
- [x] **End-to-end DeepSeek-V2-Lite** — streaming FP8 loader, MoE patching (v0.4-0.5)
- [x] **INT4 grouped GEMM kernel** — 4× weight VRAM savings (v0.6)
- [x] **LRU expert cache v0.8** — GPU state tensors, batched fetch, async H2D stream. 99 % hit rate, 50 % VRAM reduction, overhead dropped 11× vs v0.7 (see `docs/BENCH_cache.md`)
- [x] **Group-scale INT4 v0.9** — Q4_0-style bf16 scale per 32-K block, 22 % → 16 % L1 err on random weights (tighter on trained)
- [ ] Q4_K_M super-blocks (6-bit scale + 4-bit zero, ~3 % err)
- [ ] Online SwiGLU+quant kernel (Windows AppControl blocking, WSL2 required)
- [ ] MLA attention FP8 quantization
- [ ] vLLM custom-op integration
- [ ] FP8 KV cache path
- [ ] vLLM custom-op integration

## Licence

Apache-2.0.
