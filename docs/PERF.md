# Performance reference — RTX 5080 (Blackwell sm_120)

Driver 595.97, CUDA 12.8, Torch 2.11+cu128, Triton 3.6.0 (triton-windows).
All numbers come from `scripts/profile_v3.py` and `scripts/bench_matrix.py`.

## Per-kernel CUDA time (Qwen3-30B-A3B shape, T=1024)

```
_fused_gate_up_kernel        7.65 ms   45.91 %   ← largest single contributor
_grouped_fp8_gemm_kernel     3.84 ms   23.06 %   (down projection)
aten::index_add_             1.40 ms    8.40 %   ← replaced by Triton scatter (v0.11)
aten::mul (SwiGLU)           1.17 ms    7.01 %
_segment_amax_scale_kernel   0.88 ms    5.31 %
silu                         0.26 ms    1.57 %
sort                         0.27 ms    1.56 %
index gather                 0.26 ms    1.58 %
copy_                        0.21 ms    1.28 %
```

Total CUDA time: 16.65 ms / 10 iterations = **1.66 ms / forward**.

## Bench matrix snapshot (v0.10)

| Shape | bf16 ref | fp8_v3 fused | int4_v4 (per-ch) | int4_v5 (group) |
|---|---|---|---|---|
| Toy E=16            | 56k tok/s | **336k** | **430k** | n/a |
| OLMoE 7B            | tbd       | tbd      | tbd      | tbd |
| Qwen3-30B-A3B       | 53k       | **496k** | 107k     | 75k |
| Qwen3-MoE 57B/14B   | tbd       | tbd      | tbd      | tbd |
| Mixtral-8x7B        | 999k      | **1.61M** | tbd     | tbd |
| DeepSeek-V2-Lite    | tbd       | tbd      | tbd      | tbd |
| DeepSeek E=128      | 47k       | 142k     | 43k      | tbd |

Refresh: `make bench-matrix && cat bench_results/matrix.csv`.

## Memory profile (DeepSeek-V2-Lite end-to-end, v0.5)

```
64 routed experts × 26 layers (FP8)  : 13.7 GB
2 shared experts × 26 layers (FP8)   :  0.4 GB   (v0.5 patch)
MLA attention                        :  0.4 GB   (still bf16 — TODO)
Embed + lm_head                      :  0.8 GB
KV cache + activations               :  1.6 GB
                                      ────────
                                      16.9 GB peak
```

On a 16 GB RTX 5080, ~1 GB overflows to WDDM shared memory → ~5–7 tok/s.
Killing the MLA bf16 attention recovers that headroom.

## Bottleneck targets next

1. `_fused_gate_up_kernel` (46 %) — try persistent kernel + larger BLOCK_M
2. `aten::index_add_` (8 %) — done, replaced by `scatter_add` Triton
3. `_segment_amax_scale_kernel` (5 %) — fuse into the activation quant kernel
4. MLA attention bf16 (~1 GB VRAM) — convert to FP8 for VRAM headroom
