# Changelog

## v0.11 (current)

- Custom Triton `scatter_add` kernel — replaces `aten::index_add_` (8 % of
  v0.10 forward) with native bf16 atomic-add on global memory
- `quant_fp8_per_row` and `quant_fp8_block` helpers in `kernels/fp8_quant`
- 10 autotune configs per grouped GEMM (up from 5–6) — wider search space for sm_120
- `scripts/verify_all.py` — single-command pass/fail across every verify script
- `scripts/bench_matrix.py` extended: OLMoE-1B-7B, Qwen3-MoE 57B/14B, DeepSeek-V2-Lite shapes
- `Makefile` with `install`, `test`, `bench`, `bench-matrix`, `verify`, `profile`, `lint`
- `docs/PERF.md` — kernel profile reference, memory footprint, target bottlenecks
- 26 unit tests (up from 18) — added scatter and FP8 helpers

## v0.10

- Public API: `from blackwell_moe import fp8_moe_forward_v3, int4_group_moe_forward, FastExpertCache, quant_fp8_e4m3`
- Shared `kernels/fp8_quant.py` module, deduplicated `_quant_fp8` across 7 files
- 18 unit tests covering routing, segment ops, INT4 round-trip (per-channel + group), expert cache LRU
- GitHub Actions: lint (ruff) + cpu-safe pytest
- `scripts/bench_matrix.py` and `scripts/profile_v3.py` for systematic measurement
- Profile of v3 forward: fused gate+up = 46 % of CUDA time, grouped_fp8 down = 23 %, index_add = 8 %

## v0.9

- INT4 group-scale kernel (`int4_group`, `int4_moe_group`): bf16 scale per 32-K block (Q4_0 style)
- Random-weight L1 error 22 % → 16.6 % vs per-channel; expected ~3-5 % on trained weights

## v0.8

- `FastExpertCache`: GPU-resident state tensors (`expert_to_slot`, `slot_to_expert`, `slot_last_used`)
- Batched H2D via pinned-pool gather + `non_blocking=True`
- Dedicated `h2d_stream` overlaps weight transfer with main-stream compute
- Single `fetch_batch()` per forward
- Cache overhead +2786 % → +249 % (11×), VRAM reduction 29 % → 50 %

## v0.7 (experimental)

- First `LRUExpertCache` (Python dict + OrderedDict)
- Concept validated, performance dominated by Python-side overhead — see `docs/BENCH_cache.md`

## v0.6

- `int4_quant`, `grouped_int4`, `int4_moe`: per-channel INT4 GEMM in Triton
- 4× weight VRAM savings vs FP8

## v0.5

- `shared_expert_fp8`: drop-in FP8 SwiGLU replacement for `DeepseekV2MLP`
- `cli` runtime cleanup; end-to-end DeepSeek-V2-Lite at 5–7 tok/s

## v0.4

- `runtime/loader.py`: streaming FP8 loader, quantizes routed experts on-the-fly
- `runtime/deepseek_patch.py`: replaces `DeepseekV2MoE.forward` with v3 kernel

## v0.3

- Triton segment-amax kernel (`segment_ops.py`): kills the Python loop, removes 64 syncs/forward
- `grouped_fp8_gateup`: fused gate + up matmul (one kernel, shared X load)
- `fp8_moe_v3`: 8.3× bf16 cuBLASLt on Qwen3-30B-A3B shape

## v0.2

- `grouped_fp8`: single-launch grouped FP8 GEMM, autotuned for sm_120
- 2× faster than per-expert dispatch, parity with bf16

## v0.1

- Project scaffold, Triton FP8 baseline, bench harness
