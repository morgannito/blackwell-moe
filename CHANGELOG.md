# Changelog

## v0.19 (current)

- **Perplexity quality validation** (`scripts/eval_perplexity.py`) — WikiText-2 sliding-window PPL on DeepSeek-V2-Lite-Chat
- 8192 tokens / stride 2048:
  - FP8 + CPU offload: PPL 80.66 (NLL 4.39)
  - FP8 without offload: PPL 78.02 (NLL 4.36)
  - Delta **3.4 %** — CPU offload preserves quality
- Mixtral-8x22B expert extraction complete: **448 files** (56 layers × 8 experts), 135 GB per-expert FP8 safetensors on `J:\models\Mixtral-8x22B-fp8-experts\`
- Absolute PPL is elevated (Chat variant on raw WikiText), but both configs track identically → no FP8 regression

## v0.18

- `fp8_moe_forward_small_e` + `grouped_fp8_gemm_small_e` — Mixtral-tuned
  kernel with BLOCK_M up to 256, wide BLOCK_N up to 256, specifically for
  E ≤ 8 / top_k ≤ 2 workloads (Mixtral-8x7B, 8x22B)
- Bench matrix extended with Mixtral-8x22B shapes (prefill T=512 and
  decode T=16) at the real (D=6144, H=16384) dimensions
- Mixtral-8x22B decode on RTX 5080:
  - bf16: 2 291 tok/s
  - fp8_v3: 4 604 tok/s
  - fp8_v4: 4 440 tok/s
  - **fp8_small_e: 4 707 tok/s** ← best, 2.05× bf16
- Mixtral-8x22B prefill (T=512):
  - fp8_v4: **101 261 tok/s** (2.66× bf16)
  - fp8_small_e: 94 055 tok/s
- Mixtral-8x22B download complete (282 GB); expert extraction in progress

## v0.17

- **Async prefetch**: `ThreeTierExpertCache.prefetch_layer()` issues
  background disk → RAM loads on a `ThreadPoolExecutor`, called from
  `streaming_moe_forward(..., prefetch_next_layer=N+1)` so layer N+1's
  experts are warm by the time layer N+1 actually executes
- **Hot-history-based prefetch**: only experts that have been used before
  for that layer are prefetched (avoids 64-wide eviction storms)
- LFU eviction reverted to LRU after discovering it could evict the very
  experts we had just loaded in the same fetch (all newly-loaded entries
  share `freq=1`, ties get picked deterministically and randomly clobber)
- Strict `_prefetch_lock` covers the entire fetch loop — prefetch threads
  cannot evict an entry between its lookup and the GPU copy
- DeepSeek-V2-Lite end-to-end on streaming infrastructure:
  - 17.2 GB VRAM → **2.4 GB** (87 % less)
  - 10.4 → 2.1 tok/s (disk-bound on 1664-expert pool, expected)
  - Output quality preserved: "The capital of France is a city that is
    known for its history, its art, and its world-class art..."
- Infrastructure validated end-to-end. Ready for Mixtral-8x22B once the
  282 GB download completes.

## v0.16

- 3-tier expert cache (`disk_expert_pool.py`): GPU slots / RAM pinned /
  disk-mmap safetensors per expert
- `streaming_moe.py`: streaming MoE forward that fetches via the cache
- `extract_experts_to_disk.py`: one-shot script to convert HF checkpoints
  into per-expert FP8 safetensors files (DeepSeek + Mixtral patterns)
- `mixtral_loader.py`, `mixtral_patch.py`, `mixtral_cli.py`,
  `deepseek_streaming.py`, `streaming_cli.py`: full streaming runtime
  for both model families

## v0.15

- `runtime/cli.py` warmup pass + `reset_peak_memory_stats()` reveals the
  actual generation footprint: **15.80 GB peak post-warmup** vs 17.19 GB
  before. The 1.4 GB delta was Triton autotune scratch buffers from the
  very first JIT — they get freed and never re-allocated
- **Generation throughput on RTX 5080 jumps 6.4 → 10.4 tok/s (+62 %)** on
  DeepSeek-V2-Lite-Chat, because we no longer overflow into WDDM shared
  memory once the autotune is primed
- Output quality preserved: "In the midst of the sea, the world is a
  gentle whisper, A place of stone and a heart's understanding…"

## v0.14

- `runtime/cpu_offload.py`: `CPUEmbedding` and `CPULinear` wrappers keep
  `embed_tokens` and `lm_head` weights on CPU permanently, transferring only
  activations across PCIe — **838 MB of GPU steady-state memory freed** on
  DeepSeek-V2-Lite-Chat
- Quality preserved end-to-end: "The capital of France is a passionate,
  beautiful, and at times, a very unforgiving world..." — coherent text
- `fp8_moe_forward_v4` (mega-fusion SwiGLU+quant) is **excellent for prefill
  benches but degrades generation quality** on real models — likely a
  numerical issue in the `tl.atomic_max` amax accumulation when some experts
  receive zero tokens. Reverted to v3 in the runtime patch; v4 stays
  available in `blackwell_moe.fp8_moe_forward_v4` for prefill-heavy
  experiments
- VRAM after MoE patch + CPU offload: 16.61 → **15.77 GB steady state**
  (peak during forward stays at 17.19 GB — Triton autotune scratch + sort
  intermediates account for the spike)

## v0.13

- **Mega-fusion `fused_swiglu_quant`**: two Triton passes replace the v0.12
  `mul` + `segment_amax` + `segment_quant` triplet, eliminating the bf16 `h`
  intermediate materialization
- Pass 1 (`_swiglu_amax_kernel`) computes silu(g)*u in registers and accumulates
  per-expert amax via `tl.atomic_max` on a global [E] fp32 tensor
- Pass 2 (`_swiglu_quant_kernel`) recomputes silu(g)*u with the fixed scale and
  writes h_fp8 directly
- New forward `fp8_moe_forward_v4` chains the existing kernels with the fused
  SwiGLU+quant
- **Bench (Qwen3-30B-A3B, T=1024)**: v3 525k → v4 538k tok/s (+2.3 %),
  peak VRAM 2838 → 2654 MB (-185 MB, the bf16 `h` is gone)
- Profile delta: `aten::mul` 1.17 → 0.74 ms (-37 %), `segment_amax` 0.88 →
  0.50 ms (-43 %), total CUDA 15.46 → 15.09 ms (-2.4 %)
- Triton kernel that previously failed to compile under Windows AppControl now
  loads cleanly in the .venv312 toolchain — no AppControl block this time

## v0.12

- Wired `scatter_add` Triton kernel into `fp8_moe_forward_v3` (was still
  calling `aten::index_add_`). End result: **+5 % throughput** on
  Qwen3-30B-A3B shape (493k → 517k tok/s), **-82 % time** on the scatter op
- Wider autotune search (20+ configs per grouped GEMM) ran but found no
  better tile for sm_120 — confirmed we hit the FP8 tensor-core throughput
  ceiling on consumer Blackwell for this shape

## v0.11

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
