"""LRU expert cache — pinned CPU pool + N GPU slots, async H2D on miss.

Problem: MoE with E=64..256 experts in FP8 can exceed GPU VRAM for larger
models (Mixtral-8x22B, DeepSeek-V2 full, Qwen3-235B-A22B). Expert activation
follows a power-law: a handful of "hot" experts get hit 10-100× more than
cold ones. Keep hot experts on GPU, stream cold ones from pinned CPU on demand.

Design
------
* All E expert weight tensors live in **pinned CPU memory** (page-locked, enables
  peak PCIe 5.0 x16 throughput ~64 GB/s)
* A GPU buffer holds N < E experts at a time (the "slots")
* `expert_to_slot` maps logical expert id → slot index (or -1 if cold)
* On `fetch_experts(required_ids)`:
    - hot hits: return their slot ids directly
    - cold misses: evict least-recently-used slots, async H2D copy, update map
* Router indices [T, K] get **remapped** to slot indices before kernel dispatch
* An aux tensor `slot_to_expert` [N] tracks which expert is in each slot

PCIe math for DeepSeek-V2-Lite-sized experts (~8 MB FP8 each):
  1 cold-miss transfer = 8 MB / 64 GB/s ≈ 125 µs
  Per forward, 26 layers × top-6 misses worst-case = 19.5 ms pure PCIe
  With 80 % cache hit rate → 156 × 0.2 × 0.125 ms ≈ 3.9 ms / forward

Trade-off: for models that already fit in VRAM, caching adds pure overhead.
The wins land for models whose expert pool > VRAM (Mixtral-8x22B, V2 full).
"""

from __future__ import annotations

from collections import OrderedDict

import torch


class LRUExpertCache:
    """N-slot LRU cache of FP8 expert weights (gate, up, down) + scales."""

    def __init__(
        self,
        experts_cpu_fp8_gate: torch.Tensor,  # [E, D, H] fp8, pinned cpu
        experts_cpu_fp8_up: torch.Tensor,
        experts_cpu_fp8_down: torch.Tensor,
        scales_cpu_gate: torch.Tensor,       # [E] fp32 pinned
        scales_cpu_up: torch.Tensor,
        scales_cpu_down: torch.Tensor,
        gpu_slots: int,
        device: str = "cuda",
    ):
        E = experts_cpu_fp8_gate.shape[0]
        assert gpu_slots <= E
        self.E = E
        self.N = gpu_slots
        self.device = device

        # Pinned CPU reservoir (memory-mapped via pin_memory)
        self.cpu_g = experts_cpu_fp8_gate
        self.cpu_u = experts_cpu_fp8_up
        self.cpu_d = experts_cpu_fp8_down
        self.cpu_sg = scales_cpu_gate
        self.cpu_su = scales_cpu_up
        self.cpu_sd = scales_cpu_down

        _, D, H = experts_cpu_fp8_gate.shape
        # GPU slot buffers
        self.gpu_g = torch.empty((gpu_slots, D, H), device=device, dtype=torch.float8_e4m3fn)
        self.gpu_u = torch.empty((gpu_slots, D, H), device=device, dtype=torch.float8_e4m3fn)
        self.gpu_d = torch.empty((gpu_slots, H, D), device=device, dtype=torch.float8_e4m3fn)
        self.gpu_sg = torch.empty(gpu_slots, device=device, dtype=torch.float32)
        self.gpu_su = torch.empty(gpu_slots, device=device, dtype=torch.float32)
        self.gpu_sd = torch.empty(gpu_slots, device=device, dtype=torch.float32)

        self.expert_to_slot: dict[int, int] = {}
        self.slot_to_expert: list[int] = [-1] * gpu_slots
        self.lru: OrderedDict[int, None] = OrderedDict()   # eid → None, ordered by touch
        self.free_slots: list[int] = list(range(gpu_slots))

        self.n_hits = 0
        self.n_misses = 0

    def warmup(self, expert_ids: list[int]) -> None:
        """Pre-load a batch of experts into GPU (e.g. predicted hot set)."""
        for eid in expert_ids[: self.N]:
            self._load(eid)

    def _load(self, eid: int) -> int:
        """Copy expert eid from CPU → next free or LRU-evicted slot. Returns slot."""
        if self.free_slots:
            slot = self.free_slots.pop()
        else:
            # Evict LRU
            evict_eid, _ = self.lru.popitem(last=False)
            slot = self.expert_to_slot.pop(evict_eid)

        self.gpu_g[slot].copy_(self.cpu_g[eid], non_blocking=True)
        self.gpu_u[slot].copy_(self.cpu_u[eid], non_blocking=True)
        self.gpu_d[slot].copy_(self.cpu_d[eid], non_blocking=True)
        self.gpu_sg[slot] = self.cpu_sg[eid].to(self.device, non_blocking=True)
        self.gpu_su[slot] = self.cpu_su[eid].to(self.device, non_blocking=True)
        self.gpu_sd[slot] = self.cpu_sd[eid].to(self.device, non_blocking=True)

        self.expert_to_slot[eid] = slot
        self.slot_to_expert[slot] = eid
        self.lru[eid] = None
        return slot

    def fetch(self, needed_ids: torch.Tensor) -> torch.Tensor:
        """Ensure all needed expert ids are on GPU. Returns slot remap tensor.

        If the batch touches more unique experts than we have GPU slots, raise
        — higher layer should batch-split or enlarge the cache.
        """
        unique_ids = torch.unique(needed_ids).tolist()
        if len(unique_ids) > self.N:
            raise RuntimeError(
                f"Batch requires {len(unique_ids)} unique experts but cache has "
                f"only {self.N} GPU slots — raise `gpu_slots` or split the batch."
            )

        # First: mark all hits as recently-used (avoids them being evicted
        # when we load misses later in this same call)
        hits_now = []
        for eid in unique_ids:
            if eid in self.expert_to_slot:
                hits_now.append(eid)
                self.lru.move_to_end(eid)
                self.n_hits += 1
        # Then load misses — LRU victims are guaranteed to not be in this batch
        for eid in unique_ids:
            if eid not in self.expert_to_slot:
                self.n_misses += 1
                self._load(eid)

        torch.cuda.synchronize()

        remap = torch.empty(self.E, device=self.device, dtype=torch.int32)
        remap.fill_(-1)
        for eid, slot in self.expert_to_slot.items():
            remap[eid] = slot
        return remap

    def stats(self) -> dict:
        total = self.n_hits + self.n_misses
        return {
            "hits": self.n_hits,
            "misses": self.n_misses,
            "hit_rate": self.n_hits / max(total, 1),
            "slots_used": len(self.expert_to_slot),
        }
