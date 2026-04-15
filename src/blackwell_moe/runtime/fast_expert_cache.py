"""FastExpertCache — GPU-resident LRU state, batched fetch, async H2D stream.

Replaces the Python-dict based `LRUExpertCache` with a pure-tensor design.
All cache state lives on the GPU; the only CPU-side work per forward is a
single Tensor.cpu() roundtrip on the miss list (small, typically <32 items).

Key improvements vs v0.7 `LRUExpertCache`
-----------------------------------------
1. State is on GPU: `expert_to_slot [E]`, `slot_to_expert [N]`, `slot_last_used [N]`
   — no Python dict, no OrderedDict LRU, no per-expert .item() call
2. `fetch_batch(all_layer_ids)` processes every layer's routing needs in one
   pass, so we pay a single sync cost per forward rather than per layer
3. CPU→GPU transfers are gathered: `batch_weights = cpu_pool[miss_cpu_idx]`
   followed by `gpu_slots[slot_idx] = batch_weights.to(gpu, non_blocking)`
   — one H2D copy instead of 3 per miss
4. H2D runs on a dedicated CUDA stream, overlapping compute on the main stream
5. LRU victim selection is `torch.argmin(slot_last_used)` on-device
"""

from __future__ import annotations

import torch


class FastExpertCache:
    def __init__(
        self,
        cpu_g: torch.Tensor,        # [E, D, H] fp8, pinned
        cpu_u: torch.Tensor,
        cpu_d: torch.Tensor,
        cpu_sg: torch.Tensor,       # [E] fp32, pinned
        cpu_su: torch.Tensor,
        cpu_sd: torch.Tensor,
        gpu_slots: int,
        device: str = "cuda",
    ):
        E, D, H = cpu_g.shape
        assert gpu_slots <= E
        self.E = E
        self.N = gpu_slots
        self.device = device

        self.cpu_g = cpu_g
        self.cpu_u = cpu_u
        self.cpu_d = cpu_d
        self.cpu_sg = cpu_sg
        self.cpu_su = cpu_su
        self.cpu_sd = cpu_sd

        self.gpu_g = torch.empty((gpu_slots, D, H), device=device, dtype=torch.float8_e4m3fn)
        self.gpu_u = torch.empty((gpu_slots, D, H), device=device, dtype=torch.float8_e4m3fn)
        self.gpu_d = torch.empty((gpu_slots, H, D), device=device, dtype=torch.float8_e4m3fn)
        self.gpu_sg = torch.empty(gpu_slots, device=device, dtype=torch.float32)
        self.gpu_su = torch.empty(gpu_slots, device=device, dtype=torch.float32)
        self.gpu_sd = torch.empty(gpu_slots, device=device, dtype=torch.float32)

        self.expert_to_slot = torch.full((E,), -1, dtype=torch.int32, device=device)
        self.slot_to_expert = torch.full((gpu_slots,), -1, dtype=torch.int32, device=device)
        self.slot_last_used = torch.zeros(gpu_slots, dtype=torch.int64, device=device)
        self.step = 0

        self.h2d_stream = torch.cuda.Stream(device=device)

        self._hits_t = torch.zeros(1, dtype=torch.int64, device=device)
        self._misses_t = torch.zeros(1, dtype=torch.int64, device=device)

    # ------------------------------------------------------------------ warmup

    def warmup(self, expert_ids: list[int]) -> None:
        ids = expert_ids[: self.N]
        miss_cpu = torch.tensor(ids, dtype=torch.long)
        slots = torch.arange(len(ids), dtype=torch.int32, device=self.device)
        self._h2d_batch(miss_cpu, slots)
        torch.cuda.current_stream().wait_stream(self.h2d_stream)
        self.expert_to_slot[miss_cpu.to(self.device)] = slots
        self.slot_to_expert[: len(ids)] = miss_cpu.to(self.device, dtype=torch.int32)
        self.slot_last_used[: len(ids)] = self.step

    # ------------------------------------------------------------------ batch

    @torch.inference_mode()
    def fetch_batch(self, all_layer_ids: torch.Tensor) -> torch.Tensor:
        """Ensure all expert ids across all layers are on GPU.

        Args
        ----
        all_layer_ids : int32/int64 tensor, any shape — contains every expert id
                        needed for an entire forward pass (all layers flattened)

        Returns a [E] remap tensor: remap[expert_id] = slot (-1 if not loaded).
        """
        self.step += 1
        unique_ids = torch.unique(all_layer_ids.to(self.device).flatten()).to(torch.int32)
        if unique_ids.numel() > self.N:
            raise RuntimeError(
                f"Batch needs {unique_ids.numel()} experts, cache has {self.N} slots."
            )

        cur_slots = self.expert_to_slot[unique_ids.to(torch.long)]  # [U] int32
        hit_mask = cur_slots >= 0
        miss_ids = unique_ids[~hit_mask]
        hit_slots = cur_slots[hit_mask]

        # Stats accumulate as GPU tensors — no .item() sync in the hot path
        self._hits_t += hit_mask.sum()
        self._misses_t += (~hit_mask).sum()

        # Touch hits (update LRU timestamp)
        if hit_slots.numel() > 0:
            self.slot_last_used[hit_slots.to(torch.long)] = self.step

        # Handle misses: find victim slots and issue H2D
        if miss_ids.numel() > 0:
            n_miss = miss_ids.numel()
            # Least-recently-used slots (top-k argmin)
            _, victim_slots = torch.topk(self.slot_last_used, n_miss, largest=False)
            victim_slots = victim_slots.to(torch.int32)

            # Invalidate old expert→slot mappings for victims (vectorized, no sync)
            # Valid old_experts (>=0) get -1; invalid are masked via where.
            old_experts = self.slot_to_expert[victim_slots.to(torch.long)].to(torch.long)
            # Clamp to valid index range; entries where old_experts == -1 write to slot 0,
            # so we use scatter with a mask-based override pattern:
            # simpler: only scatter where old_experts >= 0 via index_put_ with mask
            mask = old_experts >= 0
            safe_idx = torch.where(mask, old_experts, torch.zeros_like(old_experts))
            # Write -1 to safe_idx positions; the mask-false writes overwrite slot-0 mapping
            # which is harmless if we immediately re-overwrite it below for the fresh mapping.
            self.expert_to_slot.scatter_(
                0, safe_idx, torch.where(mask, torch.full_like(safe_idx, -1, dtype=torch.int32),
                                          self.expert_to_slot.gather(0, safe_idx)).to(torch.int32)
            )

            # Install new mapping
            self.expert_to_slot[miss_ids.to(torch.long)] = victim_slots
            self.slot_to_expert[victim_slots.to(torch.long)] = miss_ids
            self.slot_last_used[victim_slots.to(torch.long)] = self.step

            # H2D transfer on dedicated stream (overlaps with main-stream compute)
            miss_cpu = miss_ids.cpu().to(torch.long)
            self._h2d_batch(miss_cpu, victim_slots)

        # Make main stream wait on H2D stream before kernel launches
        torch.cuda.current_stream().wait_stream(self.h2d_stream)

        return self.expert_to_slot

    def _h2d_batch(self, expert_ids_cpu: torch.Tensor, slot_ids_gpu: torch.Tensor) -> None:
        with torch.cuda.stream(self.h2d_stream):
            slot_long = slot_ids_gpu.to(torch.long)
            g_batch = self.cpu_g[expert_ids_cpu]  # [M, D, H] pinned
            u_batch = self.cpu_u[expert_ids_cpu]
            d_batch = self.cpu_d[expert_ids_cpu]
            sg_batch = self.cpu_sg[expert_ids_cpu]
            su_batch = self.cpu_su[expert_ids_cpu]
            sd_batch = self.cpu_sd[expert_ids_cpu]
            self.gpu_g[slot_long] = g_batch.to(self.device, non_blocking=True)
            self.gpu_u[slot_long] = u_batch.to(self.device, non_blocking=True)
            self.gpu_d[slot_long] = d_batch.to(self.device, non_blocking=True)
            self.gpu_sg[slot_long] = sg_batch.to(self.device, non_blocking=True)
            self.gpu_su[slot_long] = su_batch.to(self.device, non_blocking=True)
            self.gpu_sd[slot_long] = sd_batch.to(self.device, non_blocking=True)

    # ------------------------------------------------------------------ stats

    def stats(self) -> dict:
        hits = int(self._hits_t.item())
        misses = int(self._misses_t.item())
        total = hits + misses
        return {
            "hits": hits,
            "misses": misses,
            "hit_rate": hits / max(total, 1),
        }
