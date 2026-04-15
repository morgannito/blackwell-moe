"""3-tier expert weight pool: GPU slots / RAM pinned cache / disk mmap.

For models whose total expert pool exceeds CPU RAM (Mixtral-8x22B = 134 GB
FP8 experts, DeepSeek-V2 = 200 GB), we cannot keep everything in pinned
RAM. Instead:

  Tier 0 — GPU slots         (fastest, ~10 GB, hot experts)
  Tier 1 — RAM pinned page    (medium, ~20-50 GB, recently-used)
  Tier 2 — Disk safetensors   (slowest, all weights, mmap-backed)

A miss bubbles up: RAM → GPU (PCIe 64 GB/s), or Disk → RAM → GPU.

Layout on disk: each expert is stored as one safetensors file
  expert_L<layer>_E<idx>.safetensors  containing keys
    gate_q, up_q, down_q, scale_g, scale_u, scale_d  (FP8 + scales)

This way an individual expert can be mmap-loaded in isolation without
touching the rest of the model.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def save_expert_to_disk(
    out_dir: str | Path,
    layer_idx: int,
    expert_idx: int,
    g_q: torch.Tensor, u_q: torch.Tensor, d_q: torch.Tensor,
    s_g: torch.Tensor, s_u: torch.Tensor, s_d: torch.Tensor,
) -> str:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"expert_L{layer_idx:03d}_E{expert_idx:03d}.safetensors"
    save_file({
        "gate_q": g_q.cpu(),
        "up_q":   u_q.cpu(),
        "down_q": d_q.cpu(),
        "scale_g": s_g.cpu().to(torch.float32),
        "scale_u": s_u.cpu().to(torch.float32),
        "scale_d": s_d.cpu().to(torch.float32),
    }, str(fname))
    return str(fname)


class ExpertPath:
    """Locate an expert's safetensors path on disk."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def __call__(self, layer_idx: int, expert_idx: int) -> Path:
        return self.root / f"expert_L{layer_idx:03d}_E{expert_idx:03d}.safetensors"


def load_expert_to_pinned_ram(
    path: Path,
) -> dict[str, torch.Tensor]:
    """Memory-map a single expert from disk into pinned CPU buffers."""
    out: dict[str, torch.Tensor] = {}
    with safe_open(str(path), framework="pt", device="cpu") as f:
        for k in f.keys():
            t = f.get_tensor(k)
            if t.dtype == torch.float8_e4m3fn or t.dtype.is_floating_point:
                # Pin for fast H2D
                out[k] = t.pin_memory() if not t.is_pinned() else t
            else:
                out[k] = t
    return out


class ThreeTierExpertCache:
    """GPU slots (hot) + RAM page (warm) + disk (cold).

    Keys are (layer_idx, expert_idx). Lookups always return a dict of
    GPU-resident FP8 tensors {gate_q, up_q, down_q, scale_g, scale_u, scale_d}.
    """

    def __init__(
        self,
        expert_root: str | Path,
        n_layers: int,
        n_experts_per_layer: int,
        gpu_slots: int,
        ram_slots: int,
        gpu_buffer_specs: dict,   # {gate_q: (D, H), up_q: (D, H), down_q: (H, D)}
        device: str = "cuda",
    ):
        self.path_for = ExpertPath(expert_root)
        self.n_layers = n_layers
        self.n_experts = n_experts_per_layer
        self.gpu_slots = gpu_slots
        self.ram_slots = ram_slots
        self.device = device

        # Pre-allocate GPU tier buffers
        self.gpu = {
            "gate_q": torch.empty((gpu_slots,) + gpu_buffer_specs["gate_q"],
                                   device=device, dtype=torch.float8_e4m3fn),
            "up_q":   torch.empty((gpu_slots,) + gpu_buffer_specs["up_q"],
                                   device=device, dtype=torch.float8_e4m3fn),
            "down_q": torch.empty((gpu_slots,) + gpu_buffer_specs["down_q"],
                                   device=device, dtype=torch.float8_e4m3fn),
            "scale_g": torch.empty(gpu_slots, device=device, dtype=torch.float32),
            "scale_u": torch.empty(gpu_slots, device=device, dtype=torch.float32),
            "scale_d": torch.empty(gpu_slots, device=device, dtype=torch.float32),
        }

        # Maps (layer, expert) → slot
        self.gpu_map: OrderedDict[tuple[int, int], int] = OrderedDict()
        self.ram_map: OrderedDict[tuple[int, int], dict[str, torch.Tensor]] = OrderedDict()

        self.h2d_stream = torch.cuda.Stream(device=device)

        self.stats = {"gpu_hits": 0, "ram_hits": 0, "disk_loads": 0}

    def _evict_gpu_lru(self) -> int:
        if len(self.gpu_map) < self.gpu_slots:
            return len(self.gpu_map)
        old_key, slot = self.gpu_map.popitem(last=False)
        return slot

    def _evict_ram_lru(self) -> None:
        if len(self.ram_map) >= self.ram_slots:
            self.ram_map.popitem(last=False)

    def _ram_to_gpu(self, key: tuple[int, int],
                     tensors_pinned: dict[str, torch.Tensor]) -> int:
        slot = self._evict_gpu_lru()
        with torch.cuda.stream(self.h2d_stream):
            for k in ("gate_q", "up_q", "down_q"):
                self.gpu[k][slot].copy_(tensors_pinned[k], non_blocking=True)
            for k in ("scale_g", "scale_u", "scale_d"):
                self.gpu[k][slot] = tensors_pinned[k].to(self.device, non_blocking=True)
        self.gpu_map[key] = slot
        return slot

    def _disk_to_ram(self, key: tuple[int, int]) -> dict[str, torch.Tensor]:
        path = self.path_for(*key)
        tensors = load_expert_to_pinned_ram(path)
        self._evict_ram_lru()
        self.ram_map[key] = tensors
        return tensors

    def fetch(self, layer_idx: int, expert_ids: list[int]) -> torch.Tensor:
        """Ensure all listed experts are GPU-resident. Return [E_total] remap."""
        slots_for_request: list[int] = []
        for e in expert_ids:
            key = (layer_idx, e)
            if key in self.gpu_map:
                slot = self.gpu_map[key]
                self.gpu_map.move_to_end(key)
                self.stats["gpu_hits"] += 1
            elif key in self.ram_map:
                pinned = self.ram_map[key]
                self.ram_map.move_to_end(key)
                slot = self._ram_to_gpu(key, pinned)
                self.stats["ram_hits"] += 1
            else:
                pinned = self._disk_to_ram(key)
                slot = self._ram_to_gpu(key, pinned)
                self.stats["disk_loads"] += 1
            slots_for_request.append(slot)
        torch.cuda.current_stream().wait_stream(self.h2d_stream)
        return torch.tensor(slots_for_request, dtype=torch.int32, device=self.device)

    def warmup_layer(self, layer_idx: int, expert_ids: list[int]) -> None:
        """Pre-load given experts of a layer into RAM (no GPU push)."""
        for e in expert_ids:
            key = (layer_idx, e)
            if key not in self.ram_map and key not in self.gpu_map:
                self._disk_to_ram(key)
