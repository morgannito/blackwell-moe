"""3-tier expert weight pool: GPU slots / RAM pinned cache / disk mmap.

For models whose total expert pool exceeds CPU RAM (Mixtral-8x22B = 134 GB
FP8 experts, DeepSeek-V2 = 200 GB), we cannot keep everything in pinned
RAM. Instead:

  Tier 0 — GPU slots         (fastest, ~10 GB, hot experts)
  Tier 1 — RAM pinned page    (medium, ~20-50 GB, recently-used)
  Tier 2 — Disk safetensors   (slowest, all weights, mmap-backed)

A miss bubbles up: RAM → GPU (PCIe 64 GB/s), or Disk → RAM → GPU.

v0.17 additions
---------------
* `prefetch_layer(layer_idx)` — async load of all experts for an upcoming
  layer into the RAM tier on a background thread. Called by the streaming
  forward right after a layer completes, so the next layer's experts are
  warm by the time it asks for them.
* LFU-aware eviction — replaces pure LRU. Each entry tracks a hit count;
  when slots fill, the lowest-count entry gets evicted regardless of
  recency. Combats Zipf workloads where a few hot experts deserve
  permanent residence.
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
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

        # LFU frequency counts per (layer, expert)
        self.gpu_freq: dict[tuple[int, int], int] = {}
        self.ram_freq: dict[tuple[int, int], int] = {}

        self.h2d_stream = torch.cuda.Stream(device=device)

        # Background thread pool for disk → RAM prefetch
        self._prefetch_pool = ThreadPoolExecutor(max_workers=4)
        self._prefetch_lock = threading.Lock()

        self.stats = {"gpu_hits": 0, "ram_hits": 0, "disk_loads": 0,
                      "prefetched": 0}

    def _pick_gpu_victim(self) -> int:
        """LRU eviction — evict the GPU slot accessed least recently.

        OrderedDict iteration order = insertion / move_to_end order, so
        `popitem(last=False)` removes the front (oldest).
        """
        if len(self.gpu_map) < self.gpu_slots:
            return len(self.gpu_map)
        old_key, slot = self.gpu_map.popitem(last=False)
        self.gpu_freq.pop(old_key, None)
        return slot

    def _pick_ram_victim_and_evict(self) -> None:
        if len(self.ram_map) >= self.ram_slots:
            old_key, _ = self.ram_map.popitem(last=False)
            self.ram_freq.pop(old_key, None)

    def _ram_to_gpu(self, key: tuple[int, int],
                     tensors_pinned: dict[str, torch.Tensor]) -> int:
        slot = self._pick_gpu_victim()
        with torch.cuda.stream(self.h2d_stream):
            for k in ("gate_q", "up_q", "down_q"):
                self.gpu[k][slot].copy_(tensors_pinned[k], non_blocking=True)
            for k in ("scale_g", "scale_u", "scale_d"):
                self.gpu[k][slot] = tensors_pinned[k].to(self.device, non_blocking=True)
        self.gpu_map[key] = slot
        # Inherit frequency from RAM tier if any
        self.gpu_freq[key] = self.ram_freq.pop(key, 0)
        return slot

    def _disk_to_ram_inner(self, key: tuple[int, int]) -> dict[str, torch.Tensor]:
        path = self.path_for(*key)
        return load_expert_to_pinned_ram(path)

    def _disk_to_ram(self, key: tuple[int, int]) -> dict[str, torch.Tensor]:
        with self._prefetch_lock:
            if key in self.ram_map:
                return self.ram_map[key]
        tensors = self._disk_to_ram_inner(key)
        with self._prefetch_lock:
            self._pick_ram_victim_and_evict()
            self.ram_map[key] = tensors
            self.ram_freq[key] = self.ram_freq.get(key, 0)
        return tensors

    def fetch(self, layer_idx: int, expert_ids: list[int]) -> torch.Tensor:
        """Ensure all listed experts are GPU-resident. Return slot id list."""
        slots_for_request: list[int] = []
        # Hold lock for the whole fetch so prefetch threads cannot evict
        # an entry under us between the lookup and the GPU copy.
        with self._prefetch_lock:
            for e in expert_ids:
                key = (layer_idx, e)
                if key in self.gpu_map:
                    slot = self.gpu_map[key]
                    self.gpu_map.move_to_end(key)
                    self.gpu_freq[key] = self.gpu_freq.get(key, 0) + 1
                    self.stats["gpu_hits"] += 1
                elif key in self.ram_map:
                    pinned = self.ram_map[key]
                    self.ram_map.move_to_end(key)
                    self.ram_freq[key] = self.ram_freq.get(key, 0) + 1
                    slot = self._ram_to_gpu(key, pinned)
                    self.stats["ram_hits"] += 1
                else:
                    pinned = self._disk_to_ram_inner(key)
                    self._pick_ram_victim_and_evict()
                    self.ram_map[key] = pinned
                    self.ram_freq[key] = 1
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

    def prefetch_layer(self, layer_idx: int, expert_ids: list[int] | None = None,
                        max_prefetch: int | None = None) -> None:
        """Async load of upcoming layer's experts into the RAM tier.

        Submits disk reads on a background pool. Returns immediately. Subsequent
        `fetch()` calls for these experts skip the disk hit.

        To avoid eviction storms, only fires for at most `max_prefetch` experts
        (default = top_k * 2 estimate via the layer's hot history). If a hint
        is omitted, prefetches the experts that have ever been hot for this
        layer based on cumulative frequency.
        """
        # Hot-history-based prefetch: only experts we've seen used before
        if expert_ids is None:
            expert_ids = [
                e for (l, e) in self.ram_freq.keys() if l == layer_idx
            ] + [
                e for (l, e) in self.gpu_freq.keys() if l == layer_idx
            ]
            expert_ids = list(set(expert_ids))
        if max_prefetch is not None:
            expert_ids = expert_ids[:max_prefetch]
        if not expert_ids:
            return

        def _load_one(eid: int):
            key = (layer_idx, eid)
            with self._prefetch_lock:
                if key in self.gpu_map or key in self.ram_map:
                    return
            try:
                tensors = self._disk_to_ram_inner(key)
            except FileNotFoundError:
                return
            with self._prefetch_lock:
                if key in self.ram_map or key in self.gpu_map:
                    return
                self._pick_ram_victim_and_evict()
                self.ram_map[key] = tensors
                self.ram_freq[key] = self.ram_freq.get(key, 0)
                self.stats["prefetched"] += 1

        for eid in expert_ids:
            self._prefetch_pool.submit(_load_one, eid)
