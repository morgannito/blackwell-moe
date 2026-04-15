"""INT4 symmetric quantization with per-output-channel bf16 scale.

Packing:
  Input  : W  [K, N] bf16       (weight matrix)
  Output : W_q [K, N/2] uint8   (two 4-bit values per byte, low-nibble first)
           scales [N] bf16       (per output channel)

Dequant formula (inside kernel):
  val_int4 = (byte >> (4 * (j & 1))) & 0xF       # unpack nibble
  val_s    = val_int4 - 8                         # center to signed [-8, 7]
  val_bf16 = val_s * scales[n]
"""

from __future__ import annotations

import torch


def quantize_int4_per_channel(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize w [K, N] bf16 -> packed int4 [K, N/2] uint8, scales [N] bf16.

    Symmetric: scale[n] = max(|w[:, n]|) / 7
    Stored as unsigned nibbles with implicit offset 8 so (nibble - 8) in [-8, 7].
    """
    K, N = w.shape
    assert N % 2 == 0, "N must be even for int4 packing"

    w_f32 = w.to(torch.float32)
    per_ch_max = w_f32.abs().amax(dim=0).clamp(min=1e-6)   # [N]
    scales = (per_ch_max / 7.0).to(torch.bfloat16)          # [N]

    # Quantize to [-8, 7], offset to [0, 15]
    w_q = (w_f32 / scales.to(torch.float32)).round().clamp(-8, 7).to(torch.int32) + 8
    w_q = w_q.to(torch.uint8)  # [K, N] in [0, 15]

    # Pack pairs of nibbles: even -> low 4 bits, odd -> high 4 bits
    low = w_q[:, 0::2] & 0xF
    high = (w_q[:, 1::2] & 0xF) << 4
    packed = (low | high).to(torch.uint8)  # [K, N/2]
    return packed, scales


def dequantize_int4_per_channel(packed: torch.Tensor, scales: torch.Tensor,
                                 N: int) -> torch.Tensor:
    """Reverse of quantize_int4_per_channel, for correctness checks."""
    K, N_half = packed.shape
    assert N == N_half * 2
    low = packed & 0xF
    high = (packed >> 4) & 0xF
    w_q = torch.empty((K, N), device=packed.device, dtype=torch.int32)
    w_q[:, 0::2] = low.to(torch.int32)
    w_q[:, 1::2] = high.to(torch.int32)
    w_s = (w_q - 8).to(torch.float32)
    return (w_s * scales.to(torch.float32)).to(torch.bfloat16)
