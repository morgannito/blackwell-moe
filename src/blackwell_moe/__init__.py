"""Blackwell-optimized FP8 MoE inference kernels."""

__version__ = "0.1.0"

from blackwell_moe.kernels.fp8_moe import fp8_moe_forward
from blackwell_moe.kernels.routing import top_k_router

__all__ = ["fp8_moe_forward", "top_k_router"]
