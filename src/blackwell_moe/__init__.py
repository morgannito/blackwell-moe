"""Blackwell-optimized FP8 & INT4 MoE inference kernels (sm_120).

Public API (v0.9+):
    from blackwell_moe import (
        fp8_moe_forward_v3,        # fastest FP8 MoE kernel
        int4_group_moe_forward,    # Q4_0-style INT4 kernel
        FastExpertCache,           # LRU CPU→GPU cache
        quant_fp8_e4m3,            # shared FP8 quantizer
    )
"""

__version__ = "0.19.0"

from blackwell_moe.kernels.fp8_moe_v3 import fp8_moe_forward_v3
from blackwell_moe.kernels.fp8_moe_v4 import fp8_moe_forward_v4
from blackwell_moe.kernels.fp8_quant import (
    FP8_MAX_E4M3,
    dequant_fp8_e4m3,
    quant_fp8_e4m3,
)
from blackwell_moe.kernels.int4_moe import int4_moe_forward
from blackwell_moe.kernels.int4_moe_group import int4_group_moe_forward
from blackwell_moe.kernels.routing import top_k_router
from blackwell_moe.runtime.fast_expert_cache import FastExpertCache

__all__ = [
    "FP8_MAX_E4M3",
    "FastExpertCache",
    "__version__",
    "dequant_fp8_e4m3",
    "fp8_moe_forward_v3",
    "fp8_moe_forward_v4",
    "int4_group_moe_forward",
    "int4_moe_forward",
    "quant_fp8_e4m3",
    "top_k_router",
]
