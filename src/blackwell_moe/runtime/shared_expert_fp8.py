"""FP8 shared expert (dense MLP with SwiGLU) via torch._scaled_mm.

Shared experts run on every token (no routing). They're regular MLPs with
SwiGLU. We quantize their weights to FP8 once at load and use torch native
scaled_mm for inference — no batched/grouped dispatch needed since there's
no expert selection.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _quant_fp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    amax = x.abs().amax().clamp(min=1e-4).to(torch.float32)
    scale = (448.0 / amax).to(torch.float32)
    q = (x.to(torch.float32) * scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return q, scale


def _scaled_mm(a_fp8, b_fp8, sa, sb, out_dtype=torch.bfloat16):
    return torch._scaled_mm(
        a_fp8, b_fp8,
        scale_a=(1.0 / sa).to(torch.float32),
        scale_b=(1.0 / sb).to(torch.float32),
        out_dtype=out_dtype,
        use_fast_accum=True,
    )


class FP8SharedExpert(nn.Module):
    """Drop-in FP8 replacement for DeepseekV2MLP used as shared_experts."""

    def __init__(self, mlp: nn.Module):
        super().__init__()
        # DeepseekV2MLP: gate_proj, up_proj [H_inter, D], down_proj [D, H_inter]
        # torch._scaled_mm wants b in column-major → pre-transpose and store
        gp = mlp.gate_proj.weight.detach().t().contiguous()  # [D, H_inter]
        up = mlp.up_proj.weight.detach().t().contiguous()
        dp = mlp.down_proj.weight.detach().t().contiguous()  # [H_inter, D]

        g_q, sg = _quant_fp8(gp.to(torch.bfloat16))
        u_q, su = _quant_fp8(up.to(torch.bfloat16))
        d_q, sd = _quant_fp8(dp.to(torch.bfloat16))

        self.register_buffer("w_gate", g_q.t().contiguous().t())  # keep col-major
        self.register_buffer("w_up", u_q.t().contiguous().t())
        self.register_buffer("w_down", d_q.t().contiguous().t())
        self.register_buffer("s_gate", sg)
        self.register_buffer("s_up", su)
        self.register_buffer("s_down", sd)

        self.hidden_size = gp.shape[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        flat = x.reshape(-1, self.hidden_size).to(torch.bfloat16)
        x_fp8, sx = _quant_fp8(flat)

        gate = _scaled_mm(x_fp8, self.w_gate, sx, self.s_gate)
        up = _scaled_mm(x_fp8, self.w_up, sx, self.s_up)
        h = F.silu(gate) * up

        h_fp8, sh = _quant_fp8(h)
        y = _scaled_mm(h_fp8, self.w_down, sh, self.s_down)
        return y.reshape(shape)


def patch_shared_experts(model: nn.Module) -> int:
    """Replace DeepseekV2MLP used as .shared_experts with FP8 version."""
    patched = 0
    for module in model.modules():
        if hasattr(module, "shared_experts") and module.__class__.__name__ == "DeepseekV2MoE":
            if module.shared_experts.__class__.__name__ == "DeepseekV2MLP":
                module.shared_experts = FP8SharedExpert(module.shared_experts).to(
                    next(module.shared_experts.parameters()).device
                )
                patched += 1
    return patched
