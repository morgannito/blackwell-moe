"""End-to-end runtime: load DeepSeek-V2-Lite, patch MoE with FP8 kernels, generate."""

from __future__ import annotations

import argparse
import time

import torch
from transformers import AutoTokenizer

from blackwell_moe.runtime.deepseek_patch import patch_deepseek_moe_with_store
from blackwell_moe.runtime.loader import load_deepseek_fp8_streaming
from blackwell_moe.runtime.shared_expert_fp8 import patch_shared_experts


def load_model(path: str, patch: bool = True, device: str = "cuda"):
    print(f"Loading tokenizer from {path}")
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model, fp8_store = load_deepseek_fp8_streaming(path, device=device)
    if patch:
        n_shared = patch_shared_experts(model)
        print(f"Patched {n_shared} shared experts to FP8")
        n = patch_deepseek_moe_with_store(model, fp8_store)
        print(f"Patched {n} DeepseekV2MoE layers with FP8 kernels")
        torch.cuda.empty_cache()
        print(f"VRAM after patching: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    return tok, model


@torch.inference_mode()
def generate(model, tok, prompt: str, max_new_tokens: int = 64):
    ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
    t0 = time.perf_counter()
    out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    n_new = out.shape[1] - ids.shape[1]
    print(f"\nGenerated {n_new} tokens in {dt:.2f}s = {n_new / dt:.1f} tok/s")
    print(f"Peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    text = tok.decode(out[0], skip_special_tokens=True)
    # Sanitize for Windows cp1252 console
    return text.encode("ascii", errors="replace").decode("ascii")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=r"J:\models\DeepSeek-V2-Lite-Chat")
    p.add_argument("--prompt", default="What is the capital of France? Answer in one sentence.")
    p.add_argument("--tokens", type=int, default=64)
    p.add_argument("--no-patch", action="store_true",
                   help="bf16 baseline, no FP8 patching")
    args = p.parse_args()

    tok, model = load_model(args.model, patch=not args.no_patch)
    print("\n=== Prompt ===")
    print(args.prompt)
    print("\n=== Output ===")
    text = generate(model, tok, args.prompt, args.tokens)
    print(text)


if __name__ == "__main__":
    main()
