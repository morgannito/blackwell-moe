"""End-to-end streaming generation CLI (DeepSeek-V2-Lite or Mixtral)."""

from __future__ import annotations

import argparse
import time

import torch
from transformers import AutoTokenizer

from blackwell_moe.runtime.cpu_offload import offload_embed_and_lm_head


@torch.inference_mode()
def generate(model, tok, prompt: str, max_new: int = 64, warmup: int = 4):
    ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
    if warmup > 0:
        print(f"\nWarmup ({warmup} tokens, primes Triton + cache)...")
        _ = model.generate(ids, max_new_tokens=warmup, do_sample=False,
                           pad_token_id=tok.eos_token_id)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    out = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    n_new = out.shape[1] - ids.shape[1]
    print(f"\nGenerated {n_new} tokens in {dt:.2f}s = {n_new / dt:.2f} tok/s")
    print(f"Peak VRAM (post-warmup): {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    text = tok.decode(out[0], skip_special_tokens=True)
    return text.encode("ascii", errors="replace").decode("ascii")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--family", choices=["deepseek", "mixtral"], default="deepseek")
    p.add_argument("--model", required=True)
    p.add_argument("--experts", required=True)
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--tokens", type=int, default=64)
    p.add_argument("--gpu-slots", type=int, default=16)
    p.add_argument("--ram-slots", type=int, default=32)
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    if args.family == "deepseek":
        from blackwell_moe.runtime.deepseek_streaming import load_deepseek_streaming
        model, cache = load_deepseek_streaming(
            args.model, args.experts,
            gpu_slots=args.gpu_slots, ram_slots=args.ram_slots,
        )
    else:
        from blackwell_moe.runtime.mixtral_loader import load_mixtral_streaming
        model, cache = load_mixtral_streaming(
            args.model, args.experts,
            gpu_slots=args.gpu_slots, ram_slots=args.ram_slots,
        )

    freed = offload_embed_and_lm_head(model)
    print(f"CPU-offloaded embed/lm_head: -{freed} MB GPU")
    print(f"VRAM after CPU offload: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    print(f"\n=== Prompt ===\n{args.prompt}\n\n=== Output ===")
    text = generate(model, tok, args.prompt, max_new=args.tokens)
    print(text)
    print(f"\nFinal cache stats: {cache.stats}")


if __name__ == "__main__":
    main()
