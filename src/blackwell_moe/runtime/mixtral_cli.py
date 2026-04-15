"""Run Mixtral-8x22B end-to-end on a 16 GB consumer GPU via streaming experts.

Usage:
    python -m blackwell_moe.runtime.mixtral_cli \\
        --model J:\\models\\Mixtral-8x22B-Instruct \\
        --experts J:\\models\\Mixtral-8x22B-Instruct-fp8-experts \\
        --prompt "Explain quantum computing in one sentence:"
"""

from __future__ import annotations

import argparse
import time

import torch
from transformers import AutoTokenizer

from blackwell_moe.runtime.cpu_offload import offload_embed_and_lm_head
from blackwell_moe.runtime.mixtral_loader import load_mixtral_streaming


@torch.inference_mode()
def generate(model, tok, prompt: str, max_new: int = 64,
              warmup_tokens: int = 4):
    ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)

    if warmup_tokens > 0:
        print(f"\nWarmup ({warmup_tokens} tokens, primes Triton + cache)...")
        _ = model.generate(ids, max_new_tokens=warmup_tokens, do_sample=False,
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
    print(f"Peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    text = tok.decode(out[0], skip_special_tokens=True)
    return text.encode("ascii", errors="replace").decode("ascii")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Mixtral HF directory")
    p.add_argument("--experts", required=True, help="Per-expert FP8 root dir")
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--tokens", type=int, default=64)
    p.add_argument("--gpu-slots", type=int, default=4,
                   help="Number of experts pinned in VRAM")
    p.add_argument("--ram-slots", type=int, default=32,
                   help="Number of experts pinned in CPU RAM")
    p.add_argument("--vram-cap", type=float, default=0.85,
                   help="Fraction of total VRAM reserved for this process")
    args = p.parse_args()

    if args.vram_cap < 1.0:
        torch.cuda.set_per_process_memory_fraction(args.vram_cap)
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM cap: {args.vram_cap*100:.0f}% = {total*args.vram_cap:.2f} GB "
              f"(total {total:.2f} GB)")

    tok = AutoTokenizer.from_pretrained(args.model)
    model, cache = load_mixtral_streaming(
        args.model, args.experts,
        gpu_slots=args.gpu_slots, ram_slots=args.ram_slots,
    )
    freed = offload_embed_and_lm_head(model)
    print(f"CPU-offloaded embed/lm_head: -{freed} MB GPU")
    print(f"VRAM after CPU offload: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    print("\n=== Prompt ===")
    print(args.prompt)
    print("\n=== Output ===")
    text = generate(model, tok, args.prompt, max_new=args.tokens)
    print(text)
    print(f"\nFinal cache stats: {cache.stats}")


if __name__ == "__main__":
    main()
