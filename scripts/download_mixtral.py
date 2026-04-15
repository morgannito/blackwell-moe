"""Background download script — Mixtral-8x22B-Instruct (282 GB bf16)."""

import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from huggingface_hub import snapshot_download

print("Downloading Mixtral-8x22B-Instruct-v0.1 to J:\\models\\Mixtral-8x22B-Instruct ...")
p = snapshot_download(
    "mistralai/Mixtral-8x22B-Instruct-v0.1",
    local_dir=r"J:\models\Mixtral-8x22B-Instruct",
    allow_patterns=["*.safetensors", "*.json", "*.txt", "tokenizer*"],
    max_workers=4,
)
print(f"Done: {p}")
