#!/usr/bin/env bash
# Sync projet vers machine GPU Blackwell (RTX 5080) et exec bench
# Usage: GPU_HOST=user@ip ./scripts/deploy_to_gpu.sh [bench|test]
set -euo pipefail

HOST="${GPU_HOST:?set GPU_HOST=user@ip.address}"
REMOTE_DIR="${REMOTE_DIR:-~/blackwell-moe}"
MODE="${1:-bench}"

rsync -az --delete \
    --exclude='.git' --exclude='__pycache__' --exclude='.venv' \
    --exclude='models' --exclude='bench_results' \
    ./ "$HOST:$REMOTE_DIR/"

ssh "$HOST" bash -lc "'
cd $REMOTE_DIR
if [ ! -d .venv ]; then
  python3.11 -m venv .venv
  .venv/bin/pip install -U pip
  .venv/bin/pip install -e \".[bench,dev]\"
fi
case \"$MODE\" in
  bench) .venv/bin/bwmoe-bench --tokens 1024 --experts 64 --topk 8 ;;
  test)  .venv/bin/pytest -v tests/ ;;
  *)     .venv/bin/python -c \"import blackwell_moe; print(blackwell_moe.__version__)\" ;;
esac
'"
