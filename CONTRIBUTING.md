# Contributing

## Local development

```bash
git clone https://github.com/morgannito/blackwell-moe
cd blackwell-moe
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

On Windows native: install `triton-windows` instead of `triton`. Python 3.12
recommended (3.13+ TCC issues, see `docs/BENCH_v0.1.md`).

## Running tests

```bash
pytest tests/ -v               # cuda tests auto-skip without GPU
pytest tests/ -v -m "not cuda" # cpu-only subset
```

CI runs ruff + cpu-safe tests on every push.

## Benchmarking

```bash
python -m blackwell_moe.bench.cli --tokens 1024 --dim 2048 --experts 64 --topk 8 --hidden 1536
python scripts/bench_matrix.py    # full kernel × shape matrix → CSV
python scripts/profile_v3.py      # torch.profiler hot-spot dump
```

Bench results live in `bench_results/`. When committing performance changes,
include before/after numbers in the commit message.

## Adding a kernel

1. Implement in `src/blackwell_moe/kernels/<name>.py`
2. Add a forward pass in `<name>_moe.py` that wires routing + dispatch
3. Add unit tests in `tests/test_<name>.py` — round-trip and parity vs bf16 reference
4. Add to `bench/cli.py` so it shows up in the matrix
5. Document in `README.md` and `CHANGELOG.md`

## Style

- `ruff check` and `ruff format` (config in `pyproject.toml`)
- Functions <25 lines where possible; comments only when intent isn't obvious
- Tensor shapes in docstrings or first-line comments
