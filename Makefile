.PHONY: install test bench bench-matrix verify profile lint clean

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

bench:
	python -m blackwell_moe.bench.cli --tokens 1024 --dim 2048 --experts 64 --topk 8 --hidden 1536

bench-matrix:
	python scripts/bench_matrix.py

verify:
	python scripts/verify_all.py

profile:
	python scripts/profile_v3.py

lint:
	ruff check src/ tests/ scripts/
	ruff format --check src/ tests/ scripts/

format:
	ruff format src/ tests/ scripts/

clean:
	rm -rf build/ dist/ *.egg-info src/*.egg-info
	find . -name "__pycache__" -exec rm -rf {} +
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache .ruff_cache .mypy_cache bench_results/
