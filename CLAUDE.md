# blackwell-moe

## Stack

Kernels d'inférence MoE FP8/INT4 en **Triton + PyTorch**, ciblés NVIDIA Blackwell grand public
(sm_120, RTX 5080/5090). Paquet Python `blackwell_moe` (layout `src/`), build **hatchling**.

- `requires-python = ">=3.11"` ; deps : `torch>=2.6`, `triton>=3.2 (sys_platform != 'win32')`,
  `transformers>=4.46`, `safetensors>=0.4.5`, `numpy`, `tqdm` (pyproject.toml).
- Extras : `fp8` (transformer-engine), `bench` (matplotlib, pandas, vllm sur Linux), `dev` (pytest, ruff, mypy).
- Entry points déclarés : `bwmoe-bench` → `blackwell_moe.bench.cli:main`, `bwmoe-run` → `blackwell_moe.runtime.cli:main`.
- Licence Apache-2.0.

## Commandes (telles qu'écrites dans le dépôt)

Makefile : `make install` (`pip install -e ".[dev]"`) · `make test` (`pytest tests/ -v`) ·
`make bench` · `make bench-matrix` (`scripts/bench_matrix.py`) · `make verify` (`scripts/verify_all.py`) ·
`make profile` (`scripts/profile_v3.py`) · `make lint` (`ruff check` + `ruff format --check` sur
`src/ tests/ scripts/`) · `make format` · `make clean`.

README / CONTRIBUTING : `pip install -e ".[bench,dev]"`, `bwmoe-bench --tokens 1024 --experts 64 --topk 8 --hidden 1536`,
`pytest tests/ -v -m "not cuda"` (sous-ensemble CPU), `GPU_HOST=user@5080-rig ./scripts/deploy_to_gpu.sh bench|test`.

Tout le bench et la quasi-totalité des tests exigent CUDA : **rien de significatif ne tourne sur le Mac.**

## Architecture

- `src/blackwell_moe/__init__.py` — API publique (`__all__`) : `fp8_moe_forward_v3/_v4`,
  `int4_moe_forward`, `int4_group_moe_forward`, `top_k_router`, `quant_fp8_e4m3`, `FastExpertCache`.
- `src/blackwell_moe/kernels/` (27 fichiers .py, 2 362 lignes) — kernels Triton. Une variante par
  fichier, versionnée : `fp8_moe.py` → `_v2` → `_v3` → `_v4`, plus `_small_e` (Mixtral E≤8),
  `_cached`, `_fastcache`, `_torch` ; briques `grouped_fp8*`, `int4_*`, `routing`, `scatter`,
  `segment_ops`, `permute`, `swiglu_*`, et `reference.py` (bf16, oracle de parité).
- `src/blackwell_moe/runtime/` (17 fichiers, 2 038 lignes) — bout-en-bout sur vrais modèles :
  loaders streaming (`loader.py` DeepSeek, `mixtral_loader.py`, `qwen_loader.py`), patches de
  `*MoE.forward` (`deepseek_patch`, `mixtral_patch`, `qwen_patch`), caches
  (`expert_cache`, `fast_expert_cache`, `disk_expert_pool` 3 niveaux GPU→RAM pinned→mmap disque),
  `cpu_offload`, et 3 CLIs : `cli.py` (DeepSeek), `mixtral_cli.py`, `streaming_cli.py` (`--family deepseek|mixtral`).
- `src/blackwell_moe/bench/cli.py` — harness bf16 vs FP8/INT4 (defaults : tokens 512, dim 2048, experts 64, topk 8, hidden 1536).
- `scripts/` (12 .py + `deploy_to_gpu.sh`) — bench, profilage, `verify_*.py`, extraction d'experts sur disque, PPL WikiText.
- `tests/` — 7 fichiers, **18 fonctions `test_`** + `conftest.py`.
- `docs/` — DESIGN.md (stratégie de scaling FP8, dispatch par permutation de tokens), PERF.md, BENCH_*.md.

## Conventions

- Ruff (line-length 100, target py311) ; `from __future__ import annotations` dans 43 des 47 fichiers de `src/`.
- Docstring de module en tête de kernel décrivant la version et la fusion effectuée ; formes de tenseurs en docstring.
- CONTRIBUTING : fonctions <25 lignes, nouveau kernel = fichier dans `kernels/` + test de parité vs `reference.py`
  + ajout dans `bench/cli.py` + entrée README/CHANGELOG ; les commits de perf doivent porter les chiffres avant/après.
- Messages de commit : `vX.Y: <résumé + mesure>` (voir `git log`). CHANGELOG.md tenu à jour par version, v0.1 → v0.21.
- Tests CUDA marqués `@pytest.mark.cuda` ; `conftest.py` les skippe automatiquement sans GPU (fixture `device`).

## Pièges

- **Version désynchronisée** : `pyproject.toml` dit `version = "0.1.0"`, `__init__.py` dit `__version__ = "0.21.0"`.
- **Windows vs Linux** : DESIGN.md liste Windows en *non-goal* (« Linux-only »), mais les mesures de PERF.md
  sont prises sous `triton-windows`, les chemins par défaut sont `J:\models\...` (`runtime/cli.py`,
  `scripts/eval_perplexity.py`, `scripts/test_streaming.py`) et le CHANGELOG v0.21 documente des contournements
  Windows (`expandable_segments` indisponible). La machine d'exécution réelle est Windows.
  `pyproject.toml` n'installe pas Triton sur win32 : CONTRIBUTING impose `triton-windows` à la main.
- **CI annoncée mais absente** : CONTRIBUTING dit « CI runs ruff + cpu-safe tests on every push », il n'y a
  aucun `.github/` ni fichier de workflow dans le dépôt.
- **`scripts/verify_all.py` se ré-exécute lui-même** : il fait `glob("verify_*.py")` sans s'exclure, donc
  `make verify` relance `verify_all.py` en sous-processus (récursion). Constat statique, non exécuté ici.
- **Qwen3.6 (résultat vedette v0.21) n'a pas de point d'entrée** : `load_qwen_streaming` et
  `patch_qwen_streaming` ne sont appelés par aucun script ni CLI du dépôt ; `runtime/qwen_patch.py`
  est de plus modifié non commité dans l'arbre de travail.
- **Versions Python contradictoires** : README `python3.11`, `deploy_to_gpu.sh` `python3.11`,
  CONTRIBUTING « Python 3.12 recommandé (3.13+ problèmes TCC) ».
- `scripts/deploy_to_gpu.sh` fait `rsync -az --delete` vers `$REMOTE_DIR` : tout ce qui est distant et
  absent en local est supprimé (exclusions : `.git`, `__pycache__`, `.venv`, `models`, `bench_results`).
- Poids et sorties de bench sont gitignorés (`models/`, `*.safetensors`, `bench_results/`, `*.csv`, `*.png`) :
  aucun artefact n'est reproductible sans la machine GPU et ses checkpoints (135 GB d'experts Mixtral, 282 GB de download).
- Les tableaux de PERF.md / README contiennent des cases `tbd` : la matrice n'est pas complète, ne pas
  la citer comme mesure exhaustive.
