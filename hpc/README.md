# HPC execution

The HPC environment stores heavy data and intermediate artifacts that are intentionally excluded from GitHub.

## Principle

GitHub contains code, configs, documentation, compact results, and final figures.

The cluster contains:

```text
data/
models/
checkpoints/
logs/
results/intermediate/
```

## Recommended workflow

```bash
git pull
python -m pip install -e .
python scripts/00_inspect_h5ad.py data/raw/<dataset>.h5ad
```

Before each major run, record the current commit:

```bash
git rev-parse HEAD
```

All production experiments should save run metadata and avoid overwriting previous outputs.
