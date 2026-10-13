# PerturbAtlas-HPC

**HPC-enabled framework for uncertainty-aware benchmarking and cross-context analysis of single-cell perturbation responses.**

## Scientific question

**When is a single-cell perturbation prediction trustworthy?**

PerturbAtlas-HPC studies the reliability of perturbational transcriptomic prediction across datasets, perturbations, cell contexts, and modelling assumptions. The project focuses on forward perturbation-response modelling, uncertainty calibration, out-of-distribution evaluation, and reproducible HPC workflows.

## Public scope

This repository is intentionally limited to:

- perturbational single-cell RNA-seq analysis;
- Scanpy / AnnData preprocessing;
- statistical and machine-learning baselines;
- PyTorch representation and response models;
- uncertainty quantification and calibration;
- cross-context generalization;
- scalable HPC execution;
- reproducible benchmarking.

It does **not** implement inverse intervention design, target-state optimization, intervention search, or lab-in-the-loop control.

## Planned research questions

1. When do nonlinear models outperform simple statistical baselines?
2. Does predictive uncertainty track actual out-of-distribution error?
3. How does calibration degrade across cellular contexts?
4. Which perturbations are intrinsically easy or difficult to predict?
5. Are perturbation-response representations transferable across datasets?

## Repository layout

```text
PerturbAtlas-HPC/
├── configs/       # experiment configuration files
├── docs/          # scientific and reproducibility documentation
├── hpc/           # HPC launchers and cluster notes
├── notebooks/     # clean exploratory / demonstration notebooks
├── scripts/       # executable pipeline stages
├── src/           # reusable Python package
├── tests/         # unit and smoke tests
├── figures/       # publication-ready figures only
└── results/       # compact derived outputs only
```

Large biological datasets, model checkpoints, logs, and intermediate results are excluded from version control.

## Development status

Early development. The first milestone is a reproducible Replogle K562 Perturb-seq pipeline with QC, perturbation-level summaries, baseline models, and a PyTorch-ready data interface.

## Reproducibility principles

Each experiment should record:

- dataset and version;
- configuration;
- random seed;
- Git commit;
- runtime;
- software environment;
- model / baseline;
- evaluation metrics.

## License

MIT.
