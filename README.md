# PerturbAtlas-HPC

**HPC-enabled framework for uncertainty-aware benchmarking of single-cell perturbation responses.**

## Scientific question

**When is a single-cell perturbation prediction trustworthy?**

PerturbAtlas-HPC studies how perturbation-response signal strength, intrinsic reproducibility, model class, and predictive uncertainty jointly determine whether a held-out single-cell perturbation response can be predicted reliably.

The current benchmark uses a harmonized Replogle K562 Perturb-seq dataset with **308,646 cells, 1,971 perturbations, and 8,563 genes**.

## Current results

The present pipeline establishes five main findings:

1. **Intrinsic perturbation reproducibility is highly heterogeneous.** Median split-half Pearson reproducibility across perturbations is approximately **0.26**.
2. **Effective signal-to-noise is strongly associated with reproducibility.** Noise-adjusted effect strength remains strongly associated with split-half reproducibility after controlling for perturbation sample size.
3. **Low-dimensional denoising is competitive with a neural model on held-out perturbations.** On a perturbation-identity split (1,380 train / 296 validation / 295 test), PCA-128 and a PyTorch denoising autoencoder achieve similar test performance; their paired difference is not statistically significant.
4. **Prediction quality depends strongly on intrinsic reliability.** For highly reproducible perturbations, held-out Pearson performance approaches ~0.8-0.9, whereas weakly reproducible perturbations remain difficult for either model class.
5. **Scale-normalized ensemble uncertainty identifies unreliable predictions.** Relative ensemble variance and directional ensemble disagreement correlate strongly with correlation-based prediction error and support selective prediction: retaining only the 20% least-uncertain test perturbations increases median Pearson from ~0.37 to ~0.80.

These results support evaluating perturbation-prediction systems relative to the reliability of the underlying biological response rather than using model accuracy alone.

## Benchmark design

The current analysis includes:

- perturbation-level response atlas construction;
- sample-size-aware effect calibration;
- empirical matched-size control nulls;
- on-target biological QC;
- split-half intrinsic reproducibility;
- sample-size deconfounding;
- held-out perturbation train/validation/test splitting;
- PCA baselines;
- PyTorch denoising autoencoder;
- paired bootstrap and Wilcoxon model comparison;
- deep ensembles;
- raw and scale-normalized uncertainty diagnostics;
- selective prediction / abstention analysis.

### Held-out split

Perturbation identities are separated into:

- **1,380 train**
- **296 validation**
- **295 test**

This prevents the same perturbation identity from appearing across train and test partitions.

## Key predictive results

| Method | Median Pearson on test |
|---|---:|
| Identity half-A baseline | 0.256 |
| PCA-128 | 0.353 |
| PyTorch denoising autoencoder | 0.345 |
| 5-member deep ensemble | 0.365 |

The PCA-128 versus PyTorch paired median difference is small (~0.0018), its bootstrap 95% interval crosses zero, and the paired Wilcoxon test is not significant.

## Uncertainty calibration

Raw ensemble variance is not a valid failure score in this setting because it is confounded by perturbation-response magnitude.

Two scale-aware alternatives perform substantially better:

- **Relative ensemble variance**
  - Spearman with correlation error: ~0.884
  - Failure-detection AUC: ~0.958
- **Pairwise directional disagreement**
  - Spearman with correlation error: ~0.889
  - Failure-detection AUC: ~0.957

Selective prediction using normalized uncertainty increases median Pearson from approximately **0.365 at 100% coverage** to approximately **0.80 at 20% coverage**.

A current limitation is that these uncertainty scores calibrate correlation-based error much better than RMSE, indicating that uncertainty is metric-dependent rather than universally calibrated.

## Technology

- Python 3.10+
- AnnData / Scanpy
- NumPy / SciPy / pandas
- scikit-learn
- PyTorch
- pytest
- Linux / Singularity
- HPC execution

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

Large biological datasets, model checkpoints, and intermediate matrices are intentionally excluded from version control.

## Reproducibility

The pipeline records:

- dataset path / source metadata;
- Git commit;
- random seeds;
- runtime environment;
- model configuration;
- evaluation metrics;
- derived tables and figures.

The current test suite contains **13 passing tests** covering the core analysis and evaluation utilities.

## Current scope and limitations

This repository currently focuses on **forward perturbation-response prediction and reliability assessment**. The present benchmark uses one harmonized K562 dataset, so cross-context validation remains a next step. Cell-level observations are not independent biological replicates; split-half analysis should therefore be interpreted as an intrinsic cell-sampling reproducibility benchmark rather than biological replicate-level validation.

Planned extensions include:

- external validation in an additional perturbation dataset;
- cross-cell-context generalization;
- stronger OOD evaluation;
- calibration across datasets;
- additional uncertainty methods.

## Public scope

This repository is intentionally limited to:

- perturbational single-cell RNA-seq analysis;
- forward perturbation-response modelling;
- uncertainty quantification and calibration;
- out-of-distribution evaluation;
- cross-context generalization;
- reproducible HPC workflows.

It does **not** implement inverse intervention design, target-state optimization, intervention search, or lab-in-the-loop control.

## License

MIT.
