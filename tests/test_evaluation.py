import pandas as pd

from perturbatlas.evaluation import paired_model_statistics, stratified_model_performance


def test_paired_model_statistics():
    df = pd.DataFrame(
        {
            "pca_128": [0.8, 0.6, 0.4, 0.2, 0.1],
            "pytorch": [0.7, 0.65, 0.35, 0.25, 0.05],
            "pearson_median": [0.9, 0.8, 0.6, 0.4, 0.2],
            "identity_half_A": [0.7, 0.5, 0.3, 0.1, 0.0],
        }
    )
    paired, summary = paired_model_statistics(df, "pca_128", "pytorch", n_bootstrap=100)
    assert paired.shape[0] == 5
    assert summary["n_matched"] == 5
    strat = stratified_model_performance(df, ["identity_half_A", "pca_128", "pytorch"])
    assert strat.shape[0] >= 1
