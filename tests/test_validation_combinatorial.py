import anndata as ad
import numpy as np
import pandas as pd

from perturbatlas.validation import target_response_summary, target_response_table


def test_target_response_table_empty_for_combinatorial_labels(tmp_path):
    atlas = ad.AnnData(
        X=np.array([[1.0, -1.0], [0.5, 0.2]], dtype=np.float32),
        obs=pd.DataFrame({"n_cells": [50, 60]}, index=["GENEA+GENEB", "GENEC+GENED"]),
        var=pd.DataFrame({"control_mean": [0.2, 0.3]}, index=["GENEA", "GENEB"]),
    )
    atlas.layers["mean_expression"] = np.array([[1.2, -0.7], [0.7, 0.5]], dtype=np.float32)
    path = tmp_path / "combo.h5ad"
    atlas.write_h5ad(path)

    table = target_response_table(path)
    assert table.empty
    assert table.index.name == "perturbation"

    summary = target_response_summary(table)
    assert summary["n_targets_measured"] == 0
