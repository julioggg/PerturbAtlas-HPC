import numpy as np
import pandas as pd
from anndata import AnnData

from perturbatlas.effects import calibrate_effects


def test_calibrate_effects(tmp_path):
    raw = AnnData(
        X=np.array(
            [
                [0.0, 0.0],
                [2.0, 2.0],
                [1.0, 1.0],
                [3.0, 3.0],
            ],
            dtype=np.float32,
        ),
        obs=pd.DataFrame(
            {"perturbation": ["control", "control", "GENEA", "GENEA"]},
            index=["c1", "c2", "c3", "c4"],
        ),
        var=pd.DataFrame(index=["g1", "g2"]),
    )
    raw_path = tmp_path / "raw.h5ad"
    raw.write_h5ad(raw_path)

    atlas = AnnData(
        X=np.array([[1.0, 1.0]], dtype=np.float32),
        obs=pd.DataFrame(
            {"n_cells": [2], "l2_effect": [np.sqrt(2)], "mean_abs_effect": [1.0]},
            index=["GENEA"],
        ),
        var=pd.DataFrame(index=["g1", "g2"]),
    )
    atlas_path = tmp_path / "atlas.h5ad"
    atlas.write_h5ad(atlas_path)

    table, summary = calibrate_effects(raw_path, atlas_path)

    assert "expected_null_l2" in table.columns
    assert "noise_adjusted_l2" in table.columns
    assert "l2_to_null_ratio" in table.columns
    assert summary.n_perturbations == 1
    assert summary.n_controls == 2
