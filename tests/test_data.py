import numpy as np
import pandas as pd
from anndata import AnnData

from perturbatlas.data import profile_h5ad


def test_profile_h5ad_target_coverage(tmp_path):
    adata = AnnData(
        X=np.array([[0, 1], [1, 0], [2, 0]], dtype=np.float32),
        obs=pd.DataFrame(
            {"perturbation": ["control", "GENEA", "GENEA"]},
            index=["c1", "c2", "c3"],
        ),
        var=pd.DataFrame(index=["GENEA", "GENEB"]),
    )
    path = tmp_path / "tiny.h5ad"
    adata.write_h5ad(path)

    profile, counts = profile_h5ad(path)

    assert profile.n_cells == 3
    assert profile.n_genes == 2
    assert profile.n_controls == 1
    assert profile.target_gene_coverage == 1
    assert profile.target_gene_coverage_fraction == 1.0
    assert counts.loc["GENEA", "n_cells"] == 2
