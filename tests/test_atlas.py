import numpy as np
import pandas as pd
from anndata import AnnData

from perturbatlas.atlas import build_response_atlas


def test_response_atlas(tmp_path):
    x = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 8.0],
            [7.0, 10.0],
        ],
        dtype=np.float32,
    )
    adata = AnnData(
        X=x,
        obs=pd.DataFrame(
            {"perturbation": ["control", "control", "GENEA", "GENEA"]},
            index=["c1", "c2", "c3", "c4"],
        ),
        var=pd.DataFrame(index=["GENEA", "GENEB"]),
    )
    path = tmp_path / "tiny.h5ad"
    adata.write_h5ad(path)

    atlas, summary = build_response_atlas(path)

    assert atlas.shape == (1, 2)
    np.testing.assert_allclose(np.asarray(atlas.X), [[4.0, 6.0]])
    np.testing.assert_allclose(atlas.var["control_mean"].to_numpy(), [2.0, 3.0])
    assert atlas.obs.loc["GENEA", "n_cells"] == 2
    assert summary.n_perturbations == 1
    assert summary.n_controls == 2
