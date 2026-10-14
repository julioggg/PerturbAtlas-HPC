import numpy as np

from perturbatlas.uncertainty import (
    pairwise_correlation_disagreement,
    relative_ensemble_variance,
)


def test_normalized_uncertainty_metrics():
    preds = np.array([
        [[1.0, 2.0, 3.0], [1.0, 0.0, -1.0]],
        [[1.1, 2.1, 3.1], [-1.0, 0.0, 1.0]],
        [[0.9, 1.9, 2.9], [0.0, 1.0, 0.0]],
    ])
    rel = relative_ensemble_variance(preds)
    dis = pairwise_correlation_disagreement(preds)
    assert rel.shape == (2,)
    assert dis.shape == (2,)
    assert dis[1] > dis[0]
