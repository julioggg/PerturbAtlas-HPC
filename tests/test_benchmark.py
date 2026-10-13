import numpy as np

from perturbatlas.benchmark import row_pearson


def test_row_pearson():
    a = np.array([[1, 2, 3], [1, 2, 3]], dtype=float)
    b = np.array([[2, 4, 6], [3, 2, 1]], dtype=float)
    r = row_pearson(a, b)
    np.testing.assert_allclose(r, [1.0, -1.0])
