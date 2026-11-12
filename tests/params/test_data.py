"""Module contains tests for the data utilities."""

import numpy as np

import pygeon as pg


def test_constant_tensor():
    K = np.array([[3.0, 1.0, 0.5], [1.0, 2.0, 0.3], [0.5, 0.3, 1.0]])
    tensor = pg.constant_tensor(4, K)

    assert tensor.values.shape == (3, 3, 4)
    assert np.allclose(tensor.values, K.reshape((3, 3, 1)))
