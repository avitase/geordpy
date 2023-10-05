import numpy as np
import pytest

from geordpy import rhumb_line


def test_argmin():
    f = lambda x: 3 * (x - np.array([1, 2, 3])) ** 2 + 4
    bounds = np.array([[-3, 3, -4], [4, 6, 2]])

    y = rhumb_line.argmin(f, bounds=bounds, n_samples=5, n_iterations=100)
    assert y == pytest.approx(np.array([1.0, 3.0, 2.0]))
