import numpy as np


def _refine_bounds(f, *, bounds, n_samples):
    a, b = bounds
    x = np.linspace(a, b, n_samples, axis=0)
    y = f(x)

    i = np.argmin(y, axis=0)
    low = np.maximum(0, i - 1)
    high = np.minimum(n_samples - 1, i + 1)

    return np.take_along_axis(x, np.stack((low, high), axis=0), axis=0)


def argmin(f, *, bounds, n_iterations=1, n_samples=2):
    for _ in range(n_iterations):
        bounds = _refine_bounds(f, bounds=bounds, n_samples=n_samples)

    y = f(bounds)
    idx = np.argmin(y, axis=0, keepdims=True)
    return np.take_along_axis(bounds, idx, axis=0).flatten()
