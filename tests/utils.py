import numpy as np

SQRT2 = np.sqrt(2.0)
SQRT3 = np.sqrt(3.0)


def latlon_from_vec(v):
    return np.arcsin(v[..., 2]), np.arctan2(v[..., 1], v[..., 0])


def gd(x):
    return np.arctan(np.sinh(x))


def igd(x):
    return np.arcsinh(np.tan(x))
