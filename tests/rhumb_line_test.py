import numpy as np
import pytest
import utils
from numpy import pi
from utils import SQRT2

from geordpy import great_circle, rhumb_line


def f(x):
    return 3 * (x - np.array([1, 2, 3])) ** 2 + 4


def test_minimize():
    bounds = np.array([[-3, 3, -4], [4, 6, 2]])

    y = rhumb_line.minimize(f, bounds=bounds, n_samples=5, n_iterations=100)

    x0 = np.array([1.0, 3.0, 2.0])
    assert y == pytest.approx(f(x0))


@pytest.mark.parametrize(
    "points",
    [
        (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (-1.0, 0.0, 0.0),
            0.0,
        ),
        (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.5, 0.5, 1 / SQRT2),
            1 / SQRT2,
        ),
        (
            (1 / SQRT2, -1 / SQRT2, 0.0),
            (1 / SQRT2, 1 / SQRT2, 0.0),
            (1 / SQRT2, 0.0, 1 / SQRT2),
            1 / SQRT2,
        ),
    ],
)
@pytest.mark.parametrize("rotate", [True, False])
def test_points(points, rotate):
    rot = (
        np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
        if rotate
        else np.eye(3)
    )

    a, b, c, exp = points
    lat1, lon1 = utils.latlon_from_vec(rot @ a)
    lat2, lon2 = utils.latlon_from_vec(rot @ b)
    lat3, lon3 = utils.latlon_from_vec(rot @ c)

    cos_dist = rhumb_line.cos_distance_segment(
        np.array([lat3]),
        np.array([lon3]),
        lat1=lat1,
        lon1=lon1,
        lat2=lat2,
        lon2=lon2,
        n_iterations=5,
        n_samples=10,
    ).squeeze(-1)

    assert cos_dist == pytest.approx(exp)


def gd(x):
    return np.arctan(np.sinh(x))


def igd(x):
    return np.arcsinh(np.tan(x))


@pytest.mark.parametrize("batch_size", [1, 5, 10])
@pytest.mark.parametrize("seed", list(range(1, 101)))
def test_random_points(batch_size, seed):
    rng = np.random.default_rng(seed)

    lat1, lat2 = rng.uniform(-np.deg2rad(85), np.deg2rad(85), size=2)
    lon1, lon2 = rng.uniform(-pi, pi, size=2)

    lat3 = rng.uniform(-np.deg2rad(85), np.deg2rad(85), size=batch_size)
    lon3 = rng.uniform(-pi, pi, size=batch_size)

    cos_dist = rhumb_line.cos_distance_segment(
        lat3,
        lon3,
        lat1=lat1,
        lon1=lon1,
        lat2=lat2,
        lon2=lon2,
        n_iterations=2,
        n_samples=2048,
    )

    for i, (latB, lonB, cos_d) in enumerate(zip(lat3, lon3, cos_dist)):
        N = 10_000

        if lat1 - lat2 < abs(lon1 - lon2):
            lonA = np.linspace(lon1, lon2, N)

            k = (lonA - lon1) / (lon2 - lon1)
            latA = gd(igd(lat1) + k * (igd(lat2) - igd(lat1)))
        else:
            latA = np.linspace(lat1, lat2, N)

            k = (igd(latA) - igd(lat1)) / (igd(lat2) - igd(lat1))
            lonA = lon1 + k * (lon2 - lon1)

        cos_dAB = great_circle.cos_distance(lat1=latA, lat2=latB, dlon=lonA - lonB)
        max_cos_dAB = np.max(cos_dAB)
        assert cos_dist[i] > max_cos_dAB - 1e-10
