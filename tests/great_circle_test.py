import functools

import numpy as np
import pytest
import scipy.optimize
import utils
from numpy import pi
from scipy.spatial.transform import Rotation as rotation
from utils import SQRT2, SQRT3

from geordpy.great_circle import cos_distance, cos_distance_segment


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
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (-0.25, -SQRT3 / 4, -SQRT3 / 2),
            -0.25,
        ),
        (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, -1.0),
            0.0,
        ),
    ],
)
@pytest.mark.parametrize("rotate", [True, False])
def test_points(points, rotate):
    rot = rotation.random(random_state=42).as_matrix() if rotate else np.eye(3)

    a, b, c, exp = points
    lat1, lon1 = utils.latlon_from_vec(rot @ a)
    lat2, lon2 = utils.latlon_from_vec(rot @ b)
    lat3, lon3 = utils.latlon_from_vec(rot @ c)
    cos_dist = cos_distance_segment(
        np.array([lat3]), np.array([lon3]), lat1=lat1, lon1=lon1, lat2=lat2, lon2=lon2
    ).squeeze(-1)

    assert cos_dist == pytest.approx(exp)


def great_circle(lon, *, latN, lonN):
    return np.arctan(np.tan(latN) * np.cos(lon - lonN))


def get_bound(a, b):
    if b < a:
        a, b = b, a

    d = b - a
    if d > pi:
        return b, a + 2 * pi

    return a, b


@pytest.mark.parametrize("batch_size", [1, 5, 10])
@pytest.mark.parametrize("seed", list(range(1, 101)))
def test_random_points(batch_size, seed):
    rng = np.random.default_rng(seed)

    az0 = rng.uniform(-pi, pi)
    latN = pi / 2 - np.abs(az0)
    lonN = rng.uniform(-pi, pi)

    lon1, lon2 = rng.uniform(-pi, pi, size=2)
    lat1 = great_circle(lon1, latN=latN, lonN=lonN)
    lat2 = great_circle(lon2, latN=latN, lonN=lonN)

    p = np.stack(
        [
            R.as_matrix() @ np.array([1.0, 0.0, 0.0])
            for R in rotation.random(random_state=seed, num=batch_size)
        ],
        axis=0,
    )
    lat, lon = utils.latlon_from_vec(p)

    cos_dist = cos_distance_segment(
        lat, lon, lat1=lat1, lon1=lon1, lat2=lat2, lon2=lon2
    )

    def _loss(x, *, batch_idx):
        latA = great_circle(x, latN=latN, lonN=lonN)
        latB = lat[batch_idx]
        lonAB = x - lon[batch_idx]
        return -cos_distance(lat1=latA, lat2=latB, dlon=lonAB)

    bound = get_bound(lon1, lon2)
    for i in range(batch_size):
        res = scipy.optimize.minimize_scalar(
            functools.partial(_loss, batch_idx=i), bounds=bound
        )
        assert res.success

        if (fun := _loss(lon1, batch_idx=i)) < res.fun:
            res.fun = fun
        elif (fun := _loss(lon2, batch_idx=i)) < res.fun:
            res.fun = fun

        assert cos_dist[i] > -res.fun or cos_dist[i] == pytest.approx(-res.fun), f"{i=}"
