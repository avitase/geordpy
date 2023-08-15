import functools

import numpy as np
import pytest
import scipy.optimize
from scipy.spatial.transform import Rotation as rotation

from geordpy.great_circle import cos_distance_segment

SQRT2 = np.sqrt(2.0)
SQRT3 = np.sqrt(3.0)


def latlon_from_vec(v):
    return np.arcsin(v[..., 2]), np.arctan2(v[..., 1], v[..., 0])


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
            (0.5, 0.5, 1.0 / SQRT2),
            1 / SQRT2,
        ),
        (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (-0.25, -SQRT3 / 4.0, -SQRT3 / 2.0),
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
    lat1, lon1 = latlon_from_vec(rot @ a)
    lat2, lon2 = latlon_from_vec(rot @ b)
    lat3, lon3 = latlon_from_vec(rot @ c)
    cos_dist = cos_distance_segment(
        np.array([lat3]),
        np.array([lon3]),
        lat1=np.array([lat1]),
        lon1=np.array([lon1]),
        lat2=np.array([lat2]),
        lon2=np.array([lon2]),
    ).squeeze(-1)

    assert cos_dist == pytest.approx(exp)


def great_circle(lon, *, latN, lonN):
    return np.arctan(np.tan(latN) * np.cos(lon - lonN))


def get_bound(a, b):
    if b < a:
        a, b = b, a

    d = b - a
    if d > np.pi:
        return b, a + 2.0 * np.pi

    return a, b


@pytest.mark.parametrize("batch_size", [5])
@pytest.mark.parametrize("seed", list(range(1, 101)))
def test_random_points(batch_size, seed):
    rng = np.random.default_rng(seed)

    az0 = rng.uniform(-np.pi, np.pi, size=batch_size)
    latN = np.pi / 2 - np.abs(az0)
    lonN = rng.uniform(-np.pi, np.pi, size=batch_size)

    lon1, lon2 = rng.uniform(-np.pi, np.pi, size=(2, batch_size))
    lat1 = great_circle(lon1, latN=latN, lonN=lonN)
    lat2 = great_circle(lon2, latN=latN, lonN=lonN)

    p = np.stack(
        [
            R.as_matrix() @ np.array([1.0, 0.0, 0.0])
            for R in rotation.random(random_state=seed, num=batch_size)
        ],
        axis=0,
    )
    lat, lon = latlon_from_vec(p)

    cos_dist = cos_distance_segment(
        lat, lon, lat1=lat1, lon1=lon1, lat2=lat2, lon2=lon2
    )

    def loss(x, *, batch_idx):
        lat1 = great_circle(x, latN=latN[batch_idx], lonN=lonN[batch_idx])
        lat2 = lat[batch_idx]
        dlon = x - lon[batch_idx]

        return -np.sin(lat1) * np.sin(lat2) - np.cos(lat1) * np.cos(lat2) * np.cos(dlon)

    for i in range(batch_size):
        bound = get_bound(lon1[i], lon2[i])
        res = scipy.optimize.minimize_scalar(
            functools.partial(loss, batch_idx=i), bounds=bound
        )
        assert res.success

        if (fun := loss(lon1[i], batch_idx=i)) < res.fun:
            res.fun = fun
        elif (fun := loss(lon2[i], batch_idx=i)) < res.fun:
            res.fun = fun

        assert cos_dist[i] > -res.fun or cos_dist[i] == pytest.approx(-res.fun), f"{i=}"
