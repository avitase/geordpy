import numpy as np
import pytest
import scipy.optimize
from scipy.spatial.transform import Rotation as rotation

from geordpy.great_circle import cos_distance_segment

SQRT2 = np.sqrt(2.0)
SQRT3 = np.sqrt(3.0)


def latlon_from_vec(v):
    return np.arcsin(v[2]), np.arctan2(v[1], v[0])


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
        lat3, lon3, lat1=lat1, lon1=lon1, lat2=lat2, lon2=lon2
    )

    assert cos_dist == pytest.approx(exp)


def great_circle(lon, *, latN, lonN):
    return np.arctan(np.tan(latN) * np.cos(lon - lonN))


def get_bound(a, b):
    if b < a:
        a, b = b, a

    d = b - a
    if d > (w := a + 2.0 * np.pi):
        return b, w

    return a, b


@pytest.mark.parametrize("seed", list(range(1, 10)))
def test_random_points(seed):
    rng = np.random.default_rng(seed)

    az0 = rng.uniform(-np.pi, np.pi)
    latN = np.pi / 2 - abs(az0)
    lonN = rng.uniform(-np.pi, np.pi)

    lon1, lon2 = rng.uniform(-np.pi, np.pi, size=2)
    lat1 = great_circle(lon1, latN=latN, lonN=lonN)
    lat2 = great_circle(lon2, latN=latN, lonN=lonN)

    p = rotation.random(random_state=seed).as_matrix() @ np.array([1.0, 0.0, 0.0])
    lat, lon = latlon_from_vec(p)

    cos_dist = cos_distance_segment(
        lat, lon, lat1=lat1, lon1=lon1, lat2=lat2, lon2=lon2
    )

    def loss(x):
        lat1 = great_circle(x, latN=latN, lonN=lonN)
        lat2 = lat

        dlon = x - lon

        return -np.sin(lat1) * np.sin(lat2) - np.cos(lat1) * np.cos(lat2) * np.cos(dlon)

    bound = get_bound(lon1, lon2)
    res = scipy.optimize.minimize_scalar(loss, bounds=bound)
    assert res.success

    if loss(lon1) < res.fun:
        res.fun = loss(lon1)
    elif loss(lon2) < res.fun:
        res.fun = loss(lon2)

    assert cos_dist > -res.fun or cos_dist == pytest.approx(-res.fun)
