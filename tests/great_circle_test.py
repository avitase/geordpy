import numpy as np
import pytest
import scipy.optimize
from scipy.spatial.transform import Rotation as rotation

from geordpy.great_circle import closest_point, northernmost

SQRT2 = np.sqrt(2.0)


def latlon_from_vec(v):
    return np.rad2deg(np.arcsin(v[2])), np.rad2deg(np.arctan2(v[1], v[0]))


@pytest.mark.parametrize(
    "points",
    [
        # on great-circle
        (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (-1.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0),
        ),
        # 45 degrees distance
        (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.5, 0.5, 1.0 / SQRT2),
            (1.0 / SQRT2, 1.0 / SQRT2, 0.0),
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
    lat4, lon4 = closest_point(lat3, lon3, lat1=lat1, lon1=lon1, lat2=lat2, lon2=lon2)

    lat_exp, lon_exp = latlon_from_vec(rot @ exp)

    assert lat4 == pytest.approx(lat_exp)
    assert lon4 == pytest.approx(lon_exp)


@pytest.mark.parametrize("flip", [True, False])
def test_northernmost_calculation(flip):
    lat1, lon1 = 60.0, 45.0
    lat2, lon2 = 45.0, 135.0

    if flip:
        lat1, lat2 = lat2, lat1
        lon1, lon2 = lon2, lon1

    lat_exp, lon_exp = np.rad2deg(np.arccos(1.0 / np.sqrt(5.0))), 75.0

    latN, lonN = northernmost(lat1=lat1, lon1=lon1, lat2=lat2, lon2=lon2)
    assert latN == pytest.approx(lat_exp)
    assert lonN == pytest.approx(lon_exp)


@pytest.mark.parametrize("seed", list(range(1, 10)))
def test_random_points(seed):
    def _init_vec(R, *, lon):
        lon = np.deg2rad(lon)
        return R @ np.array([np.cos(lon), np.sin(lon), 0.0])

    rng = np.random.default_rng(seed)
    rot1, rot2 = [rot.as_matrix() for rot in rotation.random(num=2, random_state=seed)]

    a = _init_vec(rot1, lon=0.0)
    b = _init_vec(rot1, lon=rng.uniform(0.0, 360.0))
    c = _init_vec(rot2, lon=0.0)

    lat1, lon1 = latlon_from_vec(a)
    lat2, lon2 = latlon_from_vec(b)
    lat3, lon3 = latlon_from_vec(c)
    lat4, lon4 = closest_point(lat3, lon3, lat1=lat1, lon1=lon1, lat2=lat2, lon2=lon2)

    latN12, lonN12 = northernmost(lat1=lat1, lon1=lon1, lat2=lat2, lon2=lon2)
    latN14, lonN14 = northernmost(lat1=lat1, lon1=lon1, lat2=lat4, lon2=lon4)
    assert latN14 == pytest.approx(latN12)
    assert lonN14 == pytest.approx(lonN12)

    def _dist(s):
        v = _init_vec(rot1, lon=s)
        return np.dot(v, c)

    res = scipy.optimize.minimize_scalar(_dist, bounds=(0.0, 360.0))
    assert res.success

    v = np.array(
        [np.cos(lat4) * np.cos(lon4), np.cos(lat4) * np.sin(lon4), np.sin(lat4)]
    )
    assert np.dot(v, c) <= res.fun
