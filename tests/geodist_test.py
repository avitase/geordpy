import numpy as np
import pytest
from geographiclib.geodesic import Geodesic

import geordpy
from geordpy.geordpy import geodist_point_lineseg


def test_geodist():
    start = 54.34, 11.98
    end = 54.55, 11.93
    point = 54.49, 12.09

    min_dist = geodist_point_lineseg(start, start=start, end=end)
    assert min_dist == pytest.approx(0.0, abs=1e-9)

    min_dist = geodist_point_lineseg(end, start=start, end=end)
    assert min_dist == pytest.approx(0.0, abs=1e-9)

    min_dist = geodist_point_lineseg(point, start=start, end=end)

    geod = Geodesic.WGS84
    line = geod.InverseLine(*start, *end, Geodesic.LATITUDE | Geodesic.LONGITUDE)

    dists = []
    for arc in np.linspace(0, line.a13, 10_000, endpoint=True):
        point2 = line.ArcPosition(arc, Geodesic.LATITUDE | Geodesic.LONGITUDE)
        point2 = point2["lat2"], point2["lon2"]
        dists.append(geod.Inverse(*point, *point2, Geodesic.DISTANCE)["s12"])

    min_dist_approx = min(dists)
    assert min_dist == pytest.approx(min_dist_approx, abs=1.0)


def test_filter():
    geod = Geodesic.WGS84

    p1 = 53.3601, 13.073

    p2 = geod.Direct(*p1, 45, 1000, Geodesic.LATITUDE | Geodesic.LONGITUDE)
    p2 = p2["lat2"], p2["lon2"]

    p3 = geod.Direct(*p2, -45, 1000, Geodesic.LATITUDE | Geodesic.LONGITUDE)
    p3 = p3["lat2"], p3["lon2"]

    p4 = geod.Direct(*p3, -135, 2000, Geodesic.LATITUDE | Geodesic.LONGITUDE)
    p4 = p4["lat2"], p4["lon2"]

    d13 = geod.Inverse(*p1, *p3, Geodesic.DISTANCE)["s12"]
    min_dist = geodist_point_lineseg(p2, start=p1, end=p3)

    points = [p1, p2, p3, p4]

    res = geordpy.filter(points, np.ceil(d13)).tolist()
    assert res == [True, False, False, True]

    res = geordpy.filter(points, np.floor(d13)).tolist()
    assert res == [True, False, True, True]

    res = geordpy.filter(points, np.ceil(min_dist)).tolist()
    assert res == [True, False, True, True]

    res = geordpy.filter(points, np.floor(min_dist)).tolist()
    assert res == [True, True, True, True]
