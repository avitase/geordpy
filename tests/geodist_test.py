from geographiclib.geodesic import Geodesic
import pytest
from geordpy.geordpy import geodist_point_lineseg
import numpy as np


def test_geodist():
    start = 54.34, 11.98
    end = 54.55, 11.93
    point = 54.49, 12.09

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
