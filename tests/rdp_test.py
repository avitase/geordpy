import functools

import numpy as np
import pytest
import utils

from geordpy import rdp_filter
from geordpy.great_circle import cos_distance_segment


def dist(point, *, start, end, radius):
    lat, lon = np.deg2rad(point)
    lat1, lon1 = np.deg2rad(start)
    lat2, lon2 = np.deg2rad(end)

    return radius * np.arccos(
        cos_distance_segment(
            np.array([lat]),
            np.array([lon]),
            lat1=lat1,
            lon1=lon1,
            lat2=lat2,
            lon2=lon2,
        ).squeeze()
    )


def test_great_circle():
    points = [
        (0.0, 0.0),  # point 0
        (-10.0, 10.0),  # point 1
        (10.0, 20.0),  # point 2
        (-20.0, 30.0),  # point 3
        (20.0, 40.0),  # point 4
        (-10.0, 50.0),  # point 5
        (10.0, 60.0),  # point 6
        (0.0, 70.0),  # point 7
    ]

    radius = 1000
    _dist = functools.partial(dist, radius=radius)
    _filter = functools.partial(rdp_filter, radius=radius)

    d3_07 = _dist(points[3], start=points[0], end=points[7])
    d2_03 = _dist(points[2], start=points[0], end=points[3])

    mask = _filter([], threshold=0)
    assert len(mask) == 0

    mask = _filter(points[:2], threshold=np.finfo(np.float64).max)
    assert len(mask) == 2 and all(mask)

    mask = _filter(points, threshold=0)
    assert len(mask) == len(points) and all(mask)

    threshold = np.ceil(d3_07)
    mask = _filter(points, threshold=threshold)
    assert all(mask == [True, False, False, False, False, False, False, True])

    threshold = np.ceil(d2_03)
    mask = _filter(points, threshold=threshold)
    assert all(mask == [True, False, False, True, True, False, False, True])

    threshold = np.floor(d2_03)
    mask = _filter(points, threshold=threshold)
    assert all(mask == [True, False, True, True, True, True, False, True])


def test_rhumb_line():
    def _gd(x):
        return np.rad2deg(utils.gd(np.deg2rad(x)))

    points = [
        (0.0, 0.0),
        (_gd(45.0), 45.0),
        (_gd(90.0), 90.0),
        (_gd(135.0), 135.0),
    ]

    radius = 1000

    mask = rdp_filter(points, threshold=100, radius=radius)
    assert all(mask == [True, True, True, True])

    mask = rdp_filter(points, threshold=1, radius=radius, rhumb_line_interpolation=True)
    assert all(mask == [True, False, False, True])


@pytest.mark.parametrize("rhumb_line", [True, False])
def test_degenerated_trajectory(rhumb_line):
    points = [
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
    ]

    mask = rdp_filter(points, threshold=10, rhumb_line_interpolation=rhumb_line)
    assert all(mask == [True, False, False, False, False, True])
