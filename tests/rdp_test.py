import functools

import numpy as np

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


def test_trajectory():
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

    d3_07 = _dist(points[3], start=points[0], end=points[7])
    d2_03 = _dist(points[2], start=points[0], end=points[3])

    mask = rdp_filter([], threshold=0, radius=radius)
    assert len(mask) == 0

    mask = rdp_filter(points[:2], threshold=np.finfo(np.float64).max, radius=radius)
    assert len(mask) == 2 and all(mask)

    mask = rdp_filter(points, threshold=0, radius=radius)
    assert len(mask) == len(points) and all(mask)

    threshold = np.ceil(d3_07)
    mask = rdp_filter(points, threshold=threshold, radius=radius)
    assert all(mask == [True, False, False, False, False, False, False, True])

    threshold = np.ceil(d2_03)
    mask = rdp_filter(points, threshold=threshold, radius=radius)
    assert all(mask == [True, False, False, True, True, False, False, True])

    threshold = np.floor(d2_03)
    mask = rdp_filter(points, threshold=threshold, radius=radius)
    assert all(mask == [True, False, True, True, True, True, False, True])


def test_degenerated_trajectory():
    points = [
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
    ]

    mask = rdp_filter(points, threshold=10)
    assert all(mask == [True, False, False, False, False, True])
