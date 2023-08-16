import numpy as np

from geordpy import rdp_filter
from geordpy.great_circle import cos_distance_segment


def test_filter():
    points = np.deg2rad(
        [
            (0.0, 0.0),  # point 1
            (-10.0, 10.0),  # point 2
            (10.0, 20.0),  # point 3
            (-20.0, 30.0),  # point 4
            (20.0, 40.0),  # point 5
            (-10.0, 50.0),  # point 6
            (10.0, 60.0),  # point 7
            (0.0, 70.0),  # point 8
        ]
    )

    radius = 100
    d18 = radius * np.arccos(
        cos_distance_segment(
            points[..., 0],
            points[..., 1],
            lat1=points[0, 0],
            lon1=points[0, 1],
            lat2=points[-1, 0],
            lon2=points[-1, 1],
        )
    )

    threshold = np.floor(np.min(d18[1:-1]))
    assert threshold > 0.0

    mask = rdp_filter(points, threshold=threshold, radius=radius)
    assert all(mask == [True, True, True, True, True, True, True, True])

    assert False
