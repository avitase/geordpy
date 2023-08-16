import numpy as np

from geordpy import great_circle


def _filter(points, threshold):
    n_points = points.shape[0]
    if n_points <= 2:
        return np.full(n_points, True)

    dist = -great_circle.cos_distance_segment(
        points[1:-1, 0],
        points[1:-1, 1],
        lat1=points[0, 0],
        lon1=points[0, 1],
        lat2=points[-1, 0],
        lon2=points[-1, 1],
    )
    i_max = np.argmax(dist) + 1  # dist[i] = dist(points[i+1], line seg.)
    dist_max = dist[i_max - 1]

    return (
        np.concatenate(
            [
                _filter(points[: i_max + 1], threshold)[:-1],
                _filter(points[i_max:], threshold),
            ]
        )
        if dist_max > threshold
        else np.array(
            [True] + [False] * (n_points - 2) + [True],
        )
    )


def rdp_filter(points, threshold, radius=6_371_000):
    if len(points) == 0:
        return np.empty(0, dtype=bool)

    points = np.deg2rad(np.array(points))
    threshold = -np.cos(threshold / radius)  # negate to make it a distance

    return _filter(points, threshold)
