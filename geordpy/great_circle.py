import numpy as np


def to_vec(*, lat, lon):
    return np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


def cos_distance_segment(lat, lon, *, lat1, lon1, lat2, lon2):
    a = to_vec(lat=lat1, lon=lon1)
    b = to_vec(lat=lat2, lon=lon2)
    x = to_vec(lat=lat, lon=lon)

    n = np.cross(a, b)
    d = np.dot(a, b)
    n /= np.sqrt((1.0 - d) * (1.0 + d))

    s = np.dot(n, x)

    eps = 1e-5
    if abs(s) > 1.0 - np.pi**2 / 8.0 * eps**2:  # rel. error < eps
        return 0.0  # arccos(0) = pi/2

    c = (x - s * n) / np.sqrt((1.0 - s) * (1.0 + s))
    assert abs(np.sum(c**2) - 1) < 1e-10

    cos_alpha = np.dot(b, c)
    cos_beta = np.dot(a, c)
    cos_gamma = np.dot(a, b)

    if cos_gamma < min(cos_alpha, cos_beta):  # gamma < max(alpha, beta)
        closest_point = c
    elif cos_beta >= cos_alpha:  # beta <= alpha
        closest_point = a
    else:  # beta > alpha
        closest_point = b

    return np.dot(x, closest_point)
