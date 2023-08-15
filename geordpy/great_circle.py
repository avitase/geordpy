import numpy as np


def to_vec(*, lat, lon):
    cos_lat = np.cos(lat)
    cos_lon = np.cos(lon)
    sin_lat = np.sin(lat)
    sin_lon = np.sin(lon)
    return np.stack([cos_lat * cos_lon, cos_lat * sin_lon, sin_lat], axis=-1)


def batched_dot(a, b, *, squeeze=True):
    a = np.expand_dims(a, 1)
    b = np.expand_dims(b, 2)
    res = np.matmul(a, b).squeeze(-1)
    return res.squeeze(-1) if squeeze else res


def cos_distance_segment(lat, lon, *, lat1, lon1, lat2, lon2):
    x, a, b = to_vec(
        lat=np.stack([lat, lat1, lat2], axis=0), lon=np.stack([lon, lon1, lon2], axis=0)
    )

    n = np.cross(a, b)
    d = batched_dot(a, b, squeeze=False)
    n /= np.sqrt((1.0 - d) * (1.0 + d))

    s = batched_dot(n, x, squeeze=False)

    eps = 1e-5
    sel = (np.abs(s) < 1.0 - np.pi**2 / 8.0 * eps**2).squeeze()  # rel. error < eps
    s = s[sel]

    c = np.copy(b)
    c[sel] = (x[sel] - s * n[sel]) / np.sqrt((1.0 - s) * (1.0 + s))

    cos_alpha = batched_dot(b, c)
    cos_beta = batched_dot(a, c)
    cos_gamma = batched_dot(a, b)

    closest_point = b

    sel = cos_gamma < np.min((cos_alpha, cos_beta), axis=0)  # gamma < max(alpha, beta)
    closest_point[sel] = c[sel]

    sel = ~sel & (cos_beta >= cos_alpha)  # beta <= alpha
    closest_point[sel] = a[sel]

    return batched_dot(x, closest_point)
