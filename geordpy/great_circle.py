import numpy as np


def to_vec(*, lat, lon):
    lat = np.deg2rad(lat)
    lon = np.deg2rad(lon)
    return np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


def northernmost(*, lat1, lon1, lat2, lon2):
    lat1 = np.deg2rad(lat1)
    lat2 = np.deg2rad(lat2)
    lon1 = np.deg2rad(lon1)
    lon2 = np.deg2rad(lon2)

    clat1 = np.cos(lat1)
    clat2 = np.cos(lat2)
    slat1 = np.sin(lat1)
    slat2 = np.sin(lat2)

    cdlon = np.cos(lon2 - lon1)
    sdlon = np.sin(lon2 - lon1)

    nom = clat2 * sdlon
    denom = clat1 * slat2 - slat1 * clat2 * cdlon
    az1 = np.arctan2(nom, denom)

    latN = np.arccos(np.sin(np.abs(az1)) * clat1)
    lonN = lon1 + np.sign(az1) * np.arccos(np.tan(lat1) / np.tan(latN))
    return np.rad2deg(latN), np.rad2deg(lonN)


def closest_point(lat, lon, *, lat1, lon1, lat2, lon2):
    a = to_vec(lat=lat1, lon=lon1)
    b = to_vec(lat=lat2, lon=lon2)
    p = to_vec(lat=lat, lon=lon)

    n = np.cross(a, b)
    s = np.dot(n, p)
    if abs(np.sum(s**2) - 1.0) < np.nextafter(0.0, 1.0):
        return lat1, lon1

    q = p - s * n
    lat = np.arctan2(q[2], np.hypot(q[0], q[1]))
    lon = np.arctan2(q[1], q[0])

    return np.rad2deg(lat), np.rad2deg(lon)
