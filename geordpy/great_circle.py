import numpy as np
from scipy.spatial.transform import Rotation


def constrain_angle(x):
    x = np.fmod(x + np.pi, 2.0 * np.pi)
    if x < 0:
        x += 2.0 * np.pi

    return x - np.pi


def sin_bearing(*, lat1, lon1, lat2, lon2):
    slat1 = np.sin(lat1)
    slat2 = np.sin(lat2)
    clat1 = np.cos(lat1)
    clat2 = np.cos(lat2)

    cdlon = np.cos(lon1 - lon2)

    nom = clat1 * slat2 - cdlon * clat2 * slat1
    denom = np.sqrt(1.0 - (cdlon * clat1 * clat2 + slat1 * slat2) ** 2)
    x = nom / denom

    # sin(arccos x) = sqrt(1 - x^2)
    return np.sqrt(1.0 - min(x**2, 1.0))


def lon_north(*, lat, lon, sin_bearing):
    nom = np.tan(lat)

    # tan(arccos x) = sqrt(1 - x^2) / x
    denom = np.sqrt(1.0 - sin_bearing**2) / sin_bearing

    return constrain_angle(lon + np.arccos(nom / denom))


def closest_point_great_circle(*, lat, lon, lat1, lon1, lat2, lon2):
    sin_az1 = sin_bearing(lat1=lat1, lon1=lon1, lat2=lat2, lon2=lon2)
    if sin_az1 < np.nextafter(0.0, 1.0):
        return np.array([np.cos(lon), np.sin(lon), 0.0])

    lon0 = lon_north(lat=lat1, lon=lon1, sin_bearing=sin_az1) + np.pi / 2.0
    rot_axis = np.array([np.cos(lon0), np.sin(lon0), 0.0])
    rot_angle = np.arcsin(sin_az1)

    rot = Rotation.from_rotvec(rot_angle * rot_axis)
    p = rot.as_matrix() @ np.array(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]
    )

    lon3 = np.arctan2(p[1], p[0])
    q = np.array([np.cos(lon3), np.sin(lon3), 0.0])

    return rot.inv().as_matrix() @ q
