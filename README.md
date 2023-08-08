# GeoRDPy
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyPI](https://img.shields.io/pypi/v/geordpy)](https://pypi.org/project/geordpy/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

GeoRDPy is a Python library that simplifies geodetic-coordinate polylines using the Ramer-Douglas-Peucker algorithm. It ensures accuracy by utilizing the precise distance calculations from the GeographicLib by Charles Karney on the WGS 84 reference ellipsoid.

## Features
- Simplify geodetic-coordinate polylines using the Ramer-Douglas-Peucker algorithm.
- Utilize precise distance calculations with GeographicLib on the WGS 84 reference ellipsoid.
- Easy-to-use interface with a single function call.

## Installation
`GeoRDPy` releases are available as wheel packages for macOS, Windows and Linux on [PyPI](https://pypi.org/project/geordpy/).
Install it using pip:
```bash
pip install geordpy
```

## Usage
```python
>>> import numpy as np
>>> import geordpy
>>> points = np.array([(42.0, -75.0), (42.1, -74.9), (42.2, -75.1), (42.3, -74.8)])
>>> threshold = 15_000  # meters
>>> mask = geordpy.filter(points, threshold=threshold)
>>> points[mask]
array([[ 42. , -75. ],
        [ 42.1, -74.9],
        [ 42.2, -75.1],
        [ 42.3, -74.8]])
```

For more details, check the [documentation](https://avitase.github.io/geordpy/).

## Contributing
Contributions are welcome! If you find any issues or have suggestions for improvements, please feel free to submit a pull request.
