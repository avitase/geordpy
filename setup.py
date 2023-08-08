#!/usr/bin/env python
# coding=utf-8

import os
import sys

from setuptools import find_packages, setup

here = os.path.abspath(os.path.dirname(__file__))
sys.path.append(here)

import versioneer  # noqa: E402

with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


setup(
    name="geordpy",
    description="A Python library for simplifying geodetic-coordinate polylines using the Ramer-Douglas-Peucker algorithm",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/avitase/geordpy",
    author="Nis Meinert",
    author_email="mail@nismeinert.name",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=find_packages(exclude=["docs", "tests"]),
    python_requires=">=3.10",
    install_requires=[
        "geographiclib",
        "numpy",
        "scipy",
    ],
    extras_require={
        "dev": [
            "pre-commit",
            "pytest",
            "pytest-cov",
            "sphinx",
        ],
    },
)
