"""Setuptools configuration for flood mapping package."""

from setuptools import find_packages, setup

setup(
    name="flood-mapping",
    version="0.1.0",
    description="AI and geospatial flood mapping pipeline",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[],
)
