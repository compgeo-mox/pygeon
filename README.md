![Pytest](https://github.com/compgeo-mox/pygeon/actions/workflows/run-pytest.yml/badge.svg)
![Mypy, ruff, isort](https://github.com/compgeo-mox/pygeon/actions/workflows/run-static-checks.yml/badge.svg)
![Tutorials](https://github.com/compgeo-mox/pygeon/actions/workflows/check_tutorials.yml/badge.svg)
![CodeQL](https://github.com/compgeo-mox/pygeon/workflows/CodeQL/badge.svg)
[![DOI](https://zenodo.org/badge/455087135.svg)](https://zenodo.org/badge/latestdoi/455087135)
[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

# PyGeoN: a Python package for Geo-Numerics

## Installation for Linux

PyGeoN requires Python >= 3.12

Since for many functionalities PyGeoN depends on [PorePy](https://github.com/pmgbergen/porepy), we assume that the latter is accessible in your PYTHONPATH.
To install PyGeoN
```bash
pip install -e .
```
avoid the `-e` if you do not want the editable version.
It might be useful to add PyGeoN to your PYTHONPATH.

## Docker
PyGeoN is also available through a Docker image. The image of the main branch be obtained by
```bash
docker pull pygeon/main
```
More details can be found [here](https://github.com/compgeo-mox/pygeon/tree/docker/dockerfiles), docker image is hosted on Docker Hub [here](https://hub.docker.com/r/pygeon/main).


## Issues
Create an [issue](https://github.com/compgeo-mox/pygeon/issues).

## Getting started
See the [tutorials](https://github.com/compgeo-mox/pygeon/tree/main/tutorials).

## Papers
For a list of papers that use PyGeoN see [papers](https://github.com/compgeo-mox/.github/blob/main/profile/papers.md).

## License
See [license](./LICENSE.md).
