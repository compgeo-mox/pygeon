![Pytest](https://github.com/compgeo-mox/pygeon/actions/workflows/run-pytest.yml/badge.svg)
![Mypy, black, isort, flake8](https://github.com/compgeo-mox/pygeon/actions/workflows/run-static-checks.yml/badge.svg)
![CodeQL](https://github.com/compgeo-mox/pygeon/workflows/CodeQL/badge.svg)
[![DOI](https://zenodo.org/badge/455087135.svg)](https://zenodo.org/badge/latestdoi/455087135)
[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>

# PyGeoN: a Python package for Geo-Numerics

## Installation for Linux

PyGeoN requires Python >= 3.10

Since for many functionalities PyGeoN depends on [PorePy](https://github.com/pmgbergen/porepy), we assume that the latter is accessible in your PYTHONPATH.

To install PyGeoN, first clone the current version of the code and then install the dependencies by
```bash
pip install -r requirements.txt

```
Additional dependencies can be installed by
```bash
pip install -r requirements-dev.txt

```
Then to install PyGeoN at user level (see also below) type
```bash
pip install .
```
if you want to modify PyGeoN install it in an editable way by writing instead
```bash
pip install --user -e .
```
It might be useful to add PyGeoN to your PYTHONPATH.

## Issues
Create an [issue](https://github.com/compgeo-mox/pygeon/issues).

## Getting started
See the [tutorials](https://github.com/compgeo-mox/pygeon/tree/main/tutorials).

## License
See [license](./LICENSE.md).
