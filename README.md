[![DOI](https://zenodo.org/badge/455087135.svg)](https://zenodo.org/badge/latestdoi/455087135)

# PyGeoN: a Python package for Geo-Numerics

## Installation for Linux

PyGeoN requires Python >= 3.10

Since for many functionalities PyGeoN depends on [PorePy](https://github.com/pmgbergen/porepy), we assume that the latter is accessible in your PYTHONPATH.

To install PyGeoN, first clone the current version of the code and then install the dependencies by
```bash
pip install -r requirements.txt

```
then to install PyGeoN at user level (see also below) type
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

## License
See
