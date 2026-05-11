PyGeoN Documentation
====================

.. image:: https://github.com/compgeo-mox/pygeon/actions/workflows/run-pytest.yml/badge.svg
   :target: https://github.com/compgeo-mox/pygeon/actions/workflows/run-pytest.yml
   :alt: Pytest

.. image:: https://github.com/compgeo-mox/pygeon/actions/workflows/run-static-checks.yml/badge.svg
   :target: https://github.com/compgeo-mox/pygeon/actions/workflows/run-static-checks.yml
   :alt: Mypy, ruff, isort

.. image:: https://zenodo.org/badge/455087135.svg
   :target: https://zenodo.org/badge/latestdoi/455087135
   :alt: DOI

.. image:: https://img.shields.io/badge/License-GPL%20v3-blue.svg
   :target: https://www.gnu.org/licenses/gpl-3.0
   :alt: License: GPL v3

**PyGeoN** is a Python package for Geo-Numerics, focusing on structure-preserving numerical methods for porous media and mixed-dimensional modeling.

Features
--------

* Structure-preserving discretizations
* Mixed-dimensional modeling support
* Integration with PorePy
* Comprehensive test suite
* Tutorials and examples

Installation
------------

PyGeoN requires Python >= 3.12

Since PyGeoN depends on `PorePy <https://github.com/pmgbergen/porepy>`_, ensure that it is accessible in your PYTHONPATH.

To install PyGeoN:

.. code-block:: bash

   pip install -e .

Omit the ``-e`` flag if you don't want the editable version.

Quick Start
-----------

See the `tutorials <https://github.com/compgeo-mox/pygeon/tree/main/tutorials>`_ to get started with PyGeoN.

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   tutorials
   examples

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/modules

.. toctree::
   :maxdepth: 1
   :caption: Reports

   reports/index

.. toctree::
   :maxdepth: 1
   :caption: Additional Information

   contributing
   papers
   license

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
