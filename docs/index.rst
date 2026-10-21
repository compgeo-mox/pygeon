PyGeoN
======

**PyGeoN** is a Python package for Geo-Numerics: structure-preserving numerical
methods for porous media and mixed-dimensional problems. It provides finite element,
virtual element and finite volume discretizations of the :math:`H^1`,
:math:`H(\mathrm{curl})`, :math:`H(\mathrm{div})` and :math:`L^2` spaces, the
discrete differential operators between them, spanning tree and Poincaré operators,
and the integration with `PorePy <https://github.com/pmgbergen/porepy>`_.

Installation
------------

PyGeoN requires Python >= 3.13 and installs PorePy as a dependency:

.. code-block:: bash

   git clone https://github.com/compgeo-mox/pygeon.git
   pip install -e pygeon

A `Docker image <https://hub.docker.com/r/pygeon/main>`_ of the main branch is also
available.

Getting started
---------------

The tutorials, listed in the sidebar, cover fluid flow, solid mechanics and coupled
problems. They are Jupyter notebooks in the
`tutorials <https://github.com/compgeo-mox/pygeon/tree/main/tutorials>`_ folder of the
repository, to be downloaded and run. Bugs and feature requests go to the
`issue tracker <https://github.com/compgeo-mox/pygeon/issues>`_.

.. toctree::
   :caption: Tutorials
   :maxdepth: 1
   :hidden:
   :glob:

   tutorials/fluid_flow/*
   tutorials/solid_mechanics/*
   tutorials/coupled_problems/*
   tutorials/advanced_topics/*

.. toctree::
   :caption: Reference
   :maxdepth: 1
   :hidden:

   API reference <api/pygeon>
   reports/index
   papers
