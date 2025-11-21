Installation
============

Requirements
------------

PyGeoN requires:

* Python >= 3.12
* PorePy (accessible in your PYTHONPATH)

Basic Installation
------------------

To install PyGeoN in editable mode:

.. code-block:: bash

   pip install -e .

For a regular installation (non-editable):

.. code-block:: bash

   pip install .

Adding to PYTHONPATH
--------------------

It might be useful to add PyGeoN to your PYTHONPATH. 

On Linux/macOS, add this to your ``.bashrc`` or ``.zshrc``:

.. code-block:: bash

   export PYTHONPATH="${PYTHONPATH}:/path/to/pygeon/src"

Dependencies
------------

PyGeoN depends on:

* PorePy (development and testing versions)
* NumPy
* SciPy
* Other dependencies are automatically installed via the PorePy installation

Verification
------------

To verify your installation:

.. code-block:: python

   import pygeon as pg
   print(pg.__version__)

Development Installation
------------------------

For development, install with additional testing dependencies:

.. code-block:: bash

   pip install -e ".[development,testing]"
