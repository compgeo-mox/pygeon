Contributing
============

We welcome contributions to PyGeoN! This guide will help you get started.

Getting Started
---------------

1. Fork the repository on GitHub
2. Clone your fork locally
3. Create a new branch for your feature or bugfix
4. Make your changes
5. Run tests to ensure everything works
6. Submit a pull request

Development Setup
-----------------

To set up a development environment:

.. code-block:: bash

   git clone https://github.com/your-username/pygeon.git
   cd pygeon
   pip install -e .

Code Style
----------

PyGeoN follows standard Python coding conventions:

* PEP 8 style guide
* Type hints where appropriate
* Comprehensive docstrings

We use the following tools for code quality:

* ``ruff`` for linting
* ``mypy`` for type checking
* ``isort`` for import sorting

Running Tests
-------------

Run the test suite with pytest:

.. code-block:: bash

   pytest

Run static checks:

.. code-block:: bash

   ruff check .
   mypy src/pygeon
   isort --check src/pygeon

Documentation
-------------

When adding new features, please include:

* Docstrings for all public functions and classes
* Updates to relevant documentation pages
* Examples or tutorials if appropriate

Submitting Issues
-----------------

If you encounter a bug or have a feature request, please `create an issue <https://github.com/compgeo-mox/pygeon/issues>`_ on GitHub.

Include:

* A clear description of the issue or feature
* Steps to reproduce (for bugs)
* Expected vs. actual behavior
* Python version and relevant package versions

Pull Request Guidelines
-----------------------

* Keep pull requests focused on a single feature or bugfix
* Include tests for new functionality
* Update documentation as needed
* Ensure all tests pass
* Follow the existing code style

Code of Conduct
---------------

Please be respectful and constructive in all interactions with the PyGeoN community.
