# Contributing to PyGeoN

## Structure of the tests

The folder `tests` mirrors the structure of `src/pygeon`: every source file has its
own tests, which test the features of that file and only those.

- A source file `src/pygeon/a/b.py` is tested in `tests/a/test_b.py`, e.g.
  `src/pygeon/grids/voronoi.py` in `tests/grids/test_voronoi.py`.
- A source file that defines several discretizations is tested in a folder
  `tests/a/b/`, with one test file per class and, if needed, one for their common
  features, e.g. `src/pygeon/discretizations/fem/h1.py` in
  `tests/discretizations/fem/h1/`, with `test_h1.py`, `test_lagrange1.py` and
  `test_lagrange2.py`.
- The tests that combine several source files are in `tests/patch_tests`, which check
  the exactness of the discretizations where it is expected, and in `tests/tutorials`.

The structure is checked by `tests/test_structure.py`, which lists the current
exceptions.
