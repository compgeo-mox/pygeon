# PyGeoN

Python library of finite element and virtual element discretizations on
(mixed-dimensional) grids, built on PorePy. Code, comments, docstrings, commit
messages and PR texts are in English.

## Repository layout

- `src/pygeon/discretizations/`: the discretizations, in `fem/`, `vem/` and `fvm/`.
- `src/pygeon/grids/`: the grids and their topology, refinement and regularization.
- `src/pygeon/numerics/`: differentials, inner products, projections, linear systems,
  Poincaré operators and spanning trees.
- `src/pygeon/filters/`, `params/`, `utils/`, `viz/`: conversion from PorePy, data,
  helpers and plotting.
- `tests/pygeon/`: the tests of the source files; `tests/patch_tests/` and
  `tests/tutorials/`: the tests that combine them.
- `docs/`: the documentation, with the report on the dofs in `docs/reports/dofs.md`.

The grids of PyGeoN are PorePy grids converted with `pg.convert_from_pp`. Before
calling an API of PorePy, or of any other library, read its signature in the installed
source.

## Checks before every commit

The CI runs these, and coverage must not drop below the one of main:

    ruff check src tests
    ruff format --check src tests tutorials
    isort --check src tests
    mypy src
    pytest tests
    SPHINXOPTS=-W make -C docs html

Run them locally before pushing: do not use the CI as the test runner. Line length is
88. New code is covered by tests.

## Tests

- `tests/pygeon` mirrors `src/pygeon`: every source file has its own tests, in
  `tests/pygeon/a/test_b.py`, or in the folder `tests/pygeon/a/b/` for the files that
  define several discretizations. See `CONTRIBUTING.md`; `tests/test_structure.py`
  checks it and lists the current exceptions.
- Tests that combine several source files go in `tests/patch_tests`.
- A test compares the result with an exact value or with an independently computed
  one. The test of a fix fails without the fix.
- Every assert has a message with the values that make it fail.

## Code

- Keep it minimal: no options, parameters or abstractions that are not needed.
- Fix bugs at their root cause. Report the bugs found outside the scope of a change,
  and open an issue for each one that is not fixed.
- Type hints on all function signatures, for the parameters and the return types.
- Docstrings on public functions in Google style (`Args:`, `Returns:`, `Raises:`).
- Imports at the top of the module, unless an import breaks a circular dependency.
- Raise an exception with a message for invalid input, e.g. `ValueError` or
  `NotImplementedError`. Use `assert` only for internal invariants.
- Do not loop in Python over the cells, faces, nodes or dofs of a grid: vectorize with
  NumPy and SciPy.
- Sparse matrices as `sps.csc_array`/`sps.csr_array`, not `*_matrix`.
- The topological maps of a grid (`cell_faces`, `face_ridges`, `ridge_peaks`,
  `edge_nodes`) are sorted in `compute_geometry`: the order of their indices may define
  the order of local entities and dofs. `face_nodes` is not sorted, since its order
  defines the geometry of the faces. Note that `abs()`, `**`, `max`, comparisons, etc.
  of a sparse array with unsorted indices sort them in place.
- Comments and docstrings describe the current code, not its earlier versions: the
  history goes in the commits and the PRs.
- Docstrings, comments, `docs/reports/dofs.md` and PR descriptions stay consistent
  with the code: a change to the dofs of a discretization updates the report too.

## Commits and PRs

- One logical change per commit, with its tests and docs.
- No `Co-Authored-By` or other attribution lines.
- Do not state from memory what a paper contains: read it first, and check references
  on Crossref.
