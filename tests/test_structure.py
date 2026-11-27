"""
Module contains the check that the tests mirror the structure of the source, see
CONTRIBUTING.md.

The rule is that every source file in src/pygeon has its own tests, at the mirrored
path in tests/pygeon:

- src/pygeon/a/b.py is tested in tests/pygeon/a/test_b.py, e.g.
  src/pygeon/grids/voronoi.py in tests/pygeon/grids/test_voronoi.py;
- or, for the source files that define several discretizations, in the folder
  tests/pygeon/a/b/, e.g. src/pygeon/discretizations/fem/h1.py in
  tests/pygeon/discretizations/fem/h1/, with test_h1.py, test_lagrange1.py and
  test_lagrange2.py.

The tests outside tests/pygeon combine several source files, and so mirror none of
them: the patch tests check the exactness of the discretizations where it is
expected, the tutorials run the notebooks.

The check goes both ways: every source file must have its tests, and every test file
must mirror a source file. The files that do not follow the rule yet are listed below
as exceptions. The lists can only shrink: when a file starts following the rule, the
check fails until it is removed from its list, so that no exception is left behind.
"""

from pathlib import Path

# The root of the source and of the tests that mirror it
SRC = Path(__file__).parents[1] / "src" / "pygeon"
TESTS = Path(__file__).parent / "pygeon"

# Exceptions: the source files without tests at the mirrored path, relative to
# src/pygeon. See #356 for the plan to remove them
SRC_WITHOUT_TESTS = {
    # Tested in files named otherwise, which #358 and #359 rename
    "discretizations/mat_discretization.py",
    # Without tests of their own, covered by the tests of other files. Their tests are
    # to be added after the restructuring
    "discretizations/fem/symmat_l2.py",
    "discretizations/fvm/fvm_disc.py",
    "discretizations/fvm/tpfa.py",
    "discretizations/vec_discretization.py",
    "grids/mortar_grid.py",
    "numerics/restrictions.py",
    "utils/common_constants.py",
    "utils/set_membership.py",
    "utils/sort_points.py",
}

# Exceptions: the test files that mirror no source file, relative to tests/pygeon. They
# test the source files listed above as named otherwise, and #358 and #359 rename them
TESTS_WITHOUT_SRC = {
    "discretizations/test_matdiscretization.py",
    "grids/test_grid_edge_tangents.py",
    "grids/test_subvolumes.py",
}


def mirrored_source(test_file: Path) -> str | None:
    """
    Returns the source file mirrored by a test file: tests/pygeon/a/test_b.py and the
    files in the folder tests/pygeon/a/b/ mirror src/pygeon/a/b.py.

    Args:
        test_file (Path): The test file.

    Returns:
        str | None: The path of the source file relative to src/pygeon, or None if the
        test file mirrors no source file.
    """
    # The folder of the test file, relative to tests/pygeon, e.g. discretizations/fem/h1
    # for tests/pygeon/discretizations/fem/h1/test_lagrange1.py
    folder = test_file.parent.relative_to(TESTS)

    # The two candidates for the source file:
    # - the folder itself, for the folders of the discretizations, e.g.
    #   discretizations/fem/h1.py for
    #   tests/pygeon/discretizations/fem/h1/test_lagrange1.py;
    # - the name of the test file without the prefix test_, e.g. grids/voronoi.py for
    #   tests/pygeon/grids/test_voronoi.py
    candidates = (folder.with_suffix(".py"), folder / (test_file.stem[5:] + ".py"))

    # The test file mirrors the first candidate that exists in the source
    for source in candidates:
        if (SRC / source).is_file():
            return source.as_posix()
    return None


def structured_test_files() -> list[Path]:
    """
    Returns the test files subject to the structure, i.e. the ones in tests/pygeon.

    Returns:
        list[Path]: The test files.
    """
    return list(TESTS.rglob("test_*.py"))


def test_sources_have_tests():
    # All the source files, relative to src/pygeon, except the __init__.py files, which
    # only collect the imports of their package
    sources = {f.relative_to(SRC).as_posix() for f in SRC.rglob("*.py")}
    sources -= {s for s in sources if s.endswith("__init__.py")}

    # The source files mirrored by at least one test file
    tested = {mirrored_source(f) for f in structured_test_files()}

    # The source files without tests must be exactly the exceptions: a new source file
    # without tests makes the check fail, and so does an exception that now has tests,
    # which must be removed from the list
    missing = sources - tested
    new_missing = sorted(missing - SRC_WITHOUT_TESTS)
    old_exceptions = sorted(SRC_WITHOUT_TESTS - missing)
    assert missing == SRC_WITHOUT_TESTS, (
        f"Source files without tests: {new_missing}. "
        "Exceptions that now have tests, to remove from SRC_WITHOUT_TESTS: "
        f"{old_exceptions}"
    )


def test_tests_have_sources():
    # The test files, relative to tests/pygeon, that mirror no source file
    unmirrored = {
        f.relative_to(TESTS).as_posix()
        for f in structured_test_files()
        if mirrored_source(f) is None
    }

    # They must be exactly the exceptions: a new test file that mirrors no source file
    # makes the check fail, and so does an exception that now mirrors one, e.g. after a
    # rename, which must be removed from the list
    new_unmirrored = sorted(unmirrored - TESTS_WITHOUT_SRC)
    old_exceptions = sorted(TESTS_WITHOUT_SRC - unmirrored)
    assert unmirrored == TESTS_WITHOUT_SRC, (
        f"Test files that mirror no source file: {new_unmirrored}. "
        "Exceptions that now mirror a source file, to remove from TESTS_WITHOUT_SRC: "
        f"{old_exceptions}"
    )
