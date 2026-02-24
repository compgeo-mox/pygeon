"""Module contains specific tests for the polynomial matrix discretization."""

import pytest

import pygeon as pg


@pytest.fixture
def discr() -> pg.MatPwPolynomials:
    return pg.MatPwPolynomials()
