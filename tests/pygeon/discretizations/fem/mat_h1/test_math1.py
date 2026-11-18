"""Module contains specific tests for the matrix Lagrange1 discretization."""

import numpy as np
import porepy as pp
import pytest
import scipy.sparse as sps

import pygeon as pg


@pytest.fixture(params=[pg.MatLagrange1])
def discr(request: pytest.FixtureRequest) -> pg.Discretization:
    return request.param("test")


@pytest.fixture
def vector_field() -> np.ndarray:
    return np.array([[1], [1], [1]])


def test_assemble_adv_matrix(
    discr: pg.MatLagrange1, ref_sd: pg.Grid, vector_field: np.ndarray
):
    data = pp.initialize_data({}, "test", {pg.VECTOR_FIELD: vector_field})
    M = discr.assemble_adv_matrix(ref_sd, data=data)

    scalar_adv = pg.Lagrange1("test").assemble_adv_matrix(ref_sd, data)
    M_known = sps.kron(sps.eye_array(ref_sd.dim**2), scalar_adv)

    assert np.allclose((M_known - M).data, 0)


def test_undefined_range_class(discr):
    with pytest.raises(NotImplementedError):
        discr.get_range_discr_class(3)
