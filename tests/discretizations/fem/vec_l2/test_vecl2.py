"""Module contains general tests for Vector L2 discretizations."""

import numpy as np
import pytest

import pygeon as pg


@pytest.fixture(
    params=[
        pg.VecPwConstants,
        pg.VecPwLinears,
        pg.VecPwQuadratics,
    ]
)
def discr(request: pytest.FixtureRequest) -> pg.VecPwPolynomials:
    return request.param("test")


def test_assemble_local_dofs(discr, ref_sd):
    # Coverage test
    dofs = discr.local_dofs_of_cell(ref_sd, 0)
    known_dofs = ref_sd.num_cells * np.arange(discr.ndof_per_cell(ref_sd))

    assert np.all(dofs == known_dofs)
