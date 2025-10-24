import pytest

import numpy as np
import pygeon as pg


@pytest.fixture(
    params=[
        pg.PwConstants,
        pg.PwLinears,
        pg.PwQuadratics,
    ]
)
def discr(request):
    return request.param("test")


def test_get_range_discr_class(discr):
    with pytest.raises(NotImplementedError):
        discr.get_range_discr_class(2)


@pytest.mark.parametrize("discr", [pg.PwConstants, pg.PwLinears], indirect=True)
def test_proj_to_higherPwPolynomials(discr, unit_sd):
    proj = discr.proj_to_higher_PwPolynomials(unit_sd)
    mass = discr.assemble_mass_matrix(unit_sd)

    discr_higher = pg.get_PwPolynomials(discr.poly_order + 1, 0)
    mass_higher = discr_higher("test").assemble_mass_matrix(unit_sd)

    diff = proj.T @ mass_higher @ proj - mass

    assert np.allclose(diff.data, 0.0)


if __name__ == "__main__":
    pytest.main([__file__])
