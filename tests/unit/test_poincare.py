import numpy as np
import pytest

import porepy as pp
import pygeon as pg  # type: ignore[import-untyped]


@pytest.fixture(params=[(dim, is_md) for is_md in [False, True] for dim in [2, 3]])
def poincare_and_flist(request):
    """
    Compute a Poincare object and a list of
    distributions to test with. The parametrization includes
    2D and 3D, in mixed and fixed dimensions.
    """
    dim, is_md = request.param

    if is_md:
        mesh_args = {"cell_size": 0.5, "cell_size_fracture": 0.5}
        if dim == 2:
            mdg, _ = pp.mdg_library.square_with_orthogonal_fractures(
                "simplex", mesh_args, range(dim)
            )
        elif dim == 3:
            mdg, _ = pp.mdg_library.cube_with_orthogonal_fractures(
                "simplex", mesh_args, range(dim)
            )
        pg.convert_from_pp(mdg)
    else:
        mdg = pg.unit_grid(dim, 1 / 3)

    mdg.compute_geometry()
    poin = pg.Poincare(mdg)

    sd = mdg.subdomains(dim=mdg.dim_max())[0]
    np.random.seed(0)

    f_list = [None] * (dim + 1)
    f_list[0] = np.random.rand(sd.num_nodes)
    f_list[dim - 2] = np.random.rand(mdg.num_subdomain_ridges())
    f_list[dim - 1] = np.random.rand(mdg.num_subdomain_faces())
    f_list[dim] = np.random.rand(mdg.num_subdomain_cells())

    return poin, f_list


def test_chain_property(poincare_and_flist):
    """
    Check the chain property, i.e. whether pp=0
    """

    poin, f_list = poincare_and_flist

    for k, f in enumerate(f_list):
        if k > 0:
            pf = poin.apply(k, f)
            ppf = poin.apply(k - 1, pf)
            assert np.allclose(ppf, 0)


def test_decomposition(poincare_and_flist):
    """
    For given f, check whether the decomposition
    (pd + pd) f = f
    holds
    """

    poin, f_list = poincare_and_flist

    for k, f in enumerate(f_list):
        pdf, dpf = poin.decompose(k, f)
        assert np.allclose(f, pdf + dpf)


if __name__ == "__main__":
    pytest.main([__file__])
