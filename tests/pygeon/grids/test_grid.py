"""Module contains the tests of the Grid class."""

import numpy as np
import porepy as pp
import pytest
import scipy.sparse as sps

import pygeon as pg


def test_sorted_topological_maps(unit_sd: pg.Grid):
    for name in ("cell_faces", "face_ridges", "ridge_peaks", "edge_nodes"):
        conn = getattr(unit_sd, name)
        assert conn.has_sorted_indices, f"{name} has unsorted indices"


def test_grid_0d(ref_sd_0d):
    # no ridges or peaks are defined in 0D, we should obtain an empty map with
    # correct size

    assert ref_sd_0d.num_ridges == 0
    assert ref_sd_0d.num_peaks == 0

    assert ref_sd_0d.ridge_peaks.shape == (0, 0)
    assert ref_sd_0d.face_ridges.shape == (0, 0)


def test_grid_1d(unit_sd_1d):
    # no ridges or peaks are defined in 1D, we should obtain an empty map with
    # correct size

    assert unit_sd_1d.num_ridges == 0
    assert unit_sd_1d.num_peaks == 0

    assert unit_sd_1d.ridge_peaks.shape == (0, 0)
    assert unit_sd_1d.face_ridges.shape == (0, 5)


def test_grid_2d_cart():
    N = 2
    sd = pp.CartGrid([N] * 2, [1] * 2)
    pg.convert_from_pp(sd)
    sd.compute_geometry()

    assert sd.num_ridges == (N + 1) ** 2
    assert sd.num_peaks == 0

    assert sd.ridge_peaks.shape == (0, (N + 1) ** 2)


def test_grid_2d_tris():
    N = 2
    sd = pp.StructuredTriangleGrid([N] * 2)
    pg.convert_from_pp(sd)
    sd.compute_geometry()

    assert sd.num_ridges == (N + 1) ** 2
    assert sd.num_peaks == 0

    assert sd.ridge_peaks.shape == (0, (N + 1) ** 2)


def test_grid_3d_cart():
    N = 2
    sd = pp.CartGrid([N] * 3, [1] * 3)
    pg.convert_from_pp(sd)
    sd.compute_geometry()

    assert sd.num_ridges == 3 * N * (N + 1) ** 2
    assert sd.num_peaks == (N + 1) ** 3


def test_grid_3d_tet():
    N = 1
    sd = pp.StructuredTetrahedralGrid([N] * 3)
    pg.convert_from_pp(sd)
    sd.compute_geometry()

    assert sd.num_ridges == 7 * N**3 + 9 * N**2 + 3 * N
    assert sd.num_peaks == (N + 1) ** 3


def test_mdg_2d(mdg_embedded_frac_2d):
    def known_face_ridges():
        data = np.array([1, -1])
        indices = np.array([10, 11])
        indptr = np.array([0, 0, 0, 2])

        return sps.csc_array((data, indices, indptr), (16, 3))

    mg = mdg_embedded_frac_2d.interfaces()[0]

    assert mg.ridge_peaks.shape == (0, 0)
    assert (mg.face_ridges - known_face_ridges()).nnz == 0


def test_mdg_3d(mdg):
    for mg in mdg.interfaces():
        sd_down = mg.sd_pair[1]

        # Check that the lower-dimensional faces occur twice, except at tips.
        if mg.dim >= 1:
            sums = np.sum(np.abs(mg.face_ridges), axis=0)
            assert np.allclose(sums[sd_down.tags["tip_faces"]], 0)
            assert np.allclose(sums[~sd_down.tags["tip_faces"]], 2)

        # Check that the lower-dimensional ridges occur twice, except at tips.
        if mg.dim >= 2:
            sums = np.sum(np.abs(mg.ridge_peaks), axis=0)
            assert np.allclose(sums[sd_down.tags["tip_nodes"]], 0)
            assert np.allclose(sums[~sd_down.tags["tip_nodes"]], 2)


def test_mdg_3d_itsc(_mdg_dict):
    mdg = _mdg_dict["fracs_3D"]

    for mg in mdg.interfaces(dim=1):
        assert mg.ridge_peaks.shape == (0, 0)
        assert mg.face_ridges.shape == (20, 2)

    for mg in mdg.interfaces(dim=2):
        assert mg.ridge_peaks.shape == (112, 20)
        assert mg.face_ridges.shape == (392, 32)


def test_opposite_nodes(unit_sd):
    opposite_node = unit_sd.compute_opposite_nodes()
    assert opposite_node.nnz == unit_sd.num_cells * (unit_sd.dim + 1)

    faces, cells, nodes = sps.find(opposite_node)

    cell_nodes = unit_sd.cell_nodes()
    assert np.all(cell_nodes[nodes, cells])
    assert not np.any(unit_sd.face_nodes[nodes, faces])


def test_opposite_nodes_0d_grid(ref_sd_0d):
    opposite_node = ref_sd_0d.compute_opposite_nodes()
    assert opposite_node.nnz == 0
    assert opposite_node.shape == (0, 1)


def test_non_simplicial_grid(ref_square):
    with pytest.raises(NotImplementedError):
        ref_square.compute_opposite_nodes()


@pytest.fixture
def concave_quad() -> pg.Grid:
    nodes = np.array([[0, 0.5, 1, 0.5], [0, 0.5, 0, 1], np.zeros(4)])
    indices = np.array([0, 1, 1, 2, 2, 3, 3, 0])
    face_nodes = sps.csc_array((np.ones(8), indices, np.arange(0, 9, 2)))
    cell_faces = sps.csc_array(np.ones((4, 1)))

    sd = pp.Grid(2, nodes, face_nodes, cell_faces, "concave")
    pg.convert_from_pp(sd)
    sd.compute_geometry()

    return sd


@pytest.fixture
def multiple_concave_quads() -> pg.Grid:
    nodes = np.array(
        [[0, 0.5, 1, 0.5, 0.5, 0.5], [0, 0.5, 0, 1, -0.5, -1], np.zeros(6)]
    )
    indices = np.array([0, 1, 1, 2, 2, 3, 3, 0, 0, 5, 5, 2, 2, 4, 4, 0])
    face_nodes = sps.csc_array((np.ones(16), indices, np.arange(0, 17, 2)))
    cell_faces_j = np.repeat(np.arange(3), 4)
    cell_faces_i = np.array([0, 1, 2, 3, 7, 6, 1, 0, 4, 5, 6, 7])
    cell_faces_v = np.array([1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1])
    cell_faces = sps.csc_array((cell_faces_v, (cell_faces_i, cell_faces_j)))

    sd = pp.Grid(2, nodes, face_nodes, cell_faces, "concave")
    pg.convert_from_pp(sd)
    sd.compute_geometry()

    return sd


def test_quads(unit_cart_sd):
    sub_volumes, sub_simplices = unit_cart_sd.compute_subvolumes(True)

    assert np.allclose(unit_cart_sd.cell_volumes, np.sum(sub_volumes, 0))

    sub_simplices_known = unit_cart_sd.cell_faces.copy().astype(float)
    factor = [0, 2, 4, 6]
    sub_simplices_known.data[:] = (
        unit_cart_sd.cell_volumes[0] / factor[unit_cart_sd.dim]
    )

    assert np.allclose((sub_simplices - sub_simplices_known).data, 0)


def test_simplices(unit_sd):
    sub_volumes, sub_simplices = unit_sd.compute_subvolumes(True)

    assert np.allclose(unit_sd.cell_volumes, np.sum(sub_volumes, 0))
    assert np.allclose(unit_sd.cell_volumes, np.sum(sub_simplices, 0))


def test_poly_grids(unit_poly_sd):
    sub_volumes, sub_simplices = unit_poly_sd.compute_subvolumes(True)

    assert np.allclose(unit_poly_sd.cell_volumes, np.sum(sub_volumes, 0))
    assert np.allclose(unit_poly_sd.cell_volumes, np.sum(sub_simplices, 0))


def test_hitchhiker_pentagon(pentagon_sd):
    sub_volumes = pentagon_sd.compute_subvolumes()
    assert np.allclose(pentagon_sd.cell_volumes, np.sum(sub_volumes, 0))


def test_subvolumes_concave_quad(concave_quad):
    sd = concave_quad
    sub_volumes = sd.compute_subvolumes()
    assert np.allclose(sd.cell_volumes, np.sum(sub_volumes, 0))


def test_subvolumes_multiple_concave_quads(multiple_concave_quads):
    sd = multiple_concave_quads
    sub_volumes = sd.compute_subvolumes()

    known_sub_volumes = np.array(
        [
            [6.25e-02, 1.25e-01, 6.25e-02],
            [0, 1.25e-01, 0],
            [6.25e-02, 1.25e-01, 6.25e-02],
            [1.25e-01, 0, 0],
            [0, 1.25e-01, 0],
            [0, 0, 1.25e-01],
        ]
    )
    assert np.allclose(sub_volumes.todense(), known_sub_volumes)


def test_subsimplices_cartgrids(unit_cart_sd: pg.Grid):
    _, volumes = unit_cart_sd.compute_simplicial_subgrid()
    assert np.allclose(volumes, volumes[0]), f"Subvolumes {volumes} are not all equal"


def test_subsimplices_multiconcave(multiple_concave_quads):
    _, volumes = multiple_concave_quads.compute_simplicial_subgrid()

    assert np.any(np.isclose(volumes, 0)), f"No zero subvolume in {volumes}"
    assert np.isclose(np.sum(volumes), 1), f"Sum of subvolumes {np.sum(volumes)}, not 1"


def test_edge_tangents_0d(ref_sd_0d):
    assert ref_sd_0d.edge_nodes.shape == (ref_sd_0d.num_nodes, 0)
    assert len(ref_sd_0d.edge_tangents) == 0
    assert len(ref_sd_0d.edge_lengths) == 0
    assert ref_sd_0d.mesh_size == 0


def test_cartgrids(unit_cart_sd):
    h = unit_cart_sd.edge_lengths[0]
    assert np.allclose(unit_cart_sd.edge_lengths, h)
    assert np.allclose(unit_cart_sd.mesh_size, h)


def test_edge_tangents_2d_tris():
    sd = pp.StructuredTriangleGrid([1] * 2)
    pg.convert_from_pp(sd)
    sd.compute_geometry()

    unique_lengths = np.unique(sd.edge_lengths)
    known_lengths = np.sqrt(np.arange(1, 3))

    assert np.allclose(unique_lengths, known_lengths)
    assert np.allclose(sd.edge_lengths, sd.face_areas)
    assert np.allclose(np.sum(sd.edge_tangents * sd.face_normals, axis=0), 0)

    known_meshsize = (4 + np.sqrt(2)) / 5
    assert np.allclose(sd.mesh_size, known_meshsize)


def test_grid_3d_tets():
    sd = pp.StructuredTetrahedralGrid([1] * 3)
    pg.convert_from_pp(sd)
    sd.compute_geometry()

    unique_lengths = np.unique(sd.edge_lengths)
    known_lengths = np.sqrt(np.arange(1, 4))

    assert np.allclose(unique_lengths, known_lengths)

    known_meshsize = (12 + 6 * np.sqrt(2) + np.sqrt(3)) / 19
    assert np.allclose(sd.mesh_size, known_meshsize)


@pytest.mark.parametrize(
    "dim, known",
    [(1, [2, 1, 0, 0]), (2, [3, 3, 1, 0]), (3, [4, 6, 4, 1])],
)
def test_num_entities_of_reference_element(_ref_elements_dict, dim, known):
    assert np.array_equal(_ref_elements_dict[dim].num_entities(), known)


def test_num_entities(unit_sd):
    num_entities = unit_sd.num_entities()

    assert num_entities[0] == unit_sd.num_nodes
    assert num_entities[1] == unit_sd.num_edges
    assert num_entities[unit_sd.dim] == unit_sd.num_cells
    assert np.all(num_entities[unit_sd.dim + 1 :] == 0)


def test_num_entities_of_point_grid(ref_sd_0d):
    # a point grid has a single cell and no nodes
    assert np.array_equal(ref_sd_0d.num_entities(), [1, 0, 0, 0])
