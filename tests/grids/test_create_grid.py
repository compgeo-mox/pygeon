"""Module contains a unit test for the standard grid computation."""

import numpy as np
import porepy as pp

import pygeon as pg


def check_grid(sd: pg.Grid, dim: int, boundary: np.ndarray):
    assert sd.dim == dim

    # check if the boundary faces belong to the imposed boundary
    faces = sd.tags["domain_boundary_faces"]
    bf = sd.face_centers[:, faces]

    # check if the boundary nodes belong to the imposed boundary
    nodes = sd.tags["domain_boundary_nodes"]
    bn = sd.nodes[:, nodes]

    face_on_boundary = np.zeros(faces.sum(), dtype=bool)
    node_on_boundary = np.zeros(nodes.sum(), dtype=bool)
    for side in boundary:
        if dim == 2:
            dist_faces, _ = pp.geometry.distances.points_segments(
                bf[:2, :], side[:, 0], side[:, 1]
            )
            dist_nodes, _ = pp.geometry.distances.points_segments(
                bn[:2, :], side[:, 0], side[:, 1]
            )
        else:
            dist_faces, _, _ = pp.geometry.distances.points_polygon(bf, side)
            dist_nodes, _, _ = pp.geometry.distances.points_polygon(bn, side)

        face_on_boundary[np.isclose(dist_faces, 0).ravel()] = True
        node_on_boundary[np.isclose(dist_nodes, 0).ravel()] = True

    assert np.all(face_on_boundary)
    assert np.all(node_on_boundary)


def test_unit_square_mdg():
    mdg = pg.unit_grid(2, mesh_size=0.5)
    sd = mdg.subdomains(dim=2)[0]

    side_1 = np.array([[0, 1], [0, 0]])
    side_2 = np.array([[1, 1], [0, 1]])
    side_3 = np.array([[1, 0], [1, 1]])
    side_4 = np.array([[0, 0], [1, 0]])
    boundary = [side_1, side_2, side_3, side_4]

    check_grid(sd, 2, boundary)


def test_unit_square():
    sd = pg.unit_grid(2, mesh_size=0.5, as_mdg=False)

    side_1 = np.array([[0, 1], [0, 0]])
    side_2 = np.array([[1, 1], [0, 1]])
    side_3 = np.array([[1, 0], [1, 1]])
    side_4 = np.array([[0, 0], [1, 0]])
    boundary = [side_1, side_2, side_3, side_4]

    check_grid(sd, 2, boundary)


def test_unit_cube():
    sd = pg.unit_grid(3, mesh_size=0.5, as_mdg=False)

    side_1 = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]])
    side_2 = np.array([[1, 1, 1, 1], [0, 1, 1, 0], [0, 0, 1, 1]])
    side_3 = np.array([[0, 1, 1, 0], [0, 0, 0, 0], [0, 0, 1, 1]])
    side_4 = np.array([[0, 0, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]])
    side_5 = np.array([[0, 1, 1, 0], [1, 1, 1, 1], [0, 0, 1, 1]])
    side_6 = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [1, 1, 1, 1]])
    boundary = [side_1, side_2, side_3, side_4, side_5, side_6]

    check_grid(sd, 3, boundary)


def test_triangle():
    side_1 = np.array([[0, 1], [0, 0]])
    side_2 = np.array([[1, 0], [0, 1]])
    side_3 = np.array([[0, 0], [1, 0]])
    boundary = [side_1, side_2, side_3]

    domain = pp.Domain(polytope=boundary)
    sd = pg.grid_from_domain(domain, mesh_size=0.5, as_mdg=False)

    check_grid(sd, 2, boundary)


def test_concave_pentagon():
    side_1 = np.array([[0, 1], [0, 0]])
    side_2 = np.array([[1, 1], [0, 1]])
    side_3 = np.array([[1, 0.5], [1, 0.5]])
    side_4 = np.array([[0.5, 0], [0.5, 1]])
    side_5 = np.array([[0, 0], [1, 0]])
    boundary = [side_1, side_2, side_3, side_4, side_5]

    domain = pp.Domain(polytope=boundary)
    sd = pg.grid_from_domain(domain, mesh_size=0.5, as_mdg=False)

    check_grid(sd, 2, boundary)


def test_triangle_from_pts():
    pts = np.array([[0, 1, 0], [0, 0, 1]])
    sd = pg.grid_from_boundary_pts(pts, mesh_size=0.5, as_mdg=False)

    side_1 = np.array([[0, 1], [0, 0]])
    side_2 = np.array([[1, 0], [0, 1]])
    side_3 = np.array([[0, 0], [1, 0]])
    boundary = [side_1, side_2, side_3]

    check_grid(sd, 2, boundary)


def test_concave_pentagon_from_pts():
    pts = np.array([[0, 1, 1, 0.5, 0], [0, 0, 1, 0.5, 1]])
    sd = pg.grid_from_boundary_pts(pts, mesh_size=0.5, as_mdg=False)

    side_1 = np.array([[0, 1], [0, 0]])
    side_2 = np.array([[1, 1], [0, 1]])
    side_3 = np.array([[1, 0.5], [1, 0.5]])
    side_4 = np.array([[0.5, 0], [0.5, 1]])
    side_5 = np.array([[0, 0], [1, 0]])
    boundary = [side_1, side_2, side_3, side_4, side_5]

    check_grid(sd, 2, boundary)


def test_unit_cube2(unit_sd_3d):
    # This hardcoded test may fail if gmsh or porepy decides to update their meshing
    # algorithm.

    assert np.isclose(unit_sd_3d.num_cells, 100)
    assert np.isclose(unit_sd_3d.num_faces, 242)
    assert np.isclose(unit_sd_3d.num_nodes, 45)
