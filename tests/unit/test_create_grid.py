""" 
Module contains a unit test for the standard grid computation.
"""
import unittest
import numpy as np

import porepy as pp
import pygeon as pg


class StandardGridTest(unittest.TestCase):
    def check_grid(self, sd, dim, boundary):
        self.assertTrue(sd.dim == dim)

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

        self.assertTrue(np.all(face_on_boundary))
        self.assertTrue(np.all(node_on_boundary))

    def test_unit_square(self):
        mesh_size = 0.5
        sd = pg.unit_grid(2, mesh_size, as_mdg=False)

        side_1 = np.array([[0, 1], [0, 0]])
        side_2 = np.array([[1, 1], [0, 1]])
        side_3 = np.array([[1, 0], [1, 1]])
        side_4 = np.array([[0, 0], [1, 0]])
        boundary = [side_1, side_2, side_3, side_4]
        self.check_grid(sd, 2, boundary)

    def test_unit_cube(self):
        mesh_size = 0.5
        sd = pg.unit_grid(3, mesh_size, as_mdg=False)

        side_1 = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]])
        side_2 = np.array([[1, 1, 1, 1], [0, 1, 1, 0], [0, 0, 1, 1]])
        side_3 = np.array([[0, 1, 1, 0], [0, 0, 0, 0], [0, 0, 1, 1]])
        side_4 = np.array([[0, 0, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]])
        side_5 = np.array([[0, 1, 1, 0], [1, 1, 1, 1], [0, 0, 1, 1]])
        side_6 = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [1, 1, 1, 1]])
        boundary = [side_1, side_2, side_3, side_4, side_5, side_6]
        self.check_grid(sd, 3, boundary)

    def test_triangle(self):
        side_1 = np.array([[0, 1], [0, 0]])
        side_2 = np.array([[1, 0], [0, 1]])
        side_3 = np.array([[0, 0], [1, 0]])
        boundary = [side_1, side_2, side_3]
        domain = pp.Domain(polytope=boundary)

        mesh_size = 0.5
        sd = pg.grid_from_domain(domain, mesh_size, as_mdg=False)

        self.check_grid(sd, 2, boundary)

    def test_concave_pentagon(self):
        side_1 = np.array([[0, 1], [0, 0]])
        side_2 = np.array([[1, 1], [0, 1]])
        side_3 = np.array([[1, 0.5], [1, 0.5]])
        side_4 = np.array([[0.5, 0], [0.5, 1]])
        side_5 = np.array([[0, 0], [1, 0]])
        boundary = [side_1, side_2, side_3, side_4, side_5]
        domain = pp.Domain(polytope=boundary)

        mesh_size = 0.5
        sd = pg.grid_from_domain(domain, mesh_size, as_mdg=False)

        self.check_grid(sd, 2, boundary)

    def test_triangle_from_pts(self):
        pts = np.array([[0, 1, 0], [0, 0, 1]])

        mesh_size = 0.5
        sd = pg.grid_from_boundary_pts(pts, mesh_size, as_mdg=False)

        side_1 = np.array([[0, 1], [0, 0]])
        side_2 = np.array([[1, 0], [0, 1]])
        side_3 = np.array([[0, 0], [1, 0]])
        boundary = [side_1, side_2, side_3]
        self.check_grid(sd, 2, boundary)

    def test_concave_pentagon_from_pts(self):
        pts = np.array([[0, 1, 1, 0.5, 0], [0, 0, 1, 0.5, 1]])
        mesh_size = 0.5
        sd = pg.grid_from_boundary_pts(pts, mesh_size, as_mdg=False)

        side_1 = np.array([[0, 1], [0, 0]])
        side_2 = np.array([[1, 1], [0, 1]])
        side_3 = np.array([[1, 0.5], [1, 0.5]])
        side_4 = np.array([[0.5, 0], [0.5, 1]])
        side_5 = np.array([[0, 0], [1, 0]])
        boundary = [side_1, side_2, side_3, side_4, side_5]
        self.check_grid(sd, 2, boundary)

    def test_unit_cube(self):
        mesh_size = 0.5
        sd = pg.unit_grid(3, mesh_size, as_mdg=False)

        self.assertTrue(np.isclose(sd.num_cells, 100))
        self.assertTrue(np.isclose(sd.num_faces, 242))
        self.assertTrue(np.isclose(sd.num_nodes, 45))


if __name__ == "__main__":
    unittest.main()
