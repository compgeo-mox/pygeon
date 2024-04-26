""" Module contains a unit test for the Einstein grid class.
"""

import unittest
import os
import numpy as np
import scipy.sparse as sps

# import numpy as np
# import scipy.sparse as sps
import porepy as pp

import pygeon as pg


class TestRegularizer(unittest.TestCase):
    def test_lloyd(self):
        sd = pg.VoronoiGrid(30, seed=0)
        sd.compute_geometry()
        sd = pg.lloyd_regularization(sd, 15)

        self.assertTrue(sd.num_cells == 30)

    def test_graph_laplace(self):
        sd = pg.VoronoiGrid(30, seed=0)
        sd.compute_geometry()
        fr_known = sd.face_ridges
        cf_known = sd.cell_faces

        sd = pg.graph_laplace_regularization(sd)

        self.assertTrue((sd.face_ridges - fr_known).nnz == 0)
        self.assertTrue((sd.cell_faces - cf_known).nnz == 0)

    def test_elasticity(self):
        sd = pg.VoronoiGrid(100, seed=0)
        sd.compute_geometry()

        cond_old = self.compute_cond(sd)

        for _ in np.arange(20):
            sd = pg.elasticity_regularization(sd, 10, is_square=False)
            pp.plot_grid(sd, alpha=0)

        cond_new = self.compute_cond(sd)

        self.assertTrue(cond_old > cond_new)

    def test_einstein(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        folder = os.path.join(dir_path, "einstein_svg_grids")
        file_name = os.path.join(folder, "H2.svg")

        sd = pg.EinSteinGrid(file_name)
        sd.compute_geometry()
        sd = pg.graph_laplace_regularization(sd)

        pp.plot_grid(sd, alpha=0)

        pass

    def compute_cond(self, sd):
        discr = pg.VLagrange1("dumb")
        A = discr.assemble_stiff_matrix(sd)

        ew1 = sps.linalg.eigsh(A, 1, which="LM", return_eigenvectors=False)
        ew2 = sps.linalg.eigsh(A, 2, which="SM", return_eigenvectors=False).max()

        return ew1 / ew2


if __name__ == "__main__":
    unittest.main()
