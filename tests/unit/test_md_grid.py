import unittest
import numpy as np

import porepy as pp
import pygeon as pg


class MDGridTest(unittest.TestCase):
    def test0(self):
        mdg = pg.MixedDimensionalGrid()

        mesh_args = {"cell_size": 0.25, "cell_size_fracture": 0.125}
        x_endpoints = [np.array([0, 0.5])]
        mdg, _ = pp.mdg_library.square_with_orthogonal_fractures(
            "simplex", mesh_args, [1], x_endpoints
        )
        pg.convert_from_pp(mdg)
        mdg.compute_geometry()

        self.assertEqual(mdg.num_subdomain_faces(), 132)
        self.assertEqual(mdg.num_subdomain_ridges(), 52)


if __name__ == "__main__":
    unittest.main()
