import unittest
import numpy as np

import porepy as pp
import pygeon as pg


class ConvertFromPpTest(unittest.TestCase):
    def test_convert_graph(self):
        class DummyGraph:
            pass

        g = DummyGraph()

        self.assertRaises(TypeError, pg.convert_from_pp, g)
        self.assertRaises(ValueError, pg.as_mdg, g)

        g.__class__ = pg.Graph

        pg.convert_from_pp(g)
        # No conversion is needed, so the object should remain unchanged
        self.assertIsInstance(g, pg.Graph)

    def test_convert_grid(self):
        dim = 2
        sd = pp.StructuredTriangleGrid([2] * dim, [1] * dim)

        mdg = pg.as_mdg(sd)
        self.assertIsInstance(mdg, pp.MixedDimensionalGrid)

        pg.convert_from_pp(sd)
        self.assertIsInstance(sd, pg.Grid)

    def test_convert_mortar_grid(self):
        mesh_args = {"cell_size": 0.25, "cell_size_fracture": 0.125}
        x_endpoints = [np.array([0, 0.5])]
        mdg, _ = pp.mdg_library.square_with_orthogonal_fractures(
            "simplex", mesh_args, [1], x_endpoints
        )

        mg = mdg.interfaces()[0]
        pg.convert_from_pp(mg)

        # The object should be converted to pg.MortarGrid
        self.assertIsInstance(mg, pg.MortarGrid)

    def test_convert_mixed_dimensional_grid(self):
        mesh_args = {"cell_size": 0.25, "cell_size_fracture": 0.125}
        x_endpoints = [np.array([0, 0.5])]
        mdg, _ = pp.mdg_library.square_with_orthogonal_fractures(
            "simplex", mesh_args, [1], x_endpoints
        )

        mdg = pg.as_mdg(mdg)
        self.assertIsInstance(mdg, pp.MixedDimensionalGrid)

        pg.convert_from_pp(mdg)

        # The subdomains and interfaces should be recursively converted
        self.assertIsInstance(mdg.subdomains(dim=2)[0], pg.Grid)
        self.assertIsInstance(mdg.subdomains(dim=1)[0], pg.Grid)
        self.assertIsInstance(mdg.interfaces()[0], pg.MortarGrid)

        # The object should be converted to pg.MixedDimensionalGrid
        self.assertIsInstance(mdg, pg.MixedDimensionalGrid)


if __name__ == "__main__":
    unittest.main()
