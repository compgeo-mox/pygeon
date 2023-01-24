import numpy as np
import porepy as pp
import pygeon as pg
import unittest


class VoronoiTest(unittest.TestCase):
    def test_simple_voronoi_grid(self):
        pts = np.array(
            [
                [0.1, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 1.5],
                [0.1, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 1.5],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )

        sd = pg.VoronoiGrid(4, 200, 0)

        sd.compute_geometry()

        # pp.plot_grid(sd, info="all", alpha=0)
        # self.assertTrue(sd.num_faces == 9)
        exp = pp.Exporter(sd, "voronoi")
        exp.write_vtu()


if __name__ == "__main__":
    VoronoiTest().test_simple_voronoi_grid()
