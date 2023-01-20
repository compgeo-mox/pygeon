import numpy as np
import porepy as pp
import pygeon as pg
import unittest


class VLagrange1Test(unittest.TestCase):
    def test_simple_voronoi_grid(self):
        pts = np.array(
            [
                [0.1, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 1.5],
                [0.1, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 1.5],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        np.random.seed(0)
        pts = np.random.rand(2, 50)

        sd = pg.VoronoiGrid(pts)

        sd.compute_geometry()

        pp.plot_grid(sd, info="all", alpha=0)
        self.assertTrue(sd.num_faces == 9)


if __name__ == "__main__":
    VLagrange1Test().test_simple_voronoi_grid()
