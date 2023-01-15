import unittest
import numpy as np

import porepy as pp
import pygeon as pg
import scipy.sparse as sps


class OctGridTest(unittest.TestCase):
    def test_simple_case(self):
        nx = [5, 5]
        sd = pg.OctGrid(nx)
        sd.compute_geometry()

        pp.plot_grid(sd, info="all", alpha=0)


if __name__ == "__main__":
    OctGridTest().test_simple_case()
