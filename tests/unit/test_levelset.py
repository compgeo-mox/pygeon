import unittest

import numpy as np
import porepy as pp
import scipy.sparse as sps

import pygeon as pg
import porepy as pp


def line_level_set_1(x):
    return x[1] - 0.75


def line_level_set_2(x):
    return x[1] - 0.7


def circle_level_set(x):
    return 0.41 - np.linalg.norm(x - np.array([0.5, 0.5, 0]))


# sd = pg.unit_g alpha=0, plot_2d=False)


class LevelsetGridTest(unittest.TestCase):
    def trianglegrid_test(self):
        # sd = pg.unit_grid(2, 1 / 5, as_mdg=False)
        # sd = pp.StructuredTriangleGrid([2] * 2, [1] * 2)
        sd = pp.CartGrid([17] * 2, [1] * 2)

        sd = pg.levelset_remesh(sd, line_level_set_1)
        sd = pg.levelset_remesh(sd, circle_level_set)

        pp.plot_grid(sd, info="n", alpha=0)

        pass


if __name__ == "__main__":
    LevelsetGridTest().trianglegrid_test()
