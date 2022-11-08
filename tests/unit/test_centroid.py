""" Module contains a dummy unit test that always passes.
"""
import unittest
import numpy as np

import porepy as pp
import pygeon as pg


class CentroidTest(unittest.TestCase):
    def test_centroid(self):
        sd = pp.CartGrid([1, 1])
        pg.convert_from_pp(sd)
        sd.compute_geometry()


if __name__ == "__main__":
    CentroidTest().test_centroid()
    # unittest.main()
