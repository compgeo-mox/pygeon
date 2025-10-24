import unittest

import pygeon as pg


class MatPwConstantsTest(unittest.TestCase):
    def test_ndof(self):
        dim = 2
        sd = pg.reference_element(dim)
        sd.compute_geometry()

        discr = pg.MatPwConstants()
        self.assertTrue(discr.ndof(sd) == 4)


if __name__ == "__main__":
    unittest.main()
