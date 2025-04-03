import unittest
import numpy as np
import porepy as pp
import pygeon as pg  # type: ignore[import-untyped]


class PwConstantsTest(unittest.TestCase):
    def create_cart_grid(self, dim, num_cells):
        """
        Create a grid for testing.
        """
        sd = pp.CartGrid([num_cells] * dim, [1] * dim)
        sd.compute_geometry()

        return sd

    def test_pwconstants_assemble_mass_matrix(self):
        """
        Test the assembly of the mass matrix for the piecewise constants.
        """
        for dim in np.arange(1, 4):
            sd = self.create_cart_grid(dim, 2)

            discr = pg.PwConstants("test")
            M = discr.assemble_mass_matrix(sd)

            assert np.allclose(M.diagonal(), 1 / sd.cell_volumes)

    def test_pwconstants_lumped_mass_matrix(self):
        """
        Test the lumped mass matrix for the piecewise constants.
        """
        for dim in np.arange(1, 4):
            sd = self.create_cart_grid(dim, 2)

            discr = pg.PwConstants("test")
            M = discr.assemble_lumped_matrix(sd)

        assert np.allclose(M.diagonal(), 1 / sd.cell_volumes)

    def test_pwconstants_assemble_diff_matrix(self):
        """
        Test the assembly of the differential matrix for the piecewise constants.
        """
        for dim in np.arange(1, 4):
            sd = self.create_cart_grid(dim, 2)

            discr = pg.PwConstants("test")
            assert discr.assemble_diff_matrix(sd).nnz == 0

    def test_pwconstants_assemble_stiff_matrix(self):
        """
        Test the assembly of the stiffness matrix for the piecewise constants.
        """
        for dim in np.arange(1, 4):
            sd = self.create_cart_grid(dim, 2)

            discr = pg.PwConstants("test")
            assert discr.assemble_stiff_matrix(sd).nnz == 0

    def test_pwconstants_interpolate(self):
        """
        Test the interpolation of a function to the piecewise constants.
        """

        def func(x):
            return x[0] + x[1]

        for dim in np.arange(1, 4):
            sd = self.create_cart_grid(dim, 2)

            discr = pg.PwConstants("test")

            num_sol = discr.interpolate(sd, func)
            ana_sol = np.array([func(x) for x in sd.cell_centers.T]) * sd.cell_volumes

            assert np.allclose(num_sol, ana_sol)

    def test_pwconstants_assemble_nat_bc(self):
        """
        Test the assembly of the natural boundary conditions for the piecewise
        constants.
        """
        for dim in np.arange(1, 4):
            sd = self.create_cart_grid(dim, 2)

            discr = pg.PwConstants("test")
            assert np.allclose(
                discr.assemble_nat_bc(sd, None, None), np.zeros(sd.num_cells)
            )


if __name__ == "__main__":
    unittest.main()
