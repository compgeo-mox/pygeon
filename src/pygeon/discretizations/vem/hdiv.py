from typing import Callable, Optional

import numpy as np
import porepy as pp
import scipy.sparse as sps

import pygeon as pg


class MVEM(pg.Discretization, pp.MVEM):
    """
    MVEM class for the mixed virtual element method (MVEM) discretization.

    Each degree of freedom is the integral over a mesh face.

    Args:
        keyword (str): The keyword for the discretization.

    Attributes:
        keyword (str): The keyword for the discretization.

    Methods:
        ndof(sd: pg.Grid) -> int:
            Returns the number of faces.

        assemble_mass_matrix(sd: pg.Grid, data: Optional[dict] = None) -> sps.csc_matrix:
            Assembles the mass matrix.

        assemble_lumped_matrix(sd: pg.Grid, data: Optional[dict] = None) -> sps.csc_matrix:
            Assembles the lumped mass matrix.

        assemble_diff_matrix(sd: pg.Grid) -> sps.csc_matrix:
            Assembles the matrix corresponding to the differential operator.

        interpolate(sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
            Interpolates a function onto the finite element space.

        eval_at_cell_centers(sd: pg.Grid, data: Optional[dict] = None) -> sps.csc_matrix:
            Assembles the matrix.

        assemble_nat_bc(sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray], b_faces: np.ndarray) -> np.ndarray: Assembles the natural boundary condition term.
        get_range_discr_class(dim: int) -> pg.Discretization: Returns the range discretization class for the given dimension.
    """

    def __init__(self, keyword: str) -> None:
        """
        Initialize the MVEM class.

        Args:
            keyword (str): The keyword for the discretization.

        Returns:
            None
        """
        pg.Discretization.__init__(self, keyword)
        pp.MVEM.__init__(self, keyword)

    def ndof(self, sd: pg.Grid) -> int:
        """
        Returns the number of faces.

        Args:
            sd (pg.Grid): grid, or a subclass.

        Returns:
            int: the number of degrees of freedom.
        """
        return sd.num_faces

    def assemble_mass_matrix(
        self, sd: pg.Grid, data: Optional[dict] = None
    ) -> sps.csc_matrix:
        """
        Assembles the mass matrix.

        Args:
            sd: grid, or a subclass.
            data: optional dictionary with physical parameters for scaling.

        Returns:
            sps.csc_matrix: the mass matrix.
        """
        rt0 = pg.RT0(self.keyword)
        data = rt0.create_dummy_data(sd, data)

        pp.MVEM.discretize(self, sd, data)
        return data[pp.DISCRETIZATION_MATRICES][self.keyword][self.mass_matrix_key]

    def assemble_lumped_matrix(
        self, sd: pg.Grid, data: Optional[dict] = None
    ) -> sps.csc_matrix:
        """
        Assembles the lumped mass matrix L such that
        B^T L^{-1} B is a TPFA method.

        Args:
            sd (pg.Grid): Grid object or a subclass.
            data (Optional[dict]): Optional dictionary with physical parameters for scaling.

        Returns:
            sps.csc_matrix: The lumped mass matrix.
        """
        rt0 = pg.RT0(self.keyword)
        return rt0.assemble_lumped_matrix(sd, data)

    def assemble_diff_matrix(self, sd: pg.Grid) -> sps.csc_matrix:
        """
        Assembles the matrix corresponding to the differential operator.

        Args:
            sd (pg.Grid): Grid object representing the discretized domain.

        Returns:
            sps.csc_matrix: The differential matrix.
        """
        return sd.cell_faces.T.tocsc()

    def interpolate(
        self, sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray]
    ) -> np.ndarray:
        """
        Interpolates a function onto the finite element space.

        Args:
            sd (pg.Grid): Grid, or a subclass.
            func (Callable[[np.ndarray], np.ndarray]): A function that returns the function
                values at coordinates.

        Returns:
            np.ndarray: The values of the degrees of freedom.
        """
        vals = [
            np.inner(func(x).flatten(), normal)
            for (x, normal) in zip(sd.face_centers.T, sd.face_normals.T)
        ]
        return np.array(vals)

    def eval_at_cell_centers(
        self, sd: pg.Grid, data: Optional[dict] = None
    ) -> sps.csc_matrix:
        """
        Assembles the matrix.

        Args:
            sd (pg.Grid): Grid or a subclass.
            data (Optional[dict]): Optional data dictionary.

        Returns:
            sps.csc_matrix: The evaluation matrix.
        """
        rt0 = pg.RT0(self.keyword)
        data = rt0.create_dummy_data(sd, data)

        pp.MVEM.discretize(self, sd, data)
        return data[pp.DISCRETIZATION_MATRICES][self.keyword][self.vector_proj_key]

    def assemble_nat_bc(
        self, sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray], b_faces: np.ndarray
    ) -> np.ndarray:
        """
        Assembles the natural boundary condition term
        (n dot q, func)_\Gamma.

        Args:
            sd (pg.Grid): The grid object representing the computational domain.
            func (Callable[[np.ndarray], np.ndarray]): The function to be
                evaluated on the boundary.
            b_faces (np.ndarray): The array of boundary faces.

        Returns:
            np.ndarray: The assembled natural boundary condition term.
        """
        rt0 = pg.RT0(self.keyword)
        return rt0.assemble_nat_bc(sd, func, b_faces)

    def get_range_discr_class(self, dim: int) -> pg.Discretization:
        """
        Returns the range discretization class for the given dimension.

        Args:
            dim (int): The dimension of the range.

        Returns:
            pg.Discretization: The range discretization class.
        """
        return pg.PwConstants


class VBDM1(pg.Discretization):
    """
    Virtual Element Method (VEM) based on the BDM1 (Brezzi-Douglas-Marini) discretization
    for the H(div) space.

    This class implements the VEM discretization for the H(div) space.
    It provides methods for assembling the mass matrix, projecting to VRT0 space,
    assembling the differential matrix, evaluating at cell centers, interpolating
    a function, assembling the natural boundary condition term, and more.

    Attributes:
        keyword (str): The keyword associated with the discretization.

    Methods:
        ndof(sd: pg.Grid) -> int:
            Returns the number of faces time the dimension.

        assemble_mass_matrix(sd: pg.Grid, data: Optional[dict] = None) -> sps.csc_matrix:
            Assembles the mass matrix.

        assemble_lumped_matrix(sd: pg.Grid, data: Optional[dict] = None) -> sps.csc_matrix:
            Assembles the lumped mass matrix.

        assemble_diff_matrix(sd: pg.Grid) -> sps.csc_matrix:
            Assembles the matrix corresponding to the differential operator.

        interpolate(sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
            Interpolates a function onto the finite element space.

        eval_at_cell_centers(sd: pg.Grid, data: Optional[dict] = None) -> sps.csc_matrix:
            Assembles the matrix.

        assemble_nat_bc(sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray], b_faces: np.ndarray) -> np.ndarray:
            Assembles the natural boundary condition term.

        get_range_discr_class(dim: int) -> pg.Discretization:
            Returns the range discretization class for the given dimension.
    """

    def ndof(self, sd: pg.Grid) -> int:
        """
        Return the number of degrees of freedom associated to the method.
        In this case, it returns the number of faces times the dimension.

        Args:
            sd (pg.Grid): The grid object for which to calculate the number
                of degrees of freedom.

        Returns:
            int: The number of degrees of freedom.

        Raises:
            ValueError: If the input `sd` is not an instance of `pg.Grid`.
        """
        if isinstance(sd, pg.Grid):
            return sd.face_nodes.nnz
        else:
            raise ValueError

    def assemble_mass_matrix(
        self, sd: pg.Grid, data: Optional[dict] = None
    ) -> sps.csc_matrix:
        """
        Computes the mass matrix for the Virtual Element Method (VEM).

        Args:
            sd (pg.Grid): The grid object representing the computational domain.
            data (Optional[dict]): Optional data dictionary.

        Returns:
            sps.csc_matrix: The assembled mass matrix.

        Notes:
            The mass matrix is computed using the VEM approach.
            The mass matrix is a sparse matrix in compressed sparse column (CSC) format.
        """
        # Allocate the data to store matrix entries
        cell_nodes = sd.cell_nodes()
        size = int(np.sum(np.square(2 * np.sum(cell_nodes, 0))))

        rows_I = np.empty(size, dtype=int)
        cols_J = np.empty(size, dtype=int)
        data_V = np.empty(size)
        idx = 0

        dof = self.get_dof_enumeration(sd)
        disc_VL1 = pg.VLagrange1("dummy")

        tangents = sd.nodes * sd.face_ridges
        cell_diams = sd.cell_diameters(cell_nodes)

        for cell, diam in enumerate(cell_diams):
            faces_loc = sd.cell_faces[:, cell].indices

            # Obtain local indices of dofs, ordered by associated node number
            local_dof = dof[:, faces_loc].tocsr().tocoo()
            dof_indx = local_dof.data
            dof_node = local_dof.row
            dof_face = faces_loc[local_dof.col]

            # Compute the values of the basis functions
            swapper = np.arange(dof_face.size)
            swapper[::2] += 1
            swapper[1::2] -= 1
            swapped_tangents = tangents[:, dof_face[swapper]]

            BDM_basis = swapped_tangents / np.sum(
                swapped_tangents * sd.face_normals[:, dof_face], axis=0
            )

            vals = BDM_basis.T @ BDM_basis
            VL_mass = disc_VL1.assemble_loc_mass_matrix(sd, cell, diam, dof_node[::2])
            VL_mass = np.kron(VL_mass, np.ones((2, 2)))

            A = np.multiply(vals, VL_mass)

            # Save values for the local matrix in the global structure
            cols = np.tile(dof_indx, (dof_indx.size, 1))
            loc_idx = slice(idx, idx + cols.size)

            rows_I[loc_idx] = cols.T.ravel()
            cols_J[loc_idx] = cols.ravel()
            data_V[loc_idx] = A.ravel()
            idx += cols.size

        # Construct the global matrices
        return sps.csc_matrix((data_V, (rows_I, cols_J)))

    def proj_to_VRT0(self, sd: pg.Grid) -> sps.csc_matrix:
        """
        Project the degrees of freedom to the space VRT0.

        Args:
            sd (pg.Grid): The grid on which to project the degrees of freedom.

        Returns:
            sps.csc_matrix: The projection matrix.
        """
        dof = self.get_dof_enumeration(sd).tocoo()
        return sps.csc_matrix((np.ones(self.ndof(sd)), (dof.col, dof.data))) / 2

    def proj_from_RT0(self, sd: pg.Grid) -> sps.csc_matrix:
        """
        Project the RT0 finite element space onto the H(div) finite element space.

        Args:
            sd (pg.Grid): The grid on which the projection is performed.

        Returns:
            sps.csc_matrix: The projection matrix.

        Raises:
            NotImplementedError: This method is not implemented and should be
                overridden in a subclass.
        """
        raise NotImplementedError

    def assemble_diff_matrix(self, sd: pg.Grid) -> sps.csc_matrix:
        """
        Assembles the matrix corresponding to the differential operator for the H(div) space.

        Args:
            sd (pg.Grid): The grid or a subclass.

        Returns:
            sps.csc_matrix: The differential matrix.
        """
        mvem = pg.MVEM(self.keyword)
        VRT0_diff = mvem.assemble_diff_matrix(sd)

        proj_to_vrt0 = self.proj_to_VRT0(sd)
        return VRT0_diff @ proj_to_vrt0

    def eval_at_cell_centers(self, sd: pg.Grid) -> sps.csc_matrix:
        """
        Evaluate the function at the cell centers of the given grid.

        Args:
            sd (pg.Grid): The grid on which to evaluate the function.

        Returns:
            sps.csc_matrix: The evaluated function values at the cell centers.

        Raises:
            NotImplementedError: This method is not implemented and should be
                overridden in a subclass.
        """
        raise NotImplementedError

    def interpolate(
        self, sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray]
    ) -> np.ndarray:
        """
        Interpolates a function onto the given grid.

        Args:
            sd (pg.Grid): The grid onto which the function will be interpolated.
            func (Callable[[np.ndarray], np.ndarray]): The function to be interpolated.

        Returns:
            np.ndarray: The interpolated values on the grid.

        Raises:
            NotImplementedError: This method is not implemented and should be
                overridden in a subclass.
        """
        raise NotImplementedError

    def assemble_nat_bc(
        self, sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray], b_faces: np.ndarray
    ) -> np.ndarray:
        """
        Assembles the natural boundary condition term
        (n dot q, func)_\Gamma

        Args:
            sd (pg.Grid): The grid object representing the computational domain.
            func (Callable[[np.ndarray], np.ndarray]): The function used to evaluate
                the values on the boundary.
            b_faces (np.ndarray): The array of boundary faces.

        Returns:
            np.ndarray: The assembled natural boundary condition term.
        """
        if b_faces.dtype == "bool":
            b_faces = np.where(b_faces)[0]

        p1 = pg.Lagrange1(self.keyword)
        local_mass = p1.local_mass(np.ones(1), sd.dim - 1)

        dof = self.get_dof_enumeration(sd)
        vals = np.zeros(self.ndof(sd))
        for face in b_faces:
            sign = np.sum(sd.cell_faces.tocsr()[face, :])
            nodes_loc = sd.face_nodes[:, face].indices
            loc_vals = np.array([func(sd.nodes[:, node]) for node in nodes_loc])
            dof_loc = dof[nodes_loc, face].data

            vals[dof_loc] = sign * local_mass @ loc_vals

        return vals

    def get_range_discr_class(self, dim: int) -> pg.Discretization:
        """
        Returns the range discretization class based on the given dimension.

        Args:
            dim (int): The dimension of the range.

        Returns:
            pg.Discretization: The range discretization class.
        """
        return pg.PwConstants

    def get_dof_enumeration(self, sd: pg.Grid) -> np.ndarray:
        """
        Get the degree of freedom enumeration for a given grid.

        Args:
            sd (pg.Grid): The grid for which to compute the degree of freedom enumeration.

        Returns:
            np.ndarray: The degree of freedom enumeration array.
        """
        dof = sd.face_nodes.copy()
        dof.data = np.arange(sd.face_nodes.nnz)
        return dof

    def assemble_lumped_matrix(
        self, sd: pg.Grid, data: Optional[dict] = None
    ) -> sps.csc_matrix:
        """
        Assembles the lumped matrix for the given grid and data.

        Args:
            sd (pg.Grid): The grid for which the lumped matrix is assembled.
            data (Optional[dict]): Optional data required for the assembly.

        Returns:
            sps.csc_matrix: The assembled lumped matrix.

        Raises:
            NotImplementedError: This method is not implemented and should be
                overridden in a subclass.
        """
        raise NotImplementedError
