from typing import Dict, Tuple

import numpy as np
import scipy.sparse as sps

import porepy as pp
import pygeon as pg


class Lagrange:
    def __init__(self, keyword: str) -> None:
        self.keyword = keyword

        # Keywords used to identify individual terms in the discretization matrix dictionary
        # Discretization of stiffness matrix
        self.stiffness_matrix_key = "stiffness"

    def ndof(self, g: pp.Grid) -> int:
        """
        Return the number of degrees of freedom associated to the method.
        In this case number of nodes.

        Parameter
        ---------
        g: grid, or a subclass.

        Return
        ------
        dof: the number of degrees of freedom.

        """
        if isinstance(g, pp.Grid):
            return g.num_nodes
        else:
            raise ValueError

    def assemble_matrix_rhs(
        self, g: pp.Grid, data: Dict
    ) -> Tuple[sps.csr_matrix, np.ndarray]:

        """Return the matrix and right-hand side for a discretization of a second
        order elliptic equation using P1 method on simplices.

        We assume the following sub-dictionary to be present in the data dictionary:
            parameter_dictionary, storing all parameters.
                Stored in data[pp.PARAMETERS][self.keyword].

        parameter_dictionary contains the entries:
            second_order_tensor: (pp.SecondOrderTensor) Permeability defined cell-wise.
            bc: (BoundaryConditionNode) node-wise boundary conditions.
            bc_values: array (self.ndof) boundary condition values.

        Parameters
        ----------
        g: grid, or a subclass, with geometry fields computed.
        data: dictionary to store the data.

        Returns
        ------
        matrix: sparse csr (g.num_nodes, g.num_nodes) Matrix obtained from the
            discretization.
        rhs: array (g.num_nodes) Right-hand side which contains the boundary conditions.
        """
        # First assemble the matrix
        M = self.assemble_matrix(g, data)

        # Assemble right hand side term
        return M, self.assemble_rhs(g, data)

    def assemble_matrix(self, g: pp.Grid, data: Dict) -> sps.csr_matrix:
        """Assemble matrix from an existing discretization."""
        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES][self.keyword]

        return matrix_dictionary[self.stiffness_matrix_key]

    def discretize(self, g: pp.Grid, data: Dict) -> None:
        """Return the matrix for a discretization of a second order elliptic equation
        using P1 method.

        Parameters
        ----------
        g: grid, or a subclass, with geometry fields computed.
        data: dictionary to store the data. See self.matrix_rhs for required contents.

        Returns
        ------
        matrix: sparse csr (g.num_nodes, g.num_nodes)
            Matrix obtained from the discretization.

        Raises:
        ------
        ValueError if the boundary condition is not defined node-wise.
        """
        # Allow short variable names in backend function
        # pylint: disable=invalid-name

        # Get dictionary for discretization matrix storage
        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES][self.keyword]

        # If a 0-d grid is given then we return an identity matrix
        if g.dim == 0:
            matrix_dictionary[self.stiffness_matrix_key] = sps.csr_matrix((1, 1))
            return

        # Get dictionary for parameter storage
        parameter_dictionary = data[pp.PARAMETERS][self.keyword]
        # Retrieve the permeability, boundary conditions
        k = parameter_dictionary["second_order_tensor"]
        bc = parameter_dictionary["bc"]
        if not isinstance(bc, pg.BoundaryConditionNode):
            raise ValueError("Consider BoundaryConditionNode to assign bc")

        # Map the domain to a reference geometry (i.e. equivalent to compute
        # surface coordinates in 1d and 2d)
        c_centers, f_normals, f_centers, R, dim, node_coords = pp.map_geometry.map_grid(
            g
        )

        if not data.get("is_tangential", False):
            # Rotate the permeability tensor and delete last dimension
            if g.dim < 3:
                k = k.copy()
                k.rotate(R)
                remove_dim = np.where(np.logical_not(dim))[0]
                k.values = np.delete(k.values, (remove_dim), axis=0)
                k.values = np.delete(k.values, (remove_dim), axis=1)

        # Allocate the data to store matrix entries, that's the most efficient
        # way to create a sparse matrix.
        size = np.power(g.dim + 1, 2) * g.num_cells
        I = np.empty(size, dtype=np.int)
        J = np.empty(size, dtype=np.int)
        dataIJ = np.empty(size)
        idx = 0

        cell_nodes = g.cell_nodes()
        nodes, cells, _ = sps.find(cell_nodes)

        for c in np.arange(g.num_cells):
            # For the current cell retrieve its nodes
            loc = slice(cell_nodes.indptr[c], cell_nodes.indptr[c + 1])

            nodes_loc = nodes[loc]
            coord_loc = node_coords[:, nodes_loc]

            # Compute the stiff-H1 local matrix
            A = self.stiffH1(
                k.values[0 : g.dim, 0 : g.dim, c], g.cell_volumes[c], coord_loc, g.dim
            )

            # Save values for stiff-H1 local matrix in the global structure
            cols = np.tile(nodes_loc, (nodes_loc.size, 1))
            loc_idx = slice(idx, idx + cols.size)
            I[loc_idx] = cols.T.ravel()
            J[loc_idx] = cols.ravel()
            dataIJ[loc_idx] = A.ravel()
            idx += cols.size

        # Construct the global matrices
        M = sps.csr_matrix((dataIJ, (I, J)))

        # assign the Dirichlet boundary conditions
        if bc and np.any(bc.is_dir):
            dir_nodes = np.where(bc.is_dir)[0]
            # set in an efficient way the essential boundary conditions, by
            # clear the rows and put norm in the diagonal
            for row in dir_nodes:
                M.data[M.indptr[row] : M.indptr[row + 1]] = 0.0

            d = M.diagonal()
            d[dir_nodes] = 1.0
            M.setdiag(d)

        matrix_dictionary[self.stiffness_matrix_key] = M

    # ------------------------------------------------------------------------------#

    def local_grads(self, coord, dim):
        Q = np.hstack((np.ones((dim + 1, 1)), coord.T))
        invQ = np.linalg.inv(Q)
        return invQ[1:, :]

    def local_mass(self, c_volume, dim):
        """Compute the local mass H1 matrix using the P1 Lagrangean approach.

        Parameters
        ----------
        c_volume : scalar
            Cell volume.

        Return
        ------
        out: ndarray (num_faces_of_cell, num_faces_of_cell)
            Local mass Hdiv matrix.
        """
        # Allow short variable names in this function
        # pylint: disable=invalid-name

        M = np.ones((dim + 1, dim + 1)) + np.identity(dim + 1)
        return c_volume * M / ((dim + 1) * (dim + 2))

    def stiffH1(self, K, c_volume, coord, dim):
        """Compute the local stiffness H1 matrix using the P1 Lagrangean approach.

        Parameters
        ----------
        K : ndarray (g.dim, g.dim)
            Permeability of the cell.
        c_volume : scalar
            Cell volume.

        Return
        ------
        out: ndarray (num_faces_of_cell, num_faces_of_cell)
            Local mass Hdiv matrix.
        """
        # Allow short variable names in this function
        # pylint: disable=invalid-name

        dphi = self.local_grads(coord)

        return c_volume * np.dot(dphi.T, np.dot(K, dphi))

    # ------------------------------------------------------------------------------#
