from typing import Dict, Tuple

import numpy as np
import scipy.sparse as sps

import porepy as pp
import pygeon as pg


class Nedelec1:
    def __init__(self, keyword: str) -> None:
        self.keyword = keyword

        # Keywords used to identify individual terms in the discretization matrix dictionary
        # Discretization of mass matrix
        self.mass_matrix_key = "mass"
        self.curl_matrix_key = "curl"

    def ndof(self, g: pp.Grid) -> int:
        """
        Return the number of degrees of freedom associated to the method.
        In this case number of ridges.

        Parameter
        ---------
        g: grid, or a subclass.

        Return
        ------
        dof: the number of degrees of freedom.

        """
        if isinstance(g, pg.Grid):
            return 2 * g.num_ridges
        else:
            raise ValueError

    def discretize(self, g: pg.Grid, data: Dict):
        """Compute the mass matrix for a first-order Nedelec discretization

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

        assert g.dim == 3

        # Allocate the data to store matrix entries, that's the most efficient
        # way to create a sparse matrix.
        size = 9 * 4 * g.num_cells
        I = np.empty(size, dtype=np.int)
        J = np.empty(size, dtype=np.int)
        dataIJ = np.empty(size)
        idx = 0

        cell_ridges = g.face_ridges.astype(bool) * g.cell_faces.astype(bool)
        ridges, cells, _ = sps.find(cell_ridges)

        ridge_peaks = g.ridge_peaks

        for c in np.arange(g.num_cells):
            # For the current cell retrieve its ridges and
            # determine the location of the dof
            loc = slice(cell_ridges.indptr[c], cell_ridges.indptr[c + 1])
            ridges_loc = ridges[loc]
            dof_loc = np.reshape(
                ridge_peaks[:, ridges_loc].indices, (2, -1), order="F"
            ).flatten()

            # Find the nodes of the cell and their coordinates
            nodes_uniq, indices = np.unique(dof_loc, return_inverse=True)
            coords = g.nodes[:, nodes_uniq]

            # Compute the gradients of the Lagrange basis functions
            dphi = self.local_grads(coords)

            # Compute the local Nedelec basis functions and global indices
            Ne_basis = np.roll(dphi[:, indices], 6, axis=1)
            Ne_indices = np.concatenate((ridges_loc, ridges_loc + g.num_ridges))

            # Compute the inner products around each node
            for node in nodes_uniq:
                bf_is_at_node = dof_loc == node
                grads = Ne_basis[:, bf_is_at_node]
                A = grads.T @ grads
                A *= g.cell_volumes[c] / 4

                loc_ind = Ne_indices[bf_is_at_node]

                # Save values for stiff-H1 local matrix in the global structure
                cols = np.tile(loc_ind, (loc_ind.size, 1))
                loc_idx = slice(idx, idx + cols.size)
                I[loc_idx] = cols.T.ravel()
                J[loc_idx] = cols.ravel()
                dataIJ[loc_idx] = A.ravel()
                idx += cols.size

        # Construct the global matrices
        M = sps.csr_matrix((dataIJ, (I, J)))

        matrix_dictionary[self.mass_matrix_key] = M
        matrix_dictionary[self.curl_matrix_key] = self.curl(g)

    def curl(self, g):
        return sps.bmat([[pg.curl(g), -pg.curl(g)]])

    def local_grads(self, coord, dim=3):
        Q = np.hstack((np.ones((dim + 1, 1)), coord.T))
        invQ = np.linalg.inv(Q)
        return invQ[1:, :]
