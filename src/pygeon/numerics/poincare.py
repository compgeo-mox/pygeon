""" Module for poincare operators. """

import numpy as np
import scipy.sparse as sps

import pygeon as pg
from pygeon.numerics.differentials import exterior_derivative as diff
from pygeon.numerics.linear_system import create_restriction


class Poincare:
    """
    Class for generating Poincaré operators p
    that satisfy pd + dp = I
    with d the exterior derivative

    Attributes:
        mdg (pg.MixedDimensionalGrid): The (mixed-dimensional) grid.
        dim (int): The ambient dimension.
        top_sd (int): The top-dimensional subdomain.
        bar_spaces (list): List of boolean arrays indicating subspaces on
            which the exterior derivative is invertible.
    """

    def __init__(self, mdg: pg.MixedDimensionalGrid) -> None:
        """
        Initializes a Poincare class

        Args:
            mdg (pg.MixedDimensionalGrid): a (mixed-dimensional) grid
        """
        self.mdg = mdg
        self.dim = mdg.dim_max()
        self.top_sd = mdg.subdomains(dim=self.dim)[0]
        self.define_bar_spaces()

        if mdg.num_subdomains() > 1:
            raise NotImplementedError

    def define_bar_spaces(self) -> None:
        """
        Flag the mesh entities that will be used to generate the Poincaré operators
        """
        # Preallocation
        self.bar_spaces = [None] * (self.dim + 1)

        # Cells
        self.bar_spaces[self.dim] = np.zeros(self.mdg.num_subdomain_cells(), dtype=bool)

        # Faces
        self.bar_spaces[self.dim - 1] = pg.SpanningTree(
            self.mdg, "all_bdry"
        ).flagged_faces

        # Edges in 3D
        if self.dim == 3:
            self.bar_spaces[1] = self.flag_edges_3d()

        # Nodes
        self.bar_spaces[0] = self.flag_nodes()

    def flag_edges_3d(self) -> np.ndarray:
        """
        Flag the edges of the grid that form a spanning tree of the nodes.
        This function only gets called in 3D.

        Returns:
            np.ndarray: boolean array with flagged edges
        """

        grad = pg.grad(self.mdg)
        incidence = grad.T @ grad

        root = self.find_central_node()

        tree = sps.csgraph.breadth_first_tree(incidence, root, directed=False)
        c_start, c_end, _ = sps.find(tree)

        rows = np.hstack((c_start, c_end))
        cols = np.hstack([np.arange(c_start.size)] * 2)
        vals = np.ones_like(rows)

        edge_finder = sps.csc_array(
            (vals, (rows, cols)), shape=(grad.shape[1], tree.nnz)
        )
        edge_finder = np.abs(grad) @ edge_finder
        I, _, V = sps.find(edge_finder)
        tree_edges = I[V == 2]

        flagged_edges = np.ones(grad.shape[0], dtype=bool)
        flagged_edges[tree_edges] = False

        return flagged_edges

    def find_central_node(self) -> int:
        """
        Find the node that is closest to the center of the domain.

        Returns:
            int: index of the central node
        """

        center = np.mean(self.top_sd.nodes, axis=1, keepdims=True)
        dists = np.linalg.norm(self.top_sd.nodes - center, axis=0)

        return np.argmin(dists)

    def flag_nodes(self) -> np.ndarray:
        """
        Flag all the nodes in the top-dim domain, except for the first node

        Returns:
            np.ndarray: boolean array with flagged nodes
        """

        flagged_nodes = np.ones(self.top_sd.num_nodes, dtype=bool)
        flagged_nodes[0] = False

        return flagged_nodes

    def apply(self, k: int, f: np.ndarray) -> np.ndarray:
        """
        Apply the Poincare operator

        Args:
            k (int): order of the differential form
            f (np.ndarray): the differential form

        Returns:
            np.ndarray: the image of the Poincaré operator under f
        """
        # Nodes to the constants
        if k == 0:
            return np.full_like(f, np.mean(f))

        # For k > 0, we simply apply the operator
        pf = self._apply_op(k, f)

        # For edges to nodes, we subtract the mean
        if k == 1:
            return pf - np.mean(pf)
        else:  # Otherwise, we return pf
            return pf

    def _apply_op(self, k: int, f: np.ndarray) -> np.ndarray:
        """
        Apply the permitted Poincaré operator for k-forms

        Args:
            k (int): order of the form

        Returns:
            np.ndarray: the image of the Poincaré operator
        """
        n_minus_k = self.dim - k
        _diff = diff(self.mdg, n_minus_k + 1)

        R_0 = create_restriction(~self.bar_spaces[k])
        R_bar = create_restriction(self.bar_spaces[k - 1])

        pi_0_d_bar = R_0 @ _diff @ R_bar.T

        return R_bar.T @ sps.linalg.spsolve(pi_0_d_bar, R_0 @ f)

    def decompose(self, k: int, f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Use the Poincaré operators to decompose f = pd(f) + dp(f)

        Args:
            k (int): order of the k-form f
            f (np.ndarray): the function to be decomposed

        Returns:
            tuple(np.ndarray): the decomposition of f as (dp(f), pd(f))
        """
        n_minus_k = self.dim - k

        if k == self.dim:  # then df = 0
            pdf = np.zeros_like(f)
        else:
            df = diff(self.mdg, n_minus_k) @ f
            pdf = self.apply(k + 1, df)

        if k == 0:  # then dpf = mean(f)
            dpf = self.apply(k, f)
        else:
            pf = self.apply(k, f)
            dpf = diff(self.mdg, n_minus_k + 1) @ pf

        return pdf, dpf
