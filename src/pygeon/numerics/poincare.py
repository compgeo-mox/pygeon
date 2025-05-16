"""Module for poincare operators."""

import warnings
from typing import Callable, Tuple

import numpy as np
import scipy.sparse as sps

import pygeon as pg
from pygeon.numerics.differentials import exterior_derivative as diff
from pygeon.numerics.linear_system import create_restriction


class Poincare:
    """
    Class for generating Poincaré operators p
    that satisfy pd + dp = I
    with d the exterior derivative, following
    the construction from https://arxiv.org/abs/2410.08830

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

        self.check_grid_admissibility()

        self.bar_spaces, self.zer_spaces = self.define_subspaces()
        self.zero_out_tips()

        self.check_cohomology()

    def define_subspaces(self) -> None:
        """
        Flag the mesh entities that will be used to generate the Poincaré operators
        """
        # Preallocation
        bar_spaces = np.array([None] * (self.dim + 1))
        zer_spaces = np.array([None] * (self.dim + 1))

        # Cells
        bar_spaces[self.dim] = np.zeros(self.mdg.num_subdomain_cells(), dtype=bool)

        # Faces
        bar_spaces[self.dim - 1] = pg.SpanningTree(self.mdg, "all_bdry").flagged_faces

        # Edges in 3D
        if self.dim == 3:
            bar_spaces[1] = self.flag_edges_3d()

        # Nodes
        bar_spaces[0] = self.flag_nodes()

        # Define the complementary set
        zer_spaces[:] = [~bar for bar in bar_spaces]

        return bar_spaces, zer_spaces

    def zero_out_tips(self) -> None:
        tip_ridges = np.concatenate(
            [sd.tags["tip_ridges"] for sd in self.mdg.subdomains()]
        )
        self.bar_spaces[self.dim - 2][tip_ridges] = False
        assert np.all(self.zer_spaces[self.dim - 2][tip_ridges] == False)

        tip_faces = np.concatenate(
            [sd.tags["tip_faces"] for sd in self.mdg.subdomains()]
        )

        assert np.all(self.bar_spaces[self.dim - 1][tip_faces] == False)
        self.zer_spaces[self.dim - 1][tip_faces] = False

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
        shape = (grad.shape[1], tree.nnz)
        edge_finder = abs(grad) @ sps.csc_array((vals, (rows, cols)), shape=shape)

        edge, _, nr_common_nodes = sps.find(edge_finder)
        tree_edges = edge[nr_common_nodes == 2]

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

        return int(np.argmin(dists))

    def flag_nodes(self) -> np.ndarray:
        """
        Flag all the nodes in the top-dim domain, except for the first node

        Returns:
            np.ndarray: boolean array with flagged nodes
        """
        flagged_nodes = np.ones(self.top_sd.num_nodes, dtype=bool)
        flagged_nodes[0] = False

        return flagged_nodes

    def check_cohomology(self):
        dim_bar = [np.sum(bar) for bar in self.bar_spaces]
        dim_zer = [np.sum(zer) for zer in self.zer_spaces]

        assert dim_bar[self.dim - 1] == dim_zer[self.dim]

        if dim_bar[self.dim - 2] == dim_zer[self.dim - 1]:
            self.prune_graph_to_tree()

    def prune_graph_to_tree(self):
        curl = pg.curl(self.mdg)
        curl *= self.zer_spaces[self.dim - 1][:, None]

        for _ in np.arange(100):
            num_faces = curl.astype(bool).sum(axis=0)
            curl *= num_faces > 1

            keep_face = curl.sum(axis=1) == 0
            curl *= keep_face[:, None]

            if np.all(keep_face):
                break
        else:
            raise LookupError("Could not prune graph to a surface in 100 iterations")

        surface = np.abs(curl).sum(axis=1).astype(bool)

        # Extract number of subdomains
        div = pg.div(self.mdg) * np.logical_not(surface)
        n_subdomains, flags = sps.csgraph.connected_components(div @ div.T)

        # Extract subdomain boundaries
        div = pg.div(self.mdg).tocsr()
        sub_bdry = np.empty((n_subdomains, div.shape[1]), dtype=int)
        for sub in range(n_subdomains):
            loc_div = div[flags == sub, :]
            sub_bdry[sub] = np.sum(loc_div, axis=0)
            sub_bdry[sub] *= surface

        # Find subdomain connectivity
        connectivity = np.zeros((n_subdomains, n_subdomains), dtype=bool)
        for sub_1 in range(n_subdomains):
            for sub_2 in range(sub_1, n_subdomains):
                connectivity[sub_1, sub_2] = np.any(
                    np.logical_and(sub_bdry[sub_1], sub_bdry[sub_2])
                )

        sub_tree = sps.csgraph.breadth_first_tree(connectivity, 0, directed=False)
        for sub_1, sub_2, _ in zip(*sps.find(sub_tree)):
            face = np.argmax(np.logical_and(sub_bdry[sub_1], sub_bdry[sub_2]))
            self.zer_spaces[self.dim - 1][face] = False

        pass

    def apply(
        self, k: int, f: np.ndarray, solver: Callable = sps.linalg.spsolve
    ) -> np.ndarray:
        """
        Apply the Poincare operator

        Args:
            k (int): order of the differential k-form that is input
            f (np.ndarray): the input differential k-form
                as an array of the degrees of freedom
            solver (Optional[Callable]): The solver function to use.
                Defaults to sps.linalg.spsolve

        Returns:
            np.ndarray: the image of f under the Poincaré operator, i.e. p(f)
        """
        # Nodes to the constants
        if k == 0:
            return np.full_like(f, np.mean(f))

        # For k > 0, we simply apply the operator
        pf = self._apply_op(k, f, solver)

        # For the edge-to-node map, we subtract the mean
        if k == 1:
            pf -= np.mean(pf)

        return pf

    def _apply_op(self, k: int, f: np.ndarray, solver: Callable) -> np.ndarray:
        """
        Apply the permitted Poincaré operator for k-forms

        Args:
            k (int): order of the form
            f (np.ndarray): the input differential k-form
                as an array of the degrees of freedom
            solver (Callable): The solver function to use.

        Returns:
            np.ndarray: the image of f under the Poincaré operator, i.e. p(f)
        """
        n_minus_k = self.dim - k
        _diff = diff(self.mdg, n_minus_k + 1)

        R_0 = create_restriction(self.zer_spaces[k])
        R_bar = create_restriction(self.bar_spaces[k - 1])

        pi_0_d_bar = R_0 @ _diff @ R_bar.T

        return R_bar.T @ solver(pi_0_d_bar, R_0 @ f)

    def decompose(self, k: int, f: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Use the Poincaré operators to decompose f = pd(f) + dp(f)

        Args:
            k (int): order of the k-form f
            f (np.ndarray): the function to be decomposed

        Returns:
            Tuple[np.ndarray]: the decomposition of f as (dp(f), pd(f))
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

    def solve_subproblem(
        self,
        k: int,
        A: sps.csc_array,
        b: np.ndarray,
        solver: Callable = sps.linalg.spsolve,
    ) -> np.ndarray:
        """
        Solve a linear system on the subspace of
        differential forms identified by the Poincare object.

        Args:
            k (int): order of the k-form
            A (sps.csc_array): the system, usually a stiffness matrix
            b (np.ndarray): the right-hand side vector
            solver (Callable): The solver function to use. Defaults to
                sps.linalg.spsolve.

        Returns:
            np.ndarray: the solution
        """

        LS = pg.LinearSystem(A, b)
        LS.flag_ess_bc(~self.bar_spaces[k], np.zeros_like(self.bar_spaces[k]))

        return LS.solve(solver=solver)

    def check_grid_admissibility(self):
        for sd in self.mdg.subdomains(dim=1):
            if sd.num_cells == 1:
                warnings.warn(
                    "There is a 1D domain with only one cell. "
                    + "Consider refining the grid."
                )
                break
