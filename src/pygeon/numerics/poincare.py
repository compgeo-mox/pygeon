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

        self.hom_spaces, self.cycles, self.hom_basis = self.define_cohomology_spaces()
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

    def define_cohomology_spaces(self):
        hom_spaces = np.array([None] * (self.dim + 1))
        hom_spaces[:] = [np.zeros_like(bar) for bar in self.bar_spaces]

        cycles = np.array([None] * (self.dim + 1))
        hom_basis = np.array([None] * (self.dim + 1))

        return hom_spaces, cycles, hom_basis

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

    def get_subspace_dimensions(self):
        dim_bar = [np.sum(bar) for bar in self.bar_spaces]
        dim_zer = [np.sum(zer) for zer in self.zer_spaces]

        return dim_bar, dim_zer

    def check_cohomology(self):
        euler_char = self.compute_euler_char()
        if euler_char != 1:
            print("The domain has non-trivial topology")

        dim_bar, dim_zer = self.get_subspace_dimensions()

        assert dim_bar[self.dim - 1] == dim_zer[self.dim]

        if dim_bar[self.dim - 2] != dim_zer[self.dim - 1]:
            self.compute_face_cohomology()

        dim_bar, dim_zer = self.get_subspace_dimensions()
        if self.dim == 3:
            if dim_bar[1] != dim_zer[2]:
                self.compute_ridge_cohomology()

        self.compute_cohomology_basis(self.dim - 1)

        dim_bar, dim_zer = self.get_subspace_dimensions()
        assert np.all(dim_bar[:-1] == dim_zer[1:])

    def compute_face_cohomology(self):
        k = self.dim - 1

        # The zero space includes a cycle in 2D or a closed surface in 3D.
        # We find these cycles by pruning the graph.
        curl = pg.curl(self.mdg)
        curl *= self.zer_spaces[k][:, None]

        surface = self.prune_graph(curl, self.dim)

        if not np.any(surface):
            return

        sub_bdry = self.divide_domain(pg.div(self.mdg), surface)
        n_subdomains = sub_bdry.shape[0]

        # Find subdomain connectivity
        connectivity = np.zeros((n_subdomains, n_subdomains), dtype=bool)
        for sub_1 in range(n_subdomains):
            for sub_2 in range(sub_1, n_subdomains):
                connectivity[sub_1, sub_2] = np.any(
                    np.logical_and(sub_bdry[sub_1], sub_bdry[sub_2])
                )

        # Create a tree that connects the subdomains.
        # Using the tree, we can move one face from each closed surface
        # from the zero space to the cohomology space
        sub_tree = sps.csgraph.breadth_first_tree(connectivity, 0, directed=False)
        for sub_1, sub_2, _ in zip(*sps.find(sub_tree)):
            face = np.argmax(np.logical_and(sub_bdry[sub_1], sub_bdry[sub_2]))
            self.zer_spaces[k][face] = False
            self.hom_spaces[k][face] = True

        # Generate a basis for the cohomology space
        cycles = sub_bdry[1:].T
        self.cycles[k] = cycles

    def prune_graph(self, incidence, dim):
        incidence = incidence.copy()
        for _ in np.arange(100):
            n_edges_of_node = incidence.astype(bool).sum(axis=0)
            incidence *= n_edges_of_node > 1

            n_nodes_of_edge = incidence.astype(bool).sum(axis=1)
            keep_edge = np.logical_or(n_nodes_of_edge == 0, n_nodes_of_edge == dim)
            incidence *= keep_edge[:, None]

            if np.all(keep_edge):
                break
        else:
            raise RuntimeError("Could not prune graph to a surface in 100 iterations")

        # The remaining faces form a closed surface
        return np.abs(incidence).sum(axis=1).astype(bool)

    def divide_domain(self, div, surface):
        # The surface divides the domain into subdomains
        div_ = div * np.logical_not(surface)
        n_subdomains, flags = sps.csgraph.connected_components(div_ @ div_.T)

        # Extract subdomain boundaries
        div = div.tocsr()
        sub_bdry = np.empty((n_subdomains, div.shape[1]), dtype=int)
        for sub in range(n_subdomains):
            loc_div = div[flags == sub, :]
            sub_bdry[sub] = np.sum(loc_div, axis=0)
            sub_bdry[sub] *= surface

        # The first subdomain boundary is equal to the negated sum of the others
        assert np.all(np.sum(sub_bdry, axis=0) == 0)

        return sub_bdry

    def compute_ridge_cohomology(self):
        bdry_faces = np.concatenate(
            [sd.tags["domain_boundary_faces"] for sd in self.mdg.subdomains()]
        )
        bdry_ridges = np.concatenate(
            [sd.tags["domain_boundary_ridges"] for sd in self.mdg.subdomains()]
        )
        bdry_nodes = np.concatenate(
            [sd.tags["domain_boundary_nodes"] for sd in self.mdg.subdomains()]
        )

        # Compute the boundary divergence
        curl = pg.curl(self.mdg).tolil()
        div = curl[np.ix_(bdry_faces, bdry_ridges)].tocsc()

        # Create the co-tree
        incidence = div @ div.T
        tree = sps.csgraph.breadth_first_tree(incidence, 0, directed=False)

        # Extract start/end cells for each edge of the tree
        c_start, c_end, _ = sps.find(tree)

        # Find the mesh faces that correspond to tree edges
        rows = np.hstack((c_start, c_end))
        cols = np.hstack([np.arange(c_start.size)] * 2)
        vals = np.ones_like(rows)

        face_finder = abs(div.T) @ sps.csc_array(
            (vals, (rows, cols)), shape=(div.shape[0], tree.nnz)
        )
        face, _, nr_common_cells = sps.find(face_finder)
        tree_faces = face[nr_common_cells == 2]

        # Flag the relevant mesh faces in the grid
        flagged_faces = np.zeros(div.shape[1], dtype=bool)
        flagged_faces[tree_faces] = True

        # Compute the boundary curl
        grad = pg.grad(self.mdg).tolil()
        curl = grad[np.ix_(bdry_ridges, bdry_nodes)].tocsc()

        # Find the edges on the cycles
        surface = self.prune_graph(curl * np.logical_not(flagged_faces)[:, None], 2)

        curl *= surface[:, None]

        import networkx as nx

        graph = curl.T @ curl
        graph.setdiag(np.zeros(graph.shape[0]))
        graph.eliminate_zeros()

        G = nx.from_scipy_sparse_array(graph)

        # Compute the cycles in terms of node lists
        cycles = nx.cycle_basis(G)

        #
        nodes = np.where(bdry_nodes)[0]
        adjacency = np.abs(pg.div(self.mdg)) @ np.abs(pg.curl(self.mdg)[:, bdry_ridges])
        adjacency = adjacency.tocsc()
        adj_cell = adjacency.indices[adjacency.indptr[:-1]]

        for cycle in cycles:
            U, P = self.generate_polygon(cycle, div, curl, nodes, adj_cell)

        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")

        indices = np.nonzero(bdry_ridges)[0]
        grad_top = pg.grad(self.mdg).tocsr()

        for edge in indices[surface.nonzero()]:
            nodes = grad_top.indices[grad_top.indptr[edge] : grad_top.indptr[edge + 1]]
            nodes = self.top_sd.nodes[:, nodes]
            ax.plot(*nodes)
        plt.show()
        pass

    def generate_polygon(self, cycle, div, curl, nodes, adj_cell):
        extended_cycle = np.append(cycle, cycle[0])
        U = np.zeros((3, 1 + 2 * len(cycle)))
        U[:, ::2] = self.top_sd.nodes[:, nodes[extended_cycle]]

        U_midpoints = []
        for start, end in zip(cycle, extended_cycle[1:]):
            edge_finder = np.zeros(curl.shape[1])
            edge_finder[[start, end]] = 1.0
            edge = np.where(np.abs(curl) @ edge_finder == 2)[0]
            U_midpoints.append(self.top_sd.cell_centers[:, adj_cell[edge]])

        U[:, 1::2] = np.hstack(U_midpoints)

        # TODO
        P = 0

        return U, P

    def compute_cohomology_basis(self, k):
        pdc, dpc = self.decompose(k, self.cycles[k], False)
        self.hom_basis[k] = self.cycles[k] - pdc - dpc

    def hom_projection(self, k: int, f: np.ndarray):
        if not np.any(self.hom_spaces[k]):
            return np.zeros_like(f)

        basis = self.cycles[k]
        coeff = np.linalg.solve(basis.T @ basis, basis.T @ f)

        return self.hom_basis[k] @ coeff

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

        R_zer = create_restriction(self.zer_spaces[k])
        R_bar = create_restriction(self.bar_spaces[k - 1])

        pizer_dbar = R_zer @ _diff @ R_bar.T

        return R_bar.T @ solver(pizer_dbar, R_zer @ f)

    def decompose(
        self, k: int, f: np.ndarray, with_cohomology=False
    ) -> Tuple[np.ndarray, np.ndarray]:
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

        if with_cohomology:
            hf = self.hom_projection(k, f)
            return pdf, dpf, hf
        else:
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

    def compute_euler_char(self):
        c = self.mdg.num_subdomain_cells()
        f = self.mdg.num_subdomain_faces()
        e = self.mdg.num_subdomain_ridges()
        p = self.mdg.num_subdomain_peaks()

        char = p - e + f - c

        if self.dim == 2:
            char *= -1

        return char

    def orthogonalize_cohomology_basis(self, k):
        candidates = self.hom_basis[k]
        D = diff(self.mdg, self.dim - k + 1)
        M = pg.face_mass(self.mdg)

        A = D.T @ M @ D
        b = D.T @ M @ candidates

        return candidates - D @ sps.linalg.spsolve(A, b)

        pass
