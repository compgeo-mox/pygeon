"""Module for poincare operators."""

import warnings
from typing import Callable, Tuple

import networkx as nx
import numpy as np
import scipy.integrate as spi
import scipy.linalg as spla
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
        bar_spaces = [None] * (self.dim + 1)
        zer_spaces = [None] * (self.dim + 1)

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
        hom_spaces = [np.zeros_like(bar) for bar in self.bar_spaces]
        hom_basis = [np.zeros((bar.size, 0)) for bar in self.bar_spaces]
        cycles = [sps.csc_array((bar.size, 0), dtype=int) for bar in self.bar_spaces]

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

    def flag_edges_3d(self, keep_node=None) -> np.ndarray:
        """
        Flag the edges of the grid that form a spanning tree of the nodes.
        This function only gets called in 3D.

        Returns:
            np.ndarray: boolean array with flagged edges
        """
        if keep_node is None:
            keep_node = np.ones(self.mdg.num_subdomain_peaks())

        grad = pg.grad(self.mdg) * keep_node
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
        flagged_nodes[self.find_central_node()] = False

        return flagged_nodes

    def get_subspace_dim_differences(self):
        return np.array(
            [
                np.sum(bar) - np.sum(zer)
                for bar, zer in zip(self.bar_spaces[:-1], self.zer_spaces[1:])
            ]
        )

    def check_cohomology(self):
        dim_diff = self.get_subspace_dim_differences()

        assert dim_diff[-1] == 0

        if dim_diff[-2] != 0:
            self.compute_face_cohomology()

        dim_diff = self.get_subspace_dim_differences()
        if self.dim == 3 and dim_diff[1] != 0:
            self.compute_ridge_cohomology()
            self.bar_spaces[1], self.zer_spaces[1], self.hom_spaces[1] = (
                self.recompute_edge_subspaces()
            )

        for k in range(self.dim + 1):
            if np.any(self.cycles[k].data):
                self.compute_cohomology_basis(k)

        dim_diff = self.get_subspace_dim_differences()
        assert np.all(dim_diff == 0)

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
        cycles = sps.csc_array(sub_bdry[1:].T)
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

        # Compute the boundary divergence and curl
        curl_dom = pg.curl(self.mdg).tolil()
        div = curl_dom[np.ix_(bdry_faces, bdry_ridges)].tocsc()

        grad_dom = pg.grad(self.mdg).tolil()
        curl = grad_dom[np.ix_(bdry_ridges, bdry_nodes)].tocsc()

        # Generate the cycles as lists of nodes
        node_cycles = self.compute_node_cycles(div, curl)

        # Extract connectivities for the boundary grid
        cell_faces = sps.lil_array(div.T)
        face_nodes = sps.lil_array(curl.T)

        U_list = []
        P_list = []
        cycle_basis = sps.lil_array(
            (self.mdg.num_subdomain_ridges(), len(node_cycles)), dtype=int
        )
        ridge_inds = np.nonzero(bdry_ridges)[0]

        for ind, cycle in enumerate(node_cycles):
            edge_cycle = self.compute_edge_cycle(cycle, face_nodes)

            U = self.generate_submerged_polygon(
                cycle, edge_cycle, bdry_ridges, bdry_nodes
            )
            U_list.append(U)

            P = self.generate_shifted_polygon(
                cycle, cell_faces, face_nodes, bdry_nodes, edge_cycle
            )
            P_list.append(P)

            cycle_basis[ridge_inds[edge_cycle], ind] = -face_nodes[
                cycle, edge_cycle
            ].todense()

        coeffs = self.find_relevant_cycles(U_list, P_list)

        self.cycles[1] = cycle_basis.tocsc() @ sps.csc_array(coeffs)

    def compute_node_cycles(self, div, curl):
        # Create the co-tree
        incidence = div @ div.T
        tree = sps.csgraph.breadth_first_tree(incidence, 0, directed=False)

        # Find the mesh faces that correspond to tree edges
        c_start, c_end, _ = sps.find(tree)
        rows = np.hstack((c_start, c_end))
        cols = np.hstack([np.arange(c_start.size)] * 2)
        vals = np.ones_like(rows)

        face_finder = abs(div.T) @ sps.csc_array(
            (vals, (rows, cols)), shape=(div.shape[0], tree.nnz)
        )
        face, _, nr_common_cells = sps.find(face_finder)

        # Flag the relevant mesh faces in the grid
        cotree_faces = np.zeros(div.shape[1], dtype=bool)
        cotree_faces[face[nr_common_cells == 2]] = True

        # Find the edges on the cycles
        curl_surf = curl[np.logical_not(cotree_faces), :]
        graph = curl_surf.T @ curl_surf
        graph.setdiag(np.zeros(graph.shape[0]))
        graph.eliminate_zeros()
        G = nx.from_scipy_sparse_array(graph)

        # Compute the cycles in terms of node lists
        cycle_basis = nx.cycle_basis(G)

        return [np.array(cycle) for cycle in cycle_basis]

    def compute_edge_cycle(self, cycle, face_nodes):
        n_start = cycle
        n_end = np.append(cycle[1:], cycle[0])

        rows = np.hstack((n_start, n_end))
        cols = np.hstack([np.arange(n_start.size)] * 2)
        vals = np.ones_like(rows)

        face_finder = sps.csc_array(
            (vals, (cols, rows)), shape=(n_start.size, face_nodes.shape[0])
        ) @ abs(face_nodes)
        _, face, nr_common_nodes = sps.find(face_finder)

        edge_cycle = face[nr_common_nodes == 2]

        return edge_cycle

    def generate_submerged_polygon(
        self, node_cycle, edge_cycle, bdry_ridges, bdry_nodes
    ):
        # We first compute the submerged polygon

        U = np.zeros((3, 1 + 2 * len(node_cycle)))
        extended_cycle = np.append(node_cycle, node_cycle[0])
        nodes = np.where(bdry_nodes)[0]
        U[:, ::2] = self.top_sd.nodes[:, nodes[extended_cycle]]

        # Find an adjacent 3D cell for each boundary ridge
        adj_faces = np.abs(pg.curl(self.mdg)[:, bdry_ridges])
        adjacency = np.abs(pg.div(self.mdg)) @ adj_faces[:, edge_cycle]
        adjacency = adjacency.tocsc()
        adj_cells = adjacency.indices[adjacency.indptr[:-1]]

        U[:, 1::2] = self.top_sd.cell_centers[:, adj_cells]

        return U

    def generate_shifted_polygon(
        self, cycle, cell_faces, face_nodes, bdry_nodes, edge_cycle
    ):
        # For the shifted polygon, we loop around the adjacent triangles
        faces = []
        node = cycle[0]
        face = edge_cycle[0]
        cell = cell_faces[face, :].col[0]

        # Save a copy of cell_faces for fast column slicing
        cf_csc = cell_faces.tocsc()

        ind = 0

        while ind <= len(cycle):
            if face == edge_cycle[ind % len(cycle)]:
                # If the next edge is on the surface, we move to the next node
                ind += 1
                node = cycle[ind % len(cycle)]
            else:
                # The edge is not on the surface, so it is added to the list
                faces.append(face)

                # Move to the cell on the opposite side of the face
                cf = cell_faces[face, :].col
                cell = cf[cf != cell][0]

            # Move to the unique other face that is adjacent
            # to the current cell and node
            cf = (cf_csc[:, cell] * face_nodes[node, :]).indices
            face = cf[cf != face][0]

        faces.append(faces[0])
        faces = np.array(faces)

        to_keep = np.concatenate(([True], faces[1:] != faces[:-1]))
        faces = faces[to_keep]

        P = self.top_sd.nodes[:, bdry_nodes] @ (np.abs(face_nodes[:, faces]) / 2)

        return P

    def compute_linking_number(self, U: np.ndarray, P: np.ndarray, n_points=5) -> float:
        def green(diff: np.ndarray) -> np.ndarray:
            return diff / np.sqrt(np.sum(np.square(diff), axis=0)) ** 3 / (4 * np.pi)

        integral = 0

        for i in range(U.shape[1] - 1):
            Z = np.cross(P[:, 1:] - P[:, :-1], U[:, i + 1] - U[:, i], axis=0)
            for j in range(P.shape[1] - 1):

                def integrand(alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
                    x = np.outer(P[:, j], 1 - alpha) + np.outer(P[:, j + 1], alpha)
                    y = np.outer(U[:, i], 1 - beta) + np.outer(U[:, i + 1], beta)
                    return np.dot(Z[:, j], green(x - y))

                def int_x(x: np.ndarray) -> np.ndarray:
                    return spi.fixed_quad(integrand, 0, 1, args=(x,), n=n_points)[0]

                integral += spi.fixed_quad(int_x, 0, 1, n=n_points)[0]

        return integral

    def find_relevant_cycles(self, U_list, P_list):
        n_cycles = len(U_list)
        G = np.empty((n_cycles,) * 2)

        for i in range(n_cycles):
            for j in range(n_cycles):
                G[i, j] = self.compute_linking_number(U_list[i], P_list[j], n_points=2)

                # Check if the results are sufficiently close to integers
                if np.abs(G[i, j] - G[i, j].round()) >= 0.25:
                    G[i, j] = self.compute_linking_number(
                        U_list[i], P_list[j], n_points=5
                    )
                    assert np.abs(G[i, j] - G[i, j].round()) <= 0.25, (
                        "Higher accuracy integration may be needed"
                    )

        coeffs = spla.null_space(G.round())

        assert np.allclose(coeffs, coeffs.round()), "Non-integer coefficients?"

        return coeffs

    def recompute_edge_subspaces(self):
        # Flag all the nodes on the cycles except the starting node
        cycle_nodes = np.abs(self.cycles[1]).T @ np.abs(pg.grad(self.mdg))
        cycle_nodes.data[cycle_nodes.indptr[:-1]] = 0
        # Remove the flagged nodes from the set for the tree computation
        keep_nodes = np.logical_not(cycle_nodes.sum(axis=0).astype(bool))

        # Compute a spanning tree for the unflagged nodes
        # The bar space is given by all the edges
        #   - that are outside of this tree
        #   - and outside of the cycles
        bar_space = self.flag_edges_3d(keep_nodes)
        bar_space[self.cycles[1].indices] = False

        # Choose a single edge per cycle
        starting_edges = self.cycles[1].indices[self.cycles[1].indptr[:-1]]

        # Define the zero space as the complement of the bar space
        # and move the starting edge to the homology space
        zer_space = np.logical_not(bar_space)
        zer_space[starting_edges] = False

        hom_space = np.zeros_like(bar_space)
        hom_space[starting_edges] = True

        return bar_space, zer_space, hom_space

    def compute_cohomology_basis(self, k):
        cycles = self.cycles[k].todense()
        pdc, dpc = self.decompose(k, cycles, False)
        self.hom_basis[k] = self.cycles[k] - pdc - dpc

    def hom_projection(self, k: int, f: np.ndarray):
        if not np.any(self.hom_spaces[k]):
            return np.zeros_like(f)

        basis = self.cycles[k]
        system = (basis.T @ basis).todense()
        coeff = np.linalg.solve(system, basis.T @ f)

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
        self, k: int, f: np.ndarray, with_cohomology=True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Use the Poincaré operators to decompose f = pd(f) + dp(f)

        Args:
            k (int): order of the k-form f
            f (np.ndarray): the function to be decomposed

        Returns:
            Tuple[np.ndarray]: the decomposition of f as (dp(f), pd(f), q(f))
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
            qf = self.hom_projection(k, f)
            return pdf, dpf, qf
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

    def orthogonalize_cohomology_basis(self, k, Mass=None):
        D = diff(self.mdg, self.dim - k + 1)

        if Mass is None:
            Mass = pg.numerics.innerproducts.mass_matrix(self.mdg, self.dim - k)

        A = D.T @ Mass @ D
        b = D.T @ Mass @ self.hom_basis[k]

        proj = self.solve_subproblem(k - 1, A, b)

        return self.hom_basis[k] - D @ proj
