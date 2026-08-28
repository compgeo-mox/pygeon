"""Module for poincare operators."""

from typing import Callable, List, Tuple, cast

import networkx as nx
import numpy as np
import scipy.integrate as spi
import scipy.linalg as spla
import scipy.sparse as sps

import pygeon as pg
from pygeon.numerics.differentials import exterior_derivative as diff
from pygeon.numerics.innerproducts import mass_matrix


class Poincare:
    """
    Class for generating Poincaré operators p and q that satisfy :math:`pd + dp
    + q = I` with d the exterior derivative, following the construction from
    https://arxiv.org/abs/2410.08830.
    """

    def __init__(self, mdg: pg.MixedDimensionalGrid) -> None:
        """
        Initializes a Poincare object on a mixed-dimensional grid.

        Args:
            mdg (pg.MixedDimensionalGrid): A mixed-dimensional grid.
        """
        self.mdg = mdg
        self.dim = mdg.dim_max()
        self.top_sd = mdg.subdomains(dim=self.dim)[0]

        self.define_subspaces()
        self.remove_tips_from_spaces()

        self.compute_cohomology()

    def define_subspaces(self) -> None:
        """
        Flags the mesh entities used to build the Poincaré operators.

        These are split into three boolean subspaces:
        - bar_spaces: domain subspace on which the differential is injective
        - zer_spaces: range subspace on which the differential is surjective
        - hom_spaces: subspace with the same dimension as the cohomology

        The method also initializes the cohomology basis and cycle containers.
        """
        # Preallocation
        bar_spaces = [np.zeros(0, dtype=bool) for _ in range(self.dim + 1)]

        # Cells
        bar_spaces[self.dim] = np.zeros(self.mdg.num_subdomain_cells(), dtype=bool)

        # Faces
        bar_spaces[self.dim - 1] = pg.SpanningTree(self.mdg, "all_bdry").flagged_faces

        # Edges in 3D
        if self.dim == 3:
            bar_spaces[1] = self.flag_edges_3d()

        # Nodes
        bar_spaces[0] = self.flag_nodes()

        # Save this boolean array list as an attribute and define the complementary sets
        self.bar_spaces = bar_spaces
        self.zer_spaces = [~bar for bar in bar_spaces]
        self.hom_spaces = [np.zeros_like(bar) for bar in bar_spaces]

        self.hom_basis = [np.zeros((bar.size, 0)) for bar in self.bar_spaces]
        self.cycles = [
            sps.csc_array((bar.size, 0), dtype=int) for bar in self.bar_spaces
        ]

    def flag_edges_3d(self, keep_node: np.ndarray | None = None) -> np.ndarray:
        """
        Flags the edges of the grid that form a spanning tree of the nodes.

        This function is only used in 3D.

        Returns:
            np.ndarray: Boolean array with flagged edges.
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
        Finds the node closest to the center of the domain.

        Returns:
            int: Index of the central node.
        """
        center = np.mean(self.top_sd.nodes, axis=1, keepdims=True)
        dists = np.linalg.norm(self.top_sd.nodes - center, axis=0)

        return int(np.argmin(dists))

    def flag_nodes(self) -> np.ndarray:
        """
        Flags all nodes in the top-dimensional domain except the central node.

        Returns:
            np.ndarray: Boolean array with flagged nodes.
        """
        flagged_nodes = np.ones(self.top_sd.num_nodes, dtype=bool)
        flagged_nodes[self.find_central_node()] = False

        return flagged_nodes

    def remove_tips_from_spaces(self) -> None:
        """
        Removes fracture tips from the subspaces.

        These are essential boundary conditions and therefore do not appear in the
        bar or zero spaces.
        """
        tip_ridges = np.concatenate(
            [sd.tags["tip_ridges"] for sd in self.mdg.subdomains()]
        )
        self.bar_spaces[self.dim - 2][tip_ridges] = False
        assert not np.any(self.zer_spaces[self.dim - 2][tip_ridges])

        tip_faces = np.concatenate(
            [sd.tags["tip_faces"] for sd in self.mdg.subdomains()]
        )
        assert not np.any(self.bar_spaces[self.dim - 1][tip_faces])
        self.zer_spaces[self.dim - 1][tip_faces] = False

    def compute_cohomology(self) -> None:
        """
        Computes the topological cycles and basis associated with the cohomology.
        """
        # We first compute the cohomology of the face-based forms
        self.compute_face_cohomology()

        # A quick dimension check to see if the domain spaces already have the
        # same dimension as the range spaces.
        dim_diff = self.get_subspace_dim_differences()

        # A ridge cohomology calculation is needed if the dimensions don't match.
        if self.dim == 3 and dim_diff[1] != 0:
            self.compute_ridge_cohomology()
            self.recompute_edge_subspaces()

        # With these spaces defined, we can
        self.compute_node_cohomology()

        dim_diff = self.get_subspace_dim_differences()
        assert np.all(dim_diff == 0)

        for k in range(self.dim + 1):
            if np.any(self.cycles[k].data):
                self.compute_cohomology_basis(k)

    def get_subspace_dim_differences(self) -> np.ndarray:
        """
        Returns the dimension mismatch between the bar and zero spaces.

        The entries correspond to the difference between the number of degrees of
        freedom in the bar space and in the complementary zero space for each
        form order, excluding the top-dimensional space. A nonzero value signals
        that cohomology still needs to be accounted for.

        Returns:
            np.ndarray: Vector of dimension differences for each form order.
        """
        return np.array(
            [
                np.sum(bar) - np.sum(zer)
                for bar, zer in zip(self.bar_spaces[:-1], self.zer_spaces[1:])
            ]
        )

    def compute_face_cohomology(self) -> None:
        """
        Computes the topological cycles associated with face-based cohomology.

        The method identifies closed surface components that correspond to holes in
        the domain and moves one representative face from the zero space into the
        cohomology space for each independent class.
        """
        k = self.dim - 1

        # The zero space includes a cycle in 2D or a closed surface in 3D.
        # We find these cycles by pruning the graph.
        curl = pg.curl(self.mdg)
        curl *= self.zer_spaces[k][:, None]

        surface = self.prune_graph(curl, self.dim)

        # The graph has been entirely pruned so there is no cohomology.
        if not np.any(surface):
            return

        # We need to identify the surfaces that surround each hole. For that, we
        # need to add the holes as cells in an extended div matrix. We first
        # find the number of closed boundaries by using the face-ridge
        # connectivity on the boundary.
        bdry_faces = np.concatenate(
            [sd.tags["domain_boundary_faces"] for sd in self.mdg.subdomains()]
        )
        bdry_ridges = np.concatenate(
            [sd.tags["domain_boundary_ridges"] for sd in self.mdg.subdomains()]
        )
        div_bdry = pg.restrict_csc_array(pg.curl(self.mdg), bdry_faces, bdry_ridges)
        incidence = div_bdry @ div_bdry.T
        n_components, ids = sps.csgraph.connected_components(incidence)

        # We now extend the divergence of the domain with the hole cells.
        div_dom = pg.div(self.mdg)
        extension = sps.csc_array(
            (np.ones_like(ids), (ids, np.flatnonzero(bdry_faces))),
            (n_components, div_dom.shape[1]),
        )
        ext_div = sps.vstack((div_dom, extension))

        # Compute the boundaries of the subdomains
        sub_bdry = self.divide_domain(ext_div, surface)

        # Find subdomain connectivity
        connectivity = np.zeros((n_components, n_components), dtype=bool)
        for sub_1 in range(n_components):
            for sub_2 in range(sub_1, n_components):
                connectivity[sub_1, sub_2] = np.any(
                    np.logical_and(sub_bdry[sub_1], sub_bdry[sub_2])
                )

        # Create a tree that connects the subdomains. Using the tree, we can
        # move one face from each closed surface from the zero space to the
        # cohomology space
        sub_tree = sps.csgraph.breadth_first_tree(connectivity, 0, directed=False)
        for sub_i, sub_j, _ in zip(*sps.find(sub_tree)):
            face = np.argmax(np.logical_and(sub_bdry[sub_i], sub_bdry[sub_j]))
            self.zer_spaces[k][face] = False
            self.hom_spaces[k][face] = True

        # Generate the topological cycles for the cohomology space
        cycles = sps.csc_array(sub_bdry[1:].T)
        self.cycles[k] = cycles

    def prune_graph(self, incidence: sps.csc_array, dim: int) -> np.ndarray:
        """
        Prunes a graph until only the closed surfaces surrounding holes remains.

        The procedure iteratively removes vertices and edges that are not part of
        the boundary manifold. The surviving rows indicate the incidence entries
        forming the relevant closed surface.

        Args:
            incidence (sps.csc_array): Incidence matrix of the graph.
            dim (int): Dimension of the ambient space.

        Returns:
            np.ndarray: Boolean mask of the remaining rows, i.e. the selected
                closed-surface faces.
        """
        incidence = incidence.astype(bool)
        for _ in np.arange(incidence.shape[0]):
            n_edges_of_node = incidence.sum(axis=0)
            incidence *= n_edges_of_node > 1

            n_nodes_of_edge = incidence.sum(axis=1)
            keep_edge = np.logical_or(n_nodes_of_edge == 0, n_nodes_of_edge == dim)
            incidence *= keep_edge[:, None]

            if np.all(keep_edge):
                break
        else:
            # I was not able to construct an unprunable graph, but let's keep
            # this safegaurd in place anyways.
            raise RuntimeError(
                "Could not prune graph to a surface."
            )  # pragma: no cover

        # The remaining faces form a closed surface
        return abs(incidence).sum(axis=1).astype(bool)

    def divide_domain(self, div: sps.csc_array, surface: np.ndarray) -> np.ndarray:
        """
        Partitions the domain by a closed surface and returns subdomain boundaries.

        Args:
            div (sps.csc_array): Divergence matrix of the domain.
            surface (np.ndarray): Boolean mask identifying the faces that form the
                dividing closed surface.

        Returns:
            np.ndarray: Array whose rows contain boundary contributions for each
                connected component of the domain split by the surface.
        """
        # The surface divides the domain into subdomains
        div_ = div * np.logical_not(surface)
        n_subdomains, flags = sps.csgraph.connected_components(div_ @ div_.T)

        # Extract subdomain boundaries
        div_csr = div.tocsr()
        sub_bdry = np.empty((n_subdomains, div_csr.shape[1]), dtype=int)
        for sub in range(n_subdomains):
            loc_div = div_csr[flags == sub, :]
            sub_bdry[sub] = np.sum(loc_div, axis=0)
            sub_bdry[sub] *= surface

        # The first subdomain boundary is equal to the negated sum of the others
        assert np.all(np.sum(sub_bdry, axis=0) == 0)

        return sub_bdry

    def compute_ridge_cohomology(self) -> None:
        """
        Computes cohomology classes associated with boundary ridge cycles.

        This is only needed in 3D when the face cohomology does not yet match the
        zero-space dimension. The method builds boundary node cycles and converts
        them to ridge-based homology representatives.
        """
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
        div = pg.restrict_csc_array(pg.curl(self.mdg), bdry_faces, bdry_ridges)
        curl = pg.restrict_csc_array(pg.grad(self.mdg), bdry_ridges, bdry_nodes)

        # Generate the cycles as lists of nodes
        node_cycles = self.compute_bdry_node_cycles(div, curl)

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

            cycle_basis[ridge_inds[edge_cycle], ind] = -np.asarray(
                face_nodes[cycle, edge_cycle]
            )

        coeffs = self.find_relevant_cycles(U_list, P_list)

        self.cycles[1] = cycle_basis.tocsc() @ sps.csc_array(coeffs)

    def compute_bdry_node_cycles(
        self, div: sps.csc_array, curl: sps.csc_array
    ) -> List[np.ndarray]:
        """
        Finds independent cycles in the boundary graph expressed by node indices.

        Args:
            div (sps.csc_array): The boundary divergence operator.
            curl (sps.csc_array): The boundary curl operator.

        Returns:
            List[np.ndarray]: A list of node cycles, one for each independent
                boundary cycle.
        """
        # Compute the number of connected components of the boundary
        incidence = div @ div.T
        n_components, ids = sps.csgraph.connected_components(incidence)
        cycle_basis = []

        for i in range(n_components):
            bdry_cells = ids == i
            loc_incidence = bdry_cells * incidence * bdry_cells[:, None]
            loc_incidence.eliminate_zeros()

            # Create the co-tree
            start_cell = int(np.argmax(bdry_cells))
            tree = sps.csgraph.breadth_first_tree(
                loc_incidence, start_cell, directed=False
            )

            # Find the mesh edges that correspond to tree edges
            c_start, c_end, _ = sps.find(tree)
            rows = np.hstack((c_start, c_end))
            cols = np.hstack([np.arange(c_start.size)] * 2)
            vals = np.ones_like(rows)

            face_finder = abs(div.T) @ sps.csc_array(
                (vals, (rows, cols)), shape=(div.shape[0], tree.nnz)
            )
            # We are only interested in the entries 2, because that means both
            # c_start and c_end coincide.
            face_finder.data = face_finder.data == 2
            face_finder.eliminate_zeros()
            face = sps.find(face_finder)[0]

            # Flag the relevant mesh edges in the grid
            cotree_faces = np.zeros(div.shape[1], dtype=bool)
            cotree_faces[face] = True

            # Find the edges on the cycles
            relevant_faces = (bdry_cells @ abs(div)).astype(bool)
            relevant_faces[cotree_faces] = False

            curl_surf = curl[relevant_faces, :]
            graph = curl_surf.T @ curl_surf
            graph.setdiag(np.zeros(graph.shape[0]))
            graph.eliminate_zeros()
            G = nx.from_scipy_sparse_array(graph)

            # Compute the cycles in terms of node lists
            cycle_basis.extend(nx.cycle_basis(G))

        return [np.array(cycle) for cycle in cycle_basis]

    def compute_edge_cycle(
        self, cycle: np.ndarray, face_nodes: sps.lil_array
    ) -> np.ndarray:
        """
        Converts a node cycle into the corresponding boundary edge cycle.

        Args:
            cycle (np.ndarray): Ordered node indices forming a cycle.
            face_nodes (sps.lil_array): Incidence matrix mapping boundary faces to
                their nodes.

        Returns:
            np.ndarray: Indices of the boundary edges belonging to the cycle.
        """
        n_start = cycle
        n_end = np.append(cycle[1:], cycle[0])

        rows = np.hstack((n_start, n_end))
        cols = np.hstack([np.arange(n_start.size)] * 2)
        vals = np.ones_like(rows)

        edge_finder = sps.csc_array(
            (vals, (cols, rows)), shape=(n_start.size, face_nodes.shape[0])
        ) @ abs(face_nodes)
        edge_finder.data = edge_finder.data == 2
        edge_finder.eliminate_zeros()
        edge_cycle = sps.find(edge_finder)[1]

        return edge_cycle

    def generate_submerged_polygon(
        self,
        node_cycle: np.ndarray,
        edge_cycle: np.ndarray,
        bdry_ridges: np.ndarray,
        bdry_nodes: np.ndarray,
    ) -> np.ndarray:
        """
        Constructs the polygon used to compute a linking number with the ridge cycle.

        Args:
            node_cycle (np.ndarray): Ordered boundary nodes describing a cycle.
            edge_cycle (np.ndarray): Indices of the boundary edges in the cycle.
            bdry_ridges (np.ndarray): Boolean mask selecting boundary ridges.
            bdry_nodes (np.ndarray): Boolean mask selecting boundary nodes.

        Returns:
            np.ndarray: Coordinates of the polygon in 3D, with alternating node and
                cell-center points.
        """
        # Preallocation
        U = np.zeros((3, 1 + 2 * len(node_cycle)))

        # Fill the even entries with grid nodes
        extended_cycle = np.append(node_cycle, node_cycle[0])
        nodes = np.where(bdry_nodes)[0]
        U[:, ::2] = self.top_sd.nodes[:, nodes[extended_cycle]]

        # Find an adjacent 3D cell for each boundary ridge
        # and fill the odd entries with the cell centers
        adj_faces = abs(pg.curl(self.mdg)[:, bdry_ridges])
        adjacency = abs(pg.div(self.mdg)) @ adj_faces[:, edge_cycle]
        adjacency = adjacency.tocsc()
        adj_cells = adjacency.indices[adjacency.indptr[:-1]]

        U[:, 1::2] = self.top_sd.cell_centers[:, adj_cells]

        return U

    def generate_shifted_polygon(
        self,
        cycle: np.ndarray,
        cell_faces: sps.lil_array,
        face_nodes: sps.lil_array,
        bdry_nodes: np.ndarray,
        edge_cycle: np.ndarray,
    ) -> np.ndarray:
        """
        Generates the shifted polygon used in the ridge linking-number test.

        The polygon is constructed by walking across adjacent boundary faces and
        collecting the face-based support of the cycle.

        Args:
            cycle (np.ndarray): Ordered node cycle on the boundary.
            cell_faces (sps.lil_array): Incidence between cells and faces.
            face_nodes (sps.lil_array): Incidence between faces and nodes.
            bdry_nodes (np.ndarray): Boolean mask selecting boundary nodes.
            edge_cycle (np.ndarray): Edge indices on the cycle.

        Returns:
            np.ndarray: Coordinates of the shifted polygon in the domain.
        """
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
        faces_arr = np.array(faces, dtype=int)

        to_keep = np.concatenate(([True], faces_arr[1:] != faces_arr[:-1]))
        faces_arr = faces_arr[to_keep]

        P = self.top_sd.nodes[:, bdry_nodes] @ (abs(face_nodes[:, faces_arr]) / 2)

        return P

    def compute_linking_number(self, U: np.ndarray, P: np.ndarray, n_points=5) -> float:
        r"""
        Computes the linking number of two closed polygons U and P.

        The linking number is the double contour integral
            1/(4*pi) * \int_P \int_U  Z \cdot (x - y) / |x - y|^3
        where x(alpha) traces P, y(beta) traces U, and Z is the cross product of
        the two segment tangents.
        """
        integral = 0.0

        # Loop over segments of U
        for i in range(U.shape[1] - 1):
            Z = np.cross(P[:, 1:] - P[:, :-1], U[:, i + 1] - U[:, i], axis=0)

            # Loop over segments of P
            for j in range(P.shape[1] - 1):
                # tangent vector of the P segment (constant over alpha)
                tau = P[:, j + 1] - P[:, j]
                tau_sqrd = tau @ tau

                def inner(beta: np.ndarray) -> np.ndarray:
                    """Closed-form alpha-integral"""
                    # y-values, given beta
                    y = np.outer(U[:, i], 1 - beta) + np.outer(U[:, i + 1], beta)

                    # Rewrite in terms of c = p_j - y
                    c = P[:, j][:, None] - y

                    # Using x = p_j + alpha * tau and Z dot tau = 0, we rewrite:
                    # Z dot (x - y) = Z dot (c + alpha * tau) = Z dot c

                    # Now the inner integral simplifies to
                    # 1/(4*pi) * \int_U (Z \cdot c) / |c + alpha * tau|^3
                    # The numerator is independent of alpha, so we move it out
                    # of the integral
                    Z_dot_c = Z[:, j] @ c

                    # The remaining integral is \int 1 / |c + alpha * tau|^3, for which
                    # Claude gave a closed formula.
                    b = tau @ c
                    c0 = np.sum(c * c, axis=0)
                    D = tau_sqrd * c0 - b**2
                    R0 = np.sqrt(c0)
                    R1 = np.sqrt(tau_sqrd + 2 * b + c0)

                    return (Z_dot_c / D) * ((tau_sqrd + b) / R1 - b / R0) / (4 * np.pi)

                # The outer integral needs numerical quadrature
                integral += spi.fixed_quad(inner, 0, 1, n=n_points)[0]

        # Check if the results are sufficiently close to integers
        rounded_int = np.round(integral)
        if np.abs(integral - rounded_int) <= 0.25:
            return rounded_int

        # Check if the number of quadrature points is reasonably small
        if n_points >= 16:
            raise RuntimeError(
                f"Linking number computation did not converge to an integer, "
                f"got {integral:.4f} with n_points={n_points}"
            )

        # If we did not find an integer, double the number of quadrature points
        return self.compute_linking_number(U, P, n_points=2 * n_points)

    def find_relevant_cycles(
        self, U_list: List[np.ndarray], P_list: List[np.ndarray]
    ) -> np.ndarray:
        """
        Selects the independent linear combinations of ridge cycles.

        The linking numbers between all candidate cycle pairs are assembled into
        a matrix, and the null space gives the coefficients of the relevant
        cycle combinations.

        Args:
            U_list (List[np.ndarray]): The submerged polygons.
            P_list (List[np.ndarray]): The shifted polygons.

        Returns:
            np.ndarray: Coefficient matrix defining the linear combinations that
                span the relevant homology classes.
        """
        n_cycles = len(U_list)
        G = np.empty((n_cycles,) * 2)

        for i in range(n_cycles):
            for j in range(n_cycles):
                G[i, j] = self.compute_linking_number(U_list[i], P_list[j], n_points=2)

        coeffs = spla.null_space(G)
        coeffs /= np.min(np.abs(coeffs[coeffs != 0]), axis=0)

        assert np.allclose(coeffs, coeffs.round()), "Non-integer coefficients?"

        return coeffs

    def recompute_edge_subspaces(self) -> None:
        """
        Updates the edge bar/zero/homology subspaces after ridge cohomology.

        The cycle-supporting edges are removed from the bar space, the remaining
        tree edges are chosen for the zero space, and one edge per cycle is moved
        into the cohomology subspace.
        """
        # Flag all the nodes on the cycles except the starting node
        cycle_nodes = abs(self.cycles[1]).T @ abs(pg.grad(self.mdg))
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

        # Overwrite the edge subspaces
        self.bar_spaces[1] = bar_space
        self.zer_spaces[1] = zer_space
        self.hom_spaces[1] = hom_space

    def compute_node_cohomology(self) -> None:
        """
        Assigns the node-level cohomology basis and eliminates it from the zero space.
        """
        self.hom_spaces[0] = self.zer_spaces[0].copy()
        self.zer_spaces[0] &= False

        cycle = np.atleast_2d(self.hom_spaces[0]).T
        self.cycles[0] = sps.csc_array(cycle, dtype=int)

    def compute_cohomology_basis(self, k: int) -> None:
        """
        Computes a basis for the cohomology space at form order k.

        Args:
            k (int): Form degree for which the cohomology basis is computed.
        """
        cycles = self.cycles[k].todense()
        pdc, dpc = self._decompose_without_cohomology(k, cycles)

        self.hom_basis[k] = cycles - pdc - dpc

    def cohom_projection(self, k: int, f: np.ndarray) -> np.ndarray:
        """
        Projects a form onto the cohomology subspace.

        Args:
            k (int): Form degree.
            f (np.ndarray): The input k-form coefficients.

        Returns:
            np.ndarray: The projection of f onto the cohomology space.
        """
        if not np.any(self.hom_spaces[k]):
            return np.zeros_like(f)

        basis = self.cycles[k]
        system = basis.T @ basis
        if system.shape == (1, 1):
            qf = self.hom_basis[k] * (basis.T @ f) / system[0, 0]
        else:
            dense_system = system.todense()
            coeff = np.linalg.solve(dense_system, basis.T @ f)
            qf = self.hom_basis[k] @ coeff

        return np.reshape(qf, f.shape)

    def apply(
        self, k: int, f: np.ndarray, solver: Callable = sps.linalg.spsolve
    ) -> np.ndarray:
        """
        Apply the Poincare operator

        Args:
            k (int): Order of the differential k-form that is input.
            f (np.ndarray): The input differential k-form
                as an array of the degrees of freedom.
            solver (Callable): The solver function to use.
                Defaults to sps.linalg.spsolve.

        Returns:
            np.ndarray: The image of f under the Poincaré operator, i.e. p(f)
        """
        if k == 0:
            return np.zeros(0)

        n_minus_k = self.dim - k
        _diff = diff(self.mdg, n_minus_k + 1)

        R_zer = pg.create_restriction(self.zer_spaces[k])
        R_bar = pg.create_restriction(self.bar_spaces[k - 1])

        pizer_dbar = R_zer @ _diff @ R_bar.T

        pf = R_bar.T @ solver(pizer_dbar, R_zer @ f)

        return pf

    def decompose(
        self,
        k: int,
        f: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Use the Poincaré operators to decompose :math:`f = pd(f) + dp(f) + q(f)`.

        Args:
            k (int): Order of the k-form f.
            f (np.ndarray): The function to be decomposed.

        Returns:
            Tuple[np.ndarray]: The decomposition of f as :math:`(dp(f), pd(f),
                q(f))`.
        """

        pdf, dpf = self._decompose_without_cohomology(k, f)
        qf = self.cohom_projection(k, f)

        return pdf, dpf, qf

    def _decompose_without_cohomology(
        self,
        k: int,
        f: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decomposes a form into the pd and dp parts without the cohomology component.

        Args:
            k (int): Form degree of the input array.
            f (np.ndarray): The k-form to decompose.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The components corresponding to p d(f)
                and d p(f).
        """

        n_minus_k = self.dim - k

        if k == self.dim:  # then df = 0
            pdf = np.zeros_like(f)
        else:
            df = diff(self.mdg, n_minus_k) @ f
            pdf = self.apply(k + 1, df)

        if k == 0:  # then pf = 0
            dpf = np.zeros_like(f)
        else:
            pf = self.apply(k, f)
            dpf = diff(self.mdg, n_minus_k + 1) @ pf

        pdf = np.reshape(pdf, f.shape)
        dpf = np.reshape(dpf, f.shape)

        return pdf, dpf

    def solve_subproblem(
        self,
        k: int,
        A: sps.csc_array,
        b: np.ndarray,
        solver: Callable = sps.linalg.spsolve,
    ) -> np.ndarray:
        """
        Solve a linear system on the subspace of differential forms identified by
        the Poincare object.

        Args:
            k (int): Order of the k-form.
            A (sps.csc_array): The system, usually a stiffness matrix.
            b (np.ndarray): The right-hand side vector.
            solver (Callable): The solver function to use. Defaults to
                sps.linalg.spsolve.

        Returns:
            np.ndarray: The solution.
        """
        LS = pg.LinearSystem(A, b)
        LS.flag_ess_bc(~self.bar_spaces[k], np.zeros_like(self.bar_spaces[k]))

        return LS.solve(solver=solver)

    def compute_euler_char(self) -> int:
        """
        Returns the Euler characteristic of the domain from the cohomology basis.

        Returns:
            int: The alternating sum of cohomology dimensions across all form degrees.
        """

        return sum((-1) ** k * self.hom_basis[k].shape[1] for k in range(self.dim + 1))

    def compute_basis_harmonic_forms(
        self, k: int, Mass: sps.csc_array | None = None
    ) -> np.ndarray:
        """
        Computes a harmonic basis for the cohomology space of degree k.

        Args:
            k (int): Degree of the cohomology space.
            Mass (sps.csc_array | None, optional): Mass matrix used for the
                orthogonalization. If not provided, a default mass matrix is
                constructed.

        Returns:
            np.ndarray: Orthonormalized harmonic basis functions for the cohomology
                space.
        """
        if k == 0:
            return self.hom_basis[k]

        D = diff(self.mdg, self.dim - k + 1)

        if Mass is None:
            Mass = cast(sps.csc_array, mass_matrix(self.mdg, self.dim - k))

        A = D.T @ Mass @ D
        b = D.T @ Mass @ self.hom_basis[k]

        proj = self.solve_subproblem(k - 1, A, b)

        orth = self.hom_basis[k] - D @ proj

        norms = np.array([v.T @ Mass @ v for v in orth.T])

        return orth / norms
