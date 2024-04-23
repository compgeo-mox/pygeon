""" Module for spanning tree computation. """

from typing import Optional, Union

import numpy as np
import porepy as pp
import scipy.sparse as sps

import pygeon as pg


class SpanningTree:
    """
    Class that can perform a spanning tree solve, aka a "grid sweep".
    Useful to rapidly compute a flux field that balances a mass source.

    Attributes:
        system (sps.csc_matrix): The matrix used in the solve,
            which is triangular up to row/column permutations.
        expand (sps.csc_matrix): Expansion matrix from tree to global ordering.

        div (sps.csc_matrix): the divergence operator on the associated mdg.
        starting_cells (np.ndarray): the first cells of the spanning tree.
        starting_faces (np.ndarray): the first faces of the spanning tree.
        tree (sps.csc_array): The incidence matrix of the spanning tree.

    Methods:
        setup_system(mdg: pg.MixedDimensionalGrid, flagged_faces: np.ndarray) -> None:
            Sets up the linear system for solving the spanning tree problem.

        find_starting_faces(mdg: pg.MixedDimensionalGrid) -> int:
            Find the starting face for the spanning tree if None is provided.

        find_starting_cells() -> int:
            Find the starting cell for the spanning tree.

        compute_tree() -> sps.csc_array:
            Construct a spanning tree of the elements.

        flag_tree_faces() -> np.ndarray:
            Flag the faces in the mesh that correspond to edges of the tree.

        solve(f: np.ndarray) -> np.ndarray:
            Perform a spanning tree solve to compute a conservative flux field
            for given mass source.

        solve_transpose(rhs: np.ndarray) -> np.ndarray:
            Post-process the pressure by performing a transposed solve.

        visualize_2d(mdg: pg.MixedDimensionalGrid, fig_name: Optional[str] = None) -> None:
            Create a graphical illustration of the spanning tree superimposed on the grid.
    """

    def __init__(
        self,
        mdg: pg.MixedDimensionalGrid,
        starting_faces: Optional[Union[np.ndarray, int]] = None,
    ) -> None:
        """
        Initializes a SpanningTree object.

        Args:
            mdg (pg.MixedDimensionalGrid): The mixed-dimensional grid.
            starting_faces (Union[np.ndarray, int], optional): Indices of the starting faces. Defaults
                to None.
        """
        self.div = pg.div(mdg)

        self.starting_faces = self.find_starting_faces(mdg, starting_faces)
        self.starting_cells = self.find_starting_cells()
        self.tree = self.compute_tree()

        flagged_faces = self.flag_tree_faces()

        self.setup_system(mdg, flagged_faces)

    def setup_system(
        self, mdg: pg.MixedDimensionalGrid, flagged_faces: np.ndarray
    ) -> None:
        """
        Sets up the linear system for solving the spanning tree problem.

        Args:
            mdg (pg.MixedDimensionalGrid): The mixed-dimensional grid.
            flagged_faces (np.ndarray): Array of flagged faces.

        Returns:
            None
        """
        self.expand = pg.numerics.linear_system.create_restriction(
            flagged_faces
        ).T.tocsc()

        # Save the sparse LU decomposition of the system
        system = pg.cell_mass(mdg) @ self.div @ self.expand
        self.system = sps.linalg.splu(system)

    @staticmethod
    def extract_top_dim_sd(mdg: Union[pg.MixedDimensionalGrid, pg.Grid]) -> pg.Grid:
        """
        Extracts the top-dimensional grid of a mixed-dimensional grid.
        Returns the grid if mdg is a pp.Grid.

        The method is static so that it can be reused by SpanningWeightedTrees.

        Args:
            mdg: The (mixed-dimensional) grid.

        Returns:
            sd: The top-dimensional grid
        """
        if isinstance(mdg, pp.Grid):
            sd = mdg
        elif isinstance(mdg, pp.MixedDimensionalGrid):
            sd = mdg.subdomains(dim=mdg.dim_max())[0]
        else:
            raise TypeError

        return sd

    def find_starting_faces(
        self,
        mdg: pg.MixedDimensionalGrid,
        starting_faces: Union[np.ndarray, int],
    ) -> np.ndarray:
        """
        Find the starting face for the spanning tree.
        By default, this method returns the indices of the boundary faces of the mesh.

        Args:
            mdg (pg.MixedDimensionalGrid): The mixed-dimensional grid object.

        Returns:
            np.ndarray or int: The indices of the starting faces for the spanning tree.

        Raises:
            TypeError: If the input argument `mdg` is not of type `pp.Grid` or
            `pp.MixedDimensionalGrid`.
        """

        # The default case
        if starting_faces is None:
            sd = self.extract_top_dim_sd(mdg)

            # Find the boundary faces and remove duplicate connections to boundary cells
            bdry_faces = np.where(sd.tags["domain_boundary_faces"])[0]
            I_cells, J_faces, _ = sps.find(self.div.tocsc()[:, bdry_faces])
            _, uniq_ind = np.unique(I_cells, return_index=True)

            return bdry_faces[J_faces[uniq_ind]]

        # If the starting_face is an integer, recast it as an array
        elif not isinstance(starting_faces, np.ndarray):
            return np.array([starting_faces])

        else:
            return starting_faces

    def find_starting_cells(self) -> np.ndarray:
        """
        Find the starting cell for the spanning tree.

        Returns:
            np.ndarray: The indices of the starting cells.
        """
        return self.div.tocsc()[:, self.starting_faces].indices

    def compute_tree(self) -> sps.csc_array:
        """
        Construct a spanning tree of the elements.

        Returns:
            sps.csc_array: The computed spanning tree as a compressed sparse column matrix.
        """
        incidence = self.div @ self.div.T

        # If a single root is given, we default to the breadth-first tree generator from scipy
        if len(self.starting_cells) == 1:
            tree = sps.csgraph.breadth_first_tree(
                incidence, self.starting_cells[0], directed=False
            )

        # For multiple roots, we use our own generator
        else:
            # Initialize the tree
            tree = sps.lil_array(incidence.shape, dtype=int)

            # Keep track of the cells that have been visited by the tree
            visited_cells = np.zeros(np.size(incidence, 0), dtype=bool)
            visited_cells[self.starting_cells] = True

            while np.any(~visited_cells):
                # Extract global indices of the (un)visited cells
                vis_glob = np.where(visited_cells)[0]
                unv_glob = np.where(~visited_cells)[0]

                # Find connectivity between visited and unvisited cells
                local_graph = incidence[np.ix_(vis_glob, unv_glob)]
                vis_cell, unv_cell, _ = sps.find(local_graph)

                assert len(unv_cell) > 0, "No new neighbors found."

                # Each unvisited cell is linked to exactly one visited cell
                unv_cell, nc_ind = np.unique(unv_cell, return_index=True)
                vis_cell = vis_cell[nc_ind]

                # Save the connectivity in the tree and mark the new cells as visited
                tree[vis_glob[vis_cell], unv_glob[unv_cell]] = 1
                visited_cells[unv_glob[unv_cell]] = True

        return sps.csc_array(tree)

    def flag_tree_faces(self) -> np.ndarray:
        """
        Flag the faces in the mesh that correspond to edges of the tree.

        Returns:
            np.ndarray: A boolean array indicating the flagged faces in the mesh.
        """
        # Extract start/end cells for each edge of the tree
        c_start, c_end, _ = sps.find(self.tree)

        # Find the mesh faces that correspond to tree edges
        rows = np.hstack((c_start, c_end))
        cols = np.hstack([np.arange(c_start.size)] * 2)
        vals = np.ones_like(rows)

        face_finder = sps.csc_array(
            (vals, (rows, cols)), shape=(self.div.shape[0], self.tree.nnz)
        )
        face_finder = np.abs(self.div.T) @ face_finder
        I, J, V = sps.find(face_finder)

        _, index = np.unique(J[V == 2], return_index=True)
        tree_faces = I[V == 2][index]

        # Flag the relevant mesh faces in the grid
        flagged_faces = np.zeros(self.div.shape[1], dtype=bool)
        flagged_faces[self.starting_faces] = True
        flagged_faces[tree_faces] = True

        return flagged_faces

    def solve(self, f: np.ndarray) -> np.ndarray:
        """
        Perform a spanning tree solve to compute a conservative flux field
        for given mass source.

        Args:
            f (np.ndarray): Mass source, integrated against PwConstants.

        Returns:
            np.ndarray: the post-processed flux field
        """
        return self.expand @ self.system.solve(f)

    def solve_transpose(self, rhs: np.ndarray) -> np.ndarray:
        """
        Post-process the pressure by performing a transposed solve.

        Args:
            rhs (np.ndarray): Right-hand side, usually the mass matrix times the flux
                              minus boundary terms.

        Returns:
            np.ndarray: the post-processed pressure field
        """
        return self.system.solve(self.expand.T.tocsc() @ rhs, "T")

    def visualize_2d(
        self, mdg: pg.MixedDimensionalGrid, fig_name: Optional[str] = None
    ):  # pragma: no cover
        """
        Create a graphical illustration of the spanning tree superimposed on the grid.

        Args:
            mdg (pg.MixedDimensionalGrid) The object representing the grid.
            fig_name (Optional[str], optional). The name of the figure file to save the
                visualization.
        """
        import matplotlib.pyplot as plt
        import networkx as nx

        assert mdg.dim_max() == 2

        graph = nx.from_scipy_sparse_array(self.tree)
        cell_centers = np.hstack([sd.cell_centers for sd in mdg.subdomains()])

        fig_num = 1

        pp.plot_grid(
            mdg,
            alpha=0,
            fig_num=fig_num,
            plot_2d=True,
            if_plot=False,
        )

        fig = plt.figure(fig_num)
        ax = fig.add_subplot(111)

        node_color = ["blue"] * cell_centers.shape[1]
        for sc in self.starting_cells:
            node_color[sc] = "green"

        ax.autoscale(False)
        nx.draw(
            graph,
            cell_centers[: mdg.dim_max(), :].T,
            node_color=node_color,
            node_size=40,
            edge_color="red",
            ax=ax,
        )

        plt.draw()
        if fig_name is not None:
            plt.savefig(fig_name, bbox_inches="tight", pad_inches=0)

        plt.close()


class SpanningWeightedTrees:
    """
    Class that can perform a spanning weighted trees solve, based on
    the previously introduced class SpanningTree.
    It works very similarly to the previous one by considering multiple
    trees instead.

    Attributes:
        sptrs (list): List of SpanningTree objects.
        avg (function): Function to compute the average of a given array.

    Methods:
        solve(f: np.ndarray) -> np.ndarray:
            Perform a spanning weighted trees solve to compute a conservative flux field
            for given mass source.

        solve_transpose(rhs: np.ndarray) -> np.ndarray:
            Post-process the pressure by performing a transposed solve.

        find_starting_faces(mdg: pg.MixedDimensionalGrid, num: int) -> np.ndarray:
            Find the starting faces for each spanning tree if None is provided in the
            constructor.
            By default, an equidistribution of boundary faces is constructed.

    """

    def __init__(
        self,
        mdg: pg.MixedDimensionalGrid,
        weights: np.ndarray,
        starting_faces: Optional[np.ndarray] = None,
    ) -> None:
        """Constructor of the class

        Args:
            mdg (pg.MixedDimensionalGrid): The mixed dimensional grid.
            weights (np.ndarray): The weights to impose for each spanning tree, they need to
                sum to 1.
            starting_faces (Optional[np.ndarray]): The set of starting faces, if not specified
                equi-distributed boundary faces are selected.
        """
        if starting_faces is None:
            num = np.asarray(weights).size
            starting_faces = self.find_starting_faces(mdg, num)

        self.sptrs = [pg.SpanningTree(mdg, f) for f in starting_faces]
        self.avg = lambda v: np.average(v, axis=0, weights=weights)

    def solve(self, f: np.ndarray) -> np.ndarray:
        """
        Perform a spanning weighted trees solve to compute a conservative flux field
        for given mass source.

        Args:
            f (np.ndarray): Mass source, integrated against PwConstants.

        Returns:
            np.ndarray: The post-processed flux field.
        """
        return self.avg([st.solve(f) for st in self.sptrs])

    def solve_transpose(self, rhs: np.ndarray) -> np.ndarray:
        """
        Post-process the pressure by performing a transposed solve.

        Args:
            rhs (np.ndarray): Right-hand side, usually the mass matrix times the flux
                              minus boundary terms.

        Returns:
            np.ndarray: The post-processed pressure field.
        """
        return self.avg([st.solve_transpose(rhs) for st in self.sptrs])

    def find_starting_faces(self, mdg: pg.MixedDimensionalGrid, num: int) -> np.ndarray:
        """
        Find the starting faces for each spanning tree if None is provided in the
        constructor.
        By default, an equidistribution of boundary faces is constructed.

        Args:
            mdg (pg.MixedDimensionalGrid): The (mixed dimensional) grid.
            num (int): Number of faces to be selected.

        Returns:
            np.ndarray: The selected faces at the boundary.
        """

        sd = SpanningTree.extract_top_dim_sd(mdg)
        faces = np.where(sd.tags["domain_boundary_faces"])[0]
        return faces[np.linspace(0, faces.size, num, endpoint=False, dtype=int)]


class SpanningTreeElasticity(SpanningTree):
    """
    Represents a class for computing the spanning tree for the elastic problem.

    Attributes:
        expand (sps.csc_matrix): The expanded matrix for spanning tree computation.
        system (sps.csc_matrix): The computed system matrix.

    Methods:
        setup_system(self, mdg: pg.MixedDimensionalGrid, flagged_faces: np.ndarray) -> None:
            Set up the system for the spanning tree algorithm.

        compute_expand(self, sd: pg.Grid, flagged_faces: np.ndarray) -> sps.csc_matrix:
            Compute the expanded matrix for spanning tree computation.

        compute_system(self, sd: pg.Grid) -> sps.csc_matrix:
            Computes the system matrix for the given grid.
    """

    def setup_system(
        self, mdg: Union[pg.MixedDimensionalGrid, pg.Grid], flagged_faces: np.ndarray
    ) -> None:
        """
        Set up the system for the spanning tree algorithm.

        Args:
            mdg (pg.MixedDimensionalGrid): The mixed-dimensional grid.
            flagged_faces (np.ndarray): Array of flagged faces.

        Returns:
            None
        """
        sd = self.extract_top_dim_sd(mdg)
        self.expand = self.compute_expand(sd, flagged_faces)

        # Save the sparse LU decomposition of the system
        system = self.compute_system(sd)
        self.system = sps.linalg.splu(system)

    def compute_expand(self, sd: pg.Grid, flagged_faces: np.ndarray) -> sps.csc_matrix:
        """
        Compute the expanded matrix for spanning tree computation.

        Args:
            sd (pg.Grid): The grid object.
            flagged_faces (np.ndarray): Array of flagged faces.

        Returns:
            sps.csc_matrix: The expanded matrix for spanning tree computation.
        """
        key = "tree"
        bdm1 = pg.BDM1(key)

        # this operator fix one dof per face of the bdm space to capture displacement
        P_div = bdm1.proj_from_RT0(sd)

        # this operator maps to div-free bdm to capture rotation
        P_rot = P_div.copy()
        P_rot.data[::2] *= -1

        fn = sd.face_normals.copy()
        fn = fn / np.linalg.norm(fn, axis=0)
        fn_x, fn_y, _ = np.split(fn.ravel(), 3)

        # scaled P_rot with the face normals to be consistent with the div operator
        P_asym = sps.vstack((P_rot @ sps.diags(fn_x), P_rot @ sps.diags(fn_y)))

        # combine all the P
        P_div = sps.block_diag([P_div] * sd.dim)
        P = sps.hstack((P_div, P_asym), format="csc")

        # restriction to the flagged faces and restrict P to them
        expand = pg.numerics.linear_system.create_restriction(flagged_faces).T.tocsc()

        return P @ sps.block_diag([expand] * 3, format="csc")

    def compute_system(self, sd: pg.Grid) -> sps.csc_matrix:
        """
        Computes the system matrix for the given grid.

        Args:
            sd (pg.Grid): The grid object representing the domain.

        Returns:
            sps.csc_matrix: The computed system matrix.
        """
        # first we assemble the B matrix
        key = "tree"
        vec_bdm1 = pg.VecBDM1(key)
        vec_p0 = pg.VecPwConstants(key)
        p0 = pg.PwConstants(key)

        M_div = vec_p0.assemble_mass_matrix(sd)
        M_asym = p0.assemble_mass_matrix(sd)

        div = M_div @ vec_bdm1.assemble_diff_matrix(sd)
        asym = M_asym @ vec_bdm1.assemble_asym_matrix(sd)

        # create the solution operator
        return sps.vstack((-div, -asym)) @ self.expand
