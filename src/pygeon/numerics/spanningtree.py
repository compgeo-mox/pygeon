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
        starting_cell (int): the first cell of the spanning tree.
        starting_face (int): the first face of the spanning tree.
        tree (sps.csc_array): The incidence matrix of the spanning tree.

    Methods:
        setup_system(mdg: pg.MixedDimensionalGrid, flagged_faces: np.ndarray) -> None:
            Sets up the linear system for solving the spanning tree problem.

        find_starting_face(mdg: pg.MixedDimensionalGrid) -> int:
            Find the starting face for the spanning tree if None is provided.

        find_starting_cell() -> int:
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
        self, mdg: pg.MixedDimensionalGrid, starting_face: Optional[int] = None
    ) -> None:
        """
        Initializes a SpanningTree object.

        Args:
            mdg (pg.MixedDimensionalGrid): The mixed-dimensional grid.
            starting_face (Optional[int], optional): The index of the starting face. Defaults
                to None.
        """
        self.div = pg.div(mdg)

        if starting_face is None:
            self.starting_face = self.find_starting_face(mdg)
        else:
            self.starting_face = starting_face

        self.starting_cell = self.find_starting_cell()
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
        self.system = pg.cell_mass(mdg) @ self.div @ self.expand

    def find_starting_face(self, mdg: pg.MixedDimensionalGrid) -> int:
        """
        Find the starting face for the spanning tree if None is provided.

        By default, this method returns the index of the first boundary face of the mesh.

        Args:
            mdg (pg.MixedDimensionalGrid): The mixed-dimensional grid object.

        Returns:
            int: The index of the starting face for the spanning tree.

        Raises:
            TypeError: If the input argument `mdg` is not of type `pp.Grid` or
            `pp.MixedDimensionalGrid`.
        """
        if isinstance(mdg, pp.Grid):
            sd = mdg
        elif isinstance(mdg, pp.MixedDimensionalGrid):
            # Extract the top-dimensional grid
            sd = mdg.subdomains()[0]
            assert sd.dim == mdg.dim_max()
        else:
            raise TypeError

        return np.argmax(sd.tags["domain_boundary_faces"])

    def find_starting_cell(self) -> int:
        """
        Find the starting cell for the spanning tree.

        Returns:
            int: The index of the starting cell.
        """
        return self.div.tocsc()[:, self.starting_face].indices[0]

    def compute_tree(self) -> sps.csc_array:
        """
        Construct a spanning tree of the elements.

        Returns:
            sps.csc_array: The computed spanning tree as a compressed sparse column matrix.
        """
        tree = sps.csgraph.breadth_first_tree(
            self.div @ self.div.T, self.starting_cell, directed=False
        )
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

        face_finder = sps.csc_array((vals, (rows, cols)))
        face_finder = np.abs(self.div.T) @ face_finder
        I, J, V = sps.find(face_finder)

        _, index = np.unique(J[V == 2], return_index=True)
        tree_faces = I[V == 2][index]

        # Flag the relevant mesh faces in the grid
        flagged_faces = np.zeros(self.div.shape[1], dtype=bool)
        flagged_faces[self.starting_face] = True
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
        return self.expand @ sps.linalg.spsolve(self.system, f)

    def solve_transpose(self, rhs: np.ndarray) -> np.ndarray:
        """
        Post-process the pressure by performing a transposed solve.

        Args:
            rhs (np.ndarray): Right-hand side, usually the mass matrix times the flux
                              minus boundary terms.

        Returns:
            np.ndarray: the post-processed pressure field
        """
        return sps.linalg.spsolve(self.system.T.tocsc(), self.expand.T.tocsc() @ rhs)

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
        node_color[self.starting_cell] = "green"

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
        if isinstance(mdg, pp.Grid):
            sd = mdg
        elif isinstance(mdg, pp.MixedDimensionalGrid):
            # Extract the top-dimensional grid
            sd = mdg.subdomains()[0]
            assert sd.dim == mdg.dim_max()
        else:
            raise TypeError

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
        # NOTE: we are assuming only one higher dimensional 2d grid
        if isinstance(mdg, pg.MixedDimensionalGrid):
            sd = mdg.subdomains(dim=mdg.dim_max())[0]
        else:
            sd = mdg

        self.expand = self.compute_expand(sd, flagged_faces)
        self.system = self.compute_system(sd)

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
