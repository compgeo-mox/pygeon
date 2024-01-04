from typing import Optional

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
    """

    def __init__(
        self, mdg: pg.MixedDimensionalGrid, starting_face: Optional[int] = None
    ) -> None:
        """
        Initializes a SpanningTree object.

        Args:
            mdg (pg.MixedDimensionalGrid): The mixed-dimensional grid.
            starting_face (Optional[int], optional): The index of the starting face. Defaults to None.
        """
        self.div = pg.div(mdg)

        if starting_face is None:
            self.starting_face = self.find_starting_face(mdg)
        else:
            self.starting_face = starting_face

        self.starting_cell = self.find_starting_cell()
        self.tree = self.compute_tree()

        flagged_faces = self.flag_tree_faces()

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
            TypeError: If the input argument `mdg` is not of type `pp.Grid` or `pp.MixedDimensionalGrid`.
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
        I, _, V = sps.find(face_finder)

        tree_faces = I[V == 2]

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
    ):
        """
        Create a graphical illustration of the spanning tree superimposed on the grid.

        Args:
            mdg (pg.MixedDimensionalGrid) The object representing the grid.
            fig_name (Optional[str], optional). The name of the figure file to save the visualization.
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
        __init__: Constructor of the class.
        solve: Perform a spanning weighted trees solve to compute a conservative flux field.
        solve_transpose: Post-process the pressure by performing a transposed solve.
        find_starting_faces: Find the starting faces for each spanning tree if None is provided.
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
            weights (np.ndarray): The weights to impose for each spanning tree, they need to sum to 1.
            starting_faces (Optional[np.ndarray]): The set of starting faces, if not specified equi-distributed boundary faces are selected.
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
