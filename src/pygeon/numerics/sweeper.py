import numpy as np
import porepy as pp
import scipy.sparse as sps

import pygeon as pg


class Sweeper:
    """
    Class that can perform a grid sweep.
    Useful to rapidly compute a flux field that balances a mass source.

    Attributes:
        system (sps.csc_matrix): The matrix used to perform the sweep,
            which is triangular up to row/column permutations.
        expand (sps.csc_matrix): Expansion matrix from tree to global ordering.
    """

    def __init__(self, mdg, starting_face=None) -> None:
        div = pg.div(mdg)
        tree, starting_face, _ = self.compute_tree(mdg, div, starting_face)

        # Extract start/end cells for each edge of the tree
        c_start, c_end, _ = sps.find(tree)

        # Find the mesh faces that correspond to tree edges
        rows = np.hstack((c_start, c_end))
        cols = np.hstack([np.arange(c_start.size)] * 2)
        vals = np.ones_like(rows)

        face_finder = sps.csc_matrix((vals, (rows, cols)))
        face_finder = np.abs(div.T) @ face_finder
        I, _, V = sps.find(face_finder)

        tree_faces = I[V == 2]

        # Flag the relevant mesh faces in the grid
        flag = np.zeros(div.shape[1], dtype=bool)
        flag[starting_face] = True
        flag[tree_faces] = True

        self.expand = pg.numerics.linear_system.create_restriction(flag).T.tocsc()
        self.system = pg.cell_mass(mdg) @ div @ self.expand

    def compute_tree(self, mdg, div, starting_face=None):
        # Find the starting face for the spanning tree
        if starting_face is None:
            # Extract the top-dimensional grid
            if isinstance(mdg, pp.Grid):
                sd = mdg
            elif isinstance(mdg, pp.MixedDimensionalGrid):
                sd = mdg.subdomains()[0]
                assert sd.dim == mdg.dim_max()
            else:
                raise TypeError

            starting_face = np.argmax(sd.tags["domain_boundary_faces"])

        # Find the starting cell for the spanning tree
        starting_cell = div.T.tocsr()[starting_face, :].indices[0]

        # Construct a spanning tree of the elements
        tree = sps.csgraph.breadth_first_tree(
            div @ div.T, starting_cell, directed=False
        )
        return tree, starting_face, starting_cell

    def sweep(self, f) -> np.ndarray:
        """
        Perform a grid sweep to compute a conservative flux field for given mass source.

        Parameters:
            f (np.ndarray): Mass source, integrated against PwConstants.

        Returns:
            np.ndarray: the post-processed pressure field
        """

        return self.expand @ sps.linalg.spsolve(self.system, f)

    def sweep_transpose(self, rhs) -> np.ndarray:
        """
        Post-process the pressure by performing a transposed sweep.

        Parameters:
            rhs (np.ndarray): Right-hand side, usually the mass matrix times the flux
                              minus boundary terms.

        Returns:
            np.ndarray: the post-processed pressure field
        """

        return sps.linalg.spsolve(self.system.T, self.expand.T @ rhs)

    def visualize(self, mdg, starting_face=None, fig_name=None):
        div = pg.div(mdg)
        tree, _, starting_cell = self.compute_tree(mdg, div, starting_face)

        import networkx as nx
        import matplotlib.pyplot as plt

        graph = nx.from_scipy_sparse_array(tree)
        cell_centers = np.hstack([sd.cell_centers for sd in mdg.subdomains()])

        fig_num = 1

        pp.plot_grid(
            mdg,
            alpha=0,
            fig_num=fig_num,
            plot_2d=mdg.dim_max() == 2,
            if_plot=False,
        )

        fig = plt.figure(fig_num)
        ax = fig.add_subplot(111)

        node_color = ["blue"] * cell_centers.shape[1]
        node_color[starting_cell] = "green"

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
