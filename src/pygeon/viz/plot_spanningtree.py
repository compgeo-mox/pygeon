import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import porepy as pp
import scipy.sparse as sps

import pygeon as pg


def plot_spanningtree(
    spt: pg.SpanningTree,
    mdg: pg.MixedDimensionalGrid | pg.Grid,
    fig_name: str | None = None,
    **kwargs,
) -> None:
    """
    Create a graphical illustration of the spanning tree superimposed on the grid.

    Args:
        spt (pg.SpanningTree): The spanningtree object
        mdg (pg.MixedDimensionalGrid | pg.Grid): The object representing the grid.
        fig_name (str): The name of the figure file to save the
            visualization.
        **kwargs: Additional options:

            - draw_grid (bool): Plot the grid.
            - draw_tree (bool): Plot the tree spanning the cells.
            - draw_cotree (bool): Plot the tree spanning the nodes.
            - start_color (str): Color of the "starting" cells, next to the boundary
    """
    mdg = pg.as_mdg(mdg)

    assert mdg.dim_max() == 2
    sd_top = mdg.subdomains()[0]

    draw_grid = kwargs.get("draw_grid", True)
    draw_tree = kwargs.get("draw_tree", True)
    draw_cotree = kwargs.get("draw_cotree", False)
    start_color = kwargs.get("start_color", "green")

    fig_num = 1

    # Draw grid
    if draw_grid:
        pp.plot_grid(
            mdg, alpha=0, fig_num=fig_num, plot_2d=True, if_plot=False, title=""
        )

    # Define the figure and axes
    fig = plt.figure(fig_num)

    # The grid is drawn by PorePy if desired
    if draw_grid:
        pp_ax = fig.gca()
        pp_ax.set_xlabel("")
        pp_ax.set_ylabel("")
        pp_ax.set_aspect("equal")
        plt.tick_params(
            left=False,
            labelleft=False,
            labelbottom=False,
            bottom=False,
        )
        ax = fig.add_subplot(111)
        ax.set_xlim(pp_ax.get_xlim())
        ax.set_ylim(pp_ax.get_ylim())

    # If there is no PorePy grid plot, we create our own axes
    else:
        ax = fig.gca()

        min_coord = np.min(sd_top.nodes, axis=1)
        max_coord = np.max(sd_top.nodes, axis=1)

        ax.set_xlim((min_coord[0], max_coord[0]))
        ax.set_ylim((min_coord[1], max_coord[1]))

    ax.set_aspect("equal")

    # Draw the tree that spans all cells
    if draw_tree:
        graph = nx.from_scipy_sparse_array(spt.tree)
        cell_centers = np.hstack([sd.cell_centers for sd in mdg.subdomains()])
        node_color = ["blue"] * cell_centers.shape[1]
        for sc in spt.starting_cells:
            node_color[sc] = start_color

        nx.draw(
            graph,
            cell_centers[: mdg.dim_max(), :].T,
            node_color=node_color,
            node_size=40,
            edge_color="red",
            ax=ax,
        )

        # Add connections from the roots to the starting faces
        num_bdry = len(spt.starting_faces)
        bdry_graph = sps.diags_array(
            np.ones(num_bdry),
            offsets=num_bdry,
            shape=(2 * num_bdry, 2 * num_bdry),
        )
        graph = nx.from_scipy_sparse_array(bdry_graph)

        face_centers = np.hstack([sd.face_centers for sd in mdg.subdomains()])
        cell_centers = np.hstack([sd.cell_centers for sd in mdg.subdomains()])
        node_centers = np.hstack(
            (
                face_centers[: mdg.dim_max(), spt.starting_faces],
                cell_centers[: mdg.dim_max(), spt.starting_cells],
            )
        ).T

        nx.draw(
            graph,
            node_centers,
            node_size=0,
            edge_color="red",
            ax=ax,
        )

    # Draw the tree that spans all nodes
    if draw_cotree:
        curl = pg.curl(mdg)[~spt.flagged_faces, :]
        incidence = curl.T @ curl
        incidence -= sps.triu(incidence)

        graph = nx.from_scipy_sparse_array(incidence)

        node_color = ["black"] * sd_top.nodes.shape[1]

        nx.draw(
            graph,
            sd_top.nodes[: mdg.dim_max(), :].T,
            node_color=node_color,
            node_size=30,
            edge_color="purple",
            width=1.5,
            ax=ax,
        )

    plt.draw()
    if fig_name is not None:
        plt.savefig(fig_name, bbox_inches="tight", pad_inches=0.1)
        plt.close()
    else:
        plt.show()
