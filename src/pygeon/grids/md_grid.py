import numpy as np
import scipy.sparse as sps
import porepy as pp

"""
Acknowledgments:
    The functionalities related to the ridge computations are modified from
    github.com/anabudisa/md_aux_precond developed by Ana Budiša and Wietse M. Boon.
"""


class MixedDimensionalGrid(pp.MixedDimensionalGrid):
    def __init__(self, *args, **kwargs):
        super(MixedDimensionalGrid, self).__init__(*args, **kwargs)

    def compute_geometry(self):
        """Compute geometric quantities for the grids."""
        for sd in self.subdomains():
            sd.compute_geometry()

        for intf in self.interfaces():
            pair = self.interface_to_subdomain_pair(intf)
            intf.compute_geometry(pair)

    def tag_leafs(self):
        return
        """
        Tag the mesh entities that correspond to a mesh entity of a lower-dimensional grid in a grid bucket.
        TODO: Use these tags to generate mixed-dimensional inner products.

        Parameters:
            gb (pp.GridBucket): The grid bucket.
        """

        for g in self.get_grids():
            # Tag the faces that correspond to a cell in a codim 1 domain
            g.tags["leaf_faces"] = g.tags["tip_faces"] + g.tags["fracture_faces"]

            # Initialize the other tags
            g.tags["leaf_ridges"] = np.zeros(g.num_ridges, dtype=bool)
            g.tags["leaf_peaks"] = np.zeros(g.num_peaks, dtype=bool)

        for e, d in self.edges():
            # Tag the ridges that correspond to a cell in a codim 2 domain
            mg = d["mortar_grid"]

            if mg.dim >= 1:
                g_down, g_up = self.nodes_of_edge(e)
                g_up.tags["leaf_ridges"] += (
                    abs(mg.face_ridges) * g_down.tags["leaf_faces"]
                ).astype("bool")

        for e, d in self.edges():
            # Tag the peaks that correspond to a codim 3 domain
            mg = d["mortar_grid"]

            if mg.dim >= 2:
                g_down, g_up = self.nodes_of_edge(e)
                g_up.tags["leaf_peaks"] += (
                    abs(mg.ridge_peaks) * g_down.tags["leaf_ridges"]
                ).astype("bool")
