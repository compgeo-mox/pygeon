import numpy as np
import porepy as pp


class MixedDimensionalGrid(pp.MixedDimensionalGrid):
    def __init__(self, *args, **kwargs):
        super(MixedDimensionalGrid, self).__init__(*args, **kwargs)

    def compute_geometry(self):
        """
        Compute geometric entities and tags for the subdomains and interfaces.
        """

        for sd in self.subdomains():
            sd.compute_geometry()

        for intf in self.interfaces():
            sd_pair = self.interface_to_subdomain_pair(intf)
            intf.compute_geometry(sd_pair)

        self.tag_leafs()

    def num_subdomain_faces(self, cond=None) -> int:
        """Compute the total number of faces of the mixed-dimensional grid.

        A function can be passed to filter subdomains and/or interfaces.

        Args:
            cond: optional, predicate with a grid as input.

        Return:
            int: the total number of faces of the mixed-dimensional grid.

        """
        if cond is None:
            cond = lambda _: True
        return np.sum(  # type: ignore
            [sd.num_faces for sd in self.subdomains() if cond(sd)], dtype=int
        )

    def num_subdomain_ridges(self, cond=None) -> int:
        """Compute the total number ridges of of the mixed-dimensional grid.

        A function can be passed to filter subdomains and/or interfaces.

        Args:
            cond: optional, predicate with a grid as input.

        Return:
            int: the total number of faces of the mixed-dimensional grid.

        """
        if cond is None:
            cond = lambda _: True
        return np.sum(  # type: ignore
            [sd.num_ridges for sd in self.subdomains() if cond(sd)], dtype=int
        )

    def tag_leafs(self):
        """
        Tag the mesh entities that correspond to a mesh entity of a lower-dimensional grid in a grid bucket.
        TODO: Use these tags to generate mixed-dimensional inner products.
        """

        for sd in self.subdomains():
            # Tag the faces that correspond to a cell in a codim 1 domain
            sd.tags["leaf_faces"] = sd.tags["tip_faces"] + sd.tags["fracture_faces"]

            # Initialize the other tags
            sd.tags["leaf_ridges"] = np.zeros(sd.num_ridges, dtype=bool)
            sd.tags["leaf_peaks"] = np.zeros(sd.num_peaks, dtype=bool)

        for intf in self.interfaces():
            # Tag the ridges that correspond to a cell in a codim 2 domain
            if intf.dim >= 1:
                sd_up, sd_down = self.interface_to_subdomain_pair(intf)
                sd_up.tags["leaf_ridges"] += (
                    abs(intf.face_ridges) * sd_down.tags["leaf_faces"]
                ).astype("bool")

        for intf in self.interfaces():
            # Tag the peaks that correspond to a codim 3 domain
            if intf.dim >= 2:
                sd_up, sd_down = self.interface_to_subdomain_pair(intf)
                sd_up.tags["leaf_peaks"] += (
                    abs(intf.ridge_peaks) * sd_down.tags["leaf_ridges"]
                ).astype("bool")
