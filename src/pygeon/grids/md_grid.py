"""Module for the MixedDimensionalGrid class."""

from typing import Callable, Optional

import numpy as np
import porepy as pp

import pygeon as pg


class MixedDimensionalGrid(pp.MixedDimensionalGrid):
    """
    Represents a mixed-dimensional grid.

    This class extends the functionality of the `pp.MixedDimensionalGrid` class.
    It provides methods for initializing the grid, computing geometry, and tagging mesh
    entities.

    Attributes:
        None

    Methods:
        compute_geometry():
            Compute geometric entities and tags for the subdomains and interfaces.

        initialize_data():
            Initializes the data for the multi-dimensional grid.

        num_subdomain_faces():
            Compute the total number of faces of the mixed-dimensional grid.

        num_subdomain_ridges():
            Compute the total number of ridges in the mixed-dimensional grid.

        tag_leafs():
            Tag the mesh entities that correspond to a mesh entity of a lower-dimensional grid
            in a grid bucket.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize a MixedDimensionalGrid object.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            None
        """
        super(MixedDimensionalGrid, self).__init__(*args, **kwargs)
        self.initialize_data()

    def compute_geometry(self) -> None:
        """
        Compute geometric entities and tags for the subdomains and interfaces.

        This method iterates over the subdomains and interfaces of the grid
        and computes their geometric entities and tags. It calls the
        `compute_geometry` method of each subdomain and interface, passing
        the necessary parameters. Finally, it tags the leaf nodes of the grid.

        Args:
            None

        Returns:
            None
        """
        for sd in self.subdomains():
            sd.compute_geometry()

        for intf in self.interfaces():
            sd_pair = self.interface_to_subdomain_pair(intf)
            intf.compute_geometry(sd_pair)  # type: ignore[call-arg]

        self.tag_leafs()

    def initialize_data(self) -> None:
        """
        Initializes the data for the multi-dimensional grid.

        This method initializes the data for each subdomain and interface
        in the multi-dimensional grid.
        It sets the parameters and discretization matrices for each subdomain and interface.

        Args:
            None

        Returns:
            None
        """
        for sd, data in self.subdomains(return_data=True):
            perm = pp.SecondOrderTensor(np.ones(sd.num_cells))
            data[pp.PARAMETERS] = pp.Parameters(
                sd,
                [pg.UNITARY_DATA],
                [
                    {"second_order_tensor": perm},
                ],
            )
            data[pp.DISCRETIZATION_MATRICES] = {pg.UNITARY_DATA: {}}

        for _, data in self.interfaces(return_data=True):
            data[pp.PARAMETERS] = pp.Parameters(
                sd,
                [pg.UNITARY_DATA],
                [
                    {"normal_diffusivity": 1.0},
                ],
            )
            data[pp.DISCRETIZATION_MATRICES] = {pg.UNITARY_DATA: {}}

    def num_subdomain_faces(
        self, cond: Optional[Callable[[pg.Grid], bool]] = None
    ) -> int:
        """
        Compute the total number of faces of the mixed-dimensional grid.

        A function can be passed to filter subdomains and/or interfaces.

        Args:
            cond: optional, predicate with a grid as input.

        Returns:
            int: the total number of faces of the mixed-dimensional grid.

        """
        if cond is None:
            cond = lambda _: True
        return np.sum([sd.num_faces for sd in self.subdomains() if cond(sd)], dtype=int)  # type: ignore[arg-type]

    def num_subdomain_ridges(
        self, cond: Optional[Callable[[pg.Grid], bool]] = None
    ) -> int:
        """
        Compute the total number of ridges in the mixed-dimensional grid.

        A function can be passed to filter subdomains and/or interfaces.

        Args:
            cond: Optional. A predicate function that takes a grid as input.

        Returns:
            int: The total number of ridges in the mixed-dimensional grid.
        """
        if cond is None:
            cond = lambda _: True
        return np.sum(
            [sd.num_ridges for sd in self.subdomains() if cond(sd)],  # type: ignore[attr-defined,arg-type]
            dtype=int,
        )

    def num_subdomain_peaks(
        self, cond: Optional[Callable[[pg.Grid], bool]] = None
    ) -> int:
        """
        Compute the total number of peaks in the mixed-dimensional grid.

        A function can be passed to filter subdomains and/or interfaces.

        Args:
            cond: Optional. A predicate function that takes a grid as input.

        Returns:
            int: The total number of peaks in the mixed-dimensional grid.
        """
        if cond is None:
            cond = lambda _: True
        return np.sum(
            [sd.num_peaks for sd in self.subdomains() if cond(sd)],  # type: ignore[attr-defined,arg-type]
            dtype=int,
        )

    def tag_leafs(self) -> None:
        """
        Tag the mesh entities that correspond to a mesh entity of a lower-dimensional
        grid in a grid bucket.
        TODO: Use these tags to generate mixed-dimensional inner products.

        Args:
            None

        Returns:
            None
        """
        for sd in self.subdomains():
            # Tag the faces that correspond to a cell in a codim 1 domain
            sd.tags["leaf_faces"] = sd.tags["tip_faces"] + sd.tags["fracture_faces"]

            # Initialize the other tags
            sd.tags["leaf_ridges"] = np.zeros(sd.num_ridges, dtype=bool)  # type: ignore[attr-defined]
            sd.tags["leaf_peaks"] = np.zeros(sd.num_peaks, dtype=bool)  # type: ignore[attr-defined]

        for intf in self.interfaces():
            # Tag the ridges that correspond to a cell in a codim 2 domain
            if intf.dim >= 1:
                sd_up, sd_down = self.interface_to_subdomain_pair(intf)
                sd_up.tags["leaf_ridges"] += (
                    abs(intf.face_ridges) @ sd_down.tags["leaf_faces"]  # type: ignore[attr-defined]
                ).astype("bool")

        for intf in self.interfaces():
            # Tag the peaks that correspond to a codim 3 domain
            if intf.dim >= 2:
                sd_up, sd_down = self.interface_to_subdomain_pair(intf)
                sd_up.tags["leaf_peaks"] += (
                    abs(intf.ridge_peaks) @ sd_down.tags["leaf_ridges"]  # type: ignore[attr-defined]
                ).astype("bool")
