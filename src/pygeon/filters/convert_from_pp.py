"""Conversion from porepy to pygeon."""

from typing import Union

import porepy as pp
import scipy.sparse as sps

import pygeon as pg


def convert_from_pp(
    obj: Union[pp.Grid, pp.MortarGrid, pp.MixedDimensionalGrid],
) -> None:
    """
    Convert an object from the PorePy library to the PyGeoN  library.

    Args:
        obj: The object to be converted. It can be one of the following types:
            - pp.Grid: Convert to pg.Grid.
            - pp.MortarGrid: Convert to pg.MortarGrid.
            - pp.MixedDimensionalGrid: Convert to pg.MixedDimensionalGrid.

    Raises:
        TypeError: If the input object is not one of the supported types.
    """
    if isinstance(obj, pp.Grid):
        obj.__class__ = pg.Grid
    elif isinstance(obj, pp.MortarGrid):
        obj.__class__ = pg.MortarGrid
    elif isinstance(obj, pp.MixedDimensionalGrid):
        # convert all the subdomains and interfaces
        for sd in obj.subdomains():
            convert_from_pp(sd)
        for intf in obj.interfaces():
            convert_from_pp(intf)

        obj.__class__ = pg.MixedDimensionalGrid
        if hasattr(obj, "initialize_data"):
            obj.initialize_data()
    else:
        raise TypeError

    if isinstance(obj, pg.Grid):
        # NOTE: it can be removed once PorePy also migrates to csc_array
        obj.face_nodes = sps.csc_array(obj.face_nodes)
        obj.cell_faces = sps.csc_array(obj.cell_faces)


def as_mdg(sd: Union[pp.MixedDimensionalGrid, pp.Grid]) -> pp.MixedDimensionalGrid:
    """
    Convert a grid object to a mixed-dimensional grid (MDG) object.

    Args:
        sd (Union[pp.MixedDimensionalGrid, pp.Grid]): The input grid object to be
            converted.

    Returns:
        pp.MixedDimensionalGrid: The converted mixed-dimensional grid object.

    Raises:
        ValueError: If the input grid object is neither a pp.MixedDimensionalGrid nor a
            pp.Grid.
    """
    if isinstance(sd, pp.MixedDimensionalGrid):
        return sd
    elif isinstance(sd, pp.Grid):
        return pp.meshing.subdomains_to_mdg([[sd]])
    else:
        raise ValueError
