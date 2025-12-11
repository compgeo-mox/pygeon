"""Conversion from porepy to pygeon."""

from typing import cast, overload

import porepy as pp
import scipy.sparse as sps

import pygeon as pg


@overload
def convert_from_pp(obj: pp.Grid) -> pg.Grid: ...


@overload
def convert_from_pp(obj: pp.MortarGrid) -> pg.MortarGrid: ...


@overload
def convert_from_pp(obj: pp.MixedDimensionalGrid) -> pg.MixedDimensionalGrid: ...


def convert_from_pp(
    obj: pp.Grid | pp.MortarGrid | pp.MixedDimensionalGrid,
) -> pg.Grid | pg.MortarGrid | pg.MixedDimensionalGrid:
    """
    Convert an object from the PorePy library to the PyGeoN  library.

    Args:
        obj: The object to be converted. It can be one of the following types:

            - pp.Grid: Convert to pg.Grid.
            - pp.MortarGrid: Convert to pg.MortarGrid.
            - pp.MixedDimensionalGrid: Convert to pg.MixedDimensionalGrid.

    Returns:
        The converted PyGeoN object (pg.Grid, pg.MortarGrid, or
        pg.MixedDimensionalGrid).

    Raises:
        TypeError: If the input object is not one of the supported types.
    """
    if isinstance(obj, pp.Grid):
        obj.__class__ = pg.Grid
        obj = cast(pg.Grid, obj)
    elif isinstance(obj, pp.MortarGrid):
        obj.__class__ = pg.MortarGrid
        obj = cast(pg.MortarGrid, obj)
    elif isinstance(obj, pp.MixedDimensionalGrid):
        # convert all the subdomains and interfaces
        for sd in obj.subdomains():
            convert_from_pp(sd)
        for intf in obj.interfaces():
            convert_from_pp(intf)

        obj.__class__ = pg.MixedDimensionalGrid
        obj = cast(pg.MixedDimensionalGrid, obj)
    else:
        raise TypeError

    if isinstance(obj, pg.Grid):
        # NOTE: it can be removed once PorePy also migrates to csc_array
        obj.face_nodes = sps.csc_array(obj.face_nodes)
        obj.cell_faces = sps.csc_array(obj.cell_faces)

    return obj


def as_mdg(sd: pp.MixedDimensionalGrid | pp.Grid) -> pp.MixedDimensionalGrid:
    """
    Convert a grid object to a mixed-dimensional grid (MDG) object.

    Args:
        sd (pp.MixedDimensionalGrid | pp.Grid): The input grid object to be
            converted.

    Returns:
        pp.MixedDimensionalGrid: The converted mixed-dimensional grid object.

    Raises:
        ValueError: If the input grid object is neither a pp.MixedDimensionalGrid
            nor a pp.Grid.
    """
    if isinstance(sd, pp.MixedDimensionalGrid):
        return sd
    elif isinstance(sd, pp.Grid):
        return pp.meshing.subdomains_to_mdg([[sd]])
    else:
        raise ValueError
