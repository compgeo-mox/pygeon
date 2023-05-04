import gmsh
import numpy as np
import porepy as pp

from pygeon.filters.convert_from_pp import convert_from_pp


def grid_unitary(dim, mesh_size, **kwargs):
    """
    Create a bi-dimensional unit square grid with a mesh size.

    Parameters:
        dim (int): the dimension of the grid
        mesh_size (np.ndarray or double): if double is the mesh size associated to
            all the four coordinates, if np.ndarray the mesh size for each coordinate.
        kwargs: refer to the function grid_2d_from_coords

        Returns:
            Either a pg.MixedDimensionalGrid or a pg.Grid.
    """

    bbox = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}
    if dim == 3:
        bbox.update({"zmin": 0, "zmax": 1})

    domain = pp.Domain(bounding_box=bbox)

    mesh_size_min = kwargs.get("mesh_size_min", mesh_size / 10)
    mesh_kwargs = {"mesh_size_frac": mesh_size, "mesh_size_min": mesh_size_min}
    mdg = pp.create_fracture_network(domain=domain).mesh(mesh_kwargs)

    convert_from_pp(mdg)
    if kwargs.get("as_mdg", True):
        return mdg
    else:
        return mdg.subdomains(dim=dim)[0]
