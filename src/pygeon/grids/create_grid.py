import porepy as pp

import pygeon as pg


def grid_from_domain(domain, mesh_size, **kwargs):
    """
    Create a grid from a domain with a mesh size.

    Parameters:
        domain (pp.Domain): the domain of the grid
        mesh_size (double): the mesh size
        kwargs: additional options are the following
            mesh_size_min (double): the minimum mesh size, default is mesh_size
            as_mdg (bool): return the grid as a mixed-dimensional grid

        Returns:
            Either a pg.MixedDimensionalGrid or a pg.Grid.
    """
    mesh_size_min = kwargs.get("mesh_size_min", mesh_size / 10)
    mesh_kwargs = {"mesh_size_frac": mesh_size, "mesh_size_min": mesh_size_min}
    mdg = pp.create_fracture_network(domain=domain).mesh(mesh_kwargs)

    pg.convert_from_pp(mdg)
    if kwargs.get("as_mdg", True):
        return mdg
    else:
        return mdg.subdomains(dim=mdg.dim_max())[0]


def unit_grid(dim, mesh_size, **kwargs):
    """
    Create a unit square or cube grid with a mesh size.

    Parameters:
        dim (int): the dimension of the grid, must be 2 or 3
        mesh_size (double): the mesh size
        kwargs: additional options are the following
            mesh_size_min (double): the minimum mesh size, default is mesh_size
            as_mdg (bool): return the grid as a mixed-dimensional grid

        Returns:
            Either a pg.MixedDimensionalGrid or a pg.Grid.
    """

    bbox = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}
    if dim == 3:
        bbox.update({"zmin": 0, "zmax": 1})

    domain = pp.Domain(bounding_box=bbox)
    return grid_from_domain(domain, mesh_size, **kwargs)
