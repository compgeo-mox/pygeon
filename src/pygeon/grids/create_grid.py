import numpy as np
import gmsh
import porepy as pp

from pygeon.filters.convert_from_pp import convert_from_pp


def grid_2d_from_coords(coords, mesh_size, **kwargs):
    """
    Create a bi-dimensional grid from a list of counter-clock wise ordered points and a mesh size.
    The grid can be immersed in 3d but its coordinates must lie on a plane.

    Parameters:
        coords (np.ndarray): ordered list of the coordinates, they must be num_points x 3
        mesh_size (np.ndarray or double): if double is the mesh size associated to all the coordinates,
            if np.ndarray the mesh size for each coordinate.
        kwargs: used parameters are
            as_mdg (bool, default = True): to return the grid as a MixedDimensionalGrid (default) or Grid.
            open_gmsh (bool, default = False): to open gmsh, useful for debugging
            file_name (str, default = "grid.msh"): file name of the gmsh file

        Returns:
            Either a pg.MixedDimensionalGrid or a pg.Grid.
    """
    # porepy flags for gmsh
    gmsh_flag = pp.fracs.gmsh_interface.PhysicalNames

    # initialize gmsh and remove the verbosity
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 1)

    # in case mesh_size is a double convert it to a vector
    mesh_size = mesh_size * np.ones(coords.shape[1])

    # add the boundary point to gmsh
    pts = np.array(
        [gmsh.model.geo.addPoint(*p[:-1], p[-1]) for p, h in zip(coords.T, mesh_size)]
    )
    # add the created points to a specific group requested by porepy
    [_add_group(0, p, gmsh_flag.DOMAIN_BOUNDARY_POINT.value + str(p)) for p in pts]

    # add the boundary lines to gmsh constructed from the boundary points
    lines = np.array(
        [gmsh.model.geo.addLine(p0, p1) for p0, p1 in zip(pts, np.roll(pts, -1))]
    )
    # add the created lines to a specific group requested by porepy
    [_add_group(1, l, gmsh_flag.DOMAIN_BOUNDARY_LINE.value + str(l)) for l in lines]

    # construct the line loop from the lines and the domain from the line loop
    loop = gmsh.model.geo.addCurveLoop(lines)
    domain = gmsh.model.geo.addPlaneSurface([loop])
    # add the created domain to a specific group requested by porepy
    _add_group(2, [domain], gmsh_flag.DOMAIN.value)

    # generate the grid
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)

    if kwargs.get("open_gmsh", False):
        gmsh.fltk.run()

    file_name = kwargs.get("file_name", "grid.msh")
    gmsh.write(file_name)
    gmsh.finalize()

    # import back the constructed grid
    mdg = pp.fracture_importer.dfm_from_gmsh(file_name, dim=2)
    convert_from_pp(mdg)

    if kwargs.get("as_mdg", True):
        return mdg
    else:
        return mdg.subdomains(dim=2)[0]


def grid_unit_square(mesh_size, **kwargs):
    """
    Create a bi-dimensional unit square grid with a mesh size.
    The coordinates are given as
    np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]])

    Parameters:
        mesh_size (np.ndarray or double): if double is the mesh size associated to all the four coordinates,
            if np.ndarray the mesh size for each coordinate.
        kwargs: refer to the function grid_2d_from_coords

        Returns:
            Either a pg.MixedDimensionalGrid or a pg.Grid.
    """
    coords = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]])
    return grid_2d_from_coords(coords, mesh_size, as_mdg=False)


def _add_group(dim, element, name):
    element = [element] if not isinstance(element, list) else element
    group = gmsh.model.addPhysicalGroup(dim, element)
    gmsh.model.setPhysicalName(dim, group, name)
