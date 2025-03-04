import gmsh  # type: ignore
import meshio  # type: ignore
import numpy as np

import pygeon as pg


class Exporter:
    def __init__(self, obj, file_name, **kwargs):
        self.obj = obj
        self.file_name = file_name
        self.folder_name = kwargs.get("folder_name", "./")

    def write_vtu(self, data=[], **kwargs):
        if isinstance(self.obj, pg.Graph):
            self._graph_write_vtu(data, **kwargs)

    def _graph_write_vtu(self, data=[], **kwargs):
        # get user defined data
        radius = kwargs.pop("radius", 0.1)
        cylinder_radius = kwargs.pop("cylinder_radius", 0.025)
        graph = self.obj.graph
        # create a single sphere grid
        sphere_cells, sphere_pts = self._sphere()

        # add all the nodes as sphere
        cells, pts, cell_data = [], [], {i: [] for i in data}
        num_cells = 0
        for n, d in graph.nodes(data=True):
            # add the cells, pts and cell_data
            cells.append(sphere_cells + num_cells)
            num_cells += sphere_pts.shape[0]
            pts.append(radius * sphere_pts + d["centre"])
            [cell_data[i].append(d[i] * np.ones(sphere_cells.shape[0])) for i in data]

        # create a single cylinder grid
        cylinder_cells, cylinder_pts = self._cylinder()
        for n0, n1 in graph.edges():
            # add the cells, pts and (zero) cell_data
            cells.append(cylinder_cells + num_cells)
            num_cells += cylinder_pts.shape[0]

            # get the geometrical information to transform the cylinder
            n0_centre = graph.nodes[n0]["centre"]
            n1_centre = graph.nodes[n1]["centre"]
            dist = np.linalg.norm(n0_centre - n1_centre)
            R = pg.transformation.rotation(n1_centre - n0_centre)
            S = pg.transformation.scaling([cylinder_radius, cylinder_radius, dist])

            pts_loc = np.dot(np.dot(R.T, S), cylinder_pts.T).T + n0_centre
            pts.append(pts_loc)
            [cell_data[i].append(np.zeros(cylinder_cells.shape[0])) for i in data]

        # group the cells and vertices
        cells = [("triangle", np.vstack(cells))]
        cell_data = {i: [np.hstack(v)] for i, v in cell_data.items()}

        # create the meshio grid
        meshio_grid = meshio.Mesh(np.vstack(pts), cells, cell_data=cell_data)
        file_name = self.folder_name + self.file_name
        meshio.write(file_name, meshio_grid, binary=True)

    def _sphere(self):
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)

        # create a sphere object
        model = gmsh.model()
        model.occ.addSphere(0, 0, 0, 1)

        # generate mesh
        model.occ.synchronize()
        model.mesh.generate(2)

        # extract mesh data
        _, _, cells = model.mesh.getElements(dim=2)
        _, pts, _ = model.mesh.getNodes()
        gmsh.finalize()

        return cells[0].reshape(-1, 3) - 1, pts.reshape(-1, 3)

    def _cylinder(self):
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)

        # create a sphere object
        model = gmsh.model()
        model.occ.addCylinder(0, 0, 0, 0, 0, 1, 1)

        # generate mesh
        model.occ.synchronize()
        model.mesh.generate(2)

        # extract mesh data
        _, _, cells = model.mesh.getElements(dim=2)
        _, pts, _ = model.mesh.getNodes()
        gmsh.finalize()

        return cells[0].reshape(-1, 3) - 1, pts.reshape(-1, 3)
