import numpy as np
import scipy.sparse as sps

import pygeon as pg


class OctagonGrid(pg.Grid):
    """
    A structured grid with octagons and squares in the interior
    and triangles near the boundary.
    """

    def __init__(self, nx: np.array, physdims={}, name="Octagon grid"):
        """Constructor for the 2D octagonal grid.

        Args:
            nx (np.ndarray): number of cells in the x and y directions
            physdims (np.ndarray or dict): the physical dimensions, either
                as a numpy array or a dict with keys "xmin", "xmax", "ymin", and "ymax"
            name (str): Name of grid.
        """

        # Define the nodes as a 3 x num_nodes array
        nodes = self.compute_nodes(nx, physdims)

        # Compute face-node connectivity
        face_nodes = self.compute_face_nodes(nx)

        # Compute cell-face connectivity
        cell_faces = self.compute_cell_faces(nx)

        super().__init__(2, nodes, face_nodes, cell_faces, name)

    def compute_nodes(self, nx, physdims):
        # Compute the off-set for the coordinates of an octagon inside a unit square
        offset = 1.0 / (2 + np.sqrt(2))

        # Compute the nodes on the horizontal faces
        horizontal_x = np.array([offset, 1 - offset] * nx[0])
        horizontal_x += np.repeat(np.arange(nx[0]), 2)
        horizontal_y = np.arange(nx[1] + 1)

        meshgrid = np.meshgrid(horizontal_x, horizontal_y)

        horizontal = np.vstack(
            (meshgrid[0].ravel(), meshgrid[1].ravel(), np.zeros(meshgrid[0].size))
        )

        # Compute the nodes on the vertical faces
        vertical_x = np.arange(nx[0] + 1)
        vertical_y = np.array([offset, 1 - offset] * nx[1])
        vertical_y += np.repeat(np.arange(nx[1]), 2)

        meshgrid = np.meshgrid(vertical_x, vertical_y)

        vertical = np.vstack(
            (meshgrid[0].ravel(), meshgrid[1].ravel(), np.zeros(meshgrid[0].size))
        )

        # Include the corners
        corners_coords = np.array(
            [[0, nx[0], 0, nx[0]], [0, 0, nx[1], nx[1]], np.zeros(4)]
        )

        # Collect all nodes and rescale according to physdims
        nodes = np.hstack((horizontal, vertical, corners_coords))
        nodes = self.rescale_nodes(nodes, nx, physdims)

        return nodes

    def rescale_nodes(self, nodes, nx, physdims):
        xmin, ymin = 0.0, 0.0

        if isinstance(physdims, dict):
            xmin = physdims.get("xmin", 0)
            ymin = physdims.get("ymin", 0)

            physdims = np.array(
                [physdims.get("xmax", 1) - xmin, physdims.get("ymax", 1) - ymin]
            )

        # Rescale according to physdims and nx
        nodes[0, :] *= physdims[0] / nx[0]
        nodes[1, :] *= physdims[1] / nx[1]

        # Shift according to xmin and ymin
        nodes[0, :] += xmin
        nodes[1, :] += ymin

        return nodes

    def compute_face_nodes(self, nx):
        n_oct = nx[0] * nx[1]
        n_hf = n_oct + nx[0]
        n_vf = n_oct + nx[1]

        # Compute the face-node connectivity

        h_first = np.arange(0, 2 * n_hf, 2)
        h_second = np.arange(1, 2 * n_hf + 1, 2)

        v_indices = (2 * n_hf + np.arange(2 * n_vf)).reshape((-1, nx[0] + 1))
        v_first = v_indices[::2, :].ravel()
        v_second = v_indices[1::2, :].ravel()
        corners = v_second[-1] + 1 + np.arange(4)

        fn_I = []

        # Horizontal
        start_end = np.vstack((h_first, h_second)).ravel("F")
        fn_I.append(start_end)

        # Vertical
        start_end = np.vstack((v_first, v_second)).ravel("F")
        fn_I.append(start_end)

        # South West
        starts = h_first[: -nx[0]]

        ends = v_first.reshape((-1, nx[0] + 1))
        ends = ends[:, :-1].ravel()

        start_end = np.vstack((starts, ends)).ravel("F")
        fn_I.append(start_end)

        # South East
        starts = h_second[: -nx[0]]

        ends = v_first.reshape((-1, nx[0] + 1))
        ends = ends[:, 1:].ravel()

        start_end = np.vstack((starts, ends)).ravel("F")
        fn_I.append(start_end)

        # North West
        starts = h_first[nx[0] :]

        ends = v_second.reshape((-1, nx[0] + 1))
        ends = ends[:, :-1].ravel()

        start_end = np.vstack((starts, ends)).ravel("F")
        fn_I.append(start_end)

        # North East
        starts = h_second[nx[0] :]

        ends = v_second.reshape((-1, nx[0] + 1))
        ends = ends[:, 1:].ravel()

        start_end = np.vstack((starts, ends)).ravel("F")
        fn_I.append(start_end)

        # Boundaries

        # South
        starts = h_second[: nx[0] - 1]
        ends = h_first[1 : nx[0]]

        start_end = np.vstack((starts, ends)).ravel("F")
        fn_I.append(start_end)

        # North
        starts = h_second[-nx[0] : -1]
        ends = h_first[n_hf - nx[0] + 1 :]

        start_end = np.vstack((starts, ends)).ravel("F")
        fn_I.append(start_end)

        # West

        starts = v_second[:: nx[0] + 1][:-1]
        ends = v_first[:: nx[0] + 1][1:]

        start_end = np.vstack((starts, ends)).ravel("F")
        fn_I.append(start_end)

        # East

        starts = v_second[nx[0] :: nx[0] + 1][:-1]
        ends = v_first[nx[0] :: nx[0] + 1][1:]

        start_end = np.vstack((starts, ends)).ravel("F")
        fn_I.append(start_end)

        # Corners
        fn_I.append([h_first[0], corners[0]])
        fn_I.append([v_first[0], corners[0]])
        fn_I.append([h_second[nx[0] - 1], corners[1]])
        fn_I.append([v_first[nx[0]], corners[1]])
        fn_I.append([h_first[-nx[0]], corners[2]])
        fn_I.append([v_second[-nx[0] - 1], corners[2]])
        fn_I.append([h_second[-1], corners[3]])
        fn_I.append([v_second[-1], corners[3]])

        fn_I = np.concatenate(fn_I)
        fn_J = np.repeat(np.arange(fn_I.size / 2), 2).astype(int)

        return sps.csc_matrix((np.ones(fn_I.size), (fn_I, fn_J)))

    def compute_cell_faces(self, nx):
        n_oct = nx[0] * nx[1]
        n_hf = n_oct + nx[0]
        n_vf = n_oct + nx[1]

        cf_I = []
        cf_J = []
        cf_V = []

        # Souths of octagons
        cf_I.append(np.arange(n_oct))
        cf_V.append(np.ones(n_oct))

        # Norths of octagons
        cf_I.append(nx[0] + np.arange(n_oct))
        cf_V.append(-np.ones(n_oct))

        # Easts of octagons
        verticals = (n_hf + np.arange(n_vf)).reshape((-1, nx[0] + 1))
        easts = verticals[:, :-1].ravel()

        cf_I.append(easts)
        cf_V.append(-np.ones(n_oct))

        # Wests of octagons
        wests = verticals[:, 1:].ravel()

        cf_I.append(wests)
        cf_V.append(np.ones(n_oct))

        # South West
        idx = n_hf + n_vf
        cf_I.append(idx + np.arange(n_oct))
        cf_V.append(-np.ones(n_oct))
        idx += n_oct

        # South East
        cf_I.append(idx + np.arange(n_oct))
        cf_V.append(np.ones(n_oct))
        idx += n_oct

        # North West
        cf_I.append(idx + np.arange(n_oct))
        cf_V.append(np.ones(n_oct))
        idx += n_oct

        # North East
        cf_I.append(idx + np.arange(n_oct))
        cf_V.append(-np.ones(n_oct))

        cf_J.append(np.tile(np.arange(n_oct), 8))

        # Squares
        n_sqrs = (nx[0] - 1) * (nx[1] - 1)

        # North East
        idx = n_hf + n_vf
        NE = np.reshape(idx + np.arange(n_oct), (-1, nx[0]))
        cf_I.append(NE[1:, 1:].ravel())
        cf_V.append(np.ones(n_sqrs))
        idx += n_oct

        # North West
        NW = np.reshape(idx + np.arange(n_oct), (-1, nx[0]))
        cf_I.append(NW[1:, :-1].ravel())
        cf_V.append(-np.ones(n_sqrs))
        idx += n_oct

        # South East
        SE = np.reshape(idx + np.arange(n_oct), (-1, nx[0]))
        cf_I.append(SE[:-1, 1:].ravel())
        cf_V.append(-np.ones(n_sqrs))
        idx += n_oct

        # South West
        SW = np.reshape(idx + np.arange(n_oct), (-1, nx[0]))
        cf_I.append(SW[:-1, :-1].ravel())
        cf_V.append(np.ones(n_sqrs))

        cf_J.append(np.tile(n_oct + np.arange(n_sqrs), 4))

        # Boundary triangles
        id_cell = n_oct + n_sqrs

        # South
        idx = n_hf + n_vf
        cf_I.append(idx + np.arange(1, nx[0]))
        cf_V.append(np.ones(nx[0] - 1))
        idx += n_oct

        cf_I.append(idx + np.arange(nx[0] - 1))
        cf_V.append(-np.ones(nx[0] - 1))
        idx += 3 * n_oct

        cf_I.append(idx + np.arange(nx[0] - 1))
        cf_V.append(np.ones(nx[0] - 1))

        cf_J.append(np.tile(id_cell + np.arange(nx[0] - 1), 3))
        id_cell += nx[0] - 1

        # North
        idx = n_hf + n_vf + 2 * n_oct
        cf_I.append(idx + n_oct + np.arange(-nx[0] + 1, 0))
        cf_V.append(-np.ones(nx[0] - 1))
        idx += n_oct

        cf_I.append(idx + n_oct + np.arange(-nx[0], -1))
        cf_V.append(np.ones(nx[0] - 1))
        idx += n_oct + nx[0] - 1

        cf_I.append(idx + np.arange(nx[0] - 1))
        cf_V.append(-np.ones(nx[0] - 1))

        cf_J.append(np.tile(id_cell + np.arange(nx[0] - 1), 3))
        id_cell += nx[0] - 1

        # West
        idx = n_hf + n_vf
        indices = np.reshape(idx + np.arange(n_oct), (-1, nx[0]))
        cf_I.append(indices[1:, 0])
        cf_V.append(np.ones(nx[1] - 1))
        idx += 2 * n_oct

        indices = np.reshape(idx + np.arange(n_oct), (-1, nx[0]))
        cf_I.append(indices[:-1, 0])
        cf_V.append(-np.ones(nx[1] - 1))
        idx += 2 * n_oct + 2 * (nx[0] - 1)

        cf_I.append(idx + np.arange(nx[1] - 1))
        cf_V.append(-np.ones(nx[1] - 1))

        cf_J.append(np.tile(id_cell + np.arange(nx[1] - 1), 3))
        id_cell += nx[1] - 1

        # East
        idx = n_hf + n_vf + n_oct
        indices = np.reshape(idx + np.arange(n_oct), (-1, nx[0]))
        cf_I.append(indices[1:, -1])
        cf_V.append(-np.ones(nx[1] - 1))
        idx += 2 * n_oct

        indices = np.reshape(idx + np.arange(n_oct), (-1, nx[0]))
        cf_I.append(indices[:-1, -1])
        cf_V.append(np.ones(nx[1] - 1))
        idx += n_oct + 2 * (nx[0] - 1) + (nx[1] - 1)

        cf_I.append(idx + np.arange(nx[1] - 1))
        cf_V.append(np.ones(nx[1] - 1))

        cf_J.append(np.tile(id_cell + np.arange(nx[1] - 1), 3))
        id_cell += nx[1] - 1

        # Corners
        idx += nx[1] - 1

        # South West
        diag = n_hf + n_vf
        cf_I.append([idx, idx + 1, diag])
        cf_J.append(np.tile(id_cell, 3))
        cf_V.append([-1, 1, 1])
        idx += 2
        id_cell += 1

        # South East
        diag = n_hf + n_vf + n_oct + nx[0] - 1
        cf_I.append([idx, idx + 1, diag])
        cf_J.append(np.tile(id_cell, 3))
        cf_V.append([1, -1, -1])
        idx += 2
        id_cell += 1

        # North West
        diag = n_hf + n_vf + 3 * n_oct - nx[0]
        cf_I.append([idx, idx + 1, diag])
        cf_J.append(np.tile(id_cell, 3))
        cf_V.append([1, -1, -1])
        idx += 2
        id_cell += 1

        # North East
        diag = n_hf + n_vf + 4 * n_oct - 1
        cf_I.append([idx, idx + 1, diag])
        cf_J.append(np.tile(id_cell, 3))
        cf_V.append([-1, 1, 1])

        # Assemble
        cf_I = np.concatenate(cf_I)
        cf_J = np.concatenate(cf_J)
        cf_V = np.concatenate(cf_V)

        return sps.csc_matrix((cf_V, (cf_I, cf_J)))
