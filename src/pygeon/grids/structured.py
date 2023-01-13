import porepy as pp
import numpy as np
import scipy.sparse as sps


class OctGrid(pp.Grid):
    """docstring for OctGrid."""

    def __init__(self, nx: np.array, physdims={}):

        nodes, face_nodes, cell_faces = self.create_grid(nx, physdims)
        name = "Octagon grid"

        super().__init__(2, nodes, face_nodes, cell_faces, name)

    def create_grid(self, nx: np.array, physdims: dict):
        xmin = physdims.get("xmin", 0)
        ymin = physdims.get("ymin", 0)
        xmax = physdims.get("xmax", 1)
        ymax = physdims.get("ymax", 1)

        n_oct = nx[0] * nx[1]
        n_hf = n_oct + nx[0]
        n_vf = n_oct + nx[1]

        h_first = np.arange(0, 2 * n_hf, 2)
        h_second = np.arange(1, 2 * n_hf + 1, 2)

        v_indices = (2 * n_hf + np.arange(2 * n_vf)).reshape((-1, nx[0] + 1))
        v_first = v_indices[::2, :].ravel()
        v_second = v_indices[1::2, :].ravel()
        corners = v_second[-1] + 1 + np.arange(4)

        # Coordinates

        factor = 1.0 / (2 + np.sqrt(2))

        horizontal_x = np.array([factor, 1 - factor] * nx[0])
        horizontal_x += np.repeat(np.arange(nx[0]), 2)
        horizontal_y = np.arange(nx[1] + 1)

        meshgrid = np.meshgrid(horizontal_x, horizontal_y)

        horizontal = np.vstack(
            (meshgrid[0].ravel(), meshgrid[1].ravel(), np.zeros(meshgrid[0].size))
        )

        vertical_x = np.arange(nx[0] + 1)
        vertical_y = np.array([factor, 1 - factor] * nx[1])
        vertical_y += np.repeat(np.arange(nx[1]), 2)

        meshgrid = np.meshgrid(vertical_x, vertical_y)

        vertical = np.vstack(
            (meshgrid[0].ravel(), meshgrid[1].ravel(), np.zeros(meshgrid[0].size))
        )

        corners_coords = np.array(
            [[0, nx[0], 0, nx[0]], [0, 0, nx[1], nx[1]], np.zeros(4)]
        )

        nodes = np.hstack((horizontal, vertical, corners_coords))

        # Face node connectivity
        fn_I = []

        # Horizontal
        start_end = np.vstack((h_first, h_second)).ravel("F")
        fn_I.append(start_end)

        # Vertical
        start_end = np.vstack((v_first, v_second)).ravel("F")
        fn_I.append(start_end)

        # SW
        starts = h_first[: -nx[0]]

        ends = v_first.reshape((-1, nx[0] + 1))
        ends = ends[:, :-1].ravel()

        start_end = np.vstack((starts, ends)).ravel("F")
        fn_I.append(start_end)

        # SE
        starts = h_second[: -nx[0]]

        ends = v_first.reshape((-1, nx[0] + 1))
        ends = ends[:, 1:].ravel()

        start_end = np.vstack((starts, ends)).ravel("F")
        fn_I.append(start_end)

        # NW
        starts = h_first[nx[0] :]

        ends = v_second.reshape((-1, nx[0] + 1))
        ends = ends[:, :-1].ravel()

        start_end = np.vstack((starts, ends)).ravel("F")
        fn_I.append(start_end)

        # NE
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

        face_nodes = sps.csc_matrix((np.ones(fn_I.size), (fn_I, fn_J)))

        # Cell faces

        cf_I = []
        cf_J = []
        cf_V = []

        # Bottoms of octagons
        cf_I.append(np.arange(n_oct))
        cf_V.append(np.ones(n_oct))

        # Tops of octagons
        cf_I.append(nx[0] + np.arange(n_oct))
        cf_V.append(-np.ones(n_oct))

        # Lefts of octagons
        verticals = (n_hf + np.arange(n_vf)).reshape((-1, nx[0] + 1))
        lefts = verticals[:, :-1].ravel()

        cf_I.append(lefts)
        cf_V.append(-np.ones(n_oct))

        # Rights of octagons
        rights = verticals[:, 1:].ravel()

        cf_I.append(rights)
        cf_V.append(np.ones(n_oct))

        # SW
        idx = n_hf + n_vf
        cf_I.append(idx + np.arange(n_oct))
        cf_V.append(-np.ones(n_oct))
        idx += n_oct

        # SE
        cf_I.append(idx + np.arange(n_oct))
        cf_V.append(np.ones(n_oct))
        idx += n_oct

        # NW
        cf_I.append(idx + np.arange(n_oct))
        cf_V.append(np.ones(n_oct))
        idx += n_oct

        # NE
        cf_I.append(idx + np.arange(n_oct))
        cf_V.append(-np.ones(n_oct))

        cf_J.append(np.tile(np.arange(n_oct), 8))

        # Squares
        n_sqrs = (nx[0] - 1) * (nx[1] - 1)

        # NE
        idx = n_hf + n_vf
        NE = np.reshape(idx + np.arange(n_oct), (-1, nx[0]))
        cf_I.append(NE[1:, 1:].ravel())
        cf_V.append(np.ones(n_sqrs))
        idx += n_oct

        # NW
        NW = np.reshape(idx + np.arange(n_oct), (-1, nx[0]))
        cf_I.append(NW[1:, :-1].ravel())
        cf_V.append(-np.ones(n_sqrs))
        idx += n_oct

        # SE
        SE = np.reshape(idx + np.arange(n_oct), (-1, nx[0]))
        cf_I.append(SE[:-1, 1:].ravel())
        cf_V.append(-np.ones(n_sqrs))
        idx += n_oct

        # SW
        SW = np.reshape(idx + np.arange(n_oct), (-1, nx[0]))
        cf_I.append(SW[:-1, :-1].ravel())
        cf_V.append(np.ones(n_sqrs))

        cf_J.append(np.tile(n_oct + np.arange(n_sqrs), 4))

        # BDRY Triangles
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

        # SW
        diag = n_hf + n_vf
        cf_I.append([idx, idx + 1, diag])
        cf_J.append(np.tile(id_cell, 3))
        cf_V.append([-1, 1, 1])
        idx += 2
        id_cell += 1

        # SE
        diag = n_hf + n_vf + n_oct + nx[0] - 1
        cf_I.append([idx, idx + 1, diag])
        cf_J.append(np.tile(id_cell, 3))
        cf_V.append([1, -1, -1])
        idx += 2
        id_cell += 1

        # NW
        diag = n_hf + n_vf + 3 * n_oct - nx[0]
        cf_I.append([idx, idx + 1, diag])
        cf_J.append(np.tile(id_cell, 3))
        cf_V.append([1, -1, -1])
        idx += 2
        id_cell += 1

        # NE
        diag = n_hf + n_vf + 4 * n_oct - 1
        cf_I.append([idx, idx + 1, diag])
        cf_J.append(np.tile(id_cell, 3))
        cf_V.append([-1, 1, 1])

        # Assemble
        cf_I = np.concatenate(cf_I)
        cf_J = np.concatenate(cf_J)
        cf_V = np.concatenate(cf_V)

        cell_faces = sps.csc_matrix((cf_V, (cf_I, cf_J)))

        return nodes, face_nodes, cell_faces
