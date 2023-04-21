import numpy as np
import scipy.sparse as sps

import porepy as pp
import pygeon as pg


class Sweeper():
    def __init__(self, sd) -> None:
        div_op = pg.div(sd)

        starting_face = np.argmax(sd.tags["domain_boundary_faces"])
        starting_cell = sd.cell_faces.tocsr()[starting_face, :].indices[0]

        tree = sps.csgraph.breadth_first_tree(
            div_op @ div_op.T, starting_cell, directed=False)
        I, J, _ = sps.find(tree)

        rows = np.hstack((I, J))
        cols = np.hstack([np.arange(I.size)] * 2)
        vals = np.ones_like(rows)

        face_finder = sps.csc_matrix((vals, (rows, cols)))
        face_finder = np.abs(sd.cell_faces) @ face_finder
        I, _, V = sps.find(face_finder)

        active_faces = I[V == 2]

        flag = np.zeros(sd.num_faces, dtype=bool)
        flag[starting_face] = True
        flag[active_faces] = True

        self.expand = pg.numerics.linear_system.create_restriction(flag).T
        self.system = div_op @ self.expand

    def sweep(self, f) -> np.ndarray:
        return self.expand @ sps.linalg.spsolve(self.system, f)


if __name__ == "__main__":
    sd = pp.CartGrid([5] * 2)
    pg.convert_from_pp(sd)

    swp = Sweeper(sd)

    f = np.random.rand(sd.num_cells)

    q_f = swp.sweep(f)

    div_op = pg.div(sd)

    assert np.allclose(div_op @ q_f, f)
