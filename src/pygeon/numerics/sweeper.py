import numpy as np
import porepy as pp
import pygeon as pg
import scipy.sparse as sps


class Sweeper:
    def __init__(self, mdg) -> None:
        div_op = pg.div(mdg)

        if isinstance(mdg, pp.Grid):
            starting_face = np.argmax(mdg.tags["domain_boundary_faces"])
        else:
            sd_first = mdg.subdomains()[0]
            assert sd_first.dim == mdg.dim_max()
            starting_face = np.argmax(sd_first.tags["domain_boundary_faces"])

        starting_cell = div_op.T.tocsr()[starting_face, :].indices[0]

        tree = sps.csgraph.breadth_first_tree(
            div_op @ div_op.T, starting_cell, directed=False
        )
        I, J, _ = sps.find(tree)

        rows = np.hstack((I, J))
        cols = np.hstack([np.arange(I.size)] * 2)
        vals = np.ones_like(rows)

        face_finder = sps.csc_matrix((vals, (rows, cols)))
        face_finder = np.abs(div_op.T) @ face_finder
        I, _, V = sps.find(face_finder)

        active_faces = I[V == 2]

        flag = np.zeros(div_op.shape[1], dtype=bool)
        flag[starting_face] = True
        flag[active_faces] = True

        self.expand = pg.numerics.linear_system.create_restriction(flag).T
        self.system = pg.cell_mass(mdg) @ div_op @ self.expand

    def sweep(self, f) -> np.ndarray:
        return self.expand @ sps.linalg.spsolve(self.system, f)
