import numpy as np
import porepy as pp
import scipy.sparse as sps

import pygeon as pg

sd = pp.CartGrid([10, 10, 10], [1, 1, 1])
sd = pg.convert_from_pp(sd)
sd.compute_geometry()

bdry_faces = sd.tags["domain_boundary_faces"]

spring = np.zeros((3, sd.num_faces))

tpsa = pg.TPSA()

data = pp.initialize_data({}, tpsa.keyword, {pg.LAME_LAMBDA: np.inf})
bcs = pg.TPSA_BC(sd, data, tpsa.keyword)

tract_indices = np.zeros((3, sd.num_faces), dtype=bool)
tract_indices[-1, bdry_faces] = True
tract_indices[:, np.isclose(sd.face_centers[-1], 1)] = False
tract_indices[:, np.isclose(sd.face_centers[-1], 0)] = False

bcs.set_traction_bcs(tract_indices)

M = tpsa.assemble_elasticity_matrix(sd, data)

gravity = np.concatenate((np.zeros(2 * sd.num_cells), np.ones(sd.num_cells)))
rhs = tpsa.assemble_body_force(sd, gravity)

x = sps.linalg.spsolve(M, rhs)
u, r, p = np.split(x, np.cumsum(tpsa.ndofs(sd))[:-1])

u_plot = u.reshape((sd.dim, -1))
r_plot = r.reshape((sd.dim, -1))

save = pp.Exporter(sd, "tpsa_sol")
save.write_vtu([("disp", u_plot), ("vort", r_plot)])

pass
