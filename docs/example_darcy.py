import numpy as np
import porepy as pp
import scipy.sparse as sps

import pygeon as pg

# creation of the grid
mesh_size = 0.1
dim = 2
sd = pg.unit_grid(dim, mesh_size, as_mdg=False)

# compute the geometrical properties of the grid
sd.compute_geometry()

# setup the discretization objects
key = "flow"

# declare the discretization objects, useful to setup the data
rt0 = pg.RT0(key)
p0 = pg.PwConstants(key)

# build the degrees of freedom
dofs = np.array([rt0.ndof(sd), p0.ndof(sd)])

# inverse of the permeability tensor
inv_perm = pp.SecondOrderTensor(np.ones(sd.num_cells))
param = {pg.SECOND_ORDER_TENSOR: inv_perm}
data = pp.initialize_data({}, key, param)

# compute the source term
mass_p0 = p0.assemble_mass_matrix(sd)
scalar_source = mass_p0 @ p0.interpolate(sd, lambda _: 1)

# construct the local matrices
A = rt0.assemble_mass_matrix(sd, data)
B = mass_p0 @ rt0.assemble_diff_matrix(sd)

# assemble the saddle point problem
spp = sps.block_array(
    [
        [A, -B.T],
        [B, None],
    ],
    format="csc",
)

# assemble the right-hand side
rhs = np.zeros(dofs.sum())
rhs[dofs[0] :] += scalar_source

# solve the problem
ls = pg.LinearSystem(spp, rhs)
x = ls.solve()

# split the solution into the components
idx = np.cumsum(dofs[:-1])
q, p = np.split(x, idx)

# post process variables
proj_q = rt0.eval_at_cell_centers(sd)
cell_q = (proj_q @ q).reshape((3, -1))
cell_p = p0.eval_at_cell_centers(sd) @ p

save = pp.Exporter(sd, "sol", folder_name="ex1")
save.write_vtu([("cell_p", cell_p), ("cell_q", cell_q)])
