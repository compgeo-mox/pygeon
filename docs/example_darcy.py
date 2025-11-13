import numpy as np
import scipy.sparse as sps

import porepy as pp
import pygeon as pg

# creation of the grid
mesh_size = 0.05
dim = 2
sd = pg.unit_grid(dim, mesh_size, as_mdg=False)

# compute the geometrical properties of the grid
sd.compute_geometry()

# setup the discretization objects
key = "flow"

# declare the discretization objects, useful to setup the data
rt0 = pg.RT0(key)
p0 = pg.PwConstants(key)

# set up the data for the flow problem
data = {}

# unitary permeability tensor
inv_perm = pp.SecondOrderTensor(np.ones(sd.num_cells))
parameters = {
    "second_order_tensor": inv_perm,
}
pp.initialize_data(sd, data, key, parameters)

# compute the source term
mass = p0.assemble_mass_matrix(sd)
scalar_source = mass @ p0.interpolate(sd, lambda _: 1)

# construct the local matrices
A = rt0.assemble_mass_matrix(sd, data)
mass_p0 = p0.assemble_mass_matrix(sd, data)
B = mass_p0 @ rt0.assemble_diff_matrix(sd)

# assemble the saddle point problem
spp = sps.block_array(
    [
        [A, -B.T],
        [B, None],
    ],
    format="csc",
)

# get the degrees of freedom for each variable
dof_p, dof_q = B.shape

# assemble the right-hand side
rhs = np.zeros(dof_p + dof_q)
rhs[dof_q:] += scalar_source

# solve the problem
ls = pg.LinearSystem(spp, rhs)
x = ls.solve()

# extract the variables
q = x[:dof_q]
p = x[-dof_p:]

# post process variables
proj_q = rt0.eval_at_cell_centers(sd)
cell_q = (proj_q @ q).reshape((3, -1))
cell_p = p0.eval_at_cell_centers(sd) @ p

save = pp.Exporter(sd, "sol", folder_name="ex1")
save.write_vtu([("cell_p", cell_p), ("cell_q", cell_q)])
