import os
import shutil
import numpy as np
import scipy.sparse as sps

import porepy as pp
import pygeon as pg

import inspect

grid_size = [5, 5]
dim = [1,1]
dt = 0.1
num_steps = 10

key = "mass"

nat_bc = []
ess_bc = []
source = []
sol = []

# For flow
RT0 = pg.RT0(key)
# For pressure 
P0 = pg.PwConstants(key)

def create_grid(grid_size, dim):
    sd = pp.StructuredTriangleGrid(grid_size, dim)
    # convert the grid into a mixed-dimensional grid
    mdg = pg.as_mdg(sd)
    # Convert to a pygeon grid
    pg.convert_from_pp(sd)
    sd.compute_geometry()
    return mdg


def first_order_tensor(grid,
        vx: np.ndarray,
        vy: np.ndarray = None,
        vz: np.ndarray = None,
        ):
    
    n_cells = vx.size
    vel = np.zeros((3, n_cells))

    vel[0, ::] = vx
            
    if vy is not None:
        vel[1, ::] = vy

    if vz is not None:
        vel[2, ::] = vz

    return vel  

def source_term1(x):
    print(x)
    # Example: a Gaussian source
    center = sd.cell_centers[:,sd.num_cells // 2]
    sigma = 0.05
    r2 = np.sum((x - center)**2)
    return np.exp(-r2 / (2 * sigma**2))


def source_term(x):

    center_cell = sd.num_cells // 2 - 1 

    bd_nodes = sd.get_all_boundary_nodes()

    node_map = sd.cell_nodes()

    center_nodes = node_map[:,center_cell].nonzero()[0]

    mask = ~np.isin(center_nodes, bd_nodes)
    center_nodes = center_nodes[mask]

    source_nodes_coord = sd.nodes.T[center_nodes]

    return 1.0 if np.any(np.all(source_nodes_coord == x, axis=1)) else 0.0

def u_bc(x):
    return 1.0 if abs(x[0]) < 1e-10 else 0.0

def export_data(sol, mdg, sd):

    output_directory = "output_directory"
    # Delete the output directory, if it exisis
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory)

    n = 0 
    save = pp.Exporter(mdg, "adv-diff", folder_name=output_directory)
    proj_u = P1.eval_at_cell_centers(sd)
    
    for u in sol:
        for _, data in mdg.subdomains(return_data=True):
            # post process variables
            cell_u = (proj_u @ u)

            pp.set_solution_values("cell_mass", cell_u, data, time_step_index = n)

            save.write_vtu(["cell_mass"], time_step=n)

            n += 1

    save.write_pvd()

mdg = create_grid(grid_size, dim)

for sd, data in mdg.subdomains(return_data=True):
    perm_w = pp.SecondOrderTensor(np.ones(sd.num_cells))
    perm_n = pp.SecondOrderTensor(np.ones(sd.num_cells))

    param = {"second_order_tensor": perm_w, "second_order_tensor": perm_n}
    pp.initialize_data(sd, data, key, param)

    # with the following steps we identify the portions of the boundary
    # to impose the boundary conditions

    left = sd.face_centers[0, :] == 0
    right = sd.face_centers[0, :] == 1
    top = sd.face_centers[1, :] == 1
    bottom = sd.face_centers[1, :] == 1
    
    ess_p_dofs = np.zeros(P0.ndof(sd), dtype=bool)

    nat_bc_faces = np.logical_or(left, right)
    ess_bc_faces = np.logical_or(top, bottom)
    
    nat_bc.append(RT0.assemble_nat_bc(sd, u_bc, nat_bc_faces))
    ess_bc.append(ess_u_dofs)


    P1.interpolate(sd, source_term1)
    mass = P1.assemble_mass_matrix(sd)
    source.append(mass @ P1.interpolate(sd, source_term))


# construct the local matrices
mass = P1.assemble_mass_matrix(sd)
adv = P1.assemble_adv_matrix(sd, data)
stiff = P1.assemble_stiff_matrix(sd, data)

# assemble the global matrix
# fmt: off
global_matrix = mass + dt*(adv + stiff)
# fmt: on

# get the degrees of freedom for u
dof_u = sd.num_nodes

# assemble the time-independent right-hand side
rhs_const = np.zeros(dof_u)
rhs_const[:dof_u] += np.hstack(nat_bc) + np.hstack(source)

# set initial conditions
u = np.zeros(dof_u)

sol.append(u)

for n in np.arange(num_steps):

    rhs = rhs_const.copy()
    rhs[:dof_u] += np.hstack(mass @ u)

    # solve the problem
    ls = pg.LinearSystem(global_matrix, rhs)

    # flag the essential boundary conditions
    ls.flag_ess_bc(np.hstack(ess_bc), np.zeros(dof_u))
    x = ls.solve()

    # extract the variables
    u = x[:dof_u]

    sol.append(u)

export_data(sol, mdg, sd)
