import math
import os
import shutil
from typing import Callable
import numpy as np
import scipy.sparse as sps

import porepy as pp
import pygeon as pg

import inspect


# Paramenter
diff_scaling = 0.009
adv_scaling = 20
source_scaling = 50000
adv_falloff_scaling = 0.12
inflow_rate = 10.0


grid_size = [200, 200]
dim = [1, 1]
dt = 0.5
num_steps = 200

key = "mass"

nat_bc = []
ess_bc = []
source = []

P1 = pg.Lagrange1(key)


def create_grid(grid_size, dim):
    sd = pp.StructuredTriangleGrid(grid_size, dim)
    # convert the grid into a mixed-dimensional grid
    mdg = pg.as_mdg(sd)
    # Convert to a pygeon grid
    pg.convert_from_pp(sd)
    sd.compute_geometry()
    return mdg


def vector_field(x: np.ndarray) -> np.ndarray:
    # Center of the domain (circular motion center)
    center = np.array([0.5, 0.5, 0.0])

    # Displacement from center
    dx = x - center

    # In 2D case, we assume motion in XY plane with zero Z component
    # Circular motion: (-dy, dx) in XY plane, 0 in Z
    vx = -dx[1]
    vy = dx[0]
    vz = 0.0

    # Gaussian falloff
    r2 = dx[0] ** 2 + dx[1] ** 2
    R = adv_falloff_scaling  # Radius of influence
    falloff = math.exp(-r2 / (2 * R**2))

    return adv_scaling * falloff * np.array([vx, vy, vz])


def source_term1(x):
    # Example: a Gaussian source
    center = sd.cell_centers[:, sd.num_cells // 2]
    sigma = 0.05
    r2 = np.sum((x - center) ** 2)
    return np.exp(-r2 / (2 * sigma**2))


def source_term(x):
    center_cell = sd.num_cells // 2 - (2 * grid_size[0] // 3)

    bd_nodes = sd.get_all_boundary_nodes()

    node_map = sd.cell_nodes()

    center_nodes = node_map[:, center_cell].nonzero()[0]

    mask = ~np.isin(center_nodes, bd_nodes)
    center_nodes = center_nodes[mask]

    source_nodes_coord = sd.nodes.T[center_nodes]

    return source_scaling if np.any(np.all(source_nodes_coord == x, axis=1)) else 0.0


def u_bc(x):
    return inflow_rate if abs(x[0]) < 1e-10 else 0.0


def export_data(sol, mdg, sd):
    output_directory = os.path.join(os.path.dirname(__file__), "adv-diff sol")
    # Delete the output directory, if it exisis
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory)

    save = pp.Exporter(mdg, "adv-diff", folder_name=output_directory)

    proj_u = P1.eval_at_cell_centers(sd)

    for n, u in enumerate(sol):
        for sd, data in mdg.subdomains(return_data=True):
            # post process variables
            cell_u = proj_u @ u

            pp.set_solution_values("mass", cell_u, data, time_step_index=0)
            save.write_vtu(["mass"], time_step=n)

    save.write_pvd(range(len(sol)))


mdg = create_grid(grid_size, dim)

for sd, data in mdg.subdomains(return_data=True):
    diff = pp.SecondOrderTensor(np.full(sd.num_cells, diff_scaling))
    vel_field = np.array([vector_field(x) for x in sd.cell_centers.T])
    vel_field[: grid_size[0] * 4, :] = np.array((4, 0, 0))
    vel_field[(sd.num_cells - 4 * grid_size[0]) :, :] = np.array((4, 0, 0))
    vel_field = vel_field.T
    # vel_field = P1.interpolate(sd, vector_field_func).T

    param = {"vector_field": vel_field, "second_order_tensor": diff}
    pp.initialize_data(sd, data, key, param)

    # with the following steps we identify the portions of the boundary
    # to impose the boundary conditions

    left = sd.face_centers[0, :] == 0
    right = sd.face_centers[0, :] == 1
    bottom = sd.nodes[1, :] == 0
    top = sd.nodes[1, :] == 1

    nat_bc_faces = np.logical_or(left, right)
    ess_bc_nodes = np.logical_or(bottom, top)

    nat_bc.append(dt * P1.assemble_nat_bc(sd, u_bc, nat_bc_faces))
    ess_bc.append(ess_bc_nodes)

    source.append(P1.source_term(sd, source_term))

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

sol = np.empty((num_steps + 1, dof_u), dtype=np.float32)

# set initial conditions
u = np.zeros(dof_u)
sol[0] = u

# TODO
# P1.error_l2()


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

    sol[n + 1] = u

export_data(sol, mdg, sd)
