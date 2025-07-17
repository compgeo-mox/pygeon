import math
import os
import shutil
import numpy as np
import porepy as pp
import pygeon as pg


# Paramenter
diff_scaling = 0.009
adv_scaling = 20
source_scaling = 50000
adv_falloff_scaling = 0.12
inflow_rate = 10.0
L = 1

grid_size = [20, 20]
dim = [1, 1]
dt = 0.5
num_steps = 50

key = "mass"

P1 = pg.Lagrange1(key)


def create_grid(grid_size, dim):
    """Create a structured grid with simplices for the problem."""
    sd = pp.StructuredTriangleGrid(grid_size, dim)

    # convert the grid into a mixed-dimensional grid
    mdg = pg.as_mdg(sd)

    # Convert to a pygeon grid
    pg.convert_from_pp(sd)
    sd.compute_geometry()

    return mdg


def vector_field_circular(x: np.ndarray) -> np.ndarray:
    """Vector field for advection, representing circular motion
    around the center of the domain.
    """

    # Center of motion
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


def source_term_gaussian(x):
    """Gaussian source term centered at domain center."""
    #  Center of the domain
    center = sd.cell_centers[:, sd.num_cells // 2]

    # Gaussian falloff
    sigma = 0.05

    # Calculate the squared distance from the center
    r2 = np.sum((x - center) ** 2)

    return np.exp(-r2 / (2 * sigma**2))


def source_term_point(x):
    """Point source term that is non-zero at the center cell."""

    center_cell = sd.num_cells // 2 - (2 * grid_size[0] // 3)

    bd_nodes = sd.get_all_boundary_nodes()

    node_map = sd.cell_nodes()

    center_nodes = node_map[:, center_cell].nonzero()[0]

    mask = ~np.isin(center_nodes, bd_nodes)
    center_nodes = center_nodes[mask]

    source_nodes_coord = sd.nodes.T[center_nodes]

    return source_scaling if np.any(np.all(source_nodes_coord == x, axis=1)) else 0.0


def nat_bc_func(x):
    """Natural boundary condition function for u, set to a constant value."""
    return inflow_rate if abs(x[0]) < 1e-10 else 0.0


def ess_bc_func(sd):
    """Essential boundary condition function for u, set to zero."""
    return np.zeros(sd.num_nodes)


def export_data(name, sol, mdg, sd):
    """Export the solution data to a pvd-file."""
    output_directory = os.path.join(os.path.dirname(__file__), "adv-diff " + name)
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
    # initialize the parameters on the grid
    diff = pp.SecondOrderTensor(np.full(sd.num_cells, diff_scaling))

    vel_field = np.array([vector_field_circular(x) for x in sd.cell_centers.T])
    # Set the velocity field to be constant in the first and last 4 rows
    # (drives the flow forward on top and bottom of the domain)
    vel_field[: grid_size[0] * 4, :] = np.array((4, 0, 0))
    vel_field[(sd.num_cells - 4 * grid_size[0]) :, :] = np.array((4, 0, 0))
    vel_field = vel_field.T

    # vel_field = P1.interpolate(sd, vector_field_func).T

    param = {"vector_field": vel_field, "second_order_tensor": diff}
    pp.initialize_data(sd, data, key, param)

    # with the following steps we identify the portions of the boundary
    # to impose the boundary conditions
    left_faces = sd.face_centers[0, :] == 0
    right_faces = sd.face_centers[0, :] == 1
    bottom_faces = sd.face_centers[1, :] == 0
    top_faces = sd.face_centers[1, :] == 1

    bottom_nodes = sd.nodes[1, :] == 0
    top_nodes = sd.nodes[1, :] == 1
    left_nodes = sd.nodes[0, :] == 0
    right_nodes = sd.nodes[0, :] == 1

    # set flags for the natural and essential boundary conditions
    nat_bc_flags = np.logical_or(left_faces, right_faces)
    ess_bc_flags = np.logical_or(bottom_nodes, top_nodes)

    # get essential boundary, natural boundary and source values
    ess_bc_vals = ess_bc_func(sd)
    nat_bc_vals = P1.assemble_nat_bc(sd, nat_bc_func, nat_bc_flags)
    source_vals = P1.source_term(sd, source_term_point)

# construct the constant local matrices
mass = P1.assemble_mass_matrix(sd, data)
adv = P1.assemble_adv_matrix(sd, data)
stiff = P1.assemble_stiff_matrix(sd, data)

# assemble the constant global matrix
# fmt: off
global_matrix = mass + dt*(adv + stiff)
# fmt: on

# get the degrees of freedom for u
dof_u = sd.num_nodes

# assemble the time-independent right-hand side
rhs_const = np.zeros(dof_u)
rhs_const += dt * nat_bc_vals @ diff + dt * source_vals

# initialize the solution array
sol = np.empty((num_steps + 1, dof_u), dtype=np.float32)

# set and store initial conditions
u = np.zeros(dof_u)
sol[0] = u

# time stepping
for n in np.arange(num_steps) + 1:
    # set time-dependent right-hand side
    rhs = rhs_const.copy()
    rhs += np.hstack(mass @ u)

    # set up the linear system
    ls = pg.LinearSystem(global_matrix, rhs)

    # flag the essential boundary conditions
    ls.flag_ess_bc(ess_bc_flags, ess_bc_vals)

    # solve the problem and store solution
    u = ls.solve()
    sol[n] = u

export_data("num_sol", sol, mdg, sd)
