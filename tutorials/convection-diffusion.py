import numpy as np
import scipy.sparse as sps

import porepy as pp
import pygeon as pg

def main():
    """
    This function sets up and solves a convection-diffusion problem using the 
    PyGeon library.
    It defines the geometry, boundary conditions, and discretization method, 
    and then solves the problem using the specified solver.
    """

    # Number of cells in each direction
    n = 10
    # Parameters for the convection-diffusion problem
    parameters = {
        "diffusion": 1.0,
        "convection": 1.0,
        "source": 1.0,
        "time_step": 0.1,
        "n_steps": 10,
    }

    # Boundary conditions
    bc_topp = 0
    bc_bottom = 0
    bc_left = 0
    bc_right = 0
    bc = [bc_topp, bc_bottom, bc_left, bc_right]

    # Initial condition
    ic = 0.0

    # Create a grid for the convection-diffusion problem
    grid = grid(n)

    params = parameters(grid, parameters)

    # Assign values to the grid
    assign_grid_valus(bc, ic, params, grid)

    # Assemble the matrix system
    spp, rhs = assemble_matrix_system()

    # Solve the problem
    q, p = solve(spp, rhs)
    
    # Post-process the results
    export_results(grid, q, p)

def grid(n):
    """
    This function creates a grid for the convection-diffusion problem.
    """

    sd = pp.StructuredTriangleGrid([n] * 2, [1] * 2)
    # convert the grid into a mixed-dimensional grid
    mdg = pg.as_mdg(sd)

    # Convert to a pygeon grid
    pg.convert_from_pp(sd)
    sd.compute_geometry()
    
    return mdg

def parameters(sd, parameters):
    # diffusion tensor
    diff = pp.SecondOrderTensor(np.ones(sd.num_cells))
    # convection tensor
    conv = pp.SecondOrderTensor(np.ones(sd.num_cells))
    params = {
        "second_order_tensor": diff, 
        "convection_tensor": conv,
    }
    return params 

def assign_grid_valus(bc, ic, params, grid):
    """
    This function assigns values to the grid for the convection-diffusion problem.
    """
    key = "mass"
    bc_val = []
    bc_ess = []
    vector_source = []

    P1 = pg.Lagrange1(key)

    for sd, data in grid.subdomains(return_data=True):
        pp.initialize_data(sd, data, key, params)

        # with the following steps we identify the portions of the boundary
        # to impose the boundary conditions
        left_right = np.logical_or(sd.face_centers[0, :] == 0, sd.face_centers[0, :] == 1)
        top_bottom = np.logical_or(sd.face_centers[1, :] == 0, sd.face_centers[1, :] == 1)
        ess_u_dofs = np.zeros(P1.ndof(sd), dtype=bool)

        def p_bc(x):
            return x[1]

        bc_val.append(-P1.assemble_nat_bc(sd, p_bc, top_bottom))
        bc_ess.append(np.hstack((left_right, ess_u_dofs)))

        def vector_source_fct(x):
            return np.array([0, -1, 0])

        mass = P1.assemble_mass_matrix(sd)
        vector_source.append(mass @ P1.interpolate(sd, vector_source_fct))

def assemble_matrix_system():
    # construct the local matrices
    mass = pg.face_mass(mdg)
    div = pg.cell_mass(mdg) @ pg.div(mdg)

    # assemble the saddle point problem
    # fmt: off
    spp = sps.block_array([[mass, -div.T],
                        [div,    None]], format="csc")
    # fmt: on

    # get the degrees of freedom for each variable
    dof_p, dof_q = div.shape

    # assemble the right-hand side
    rhs = np.zeros(dof_p + dof_q)
    rhs[:dof_q] += np.hstack(bc_val) + np.hstack(vector_source)

    return spp, rhs

def solve(spp, rhs):
    # solve the problem
    ls = pg.LinearSystem(spp, rhs)
    ls.flag_ess_bc(np.hstack(bc_ess), np.zeros(dof_q + dof_p))
    x = ls.solve()

    # extract the variables
    q = x[:dof_q]
    p = x[-dof_p:]

    return q, p

def export_results(grid, q, p):
    """
    This function exports the results of the convection-diffusion problem.
    """
    # post process variables
    proj_q = RT0.eval_at_cell_centers(sd)
    cell_q = (proj_q @ q).reshape((3, -1))
    cell_p = P0.eval_at_cell_centers(sd) @ p

    for _, data in mdg.subdomains(return_data=True):
        pp.set_solution_values("cell_q", cell_q, data, 0)
        pp.set_solution_values("cell_p", cell_p, data, 0)

    save = pp.Exporter(mdg, "darcy")
    save.write_vtu(["cell_q", "cell_p"])


if __name__ == "__main__":
    main()