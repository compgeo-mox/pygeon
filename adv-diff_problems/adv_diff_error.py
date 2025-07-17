import os
import shutil
import numpy as np
import porepy as pp
import pygeon as pg
import sympy as sp
from adv_diff_src import Advection_Diffusion


# Paramenter
D = 1  # Diffusion coefficient
V = np.array([1.0, 0.0, 0.0])  # Velocity vector

grid_sizes = [[5, 5], [10, 10], [20, 20], [40, 40], [80, 80]]  # [[50, 50]]  #
dim = [1, 1]
num_step_list = [2**6]  # 2 ** np.arange(1, 6)  #
end_time = 1

key = "mass"

l2_errors = np.zeros((len(grid_sizes)))  # np.zeros((len(num_step_list)))

P1 = pg.Lagrange1(key)
solver = Advection_Diffusion()


def manufactured_solution():
    """Manufactured solution for the advection-diffusion equation."""

    # Define variables and parameters
    x, y, t = sp.symbols("x y t")
    Vx, Vy, D = sp.symbols("Vx Vy D")

    # Define manufactured solution u
    u = t * x * y * (1 - x) * (1 - y)

    # Compute partial derivatives
    ut = sp.diff(u, t)
    ux = sp.diff(u, x)
    uy = sp.diff(u, y)

    # Advection term: divergence of V * u
    adv = Vx * ux + Vy * uy

    # Diffusion term
    diff = sp.diff(D * ux, x) + sp.diff(D * uy, y)

    # Source term f from PDE residual
    f = ut + adv - diff

    # Set the velocity field and diffusion coefficient valus
    f = f.subs({Vx: 1.0, Vy: 0.0, D: 1.0})

    # Simplify the source term
    f_simplified = sp.simplify(f)

    # Create lambdified functions for source term and solution
    source = sp.lambdify((x, y, t), f_simplified, "numpy")
    solution = sp.lambdify((x, y, t), u, "numpy")

    return source, solution


for k, num_steps in enumerate(num_step_list):
    dt = end_time / num_steps
    for j, grid_size in enumerate(grid_sizes):
        mdg, sd, data, nat_bc_flags, ess_bc_flags = solver.create_grid(
            grid_size, dim, D, V
        )

        # construct the constant local matrices
        mass = P1.assemble_mass_matrix(sd, data)
        adv = P1.assemble_adv_matrix(sd, data)
        stiff = P1.assemble_stiff_matrix(sd, data)

        # assemble the constant global matrix
        # fmt: off
        global_matrix = mass + dt*(adv + stiff)
        # fmt: on

        # get the source term and the manufactured solution
        source, solution = manufactured_solution()

        # set the essential boundary values
        ess_bc_vals = P1.interpolate(sd, lambda X: solution(X[0], X[1], 0.0))

        # get the degrees of freedom for u
        dof_u = sd.num_nodes

        # assemble the time-independent right-hand side
        rhs_const = np.zeros(dof_u)

        # initialize the solution arrays
        sol = np.empty((num_steps + 1, dof_u), dtype=np.float64)
        sol_an = np.empty((num_steps + 1, dof_u), dtype=np.float64)

        # set and store initial conditions
        u = P1.interpolate(sd, lambda X: solution(X[0], X[1], 0.0))
        sol[0] = u
        sol_an[0] = u

        for n in range(1, num_steps + 1):
            print(f"Grid size: {grid_size}")
            print(f"Time step {n} of {num_steps}, dt = {dt}")
            # assemble the time-dependent right-hand side
            rhs = rhs_const.copy()
            rhs += (
                dt * P1.source_term(sd, lambda X: source(X[0], X[1], n * dt)) + mass @ u
            )

            # set up the linear system
            ls = pg.LinearSystem(global_matrix, rhs)

            # flag the essential boundary conditions
            ls.flag_ess_bc(ess_bc_flags, ess_bc_vals)

            # solve the problem
            u = ls.solve()

            # calculate the manufactured solution
            u_an = P1.interpolate(sd, lambda X: solution(X[0], X[1], n * dt))

            # store the solutions
            sol[n] = u
            sol_an[n] = u_an

            # calculate and store the L2 error
            l2_errors[j] += (
                P1.error_l2(
                    sd, u, lambda X: solution(X[0], X[1], n * dt), relative=False
                )
                ** 2
                * dt
            )

        # final L2 error
        l2_errors[j] = np.sqrt(l2_errors[j])

        # export_data("num_sol", sol, mdg, sd)
        # export_data("ana_sol", sol_an, mdg, sd)


print("L2 errors:", l2_errors)

solver.plot_spatial_convergence(grid_sizes, l2_errors)
# solver.plot_temporal_convergence(end_time / num_step_list, l2_errors)
solver.calculate_convergence_order(l2_errors)
