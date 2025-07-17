import os
import shutil
import numpy as np
import porepy as pp
import pygeon as pg
import sympy as sp
import matplotlib.pyplot as plt
import scipy.sparse as sps
from adv_diff_src import Advection_Diffusion


"""
Solve nonlinear advection-diffusion equation using Backward Euler in time,
linearized by the L-scheme, and discretized by Finite Element Method (FEM).

PDE:
    ∂u/∂t + a · ∇u - ∇·( D(u) ∇u ) = S

Time discretization (Backward Euler):
    (u^{n+1} - u^n) / Δt + a · ∇u^{n+1} - ∇·( D(u^{n+1}) ∇u^{n+1} ) = S^{n+1}

Newton linearization at iteration i: 
    D(u^{i}) ∇u^{i} ≈ D(u^{i-1}) ∇u^{i}
    + D'(u^{i-1}) ∇u^{i-1} u^{i}
    - D'(u^{i-1}) ∇u^{i-1} u^{i-1}

Variational (weak) form at iteration i: find u^{(i)} in V such that
for all test functions φ:

    ∫_Ω u^{n,i} φ dΩ
    + Δt ∫_Ω (a · ∇u^{n,i}) φ dΩ
    + Δt ∫_Ω D(u^{i-1}) ∇u^{i} · ∇φ dΩ
    + Δt ∫_Ω D'(u^{i-1}) ∇u^{i-1} u^{i} · ∇φ dΩ
    = Δt ∫_Ω S^{n,i} φ dΩ
    + Δt ∫_Γ D(u^{n,i}) (∇u^{n,i} · ν) φ dΓ
    + ∫_Ω u^{n-1,i} φ dΩ
    + Δt ∫_Ω D'(u^{i-1}) ∇u^{i-1} u^{i-1} · ∇φ dΩ

This gives the matrix equation:
    (M + Δt A + Δt D^{i-1} + Δt D'^{i-1})(u^{n}) =
        Δt F + Δt BC + M u^{n-1} + Δt D^{i-1}' u^{i-1}

"""
# Paramenter
D = 1  # Diffusion coefficient
V = np.array([1.0, 0.0, 0.0])  # Velocity vector
L = 1e-3  # L-scheme parameter, can be adjusted
inflow_rate = 10.0  # Inflow rate for the boundary condition

grid_sizes = [[3, 3]]  # [[5, 5], [10, 10], [20, 20], [40, 40]]  # [[100, 100]]
dim = [1, 1]
num_step_list = [2**6]  #  2 ** np.arange(1, 6)
end_time = 1
iter = 50  # Number of iterations for the L-scheme linearization
# Relative and absolute tolerances for the non-linear solver
abs_tol = 1e-7
rel_tol = 1e-7

key = "mass"

l2_errors = np.zeros((len(grid_sizes)))  # np.zeros((len(num_step_list)))  #

P1 = pg.Lagrange1(key)

solver = Advection_Diffusion()


def manufactured_solution():
    """Manufactured solution for the advection-diffusion equation."""

    # Define variables and parameters
    x, y, t = sp.symbols("x y t")
    Vx, Vy = sp.symbols("Vx Vy")

    # Define manufactured solution u
    u = t * x * y * (1 - x) * (1 - y)

    D = 1 + u**2  # Nonlinear diffusion term D(u)

    # Compute partial derivatives
    ut = sp.diff(u, t)
    ux = sp.diff(u, x)
    uxx = sp.diff(ux, x)
    uy = sp.diff(u, y)
    uyy = sp.diff(uy, y)
    Dx = sp.diff(D, x)
    Dy = sp.diff(D, y)

    # Advection term: divergence of V * u
    adv = Vx * ux + Vy * uy

    # Diffusion term
    diff = Dx * ux + D * uxx + Dy * uy + D * uyy

    # Source term f from PDE residual
    f = ut + adv - diff

    # Set the velocity field values
    f = f.subs({Vx: 1.0, Vy: 0.0})

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

        # assemble the global constant matrix
        # fmt: off
        global_matrix_const = mass + dt * adv
        # fmt: on

        # get the source term and the manufactured solution
        source, solution = manufactured_solution()

        # set natural boundary values
        nat_bc_vals = P1.assemble_nat_bc(sd, solver.nat_bc_func, nat_bc_flags)

        # set essential boundary values
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

        # set projection operator for u
        proj_u = P1.eval_at_cell_centers(sd)

        for n in range(1, num_steps + 1):
            print(f"Grid size: {grid_size}")
            print(f"Time step {n} of {num_steps}, dt = {dt}")
            # set time-dependent source term
            source_vals = P1.source_term(sd, lambda X: source(X[0], X[1], n * dt))

            # assemble the time-dependent right-hand side
            rhs_fixed = rhs_const.copy()
            rhs_fixed += dt * source_vals + mass @ u

            u_prev = u.copy()

            for i in range(iter):
                # calculate the non-linear diffusion term for cell center and nodes
                u_cell = proj_u @ u_prev

                diff_cell = solver.diff_func(sd, u_cell)
                diff_prime_cell = solver.diff_func_prime(sd, u_cell) * u_cell
                diff_prime_vel = solver.diff_func_prime(sd, u_cell) * (
                    P1.assemble_diff_matrix(sd, data) @ u_prev
                )

                # update the diffusion tensor in the data
                pp.initialize_data(
                    sd,
                    data,
                    key,
                    {"second_order_tensor": pp.SecondOrderTensor(diff_prime_cell)},
                )

                stiff_prime = P1.assemble_stiff_matrix(sd, data)

                # set iterative rhs
                rhs = rhs_fixed.copy()
                rhs += dt * stiff_prime @ u_prev  # + dt * nat_bc_vals @ diff_node

                # update the diffusion tensor in the data
                pp.initialize_data(
                    sd,
                    data,
                    key,
                    {"vector_field": diff_prime_vel},
                )

                # assemble the u dependent local matrices for the current iteration
                stiff_prime_1 = P1.assemble_adv_matrix(sd, data)

                # update the diffusion tensor in the data
                pp.initialize_data(
                    sd,
                    data,
                    key,
                    {"second_order_tensor": pp.SecondOrderTensor(diff_cell)},
                )

                # assemble the u dependent local matrices for the current iteration
                stiff = P1.assemble_stiff_matrix(sd, data)

                # assemble the global matrix for the current iteration
                # fmt: off
                global_matrix = global_matrix_const.copy()
                global_matrix += dt * stiff_prime_1 + dt * stiff
                # fmt: on

                # set up the linear system
                ls = pg.LinearSystem(global_matrix, rhs)

                # flag the essential boundary conditions
                ls.flag_ess_bc(ess_bc_flags, ess_bc_vals)

                # solve the problem
                u = ls.solve()

                # Check if we have reached convergence
                rel_err = np.sqrt(np.sum(np.power(u - u_prev, 2)))
                abs_err = np.sqrt(np.sum(np.power(u_prev, 2)))

                # Log message with error and current iteration
                print(
                    "Iteration #"
                    + str(i + 1)
                    + ", error L2 relative u: "
                    + str(rel_err)
                )

                if rel_err > abs_tol + rel_tol * abs_err:
                    u_prev = u.copy()
                else:
                    break

            # calculate the analytical solution for the current time step
            u_an = P1.interpolate(sd, lambda X: solution(X[0], X[1], n * dt))

            # store the solution
            sol[n] = u_prev
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

solver.plot_spatial_convergence(grid_sizes, l2_errors)
print("L2 errors:", l2_errors)
solver.calculate_convergence_order(l2_errors)
# solver.plot_temporal_convergence(end_time / num_step_list, l2_errors)
