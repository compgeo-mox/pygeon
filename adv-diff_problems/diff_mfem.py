import os
import shutil
import numpy as np
import porepy as pp
import pygeon as pg
import sympy as sp
import matplotlib.pyplot as plt
import scipy.sparse as sps
from scipy.linalg import inv
from adv_diff_src import Advection_Diffusion

"""
Solve advection-diffusion equation using Backward Euler in time,
linearized by the L-scheme, and discretized by Mixed Finite Element Method (MFEM).

PDEs:
    ∂u/∂t - ∇·q = S
    q - D*∇u = 0

Time discretization (Backward Euler):
    (u^{n+1} - u^n) / Δt + ∇·q^{n+1} = S
    D^{-1}q^{n+1} - ∇u^{n+1} = 0

Variational (weak) form: find u in V such that
for all test functions φ_1, φ_2:

    ∫_Ω u^{n+1} φ_1 dΩ
    + Δt ∫_Ω ∇·q^{n+1} φ_1 dΩ
    = Δt ∫_Ω S^{n+1} φ_1 dΩ
    + ∫_Ω u^{n} φ_1 dΩ

    D^{-1}∫_Ω q^n+1 φ_2 dΩ  
    - ∫_Ω u^n+1 ∇·φ_2 dΩ
    = ∫_Γ u^n+1 (φ_2 · ν) dΓ

This gives the matrix equation:
    (M_u                    Δt B) (u^{n+1}) = (ΔtS + Mu^n)
    (- B^T            D^{-1} M_q) (q^{n+1}) = -BC
"""

# Paramenter
D = 1  # Diffusion coefficient
V = np.array([1.0, 0.0, 0.0])  # Velocity vector
L = 1e-1  # L-scheme parameter, can be adjusted
inflow_rate = 10.0  # Inflow rate for the boundary condition

grid_sizes = [[3, 3]]  # [[10, 10], [20, 20], [40, 40], [80, 80], [160, 160]]
dim = [1, 1]
timesteps = [0.001]  # [0.002, 0.001, 0.0005, 0.00025, 0.000125]
num_steps = 50
iter = 50  # Number of iterations for the L-scheme linearization
# Relative and absolute tolerances for the non-linear solver
abs_tol = 1e-10
rel_tol = 1e-10

key = "mass"

l2_errors = []

P0 = pg.PwConstants(key)
RT0 = pg.RT0(key)

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

    # Directional flux terms qx and qy
    q_x = Vx * u - D * ux
    q_y = Vy * u - D * uy

    # Compute the divergence of q
    q_div = sp.diff(q_x, x) + sp.diff(q_y, y)

    # Source term f from PDE residual
    f = ut - q_div

    # Set the velocity field values
    f = f.subs({Vx: 1.0, Vy: 0.0, D: 1.0})

    # Simplify the source term
    f_simplified = sp.simplify(f)

    # Create lambdified functions for source term and solution
    source = sp.lambdify((x, y, t), f_simplified, "numpy")
    solution = sp.lambdify((x, y, t), u, "numpy")

    return source, solution


for dt in timesteps:
    for grid_size in grid_sizes:
        mdg, sd, data, nat_bc_flags, ess_bc_flags = solver.create_grid(
            grid_size, dim, D, V
        )

        # construct the constant local matrices
        mass_u = P0.assemble_mass_matrix(sd, data)
        mass_q = RT0.assemble_mass_matrix(sd, data)
        div = dt * pg.cell_mass(mdg) @ pg.div(mdg)
        D_inv = solver.inverse_second_order_tensor(
            data.get(pp.PARAMETERS, {}).get(key, {}).get("second_order_tensor")
        ).values

        # assemble the saddle point problem
        # fmt: off
        spp = sps.block_array([[mass_u,            div],
                               [-div.T, D_inv * mass_q]], format="csc")
        # fmt: on

        # get the source term and the manufactured solution
        source, solution = manufactured_solution()

        # get the degrees of freedom for u
        dof_u, dof_q = div.shape

        # set natural boundary values
        nat_bc_vals = RT0.assemble_nat_bc(sd, solver.nat_bc_func, nat_bc_flags)

        # set essential boundary values
        ess_bc_vals = P0.interpolate(sd, lambda X: solution(X[0], X[1], 0.0))

        # assemble the time-independent right-hand side
        rhs_const = np.zeros(dof_u + dof_q)
        # rhs_const[dof_q:] += -D * nat_bc_vals

        # initialize the solution arrays
        sol = np.empty((num_steps + 1, dof_u), dtype=np.float64)
        sol_an = np.empty((num_steps + 1, dof_u), dtype=np.float64)

        # set and store initial conditions
        u = P0.interpolate(sd, lambda X: solution(X[0], X[1], 0.0))
        sol[0] = u
        sol_an[0] = u

        # set projection operator for u
        proj_u = P0.eval_at_cell_centers(sd)

        for n in range(1, num_steps + 1):
            rhs = rhs_const.copy()
            # assemble the time-dependent right-hand side
            rhs[:dof_q] = (
                dt * P0.source_term(sd, lambda X: source(X[0], X[1], n * dt))
                + mass_u @ u
            )

            # set up the linear system
            ls = pg.LinearSystem(spp, rhs)

            # flag the essential boundary conditions
            ls.flag_ess_bc(ess_bc_flags, ess_bc_vals)

            # solve the problem
            u = ls.solve()

            # calculate the manufactured solution
            u_an = P0.interpolate(sd, lambda X: solution(X[0], X[1], n * dt))

            # store the solutions
            sol[n] = u
            sol_an[n] = u_an

        # calculate and store the L2 error
        l2_errors.append(P0.error_l2(sd, u, lambda X: solution(X[0], X[1], n * dt)))

        # export_data("num_sol", sol, mdg, sd)
        # export_data("ana_sol", sol_an, mdg, sd)


# solver.plot_spatial_convergence(grid_sizes, l2_errors)
solver.plot_temporal_convergence(timesteps, l2_errors)
print("L2 errors:", l2_errors)
