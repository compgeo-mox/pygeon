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
Solve advection-diffusion equation using Backward Euler in time,
linearized by the L-scheme, and discretized by Mixed Finite Element Method (MFEM).

PDEs:
    ∂u/∂t + ∇·q = S
    D^{-1} q - D^{-1}a u + ∇u = 0

Time discretization (Backward Euler):
    (u^{n+1} - u^n) / Δt + ∇·q^{n+1} = S
    D^{-1}q^{n+1} - D^{-1} a u^{n+1} + ∇u^{n+1} = 0

Variational (weak) form: find u in V such that
for all test functions φ_1, φ_2:

    ∫_Ω u^{n+1} φ_1 dΩ
    + Δt ∫_Ω ∇·q^{n+1} φ_1 dΩ
    = Δt ∫_Ω S^{n+1} φ_1 dΩ
    + ∫_Ω u^{n} φ_1 dΩ

    D^{-1}∫_Ω q^{n+1} φ_2 dΩ
    - D^{-1}∫_Ω a u^{n+1} · φ_2 dΩ    
    - ∫_Ω u^n+1 ∇·φ_2 dΩ
    = - ∫_Γ u^n+1 (φ_2 · ν) dΓ

This gives the matrix equation:
    (M_u                       Δt B) (u^{n+1}) = (ΔtS^{n+1} + M_u u^{n})
    ((-B^T - D^{-1} A)   D^{-1} M_q) (q^{n+1}) = - BC
"""

# Paramenter
D = 1  # Diffusion coefficient
V = np.array([1.0, 0.0, 0.0])  # Velocity vector
inflow_rate = 10.0  # Inflow rate for the boundary condition

grid_sizes = [[10, 10], [20, 20], [40, 40], [80, 80]]  # [[100, 100]]
dim = [1, 1]
num_step_list = 2 ** np.arange(1, 5)  # [2**6]
end_time = 1
# Relative and absolute tolerances for the non-linear solver
abs_tol = 1e-10
rel_tol = 1e-10

key = "mass"

l2_errors = np.zeros((len(grid_sizes)))

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
    q_x = -D * ux + Vx * u
    q_y = -D * uy + Vy * u

    q = sp.Matrix([q_x, q_y, 0])

    # Compute the divergence of q
    q_div = sp.diff(q_x, x) + sp.diff(q_y, y)

    # Source term f from PDE residual
    f = ut + q_div

    # Set the velocity field values
    f = f.subs({Vx: 1.0, Vy: 0.0, D: 1.0})
    q = q.subs({Vx: 1.0, Vy: 0.0, D: 1.0})

    # Simplify the source term
    f_simplified = sp.simplify(f)

    # Create lambdified functions for source term and solution
    source = sp.lambdify((x, y, t), f_simplified, "numpy")
    solution_u = sp.lambdify((x, y, t), u, "numpy")
    solution_q = sp.lambdify((x, y, t), q, "numpy")

    return source, solution_u, solution_q


for j, (num_steps, grid_size) in enumerate(zip(num_step_list, grid_sizes)):
    dt = end_time / num_steps

    mdg, sd, data, nat_bc_flags, ess_bc_flags = solver.create_grid(grid_size, dim, D, V)

    # Inverse the diffusion tensor D
    D_inv = solver.inverse_second_order_tensor(
        data.get(pp.PARAMETERS, {}).get(key, {}).get("second_order_tensor")
    )

    # Initialize the second order tensor on the grid
    pp.initialize_data(
        sd,
        data,
        key,
        {"second_order_tensor": D_inv},
    )

    # Construct the constant local matrices
    mass_u = P0.assemble_mass_matrix(sd)
    mass_q = RT0.assemble_mass_matrix(sd, data)
    div = dt * pg.cell_mass(mdg) @ pg.div(mdg)
    A = RT0.assemble_adv_matrix(sd, data)

    # assemble the saddle point problem
    # fmt: off
    spp = sps.block_array([[mass_u,           div],
                            [-div.T - A.T, mass_q]], format="csc")
    # fmt: on

    # get the source term and the manufactured solutions
    source_func, solution_u, solution_q = manufactured_solution()

    # get the degrees of freedom for u
    dof_u, dof_q = div.shape

    # assemble the time-independent right-hand side
    rhs_const = np.zeros(dof_u + dof_q)

    # initialize the solution arrays
    sol_u = np.empty((num_steps + 1, dof_u), dtype=np.float64)
    sol_q = np.empty((num_steps + 1, dof_q), dtype=np.float64)
    sol_an_u = np.empty((num_steps + 1, dof_u), dtype=np.float64)
    sol_an_q = np.empty((num_steps + 1, dof_q), dtype=np.float64)

    # set and store initial conditions
    u = P0.interpolate(sd, lambda X: solution_u(X[0], X[1], 0.0))
    q = RT0.interpolate(
        sd, lambda X: np.array(solution_q(X[0], X[1], 0.0), dtype=float)
    )

    sol_u[0] = u
    sol_q[0] = q
    sol_an_u[0] = u
    sol_an_q[0] = q

    for n in range(1, num_steps + 1):
        # set essential boundary values
        ess_bc_vals = RT0.interpolate(
            sd, lambda X: np.array(solution_q(X[0], X[1], n * dt), dtype=float)
        )

        # set natural boundary values
        nat_bc_vals = RT0.assemble_nat_bc(
            sd, lambda X: solution_u(X[0], X[1], n * dt), nat_bc_flags
        )

        # get time-dependent source term
        source = P0.source_term(sd, lambda X: source_func(X[0], X[1], n * dt))

        # assemble the time-dependent right-hand side
        rhs = rhs_const.copy()
        rhs[:dof_u] += dt * source + mass_u @ u
        rhs[-dof_q:] -= nat_bc_vals

        # set up the linear system
        ls = pg.LinearSystem(spp, rhs)

        # flag the essential boundary conditions
        full_ess_bc_flags = np.concatenate([np.zeros(dof_u, dtype=bool), ess_bc_flags])
        full_ess_bc_vals = np.concatenate([np.zeros(dof_u), ess_bc_vals])

        ls.flag_ess_bc(full_ess_bc_flags, full_ess_bc_vals)

        # solve the problem
        x = ls.solve()

        # extract the variables
        u = x[:dof_u]
        q = x[-dof_q:]

        # calculate the manufactured solution
        u_an = P0.interpolate(sd, lambda X: solution_u(X[0], X[1], n * dt))
        q_an = RT0.interpolate(
            sd, lambda X: np.array(solution_q(X[0], X[1], n * dt), dtype=float)
        )

        # store the solutions
        sol_u[n] = u
        sol_q[n] = q
        sol_an_u[n] = u_an
        sol_an_q[n] = q_an

        # calculate and store the L2 error
        l2_errors[j] += (
            P0.error_l2(sd, u, lambda X: solution_u(X[0], X[1], n * dt), relative=False)
            ** 2
            * dt
        )

    # final L2 error
    l2_errors[j] = np.sqrt(l2_errors[j])

# Export the numerical and analytical solutions form final simulation
solver.export_mixed_data(mdg, sd, "num_sol", sol_an_u, sol_an_q, sol_u, sol_q)

solver.plot_spatial_convergence(grid_sizes, l2_errors)
print("L2 errors:", l2_errors)
solver.calculate_convergence_order(l2_errors)
# solver.plot_temporal_convergence(end_time / num_step_list, l2_errors)
