import numpy as np
import porepy as pp
import pygeon as pg
import sympy as sp
from adv_diff_src import Advection_Diffusion


"""
Solve nonlinear advection-diffusion equation using Backward Euler in time,
linearized by the L-scheme, and discretized by Finite Element Method (FEM).

PDE:
    ∂_t B(u) + a · ∇u - ∇·( D(u) ∇u ) = S

Time discretization (Backward Euler):
    (B(u)^{n} - B(u)^{n-1}) / Δt + a · ∇u^{n} - ∇·( D(u^{n}) ∇u^{n} ) = S^{n}

L-scheme linearization at iteration i:
    B(u^{i}) - Δt ∇·( D(u^{i}) ∇u^{i} ) ≈ B(u^{i-1}) - Δt ∇·( D(u^{i-1}) ∇u^{i} )
    + L (u^{i} - u^{i-1})

Variational (weak) form at iteration i: find u^{(i)} in V such that
for all test functions φ:

    + Δt ∫_Ω (a · ∇u^{n,i}) φ dΩ
    + Δt ∫_Ω D(u^{n,i-1}) ∇u^{n,i} · ∇φ dΩ
    + L ∫_Ω u^{n,i} φ dΩ
    = Δt ∫_Ω S^{n,i} φ dΩ
    + Δt ∫_Γ D(u^{n,i-1}) (∇u^{n,i} · ν) φ dΓ
    + ∫_Ω B(u^{n-1}) φ dΩ
    - ∫_Ω B(u^{n,i-1)} φ dΩ
    + L ∫_Ω u^{n,i-1} φ dΩ

This gives the matrix equation:
    (Δt A + Δt D^{i-1} + L M)(u^{n+1}) = Δt F + Δt BC + M B(u^{n-1}) 
    - M B(u^{i-1}) + L M u^{i-1}

"""

# Paramenter
D = 1  # Diffusion coefficient
V = np.array([1.0, 0.0, 0.0])  # Velocity vector
L = 0.1  # L-scheme parameter, can be adjusted

grid_sizes = [[5, 5], [10, 10], [20, 20]]  # , [40, 40]]
dim = [1, 1]
num_step_list = 25 * (4 ** np.arange(0, 3))  # 25 * (4 ** np.arange(0, 3))
end_time = 1

iter = 1000  # Number of iterations for the L-scheme linearization
# Relative and absolute tolerances for the non-linear solver
abs_tol = 1e-10
rel_tol = 1e-10

key = "mass"

l2_errors = np.zeros((len(grid_sizes)))

P1 = pg.Lagrange1(key)

solver = Advection_Diffusion()


def manufactured_solution():
    """Manufactured solution for the advection-diffusion equation."""

    # Define variables and parameters
    x, y, t = sp.symbols("x y t")
    Vx, Vy = sp.symbols("Vx Vy")

    # Define manufactured solution u
    u = t * x * y * (1 - x) * (1 - y)

    B = 0.1 + 0.9 / (1 + u**2) ** (0.5)  # Nonlinear mass term B(u)
    D = 1 + u**2  # Nonlinear diffusion term D(u)

    # Compute partial derivatives
    Bt = sp.diff(B, t)
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
    f = Bt + adv - diff

    # Set the velocity field values
    f = f.subs({Vx: 1.0, Vy: 0.0})

    # Simplify the source term
    f_simplified = sp.simplify(f)

    # Create lambdified functions for source term and solution
    source = sp.lambdify((x, y, t), f_simplified, "numpy")
    solution = sp.lambdify((x, y, t), u, "numpy")

    return source, solution


for j, (num_steps, grid_size) in enumerate(zip(num_step_list, grid_sizes)):
    iter_count = 0

    dt = end_time / num_steps

    mdg, sd, data, nat_bc_flags, ess_bc_flags = solver.create_grid(grid_size, dim, D, V)

    # construct the constant local matrices
    mass = P1.assemble_mass_matrix(sd)
    adv = P1.assemble_adv_matrix(sd, data)
    mass_lumped = P1.assemble_lumped_matrix(sd)

    # assemble the global constant matrix
    # fmt: off
    global_matrix_const = dt * adv + L * mass 
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
        count = 0
        iter_error = []
        print(f"Grid size: {grid_size}")
        print(f"Time step {n} of {num_steps}, dt = {dt}")
        # set time-dependent source term
        source_vals = P1.source_term(sd, lambda X: source(X[0], X[1], n * dt))

        B_fixed = solver.mass_func(u)

        # assemble the time-dependent right-hand side
        rhs_fixed = rhs_const.copy()
        rhs_fixed += dt * source_vals + mass @ B_fixed

        u_prev = u.copy()

        # L-scheme iterations
        for i in range(iter):
            count += 1
            iter_count = 0

            B_prev = solver.mass_func(u_prev)

            # B_prev = solver.mass_func(proj_u @ u_prev)
            # set iterative rhs
            rhs = rhs_fixed.copy()
            rhs += L * mass @ u_prev - mass @ B_prev
            # + dt * nat_bc_vals @ diff_node

            # calculate the non-linear diffusion term for cell center and nodes
            diff_cell = proj_u @ solver.diff_func(sd, u_prev)

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
            global_matrix += dt * stiff
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
            iter_error.append(rel_err)

            # Log message with error and current iteration
            # print("Iteration #" + str(i + 1) + ", error L2 relative u: " + str(rel_err))

            if rel_err > 100:
                break

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
            P1.error_l2(sd, u, lambda X: solution(X[0], X[1], n * dt), relative=False)
            ** 2
            * dt
        )

        if len(iter_error) >= 3:
            sum = 0
            for i in range(2, len(iter_error)):
                L_convergence = np.log(iter_error[i] / iter_error[i - 1]) / np.log(
                    iter_error[i - 1] / iter_error[i - 2]
                )
                # print(
                #    (
                #        f"L-scheme convergence order at step {n}, "
                #        f"iteration {i}: {L_convergence}"
                #    )
                # )
                sum += L_convergence
            print(f"Avg L-scheme convergence order: {sum / (len(iter_error) - 2)}")
            print(f"# iteration: {count}")

    # final L2 error
    l2_errors[j] = np.sqrt(l2_errors[j])

    # export_data("num_sol", sol, mdg, sd)
    # export_data("ana_sol", sol_an, mdg, sd)

solver.plot_spatial_convergence(grid_sizes, l2_errors)
print("L2 errors:", l2_errors)
solver.calculate_convergence_order(l2_errors)
# solver.plot_temporal_convergence(end_time / num_step_list, l2_errors)
