import os
import shutil
import numpy as np
import porepy as pp
import pygeon as pg
import sympy as sp
import matplotlib.pyplot as plt


class Advection_Diffusion:
    """
    Solve nonlinear advection-diffusion equation using Backward Euler in time and
    discretized by Finite Element Method (FEM).
    """

    def __init__(self):
        self.key = "mass"
        self.P1 = pg.Lagrange1(self.key)
        self.P0 = pg.PwConstants(self.key)
        self.RT0 = pg.RT0(self.key)
        self.m = 4  # depth for anderson acceleration
        self.res_prev = 0.0  # Previous residual for Anderson acceleration
        self.x_prev = 0.0  # Previous iterate for Anderson acceleration
        self.aa_res_diff = None  # Anderson acceleration residuals difference
        self.aa_x_diff = None  # Anderson acceleration step difference

    def nat_bc_func(self, x):
        """Natural boundary condition function."""
        return 0.0

    def ess_bc_func(self, x):
        """Essential boundary condition function."""
        return 0.0

    def source_term(self, x):
        """Source term function."""
        return 0.0

    def init_sol_func(self, x):
        """Initial condition function."""
        return 0.0

    def diff_func(self, sd, u):
        """Compute the non-linear diffusion term D(u) on grid."""
        return 1 + u**2

    def diff_func_prime(self, sd, u):
        """Compute the non-linear diffusion term D(u) on grid."""
        return 2 * u

    def mass_func(self, u):
        """Compute the non-linear mass term B(u) on grid."""
        return 0.1 + 0.9 / (1 + u**2) ** (0.5)

    def vel_func(self, sd, u):
        """Compute the non-linear advection term A(u) on grid.
        Note: Not used"""
        return 1 + u**2

    def create_grid(self, grid_size, dim, D, V):
        """Create a structured triangle grid for the problem."""
        sd = pp.StructuredTriangleGrid(grid_size, dim)

        # convert the grid into a mixed-dimensional grid
        mdg = pg.as_mdg(sd)

        # Convert to a pygeon grid
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        for sd, data in mdg.subdomains(return_data=True):
            self.aa_res_diff = np.empty((self.P1.ndof(sd), 50))
            self.aa_x_diff = np.empty((self.P1.ndof(sd), 50))

            # initialize the parameters on the grid
            diff = pp.SecondOrderTensor(np.full(sd.num_cells, D))

            vel_field = np.broadcast_to(V[:, None], (3, sd.num_cells))

            param = {"vector_field": vel_field, "second_order_tensor": diff}
            pp.initialize_data(sd, data, self.key, param)

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

            # set flags for essential and natural boundary conditions
            nat_bc_flags = np.logical_or(
                np.logical_or(right_faces, left_faces),
                np.logical_or(bottom_faces, top_faces),
            )

            ess_bc_flags = np.logical_or(
                np.logical_or(right_nodes, left_nodes),
                np.logical_or(bottom_nodes, top_nodes),
            )

            ess_bc_flags_mixed = np.logical_or(
                np.logical_or(right_faces, left_faces),
                np.logical_or(bottom_faces, top_faces),
            )

            nat_bc_flags_mixed = np.logical_or(
                np.logical_or(right_faces, left_faces),
                np.logical_or(bottom_faces, top_faces),
            )

        # change the flags to mixed-dimensional flags for MFEM
        return mdg, sd, data, nat_bc_flags, ess_bc_flags

    def inverse_second_order_tensor(self, tensor):
        """Compute the inverse of a second-order tensor."""
        D = tensor.values
        dim, _, Nc = D.shape  # shape: (dim, dim, num_cells)
        # Reshape to (num_cells, dim, dim) to apply batch inversion
        D_reshaped = D.transpose(2, 0, 1)  # (Nc, dim, dim)

        # Invert each (dim x dim) matrix
        D_inv_reshaped = np.linalg.inv(D_reshaped)  # (Nc, dim, dim)

        # Transpose back to (dim, dim, Nc)
        D_vals = D_inv_reshaped.transpose(1, 2, 0)

        # Fill tensor and return wrt to the dimension
        kxx = D_vals[0, 0, :]
        if dim == 1:
            return pp.SecondOrderTensor(kxx)

        kyy = D_vals[1, 1, :]
        kxy = D_vals[0, 1, :]

        if dim == 2:
            return pp.SecondOrderTensor(kxx, kyy=kyy, kxy=kxy)

        kzz = D_vals[2, 2, :]
        kxz = D_vals[0, 2, :]
        kyz = D_vals[1, 2, :]

        if dim == 3:
            return pp.SecondOrderTensor(
                kxx, kyy=kyy, kzz=kzz, kxy=kxy, kxz=kxz, kyz=kyz
            )

    def grad(self, sd, u):
        """Compute the gradient of the u at cell centers."""
        GradP1 = self.P1.get_range_discr_class(sd.dim)(self.key)

        # Compute the gradient at cell_centers
        grad_u = self.P1.assemble_diff_matrix(sd) @ u
        grad_u_at_cc = GradP1.eval_at_cell_centers(sd) @ grad_u

        if (
            sd.dim == 2
        ):  # The 2D differential is a rotated gradient, so we rotate back...
            grad_u_at_cc = np.concatenate(
                (
                    -grad_u_at_cc[sd.num_cells : sd.num_cells * 2],
                    grad_u_at_cc[: sd.num_cells],
                )
            )

        # The gradient is one long vector, which we have to reshape
        grad_u_at_cc = np.reshape(grad_u_at_cc, (sd.dim, -1))

        # Now each column of grad_u_at_cc gives the gradient at the cell center
        # as a d-vector

        return grad_u_at_cc

    def anderson_acc(self, x, g, i):
        """Calculate the Anderson acceleration weights."""
        res = g - x

        if i == 0:
            self.x_prev = x.copy()
            self.res_prev = res.copy()
            return g

        self.aa_res_diff[:, i] = res - self.res_prev

        self.aa_x_diff[:, i] = x - self.x_prev

        self.x_prev = x.copy()
        self.res_prev = res.copy()

        if i <= self.m:
            return g

        # min_(y=y_0,...,y_m_k-1)^T ||f_k - F_k y||_2,
        # F_k = (delta f_k-m_k, ..., delta f_k-1)
        # delta f_i = f_i+1 - f_i
        # f_i = g(x_i) - x_i
        F = self.aa_res_diff[:, i - self.m : i]
        X = self.aa_x_diff[:, i - self.m : i]

        # Use QR decomposition to solve the least squares problem
        Q, R = np.linalg.qr(F)
        y = np.linalg.solve(R, Q.T @ res)
        # return the accelerated iterate
        # x_i+1 = x_k + f_k - (X_k + F_k)y^(k)
        # X_k = (delta x_k-m_k, ..., delta x_k-1)
        # delta x_i = x_i+1 - x_i

        x_acc = x + res - (X + F) @ y

        return x_acc

    def export_data(self, name, sol, mdg, sd):
        """Export the solution data to pvd-file."""
        output_directory = os.path.join(os.path.dirname(__file__), "adv-diff " + name)
        # Delete the output directory, if it exisis
        if os.path.exists(output_directory):
            shutil.rmtree(output_directory)

        save = pp.Exporter(mdg, "adv-diff", folder_name=output_directory)

        proj_u = self.P1.eval_at_cell_centers(sd)

        for n, u in enumerate(sol):
            for sd, data in mdg.subdomains(return_data=True):
                # post process variables
                cell_u = proj_u @ u

                pp.set_solution_values("mass", cell_u, data, time_step_index=0)
                save.write_vtu(["mass"], time_step=n)

        save.write_pvd(range(len(sol)))

    def export_mixed_data(self, mdg, sd, name, sol_an_u, sol_an_q, sol_u, sol_q):
        """Export the MFEM solution data to pvd-file."""
        output_directory = os.path.join(
            os.path.dirname(__file__), "adv-diff-mixed " + name
        )
        # Delete the output directory, if it exisis
        if os.path.exists(output_directory):
            shutil.rmtree(output_directory)

        save = pp.Exporter(mdg, "adv-diff-mixed", folder_name=output_directory)

        proj_u = self.P0.eval_at_cell_centers(sd)
        proj_q = self.RT0.eval_at_cell_centers(sd)

        for n, (u, u_an, q, q_an) in enumerate(zip(sol_u, sol_an_u, sol_q, sol_an_q)):
            for sd, data in mdg.subdomains(return_data=True):
                cell_q = (proj_q @ q).reshape((3, -1))
                cell_u = proj_u @ u

                cell_an_q = (proj_q @ q_an).reshape((3, -1))
                cell_an_u = proj_u @ u_an

                pp.set_solution_values("cell_u", cell_u, data, time_step_index=0)
                pp.set_solution_values("cell_q", cell_q, data, time_step_index=0)
                pp.set_solution_values("cell_an_u", cell_an_u, data, time_step_index=0)
                pp.set_solution_values("cell_an_q", cell_an_q, data, time_step_index=0)

                save.write_vtu(
                    ["cell_u", "cell_q", "cell_an_u", "cell_an_q"], time_step=n
                )

        save.write_pvd(range(len(sol_u)))

    def plot_spatial_convergence(self, grid_sizes, l2_errors):
        """Plot the spatial convergence of the L2 error."""
        h = 1 / np.array(grid_sizes)[:, 0]  # Element size

        plt.plot(h, l2_errors, marker="o", label="Numerical error")

        C = l2_errors[0] / h[0] ** 2
        ref_line = C * h**2

        plt.plot(h, ref_line, "--", label=r"$\mathcal{O}(h^2)$", color="gray")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Element size")
        plt.ylabel("L2 error")
        plt.title("Spatial convergence for advection-diffusion equation")
        plt.grid()
        plt.show()

    def plot_temporal_convergence(self, timesteps, l2_errors):
        """Plot the temporal convergence of the L2 error."""
        plt.plot(timesteps, l2_errors, marker="o")

        C = l2_errors[0] / timesteps[0]
        ref_line = C * timesteps

        plt.plot(timesteps, ref_line, "--", label=r"$\mathcal{O}(h^2)$", color="gray")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Time step size")
        plt.ylabel("L2 error")
        plt.title("Temporal convergence for advection-diffusion equation")
        plt.grid()
        plt.show()

    def calculate_convergence_order(self, l2_errors):
        """Calculate the convergence order based on L2 errors."""
        orders = []
        for i in range(1, len(l2_errors)):
            order = np.log(l2_errors[i - 1] / l2_errors[i]) / np.log(2)
            orders.append(order)
        print("Order of convergence:", orders)
