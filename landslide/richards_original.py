import shutil
import os
import sys

import numpy as np
import scipy.sparse as sps
import sympy as sp

import porepy as pp
import pygeon as pg

from topograhy_file import *

# Set the maximum number of iterations of the non-linear solver
K = 50

# L-scheme parameter
L = 3.501e-2

# Set the mesh refinment
N = 4

# Set the number of steps (excluding the initial condition)
num_steps = 9

# Simulation time length
T = 9/48

# Time switch conditions (for the boundary condition)
dt_D = 1/16

# Fluid density
rho = 1000

# Relative and absolute tolerances for the non-linear solver
abs_tol = 1e-6
rel_tol = 1e-6

# Domain tolerance
domain_tolerance = 1 / (10 * N)

# Output directory
output_directory = 'landslide/output_evolutionary'


# Van Genuchten model parameters ( relative permeability model )
theta_s = 0.396
theta_r = 0.131

alpha = 0.423

n = 2.06
K_s = 4.96e-2

m = 1 - 1/n

char_length = 1

# Time step
dt   = (T-0)/num_steps


# Symbolic psi
psi_var = sp.Symbol('psi', negative=True)

# Symbolic Theta
theta_expression = theta_r + (theta_s - theta_r) / (1 + (-alpha * psi_var) ** n) ** m
effective_saturation = (theta_expression - theta_r) / (theta_s - theta_r)

# Symbolic Conductivity K
hydraulic_conductivity_expression = K_s * (effective_saturation ** 0.5) * ( 1 - (1 - effective_saturation ** (1 / m)) ** m ) ** 2

# Theta lambda
theta_lambda = sp.lambdify(psi_var, theta_expression, 'numpy')

# Conductivity tensor lambda
conductivity_lambda = sp.lambdify(psi_var, hydraulic_conductivity_expression, 'numpy')

# Actual (and final) theta function
def theta(psi):
    mask = np.where(psi < 0)
    res = np.ones_like(psi) * theta_s
    res[mask] = theta_lambda(psi[mask])

    return res

# Actual (and final) theta function
def conductivity(psi):
    mask = np.where(psi < 0)
    res = np.ones_like(psi) * K_s
    res[mask] = conductivity_lambda(psi[mask])

    return res

xx = np.array([-10,0,SlopeHeight/np.tan(SlopeAngle),10])
iterator_vect = np.arange(np.size(xx)-1)
irregular_pentagon = [np.array([[xx[0], xx[0]], [bedrock_func(xx[0]), topography_func(xx[0])]])]
for i in iterator_vect:
    line = np.array([[xx[i], xx[i+1]], [topography_func(xx[i]), topography_func(xx[i+1])]]) 
    irregular_pentagon.append(line)
line = np.array([[xx[-1], xx[-1]], [topography_func(xx[-1]), bedrock_func(xx[-1])]])
irregular_pentagon.append(line)

line = np.array([[xx[-1], xx[0]], [bedrock_func(xx[-1]), bedrock_func(xx[0])]])
irregular_pentagon.append(line)

domain_from_polytope = pp.Domain(polytope=irregular_pentagon)
#subdomain = pg.grid_from_domain(domain_from_polytope, char_length, as_mdg=False)

# Prepare the domain and its mesh
subdomain = pp.StructuredTriangleGrid([2*N, 3*N], [2,3])
subdomain.compute_geometry()
# Convert it to a mixed-dimensional grid
mdg = pp.meshing.subdomains_to_mdg([subdomain])


key = "flow"

# Collection of boundary conditions
bc_value = []
bc_essential = []

# Initial pressure
initial_pressure = []

# Discretizations for q and \psi
RT0 = pg.RT0(key)
P0  = pg.PwConstants(key)

# Gravity. The y-direction is set instead of the z-direction because we are in the 2D case
g_func = lambda x, t: np.array([0, -1, 0])

# Initial pressure function
initial_pressure_func = lambda x: 1-x[1]

# Fake loop to extract the grid and its data (i.e. conductivity tensor)
for subdomain, data in mdg.subdomains(return_data=True):
    # Prepare the gravity term Z by firstly projecting g_func into the RT0 space and then by 
    # multiplying it by the RT0 mass matrix
    g_proj = RT0.interpolate(subdomain, lambda x: g_func(x,0))
    gravity = RT0.assemble_mass_matrix(subdomain) @ g_proj
    
    # Prepare the inital pressure term by interpolating initial_pressure_func into the P0 space
    initial_pressure.append(P0.interpolate(subdomain, initial_pressure_func))
        
    # Get the boundary faces ids
    boundary_faces_indexes = subdomain.get_boundary_faces()

    # Gamma_D1 and Gamma_D2 boundary faces
    gamma_d1 = np.logical_and(subdomain.face_centers[0, :] > 0-domain_tolerance, np.logical_and(subdomain.face_centers[0, :] < 1+domain_tolerance, subdomain.face_centers[1, :] > 3-domain_tolerance))
    gamma_d2 = np.logical_and(subdomain.face_centers[0, :] > 2-domain_tolerance, np.logical_and(subdomain.face_centers[1, :] > 0-domain_tolerance, subdomain.face_centers[1, :] < 1+domain_tolerance))

    gamma_d  = np.logical_or(gamma_d1, gamma_d2)

    # Gamma_N is the remaining part of the boundary    
    gamma_n  = gamma_d.copy()
    gamma_n[boundary_faces_indexes] = np.logical_not(gamma_n[boundary_faces_indexes])
    
    # Set the initial conductivity tensor in data (the actual saved tensor does not matter at this stage)
    pp.initialize_data(subdomain, data, key, {
        "second_order_tensor": pp.SecondOrderTensor(np.ones(subdomain.num_cells)),
    })
    
    # Prepare the \hat{\psi} function
    def bc_gamma_d(x, t):
        if   x[0] > 2-domain_tolerance and x[1] > 0-domain_tolerance and x[1] < 1+domain_tolerance:
            res =  1 - x[1]
        elif x[1] > 3-domain_tolerance and x[0] > 0-domain_tolerance and x[0] < 1+domain_tolerance:
            res = min( 0.2, -2 + 2.2 * t / dt_D )
        else:
            res = 0
        
        return res

    # Add a lambda function that generates for each time instant the (discretized) natural boundary 
    # conditions for the problem
    bc_value.append(lambda t: - RT0.assemble_nat_bc(subdomain, lambda x: bc_gamma_d(x,t), gamma_d))

    # Set the essential boundary conditions (they will be enforced before solving the system)
    essential_pressure_dofs = np.zeros(P0.ndof(subdomain), dtype=bool)
    bc_essential = np.hstack((gamma_n, essential_pressure_dofs))

    essential_pressure_dofs = np.zeros(P0.ndof(subdomain), dtype=bool)
    bc_essential_2 = np.hstack((gamma_d, essential_pressure_dofs))

# Psi mass matrix
M_psi = P0.assemble_mass_matrix(subdomain)

# B
B = - pg.cell_mass(mdg, P0) @ pg.div(mdg)


# Psi projection
proj_psi = P0.eval_at_cell_centers(subdomain)

# q projection
proj_q = RT0.eval_at_cell_centers(subdomain)

dof_psi, dof_q = B.shape

# Assemble initial solution
initial_solution = np.zeros(dof_q + dof_psi)
initial_solution[-dof_psi:] += np.hstack(initial_pressure)

# Final solution list. Each of its elements will be the solution at a specific instant
sol = [initial_solution]

# Assemble the fixed part of the right hand side (rhs)
fixed_rhs = np.zeros(dof_q + dof_psi)
fixed_rhs[:dof_q] = gravity

# Helper function to project a function evaluated in the cell center to FEM (scalar)
def project_psi_to_fe(to_project):
    return to_project * subdomain.cell_volumes

# Delete the output directory, if it exisis
if os.path.exists(output_directory):
    shutil.rmtree(output_directory)

# Helper function to export the current_sol to a file
def export_solution(saver, current_sol, num_step):
    ins = list()

    ins.append((subdomain, "cell_q", ( proj_q @ current_sol[:dof_q] ).reshape((3, -1), order="F")))
    ins.append((subdomain, "cell_p", proj_psi @ current_sol[dof_q:(dof_q+dof_psi)]))

    saver.write_vtu(ins, time_step=num_step)

# Prepare the porepy exporter and export the initial solution
saver = pp.Exporter(mdg, 'sol', folder_name=output_directory)

export_solution(saver, current_sol=sol[-1], num_step=0)


# Time loop
for n in np.arange(num_steps):
    current_time = (n + 1) * dt
    print('Time ' + str(round(current_time, 5)))

    # Rhs that changes with time (but not with k)
    time_rhs = fixed_rhs.copy()

    # Add the (natural) boundary conditions
    time_rhs[:dof_q] += np.hstack(list(cond(current_time) for cond in bc_value))

    # Add \Theta^n:
    # 1. Convert psi DOF to cell-wise values
    # 2. Compute theta
    # 3. Project it to P0 elements
    # 4. Multiply by psi-mass
    time_rhs[-dof_psi:] += M_psi @ project_psi_to_fe( theta( proj_psi @ sol[-1][-dof_psi:] ) )

    # Solution at the previous iteration (k=0 corresponds to the solution at the previous time step)
    prev = sol[-1]
    current = None

    # Non-linear solver
    for k in np.arange(K):
        # Actual rhs
        rhs = time_rhs.copy()

        # \Theta^{n+1}_k, same steps as \Theta^n
        rhs[-dof_psi:] -= M_psi @ project_psi_to_fe( theta( proj_psi @ prev[-dof_psi:] ) )
        
        # L-term
        rhs[-dof_psi:] += L * M_psi @ prev[-dof_psi:]
    
        # Set the conductivity tensor in data (the actual saved tensor does not matter at this stage)
        pp.initialize_data(subdomain, data, key, {
            "second_order_tensor": pp.SecondOrderTensor(
                conductivity(proj_psi @ prev[-dof_psi:])
            ),
        })

        Mass_u = pg.face_mass(mdg, RT0)

        #print(Mass_u)
        #sys.exit()
        
        # Assemble the system to be solved at time n and interation k
        spp = sps.bmat(
            [[Mass_u,        B.T] ,
             [-dt * B, L * M_psi]], format="csc"
        )

        # Prepare the linear solver
        ls = pg.LinearSystem(spp, rhs)

        # Fix the essential boundary conditions
        ls.flag_ess_bc(np.hstack(bc_essential), np.zeros(dof_q + dof_psi))
        #ls.flag_ess_bc(np.hstack(bc_essential_2), np.ones(dof_q + dof_psi))

        # Solve the system
        current = ls.solve()
        
        # Check if we have reached convergence
        rel_err_psi  = np.sqrt(np.sum(np.power(current[-dof_psi:] - prev[-dof_psi:], 2)))
        abs_err_prev = np.sqrt(np.sum(np.power(prev[-dof_psi:], 2)))

        # Log message with error and current iteration
        print('Iteration #' + str(k+1) + ', error L2 relative psi: ' + str(rel_err_psi))
        
        if rel_err_psi > abs_tol + rel_tol * abs_err_prev:
            prev = current.copy()
        else:
            break
        
    sol.append( current )
    export_solution(saver, current_sol=sol[-1], num_step=(n+1))

saver.write_pvd([n * dt for n in np.arange(num_steps + 1)])