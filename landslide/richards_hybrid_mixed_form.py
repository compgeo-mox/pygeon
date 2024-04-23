import shutil
import os
import sys

import scipy.sparse as sps
import sympy as sp
import matplotlib.pyplot as plt

import porepy as pp
import pygeon as pg

from topograhy_file import theta_Chouly, penalty_factor_hyb, initial_pressure_func, generate_mesh, mark_boundary_faces, SlopeAngle, precipitation_func, there_is_gravity, theta, conductivity, DpsiDtheta, time_step, output_directory, num_steps, number_nonlin_it, L_scheme_coeff, abs_tol, rel_tol, DinvConductivityDpsi
from limit_equilibrium import *

from utils import *

# prima gmsh poi match_2d per interpolare sulla nuova 
# src/porepy/grids/match_grids.py

# per il calcolo delle intersezioni,
# cut_faces, new_nodes = intersect_faces(sd, levelset)
# cut_cells = intersect_cells(sd, cut_faces)

# per le normali
# subdomain.face_normals
# face_normals, normale pesata 
# cell_faces, +/-1 orientazione delle normali

subdomain = generate_mesh()
#pp.plot_grid(subdomain, info="all", alpha=0)
#sys.exit()

#g = pp.CartGrid([3] * 2, [1] * 2)
#g.compute_geometry()
#tree = pp.adtree.ADTree(4, 2)
#tree.from_grid(g)

#tree = pp.adtree.ADTree(4, 2)
#tree.from_grid(subdomain) 

#n = pp.adtree.ADTNode(99, [0.6, 0.6, 0.3, 0.3])
#n_nodes = tree.search(n)

#n = pp.adtree.ADTNode(99, [0.6] * 4)
#n_nodes = tree.search(n)
#assert np.allclose(n_nodes, [4])
#pp.plot_grid(g, info="all", alpha=0)


key = "flow"


# Initial pressure
initial_pressure = []


# Discretizations for q and \psi
RT0 = pg.RT0(key)
P0  = pg.PwConstants(key)

# Gravity. The y-direction is set instead of the z-direction because we are in the 2D case
g_func = lambda x, t: np.array([0, -1, 0])

# Fake loop to extract the grid and its data (i.e. conductivity tensor)
# Prepare the gravity term Z by firstly projecting g_func into the RT0 space and then by 
# multiplying it by the RT0 mass matrix
g_proj = RT0.interpolate(subdomain, lambda x: g_func(x,0))
gravity = RT0.assemble_mass_matrix(subdomain) @ g_proj

# interpolate precipitation func on RT0
prec_proj = RT0.interpolate(subdomain, lambda x: precipitation_func(x))

# Prepare the inital pressure term by interpolating initial_pressure_func into the P0 space
initial_pressure.append(P0.interpolate(subdomain, initial_pressure_func))

# mark noundary faces
bc_value_bedrock, gamma_laterals, gamma_topography, gamma_bedrock = mark_boundary_faces(subdomain, P0, RT0)

# Psi mass matrix
M_psi = P0.assemble_mass_matrix(subdomain)

# B
B = - P0.assemble_mass_matrix(subdomain) @ pg.div(subdomain)

#subdomain.cell_faces


# Psi projection
proj_psi = P0.eval_at_cell_centers(subdomain)

# q projection
proj_q = RT0.eval_at_cell_centers(subdomain)

dof_psi, dof_q = B.shape

dof_l = np.size(gamma_topography[gamma_topography])

# Set the essential boundary conditions (they will be enforced before solving the system)
bc_essential_laterals   = np.hstack((gamma_laterals,   np.zeros(P0.ndof(subdomain), dtype=bool), np.zeros(dof_l, dtype=bool)))
bc_essential_topography = np.hstack((gamma_topography, np.zeros(P0.ndof(subdomain), dtype=bool), np.zeros(dof_l, dtype=bool)))

# Assemble initial solution
initial_solution = np.zeros(dof_q + dof_psi + dof_l)
initial_solution[dof_q:-dof_l] += np.hstack(initial_pressure)
initial_solution[-dof_l:] += initial_pressure_func([subdomain.face_centers[0,:][gamma_topography], subdomain.face_centers[1,:][gamma_topography]])*subdomain.face_areas[gamma_topography]

# Final solution list. Each of its elements will be the solution at a specific instant
sol = [initial_solution]

# Assemble the fixed part of the right hand side (rhs)
fixed_rhs = np.zeros(dof_q + dof_psi + dof_l)
fixed_rhs[:dof_q] = gravity*there_is_gravity

# Helper function to project a function evaluated in the cell center to FEM (scalar)
def project_psi_to_fe(to_project):
    return to_project * subdomain.cell_volumes

# Delete the output directory, if it exisis
#if os.path.exists(output_directory):
#    shutil.rmtree(output_directory)

# Helper function to export the current_sol to a file
def export_solution(saver, current_sol, num_step):
    ins = list()

    ins.append((subdomain, "cell_q", ( proj_q @ current_sol[:dof_q] ).reshape((3, -1), order="F")))
    ins.append((subdomain, "cell_p", proj_psi @ current_sol[dof_q:(dof_q+dof_psi)]))

    saver.write_vtu(ins, time_step=num_step)

# Prepare the porepy exporter and export the initial solution
saver = pp.Exporter(subdomain, 'sol', folder_name=output_directory)

export_solution(saver, current_sol=sol[-1], num_step=0)


# get the sign of each normal to the face, here maybe you could use np.where(gamma_topography)[0] to access the vector, need to see which is the most performant
normal_vectors_sign = gamma_topography*0
cell_faces_sub = subdomain.cell_faces[gamma_topography,:]
normal_vectors_sign_it = normal_vectors_sign[gamma_topography]
normal_vectors_sign_it[cell_faces_sub.indices] = cell_faces_sub.data
normal_vectors_sign[gamma_topography] = normal_vectors_sign_it
#normal_vectors_sign[np.where(gamma_topography)[0]] 


tree = pp.adtree.ADTree(4, 2)
tree.from_grid(subdomain)

# https://github.com/pmgbergen/porepy/blob/develop/tutorials/grid_topology.ipynb
# mappa cella-cella
# subdomain.cell_faces

# point check
#n = pp.adtree.ADTNode(99, [0.1] * 4)
#n_nodes = tree.search(n)

flux_lat_bc = np.zeros(dof_q)
normal_vector_lat = -subdomain.face_normals[1,:][gamma_laterals]/subdomain.face_areas[gamma_laterals]
normal_vector_top =  normal_vectors_sign[gamma_topography]

gamma_Nitsche = penalty_factor_hyb/subdomain.face_areas[gamma_topography]

#prec_term_Nitsche = precipitation_func(subdomain.face_centers[0][gamma_topography])*subdomain.face_normals[1,:][gamma_topography]/subdomain.face_areas[gamma_topography]

#normal_vector_top*1./gamma_Nitsche*np.maximum((prec_proj[gamma_topography] - prev[:dof_q][gamma_topography])/subdomain.face_areas[gamma_topography]*0 - gamma_Nitsche*prev[-dof_psi:][subdomain.cell_faces[gamma_topography].nonzero()[1]], 0)



L_mat = np.zeros((dof_q, dof_q))  
L_mat[gamma_topography, gamma_topography] = normal_vector_top/subdomain.face_areas[gamma_topography]
L_mat = sps.csc_matrix(L_mat[gamma_topography])

cp = Matrix_Computer(subdomain, dof_q, dof_psi)
C_mat = sps.dok_matrix((dof_q, dof_psi))

zeros_mat_psi_l = sps.dok_matrix((dof_psi, dof_l))
zeros_mat_l_l   = sps.dok_matrix((dof_l,   dof_l))

Nitsche_contribution_hybrid = Nitsche_term_hybrid(subdomain, dof_q, dof_l, gamma_Nitsche)

# Time loop
for n in np.arange(num_steps):
    current_time = (n + 1) * time_step
    print('Time ' + str(round(current_time, 5)))

    # Solution at the previous iteration (k=0 corresponds to the solution at the previous time step)
    prev = sol[-1]

    # Rhs that changes with time (but not with k)
    time_rhs = fixed_rhs.copy()

    # Add the (natural) boundary conditions
    time_rhs[:dof_q] += 0 #bc_value_bedrock(current_time) 


    # Add \Theta^n:
    # 1. Convert psi DOF to cell-wise values
    # 2. Compute theta
    # 3. Project it to P0 elements
    # 4. Multiply by psi-mass
    time_rhs[dof_q:-dof_l] += M_psi @ project_psi_to_fe( theta( proj_psi @ prev[dof_q:-dof_l] ) ) /time_step

    # Solution at the previous iteration (k=0 corresponds to the solution at the previous time step)
    #prev = sol[-1]
    #current = None
    #L_scheme_coeff = np.max(np.abs(DpsiDtheta(prev[-dof_psi:])))

    # Non-linear solver
    for k in np.arange(number_nonlin_it):
        
        # Actual rhs
        rhs = time_rhs.copy()

        # Assemble Nitsche contributions
        Nitsche_contribution_hybrid.compute_Nitsche_contributions(gamma_topography, prec_proj, prev, normal_vector_top)
        rhs[-dof_l:] += prec_proj[gamma_topography]*normal_vector_top/subdomain.face_areas[gamma_topography] + Nitsche_contribution_hybrid.flux_top_bc + (Nitsche_contribution_hybrid.matrix_Nitsche_ll @ prev[-dof_l:]) + (Nitsche_contribution_hybrid.matrix_Nitsche_lq @ prev[:dof_q])

        # \Theta^{n+1}_k, same steps as \Theta^n
        rhs[dof_q:-dof_l] -= M_psi @ project_psi_to_fe( theta( proj_psi @ prev[dof_q:-dof_l] ) ) /time_step
        
        N_mat = P0.assemble_mass_matrix(subdomain, {pp.PARAMETERS: {key: {"second_order_tensor": pp.SecondOrderTensor(DpsiDtheta(proj_psi @ prev[dof_q:-dof_l]))}}, pp.DISCRETIZATION_MATRICES: {key: {}},})

        pic_term = N_mat # L_scheme_coeff*M_psi #N_mat

        # L-term
        rhs[dof_q:-dof_l] += pic_term/time_step @ prev[dof_q:-dof_l]
        
        # questa varia con le iterazioni
        Mass_u = cp.compute_RT0_mass_matrix(conductivity(proj_psi @ prev[dof_q:-dof_l]))
        #Mass_u = RT0.assemble_mass_matrix(subdomain, {pp.PARAMETERS: {key: {"second_order_tensor": pp.SecondOrderTensor(conductivity(proj_psi @ prev[dof_q:-dof_l]))}}, pp.DISCRETIZATION_MATRICES: {key: {}},})

        # add the Newton contribution 
        #C_mat = cp.compute_dual_C(subdomain, prev[:dof_q], DinvConductivityDpsi(proj_psi @ prev[-dof_psi:]))
        rhs[:dof_q] += C_mat @ prev[dof_q:-dof_l]  

        # Assemble the system to be solved at time n and interation k, for the L-scheme
        spp = sps.bmat(
            [[Mass_u,                                               B.T + C_mat,        L_mat.T                                        ],
            [-B,                                                    pic_term/time_step, zeros_mat_psi_l                                ],
            [L_mat + Nitsche_contribution_hybrid.matrix_Nitsche_lq, zeros_mat_psi_l.T,  Nitsche_contribution_hybrid.matrix_Nitsche_ll  ]], format="csc"
        )

        # Prepare the linear solver
        ls = pg.LinearSystem(spp, rhs)
        
        # Fix the essential boundary conditions
        flux_lat_bc[gamma_laterals] = normal_vector_lat*conductivity(prev[dof_q:-dof_l][subdomain.cell_faces[gamma_laterals].nonzero()[1]]/subdomain.cell_volumes[subdomain.cell_faces[gamma_laterals].nonzero()[1]]) 
        ls.flag_ess_bc(np.hstack(bc_essential_laterals  ), np.hstack((flux_lat_bc, np.zeros(dof_psi), np.zeros(dof_l))))
        #ls.flag_ess_bc(np.hstack(bc_essential_topography), np.hstack((prec_proj,   np.zeros(dof_psi), np.zeros(dof_l))))
        

        # Solve the system
        current = ls.solve()
        
        # Check if we have reached convergence
        rel_err_psi  = np.sqrt(np.sum(np.power(current[dof_q:-dof_l] - prev[dof_q:-dof_l], 2)))
        abs_err_prev = np.sqrt(np.sum(np.power(prev   [dof_q:-dof_l]                     , 2)))

        # Log message with error and current iteration
        print('Iteration #' + str(k+1) + ', error L2 relative psi: ' + str(rel_err_psi))
        
        if rel_err_psi > abs_tol + rel_tol * abs_err_prev:
            prev = current.copy()
        else:
            break
        
    sol.append( current )
    export_solution(saver, current_sol=current, num_step=(n+1))

    # now call the structural part!
    x_in  = domain_extent_right
    x_out = domain_extent_left
    #trial = np.array([5, -1, np.radians(70)])
    trial = np.array([x_in, x_out, np.radians(40)])
    #func(trial)
    #sys.exit()
    #zero = call_optimizer(trial, current[-dof_psi:], tree)
    


saver.write_pvd([n * time_step for n in np.arange(num_steps + 1)])










