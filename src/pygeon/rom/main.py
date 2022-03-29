
import numpy as np
import scipy as sp
import scipy.interpolate as interpolate
import porepy as pp
from matplotlib import pyplot as plt
import pdb
import os
from offline_porepy import *
from online_pod_porepy import *

os.system('clear')

def myprint(var):
    print('\n' + var + ' = ', eval(var))

def stop():
    print('\n\n\n\n')
    raise NotImplementedError


'''
TODO:
1) how much does ph vary wrt k                                                      DONE
2) benckmark => use different problem => create snap without PorePy => 5)           DONE
    possible benckmarks: Quarteroni, Manzoni, Negri p. 138, 
    
    POD benchmark: mi serve per verificare il codice scritto: ci possono essere errori nell'implementazione del algo pod
    
    NN benchmark: è quello che sto facendo... obj di questo studio. ci possono essere errori di implementazione?? anche no...
    
    
3) improve sampling paramter from distribution                                      todo
4) plot singular values                                                             DONE
5) parent class of offline                                                          DONE
6) parent class on online (OnlineComputations, OnlinePorePy)                        DONE
7) check physical parameters used                                                   DONE
8) online pod should be parent of online porepy                                     DONE
9) fix load_svd_matrices input and compute_pod input, they are counterintuitive...  DONE
10) name "compute_pod" may be misleading...                                         DONE
11) "load_snapshot" not clear...                                                    waiting for a better idea...
12) nell online serve un metodo "assemble A, b"                                     DONE
13) fix exporter: problem related to variable with different names                  DONE
14) save grid as python object insted of rebuild it                                 todo

15) WORKS ONLY WITH FAULTS ############################################################


STO GENERALIZZANDO POCO: RANGE MU UGUALE PER SNAP E PER ONLINE

'''



################################################################################
# clean the working folder
################################################################################
os.system("rm *pvd *vtu")



################################################################################
# offline:
################################################################################
offline = True 
n_snap_to_generate = 60  # 245 no fault # with fault
fault = True 
do_monolithic = False

if offline == True:
    generate_snap = OfflinePorePy()
    generate_snap.remove_old_data()
    generate_snap.generate_grid(add_fault=fault)
    generate_snap.generate_snapshots(n_snap_to_generate, verbose=True) #, vtu_filename="current_snap")



################################################################################
# online:
################################################################################
# fom solution:
pod = OnlinePodPorePy()
pod.generate_grid(add_fault=fault)
A, b = pod.compute_A_rhs()
sol_fom = pod.compute_fom_solution(A, b) # do you prefer a generic "compute_solution" method?

# compute velocity from fom sulution:
vel_fom = pod.compute_darcy_velocity(sol_fom)
g = pod.gb.grids_of_dimension(2)[0]
d = pod.gb.node_props(g)
exporter = pp.Exporter(g, "vel_fom")  
exporter.write_vtu({"vel_fom": vel_fom})

# settings:
n_snap_max = 60 # maximum number os snapshots to use
n_snap_to_use_list = np.arange(1, n_snap_max, 5) # either this or 
n_single_eval = 6
#n_snap_to_use_list = np.array([n_single_eval])   # this

mse_error = []

compute_reduced_basis = OfflinePorePy()

# the goal of the following loop is to plot the error vs the number of snapshot 
# used for the basis reduction
for n_snap_to_use in n_snap_to_use_list:
    print('\n\n\nn_snap_to_use = ', n_snap_to_use)
    
    # still offline:
    compute_reduced_basis.var_names = ['ph', 'pl', 'lmbda'] # not required if no "for" loop, compute_svd adds the var name 'all'
    compute_reduced_basis.load_snapshots(n_snap_to_use, shuffle=False)
    compute_reduced_basis.compute_svd(do_monolithic=do_monolithic)                                                  
    
    n_modes_to_use = n_snap_to_use # I'll always use the number of snapshots since their computation is expensive
    #n_modes_to_use = 1
    
    compute_reduced_basis.truncate_U(n_modes_to_use)
    compute_reduced_basis.save_svd_matrices()
    Phi = compute_reduced_basis.assemble_phi()

    # online:    
    sol_reduced = pod.compute_reduced_solution(A, b, Phi) # do you prefer a generic "compute_solution" method?
    sol_reconstructed = Phi@sol_reduced
    
    # error computation:
    mse_error.append( np.sum( (sol_reconstructed - sol_fom)**2 )/sol_fom.size )    

    np.set_printoptions(precision=3)
    myprint('np.array(mse_error)')
    
    # write solution and error vtu files:
    if n_snap_to_use == n_single_eval:
        pod.write_vtu(sol_fom, "sol_fom")
        pod.write_vtu(sol_reconstructed, "sol_reconstructed")
        se = ( sol_fom - sol_reconstructed )**2 
        pod.write_vtu(se, "squared_error")
        
        # compute velocity from reconstructed solution:
        vel_reconstructed = pod.compute_darcy_velocity(sol_reconstructed)
        g = pod.gb.grids_of_dimension(2)[0]
        d = pod.gb.node_props(g)
        exporter = pp.Exporter(g, "vel_reconstructed")  
        exporter.write_vtu({"vel_reconstructed": vel_reconstructed})
        
        # compute velocity error:
        vel_squared_err = ( vel_reconstructed[0] - vel_fom[0] )**2 + \
        ( vel_reconstructed[1] - vel_fom[1] )**2 # ovv you can compute the squared magnitude of vector error in paraview...
        exporter = pp.Exporter(g, "vel_squared_err")  
        exporter.write_vtu({"vel_squared_err": vel_squared_err}) 

        vel_vector_err = vel_reconstructed - vel_fom 
        exporter = pp.Exporter(g, "vel_vector_err")  
        exporter.write_vtu({"vel_vector_err": vel_vector_err})
        
# plot error vs snapshots used:
fig, ax = plt.subplots(1, 2)

ax[0].plot(n_snap_to_use_list, mse_error)
ax[0].set_ylabel('mse error')
ax[0].set_xlabel('number of snapshots used')
ax[0].grid(linestyle='--')

ax[1].plot(n_snap_to_use_list, mse_error)
ax[1].set_yscale('log')
ax[1].set_ylabel('mse error')
ax[1].set_xlabel('number of snapshots used')
ax[1].grid(linestyle='--')

# # plot singular values:
# S = np.loadtxt('./data/S_all')
# fig, ax = plt.subplots()
# ax.plot(S, marker='o')
# ax.set_ylabel('singular_values')
# ax.set_yscale('log')
# ax.grid(linestyle='--')

plt.show()



