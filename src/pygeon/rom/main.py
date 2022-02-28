
import numpy as np
import scipy as sp
import scipy.interpolate as interpolate
import porepy as pp
from matplotlib import pyplot as plt
import pdb
import os
from offline_porepy import *
from online_porepy import *

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
    
    
3) improve sampling paramter from distribution
4) plot singular values       
5) parent class of offline                                                          DONE
6) parent class on online (OnlineComputations, OnlinePorePy)                        DONE
7) check physical parameters used           
8) online pod should be parent of online porepy                                     DONE
9) fix load_svd_matrices input and compute_pod input, they are counterintuitive...  DONE
10) name "compute_pod" may be misleading...                                         DONE
11) "load_n_snapshot" not clear...  
12) nell online serve un metodo "assemble A, b"                                     DONE


'''


################################################################################
# offline (reduced basis computation with SVD algorithm)
################################################################################
offline = True
n_snap_to_generate = 50 # 245 no fault # with fault
fault = False
do_monolithic = True

if offline == True:
    generate_snap = OfflinePorePy()
    generate_snap.remove_old_data()
    generate_snap.generate_grid(add_fault=fault)
    generate_snap.generate_snapshots(n_snap_to_generate, verbose=True) #vtu_filename='current_snap', verbose=True)



################################################################################
# online (reduced model solution)
################################################################################
pod = OnlinePodPorePy()
pod.generate_mesh()
A, b = pod.compute_A_rhs()
sol_fom = pod.compute_fom_solution(A, b)

n_snap_max = 50
n_snap_to_use_list = np.arange(2, n_snap_max, 3) # either this or 
n_single_eval = 2
#n_snap_to_use_list = np.array([n_single_eval])   # this

#do_pod_block = False

mse_error = []

compute_reduced_basis = OfflinePorePy()

for n_snap_to_use in n_snap_to_use_list:
    print('\n\n\n\n\nn_snap_to_use = ', n_snap_to_use)
    
    # still offline:
    compute_reduced_basis.var_names = ['ph', 'pl', 'lmbda'] # not required if no for loop, compute_svd adds the var name 'all'
    compute_reduced_basis.load_n_snapshots(n_snap_to_use, shuffle=False)
    compute_reduced_basis.compute_svd(do_monolithic=do_monolithic)                                                  
    
    n_modes_to_use = n_snap_to_use ### I'll always use the number of snapshots since their computation is expensive
    
    compute_reduced_basis.truncate_U(n_modes_to_use)
    compute_reduced_basis.save_svd_matrices()
    Phi = compute_reduced_basis.assemble_phi()

    # online:    
    sol_reduced = pod.compute_reduced_solution(A, b, Phi)
    sol_reconstructed = Phi@sol_reduced
    
    # errors:
    mse_error.append( np.sum( (sol_reconstructed - sol_fom)**2 )/sol_fom.size )    

    np.set_printoptions(precision=3)
    myprint('np.array(mse_error)')

    
    # # plot solutions:
    # if n_snap_to_use == n_single_eval:
    #     for g, d in pod.gb:
    #         if g.dim == pod.gb.dim_max():
    #             d[pp.STATE]['sol_reconstructed'] = sol_reconstructed
    # 
    #     exporter = pp.Exporter(pod.gb, 'fom_sol')
    #     exporter.write_vtu(["ph"])        
    # 
    #     exporter = pp.Exporter(pod.gb, 'sol_reconstructed')                                    # TODO
    #     #reconstructed_ph = np.array([ reconstructed_sol_monolithic[i] for i in ph_indices ])
    #     exporter.write_vtu(["sol_reconstructed"])

stop()

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


# S = np.loadtxt('./data/S_all')
# fig, ax = plt.subplots()
# ax.plot(S, marker='o')
# ax.set_ylabel('singular_values')
# ax.set_yscale('log')
# ax.grid(linestyle='--')

plt.show()









'''
OBSERVATIONS:
1) il primo modo descrive già bene la discontinuità (va anche troppo bene!)

2) ph variability:
no fault:
max = 1 (left bc), min = 0 (right bc)
middle:     0.484... 0.4850, 0.4845, 0.4834, 0.4790, 0.4852, 0.4848, 0.4837
region one: 0.7364, 0.7271, 0.7291, 0.7363, 0.7211


'''






# # plot singular values:

# fig, ax= plt.subplots(5, 1) 
# ax[0].plot(S_ph)
# ax[0].set_title('ph singular values') 
# ax[1].plot(S_ph)
# ax[1].set_yscale('log')
# ax[2].plot(S_pl)
# ax[2].set_title('pl singular values') 
# ax[3].plot(S_lmbda)
# ax[3].set_title('lmbda singular values') 
# ax[4].plot(S_all)
# ax[4].set_title('all singular values') 
# plt.grid(visible=True, which='major', axis='both')

# myprint('S_ph')
# myprint('S_pl')
# myprint('S_lmbda')
# myprint('S_all')






'''
the error increases between 14 ( 14=28/2 ?) and 33, grid: 0.8 0.02 with fault
                            12                  31        0.2 0.02 with fault
                            15                  34        0.1 0.01 with fault

without fracture (=> block = monolithic) no bad trend

with fine mesh no fracture (5800 dof) really bad reconstruction using 30 modes
with 950 dof no fracture trend exponential then faster than exp! ?
using 400 snap sol is badly reconstructed, using

'''







































################################################################################
# Bin:
################################################################################


# try different base:
# D = np.random.rand(A.shape[0], snap_index_max) 
# for i in range(snap_index_max):
#     D[:, i] = D[:, i] - np.mean(D, axis=1) # useless...
# 
# U, S, Vh = np.linalg.svd(D)
# U = U[:, :n_modes]
# phi = U




# np.set_printoptions(precision=3)
# myprint('phi')







# I don't use the parameter affinity porperty (does it hold?) => I build the 
# matrix A of the system A*x = b for each new intance:

#snap_index_max = 212 # dof_manager.fulldof = [128  28  56], sum = 212 # snapshot 0, 1, 2, ... , snap_index_max #### NON VOGLIO CREARE UNA CLASSE IN offline.py ma mi servono dei parameteri definiti lì!


    # n_modes_to_use = 1
    
    # if do_pod_block == True:
    #     pod.load_svd_block_matrices(n_modes_to_use)
    #     reduced_sol_block, phi_block = pod.compute_pod_block()
    #     reconstructed_sol_block = phi_block @ reduced_sol_block
    #     projected_sol_block = phi_block.T @ fom_sol
    # 
    # pod.load_svd_monolithic_matrices(n_modes_to_use)
    # reduced_sol_monolithic, phi_monolithic = pod.compute_pod_monolithic()
    # reconstructed_sol_monolithic = phi_monolithic @ reduced_sol_monolithic
    # projected_sol_monolithic = phi_monolithic.T @ fom_sol


# ph_indices = np.loadtxt('./data/dof_info_ph').astype(int)
# fom_ph = np.array([ fom_sol[i] for i in ph_indices ])
# exporter.write_vtu({'pressure_h_fom': fom_ph})


# 
# if do_pod_block == True:
#     mse_error_block.append( np.sum( (reconstructed_sol_block - fom_sol)**2 )/fom_sol.size )
















