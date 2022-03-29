
import numpy as np
import scipy as sp
import scipy.interpolate as interpolate
import porepy as pp
from matplotlib import pyplot as plt
import pdb
import os
from offline_test import *
from online_pod_test import *

os.system('clear')

def myprint(var):
    print('\n' + var + ' = ', eval(var))

def stop():
    print('\n\n\n\n')
    raise NotImplementedError



################################################################################
# pod benchmark:
################################################################################

n_snap_to_generate = 50
n_snap_to_use = 6
n_modes_to_use = 2
do_monolithic = True

offline = OfflineTest()
offline.remove_old_data()
offline.generate_snapshots(n_snap_to_generate) # from analytical solution

offline.load_snapshots(n_snap_to_use, shuffle=False)
offline.compute_svd(do_monolithic)
offline.truncate_U(n_modes_to_use)
offline.save_svd_matrices()

Phi = offline.assemble_phi() # and save phi

# not necessary:
offline.plot_singular_values()



online = OnlineTest() 
A, b = online.compute_A_rhs()

# not necessary:
sol_analytical, sol_fom = online.compute_fom_solution(A, b) # analytical solution + A, b assembling for random mu_param

sol_reduced = online.compute_reduced_solution(A, b, Phi)
sol_reconstructed = Phi@sol_reduced

mse_err = np.sum( (sol_reconstructed - sol_fom)**2 )/sol_fom.size 
myprint('mse_err')
myprint('sol_analytical')
myprint('sol_fom')
myprint('sol_reconstructed')

plt.show()










'''
mmm va addirittura meglio del ref...
'''











