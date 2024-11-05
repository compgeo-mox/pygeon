import os
import shutil
from richards_mixed_form import *


# read input data from the points.dat file
file_path = os.getcwd() + '/landslide/points.dat'
with open(file_path, 'r') as f:
    w, h = [int(x) for x in next(f).split()]
    array = [[float(x) for x in line.split()] for line in f]

gamma_water = 10 # kN/m^3
gamma = 26.5  # saturated unit weight of soil [kN/m^3]
cohesion = 1. # soil coesion [kPa]
phi = np.radians(28.)  # friction angle  
Ns = 50


# Set the maximum number of iterations of the non-linear solver, if one it corresponds to the semi-implicit method
number_nonlin_it = 500
abs_tol = 1e-5 # Relative and absolute tolerances for the non-linear solver

# Initial pressure function
initial_pressure_func = lambda x: -50#50*(-x[1]+bedrock_func(x[0]))#(-50)#2-x[1] #-x[1]+bedrock_func(x[0]) #(2-(x[1]-bedrock_func(x[0]))) #-x[1] # -.2*x[1]#(x[0]*.5 + x[1]*np.sqrt(3)*.5)

# 
saving_time_interval = 20*3600#30*60#1000#.005*3600*unit_measure_transform_time
potential_time_step = 20*3600#900#1000#.0001*3600*unit_measure_transform_time
final_time = 1*3600*20

for i in range(1):
    theta_s = array[i][0]
    theta_r = array[i][1]
    alpha   = array[i][2]
    n_coeff = array[i][3]
    K_s     = 10**(array[i][4])

    # Output directory
    output_directory = 'landslide/output_evolutionary_' + str(i)
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory)
    os.makedirs(output_directory)

    richards_MFEM(theta_s, theta_r, alpha, n_coeff, K_s, gamma_water, gamma, cohesion, phi, Ns, output_directory, number_nonlin_it, abs_tol, initial_pressure_func, saving_time_interval, potential_time_step, final_time)
    