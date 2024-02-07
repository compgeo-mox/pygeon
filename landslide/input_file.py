# This is the complete input file 
import numpy as np

# Set the maximum number of iterations of the non-linear solver
K = 100

# L-scheme parameter
L = 2.5e-2

# Set the number of steps (excluding the initial condition)
num_steps = 10

# Simulation time length
T = 1

# Time step
dt = T/num_steps

# Time switch conditions (for the boundary condition)
#dt_D = 1/16

# Fluid density
rho = 1000

# Relative and absolute tolerances for the non-linear solver
abs_tol = 1e-6
rel_tol = 1e-6

# Output directory
output_directory = 'landslide/output_evolutionary'

# Soil Parameters
gamma = 18.  # saturated unit weight of soil [kN/m^3]
cohesion = 0. # soil coesion [kPa]
phi = np.radians(30.)  # friction angle in deg 
# numbers of Bishop's elements
Ns = 20


# Van Genuchten model parameters ( relative permeability model )
theta_s = 0.396
theta_r = 0.131

alpha = 0.423

n = 2.06
K_s = 4.96e-2

m = 1 - 1/n



# space discretization parameters
char_length = 1

# Domain tolerance
domain_tolerance = char_length/10

# Geometry of the slope 
SlopeHeight = 5. # meters
SlopeAngle = np.radians(50.) # degrees

domain_extent_left = -3
domain_extent_right = 8

xx_plot = np.linspace(domain_extent_left,domain_extent_right,300)
