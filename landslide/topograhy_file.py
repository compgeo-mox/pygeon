import numpy as np

import porepy as pp
import pygeon as pg
import numdifftools as nd 
from collections import Counter
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import sympy as sp
import gmsh
import os
from typing import Callable, Optional, Union
import pandas

# extract line in Paraview,
# 0.4330127018922193, -0.25
# 2.9330127018922196, 4.080127018922193

# 43.30127018922193, -25
# 45.80127018922193, -20.66987298107781

# theta_s = 0.4
# theta_r = 0.04
 
# alpha = 0.098067
 
# n_coeff = 1.2 #2.06
# K_s = 1e-6 #*(3.6e3)

# put everything in the SI unit measurement system

unit_measure_transform_space = 1#1e3 # meters to centimeters
unit_measure_transform_time = 1#1./(3600*24) # seconds to hours

# Set the maximum number of iterations of the non-linear solver, if one it corresponds to the semi-implicit method
number_nonlin_it = 500

# 
saving_time_interval = 10*3600#30*60#1000#.005*3600*unit_measure_transform_time

# 
time_step = 10*3600#900#1000#.0001*3600*unit_measure_transform_time

# Simulation time length in hours
final_time = 1*3600*500*unit_measure_transform_time

# Initial pressure function
initial_pressure_func = lambda x: (-50)*(x[1]<6)#2-x[1] #-x[1]+bedrock_func(x[0]) #(2-(x[1]-bedrock_func(x[0]))) #-x[1] # -.2*x[1]#(x[0]*.5 + x[1]*np.sqrt(3)*.5)

# m/s^2
gravity_field = 9.81

# Relative and absolute tolerances for the non-linear solver
abs_tol = 1e-5

# Output directory
output_directory = 'landslide/output_evolutionary'

# Soil Parameters
gamma = 26.5  # saturated unit weight of soil [kN/m^3]
cohesion = 0. # soil coesion [kPa]
phi = np.radians(40.)  # friction angle  
porosity = 0.4
# numbers of Bishop's elements
Ns = 20


# Van Genuchten model parameters ( relative permeability model )
theta_s = porosity
theta_r = 0.04

gamma_water = 10 #10 #9819e-3 # kN/m^3
alpha = 1./50*gamma_water/unit_measure_transform_space # 1./5*gamma_water #0.098067

n_coeff = 1.5 #2.06
K_s = 1e-6*unit_measure_transform_space/unit_measure_transform_time #*(3.6e3)

m_coeff = 1 - 1/n_coeff



# space discretization parameters
#char_length = .05*unit_measure_transform_space #0.025 


# Van Genuchten model
psi_var = sp.Symbol('psi', negative=True)
theta_expression = theta_r + (theta_s - theta_r) / (1 + (-alpha * psi_var) ** n_coeff) ** m_coeff
effective_saturation = (theta_expression - theta_r) / (theta_s - theta_r)
hydraulic_conductivity_expression = K_s * (effective_saturation ** .5) * ( 1 - (1 - effective_saturation ** (1 / m_coeff)) ** m_coeff ) ** 2

#theta_expression = theta_r + (theta_s - theta_r)*np.exp(alpha*psi_var)
#hydraulic_conductivity_expression = K_s*np.exp(alpha*psi_var)

invConductivity_expression = 1./hydraulic_conductivity_expression
DinvConductivityDpsi_expression = invConductivity_expression.diff(psi_var)
DConductivityDpsi_expression = hydraulic_conductivity_expression.diff(psi_var)
DthetaDpsi_expression = theta_expression.diff(psi_var)

theta_lambda = sp.lambdify(psi_var, theta_expression, 'numpy')
conductivity_lambda = sp.lambdify(psi_var, hydraulic_conductivity_expression, 'numpy')
DthetaDpsi_lambda = sp.lambdify(psi_var, DthetaDpsi_expression, 'numpy')
DinvConductivityDpsi_lambda = sp.lambdify(psi_var, DinvConductivityDpsi_expression, 'numpy')
DConductivityDpsi_expression_lambda = sp.lambdify(psi_var, DConductivityDpsi_expression, 'numpy')


#theta_lambda                = lambda psi_var: theta_r + (theta_s - theta_r) / (1 + (-alpha * psi_var) ** n_coeff) ** m_coeff 
#effective_saturation        = lambda psi_var: (theta(psi_var) - theta_r) / (theta_s - theta_r)
#conductivity_lambda         = lambda psi_var: K_s * (effective_saturation(psi_var) ** 0.5) * ( 1 - (1 - effective_saturation(psi_var) ** (1 / m_coeff)) ** m_coeff ) ** 2
#DpsiDtheta_lambda           = lambda psi_var: 1./((theta_s - theta_r)*m_coeff*(1 + (-alpha*psi_var)**n_coeff)**(-m_coeff-1)*n_coeff*alpha*(-alpha*psi_var)**(n_coeff-1))

thr = 1e-4 #1e-4

def theta(psi):
    mask = np.where(psi <= -thr)
    res = np.ones_like(psi) * theta_s
    res[mask] = theta_lambda(psi[mask])

    return res

def DthetaDpsi(psi):
    mask = np.where(psi <= -thr)
    res = np.ones_like(psi) * 0
    res[mask] = DthetaDpsi_lambda(psi[mask])

    return res

def DinvConductivityDpsi(psi):
    mask = np.where(psi <= -thr)
    res = np.ones_like(psi) * 0
    res[mask] = DinvConductivityDpsi_lambda(psi[mask])
    #res[res<-1e-6] = -1e-6

    return res


def conductivity(psi):
    mask = np.where(psi <= -thr)
    res = np.ones_like(psi) * K_s
    res[mask] = conductivity_lambda(psi[mask])
    #res[res<] = 1 #np.maximum(conductivity_lambda(psi[mask]), 1e-3)

    return res

def DConductivityDpsi(psi):
    mask = np.where(psi < 0)
    res = np.ones_like(psi) * 0
    res[mask] = DConductivityDpsi_expression_lambda(psi[mask])

    return res


def plot_Van_Genuchten_func():
    psi_plot = -np.concatenate((np.arange(0,1,0.1), np.arange(1,10,1), np.arange(10,100,10), np.arange(100,1000,100)))
    psi_plot = -np.concatenate((np.arange(-1,1,0.1), np.arange(1,10,1)))
    psi_plot = np.arange(-10,1.5,.01)
    #psi_plot = np.arange(-1,.1,0.1)
    plt.figure()
    plt.plot(psi_plot, conductivity(psi_plot))
    plt.show()

    plt.figure()
    plt.plot(psi_plot, 1./conductivity(psi_plot))
    plt.show()
    #plt.savefig("landslide/output_evolutionary/a.png")

    plt.figure()
    plt.plot(psi_plot, DinvConductivityDpsi(psi_plot))
    plt.show()

    plt.figure()
    plt.plot(psi_plot, DthetaDpsi(psi_plot))
    plt.show()

    plt.figure()
    plt.plot(psi_plot, theta(psi_plot))
    plt.show()

# Exponential law
#gamma_water = 10e3 # N/m^3
#alfa_coeff = .1e-4 #1/Pa
#theta_d = 0.04
#theta_s = 0.4
#theta        = lambda psi_var: np.exp(alfa_coeff*gamma_water*psi_var)*(theta_s-theta_d) + theta_d
#conductivity = lambda psi_var: np.exp(alfa_coeff*gamma_water*psi_var)*K_s
#DpsiDtheta   = lambda psi_var: 0*psi_var

there_is_gravity = 1




SlopeHeight = 3. # meters
SlopeAngle = np.radians(45.)

#L_domain = 8*unit_measure_transform_space # 5, 1
width_domain = 5*unit_measure_transform_space
domain_extent_left = -3  # meters
domain_extent_right = 5#L_domain*np.sin(SlopeAngle) + width_domain*np.cos(SlopeAngle) # meters

xx_plot = np.linspace(domain_extent_left, domain_extent_right, 500)

#x_topo = np.linspace(L_domain*np.sin(SlopeAngle), domain_extent_right, 500)
#x_bed  = np.linspace(domain_extent_left, width_domain*np.cos(SlopeAngle), 500)

def topography_func(x):
    #
    x_topo = []
    y_topo = []
    for i in topography_pts():
        x_topo.append(i[0])
        y_topo.append(i[1])

    y = np.interp(x, x_topo, y_topo)


    # ideal slope
    #y = np.maximum(2, 2+np.tan(beta_angle)*x)
    #y = np.minimum(np.tan(np.pi/2-beta_angle)*x, -np.tan(beta_angle)*(x-L_domain*np.sin(beta_angle))+L_domain*np.cos(beta_angle) )
                   
    #bedrock_offset = 2
    #y = bedrock_offset + np.maximum(0,np.minimum(np.tan(SlopeAngle)*x, SlopeHeight))
    return y


def bedrock_func(x):
    x_bed = []
    y_bed = []
    for i in bedrock_pts():
        x_bed.append(i[0])
        y_bed.append(i[1])

    x_bed.reverse()
    y_bed.reverse()
    y    = np.interp(x, x_bed,  y_bed )

    # ideal slope
    #y = x*0-1
    #y = np.maximum(-np.tan(beta_angle)*x, np.tan(np.pi/2-beta_angle)*(x-width_domain*np.cos(beta_angle)) - width_domain*np.sin(beta_angle))
    
    #y = -np.tan(beta_angle)*x
    #y = -1
    return y

def topography_pts(lc: Optional[float] = 0.1):
    pts = []

    # # rectangular domain
    # pts.append([0, 5, 0, lc])
    # pts.append([.1, 5, 0, lc])

    # scarpata
    pts.append([-3, 2, 0, lc])
    pts.append([0, 2, 0, lc])
    pts.append([3, 5, 0, lc])
    pts.append([5, 5, 0, lc])

    # reale
    #df = pandas.read_csv('landslide/Ville-San-Pietro/profilo.csv', sep=";", decimal=",")
    #x_vec = df[df.keys()[0]]
    #top_vec = df[df.keys()[4]]
    #for i in np.arange(np.size(x_vec)):
    #    pts.append([x_vec[i], top_vec[i], 0, lc])


    return pts


def bedrock_pts(lc: Optional[float] = 0.1):
    pts = []

    # # rectangular domain
    # pts.append([0, 0, 0, lc])
    # pts.append([.1, 0, 0, lc])

    # scarpata
    pts.append([5, 0, 0, lc])
    pts.append([-3, 0, 0, lc])

    # reale 
    #df = pandas.read_csv('landslide/Ville-San-Pietro/profilo.csv', sep=";", decimal=",")
    #x_vec = df[df.keys()[0]]
    #bed_vec = df[df.keys()[3]]
    #for i in reversed(np.arange(np.size(x_vec))):
    #    pts.append([x_vec[i], bed_vec[i], 0, lc])

    return pts


#plt.plot(xx_plot, topography_func(xx_plot), color="red")
#plt.plot(xx_plot, bedrock_func(xx_plot), color="blue")
#plt.axis('equal')
#plt.show()


def precipitation_func(x, t):
    # the rain intensity usually is ecpressed in mm/h
    # defines the precipitation rate, we assume it is directed always in the vertical direction
    rain_int = (x[0]>3)*(1e-5*unit_measure_transform_space/unit_measure_transform_time)#*(x[0]>0 and x[0]<3)*(t>(0*3600))
    #p = -np.ones(np.size(x))*rain_int
    return np.array([0, -rain_int, 0])


def mark_boundary_faces(subdomain, P0, RT0):
    bc_value_bedrock = []
    #bc_essential_topography = []

    # Get the boundary faces ids
    boundary_faces_indexes = subdomain.get_boundary_faces()

    topography_line = topography_func(subdomain.face_centers[0, :]) 
    bedrock_line    = bedrock_func   (subdomain.face_centers[0, :]) 

    # get the domain up and bottom
    gamma_topography  = np.isclose(subdomain.face_centers[1, :], topography_line)  
    gamma_bedrock     = np.isclose(subdomain.face_centers[1, :], bedrock_line)

    gamma_updown = np.logical_or(gamma_topography, gamma_bedrock)

    # Gamma_N is the remaining part of the boundary
    gamma_laterals  = gamma_updown.copy()
    gamma_laterals[boundary_faces_indexes] = np.logical_not(gamma_laterals[boundary_faces_indexes])
    #gamma_laterals_1     = np.isclose(subdomain.face_centers[1, :], np.tan(np.pi/2-beta_angle)*(subdomain.face_centers[0, :]-width_domain*np.cos(beta_angle)) - width_domain*np.sin(beta_angle))
    #gamma_laterals_2     = np.isclose(subdomain.face_centers[1, :], np.tan(np.pi/2-beta_angle)*subdomain.face_centers[0, :])

    #plt.plot(x_bed, y_bed, 'd')
    #plt.plot(subdomain.face_centers[0, :],bedrock_line, '*')
    #plt.plot(subdomain.face_centers[0, :],subdomain.face_centers[1, :] , 'o')

    plt.figure()
    plt.plot(subdomain.face_centers[0, :][gamma_bedrock],    subdomain.face_centers[1, :][gamma_bedrock],    'or')
    plt.plot(subdomain.face_centers[0, :][gamma_topography], subdomain.face_centers[1, :][gamma_topography], 'ob')
    plt.plot(subdomain.face_centers[0, :][gamma_laterals],   subdomain.face_centers[1, :][gamma_laterals],   'oy')
    plt.show()



    # Prepare the \hat{\psi} function
    def bc_gamma_d(x, t):
        return -50#initial_pressure_func(np.array([0,0]))
    
    # Prepare the \hat{\psi} function
    #def bc_gamma_topo(x, t):
    #    return precipitation_func(x)

    # Add a lambda function that generates for each time instant the (discretized) natural boundary 
    # conditions for the problem
    bc_value_bedrock = lambda t: - RT0.assemble_nat_bc(subdomain, lambda x: bc_gamma_d(x, t), gamma_bedrock)
    #bc_essential_topography = lambda t: - RT0.assemble_nat_bc(subdomain, lambda x: bc_gamma_topo(x), gamma_topography)
    #bc_essential_topography = np.hstack((gamma_topography, np.zeros(P0.ndof(subdomain), dtype=bool)))
    #bc_value_bedrock = np.hstack((gamma_bedrock,   np.zeros(P0.ndof(subdomain), dtype=bool)))

    return bc_value_bedrock, gamma_laterals, gamma_topography, gamma_bedrock

# def bool_extract_func(xx, input_func):
#     x_c = []
#     N = np.size(xx)
#     i = 0
#     b_extr = xx[-1]
#     while i<(N-1):
#         local_sum = xx[i]
#         k = 1
#         for j in np.arange(i+1, N):
#             l_loc = np.sqrt((xx[i]-xx[j])**2 + (input_func(xx[i])-input_func(xx[j]))**2)>=char_length
#             if l_loc:
#                 x_c = np.append(x_c, local_sum/k)
#                 break
#             else:
#                 local_sum = local_sum + xx[j]
#                 k = k + 1
#         i = j
    
#     if np.sqrt((x_c[-1]-b_extr)**2 + (input_func(x_c[-1])-input_func(b_extr))**2)>=char_length:
#         x_c = np.append(x_c, b_extr)

#     return x_c

pp_gmsh_flag = pp.fracs.gmsh_interface.PhysicalNames

def add_domain(loop):
    surface = gmsh.model.geo.addPlaneSurface([loop])
    add_group(2, surface, pp_gmsh_flag.DOMAIN.value)
    return surface


def add_group(dim, element, name):
    element = [element] if not isinstance(element, list) else element
    group = gmsh.model.addPhysicalGroup(dim, element)
    gmsh.model.setPhysicalName(dim, group, name)
    return group

def add_bd_pt(x, y, z, lc):
    p = gmsh.model.geo.addPoint(x, y, z, lc)
    add_group(0, p, pp_gmsh_flag.DOMAIN_BOUNDARY_POINT.value + str(p))
    return p


def add_bd_line(p0, p1):
    l = gmsh.model.geo.addLine(p0, p1)
    add_group(1, l, pp_gmsh_flag.DOMAIN_BOUNDARY_LINE.value + str(l))
    return l

def create_grid(coords, file_name, dim=2, show=False):

    pts = [add_bd_pt(*c) for c in coords]
    lines_pts = np.vstack((pts, np.roll(pts, -1)))
    lines = [add_bd_line(*p) for p in lines_pts.T]

    loop = gmsh.model.geo.addCurveLoop(lines)
    add_domain(loop)

    gmsh.model.geo.synchronize()

    gmsh.model.mesh.generate(dim)

    gmsh.write(file_name)

    if show:
        gmsh.fltk.run()

    gmsh.finalize()



def generate_msh():
    folder = os.getcwd() + "/landslide/output_evolutionary/"
    file_name = "mesh.msh"

    gmsh.initialize()

    lc = 0.1
    big_lc = 0.1
    dim = 2

    # orientazione in senso orario
    pts = topography_pts(lc) + bedrock_pts(big_lc)

    create_grid(pts, folder + file_name, show=True)

    mdg = pp.fracture_importer.dfm_from_gmsh(folder + file_name, dim=dim)
    subdomain = mdg.subdomains(dim=dim)[0]

    #subdomain = generate_mesh()
    #pp.plot_grid(subdomain, info="all", alpha=0)

    return subdomain



# def generate_mesh():
#     thr_points = 1e-3

#     y_spl = UnivariateSpline(x_topo,topography_func(x_topo),s=0,k=2)
#     y_spl_2d = y_spl.derivative(n=2)

#     bool_extractor = np.abs(y_spl_2d(x_topo))>thr_points
#     bool_extractor[ 0] = True
#     bool_extractor[-1] = True

#     xx = x_topo[bool_extractor]

#     xx = bool_extract_func(xx, topography_func)

#     y_spl = UnivariateSpline(x_bed,bedrock_func(x_bed),s=0,k=2)
#     y_spl_2d = y_spl.derivative(n=2)

#     bool_extractor = np.abs(y_spl_2d(x_bed))>thr_points
#     bool_extractor[ 0] = True
#     bool_extractor[-1] = True

#     xx_bed = x_bed[bool_extractor]

#     xx_bed = bool_extract_func(xx_bed, bedrock_func)

#     #ax = plt.gca()
#     #ax.plot(xx_plot,y_spl_2d(xx_plot))
#     #ax.plot(xx_plot,y_spl(xx_plot))
#     #plt.show()

#     irregular_polygon = [] #[np.array([[xx[0], xx[0]], [bedrock_func(xx[0]), topography_func(xx[0])]])]
#     for i in np.arange(np.size(xx)-1):
#         line = np.array([[xx[i], xx[i+1]], [topography_func(xx[i]), topography_func(xx[i+1])]])
#         irregular_polygon.append(line)

#     line = np.array([[xx[-1], xx_bed[-1]], [topography_func(xx[-1]), bedrock_func(xx_bed[-1])]])
#     if (xx_bed[-1]!=xx[-1]) or (topography_func(xx[-1])!=bedrock_func(xx_bed[-1])):
#         irregular_polygon.append(line) 

#     for i in reversed(np.arange(np.size(xx_bed)-1)):
#         line = np.array([[xx_bed[i+1], xx_bed[i]], [bedrock_func(xx_bed[i+1]), bedrock_func(xx_bed[i])]])
#         irregular_polygon.append(line)

#     line = np.array([[xx_bed[0], xx[0]], [bedrock_func(xx_bed[0]), topography_func(xx[0])]])
#     if (xx_bed[0]!=xx[0]) or (topography_func(xx_bed[0])!=bedrock_func(xx[0])):
#         irregular_polygon.append(line) 
    
#     for i in np.arange(int(np.size(irregular_polygon)/4)):
#         plt.plot(irregular_polygon[i][0], irregular_polygon[i][1], color="red")
#     plt.axis('equal')
#     plt.show()
    

#     domain_from_polytope = pp.Domain(polytope=irregular_polygon)
#     subdomain = pg.grid_from_domain(domain_from_polytope, char_length, as_mdg=False)
#     #subdomain.compute_geometry()

#     #N = 4
#     #subdomain = pp.StructuredTriangleGrid([2*N, 3*N], [domain_extent_right,np.max(topography_func(xx))])
#     #mdg = pp.meshing.subdomains_to_mdg([subdomain])
#     return subdomain



# def compute_size_field(nodes, triangles, oro_interp, d_z_lat_interp, d_z_lon_interp, starting_mesh_size, lat_crop_extent, lon_crop_extent, a=1):
    
    
#     tau = 0.1
    
#     xyz = nodes[triangles]
    
#     lat_interp_node, lon_interp_node = XYZ2LonLat(nodes[:,0], nodes[:,1], nodes[:,2], lon_crop_extent, a)
    
#     z_node = oro_interp((lat_interp_node, lon_interp_node), method='linear')
    
    
    
# #    plt.plot(lon_interp_node, lat_interp_node, 'o')
# #    plt.axis('equal')
# #    plt.show()
    
# #    plt.triplot(x_local, y_local, triangles)
# #    plt.tricontourf(x_local, y_local, triangles, z_node)

# #    plt.tricontourf(lon_interp_node, lat_interp_node, triangles, z_node)
# #    plt.axis('equal')
# #    plt.colorbar()
# #    plt.show()
    
    
#     z_node = z_node[triangles]
    
#     xyz_middle = (xyz[:,0,:] + xyz[:,1,:] + xyz[:,2,:])/3.
#     lat_interp_middle, lon_interp_middle = XYZ2LonLat(xyz_middle[:,0], xyz_middle[:,1], xyz_middle[:,2], lon_crop_extent, a)
#     d_z_lat_middle = d_z_lat_interp((lat_interp_middle, lon_interp_middle), method='nearest')
#     d_z_lon_middle = d_z_lon_interp((lat_interp_middle, lon_interp_middle), method='nearest')
    
    
    
#     N_elements = xyz.shape[0]
#     sf = np.zeros(xyz.shape[0])
#     z_normal = np.array([0.,0.,1.])
    
#     iCubeFace = -1
#     for iTri in range(xyz.shape[0]):
#         xyz_triangle = xyz[iTri]
#         z_node_triangle = z_node[iTri]
        
        
#         z_grad = np.array([0.,0.,0.])
#         for iEdge in range(-2,1):
#             edge_normal = np.cross(z_normal, xyz_triangle[iEdge,:] - xyz_triangle[iEdge+1,:])
            
#             if np.dot(edge_normal, xyz_triangle[iEdge+2,:] - xyz_triangle[iEdge,:]) < 0:
#                 edge_normal *= -1.
            
            
#             z_grad += edge_normal * z_node_triangle[ iEdge+2 ]
            

        
        
#         if (np.allclose(xyz_triangle[0,0],1.) and np.allclose(xyz_triangle[1,0],1.) and np.allclose(xyz_triangle[2,0],1.)):
#             iCubeFace = 0
#         elif (np.allclose(xyz_triangle[0,1],1.) and np.allclose(xyz_triangle[1,1],1.) and np.allclose(xyz_triangle[2,1],1.)):
#             iCubeFace = 1
#         elif (np.allclose(xyz_triangle[0,0],-1.) and np.allclose(xyz_triangle[1,0],-1.) and np.allclose(xyz_triangle[0,0],-1.)):
#             iCubeFace = 2
#         elif (np.allclose(xyz_triangle[0,1],-1.) and np.allclose(xyz_triangle[1,1],-1.) and np.allclose(xyz_triangle[2,1],-1.)):
#             iCubeFace = 3
#         elif (np.allclose(xyz_triangle[0,2],1.) and np.allclose(xyz_triangle[1,2],1.) and np.allclose(xyz_triangle[2,2],1.)):
#             iCubeFace = 4
#         elif (np.allclose(xyz_triangle[0,2],-1.) and np.allclose(xyz_triangle[1,2],-1.) and np.allclose(xyz_triangle[2,2],-1.)):
#             iCubeFace = 5
#         else:
#             print("Not recognized cube-face Id, STOP!")
#             sys.exit()
            
            
            
#         d_z_x_middle, d_z_y_middle = computeGrad(d_z_lat_middle[iTri], d_z_lon_middle[iTri], lon_interp_middle[iTri], lat_interp_middle[iTri], iCubeFace, a)
#         eta_k = np.linalg.norm(z_grad - np.array([d_z_x_middle, d_z_y_middle, 0.]))
        
#         candidate = np.sqrt(1./N_elements/.5 * (tau/(eta_k+1.e-8))**2.)
 
#         sf[ iTri ] = np.maximum(np.minimum(candidate, starting_mesh_size*4), starting_mesh_size/2) #np.maximum(np.minimum(candidate, starting_mesh_size*2), starting_mesh_size/2)
        
    
#     return sf





#np.gradient(topography_func(xx_plot))
#grad = nd.Gradient(topography_func)
#curvature = nd.Gradient(grad)

 
#ax = plt.gca()
#ax.plot(xx_plot, curvature(xx_plot), '--')
#plt.axis('equal')
#plt.xlim([-10,10])
#plt.ylim([0,10])
#plt.show()



#iterator_vect = np.arange(np.size(xx)-1)
#irregular_pentagon = [np.array([[xx[0], xx[0]], [bedrock_func(xx[0]), topography_func(xx[0])]])]
#for i in iterator_vect:
#    line = np.array([[xx[i], xx[i+1]], [topography_func(xx[i]), topography_func(xx[i+1])]]) 
#    irregular_pentagon.append(line)
#line = np.array([[xx[-1], xx[-1]], [topography_func(xx[-1]), bedrock_func(xx[-1])]])
#irregular_pentagon.append(line)

#line = np.array([[xx[-1], xx[0]], [bedrock_func(xx[-1]), bedrock_func(xx[0])]])
#irregular_pentagon.append(line)



# Prepare the domain and its mesh
#subdomain = pp.StructuredTriangleGrid([2*N, 3*N], [2,3])
#subdomain.compute_geometry()
# Convert it to a mixed-dimensional grid
#mdg = pp.meshing.subdomains_to_mdg([subdomain])
#pp.plot_grid(subdomain, info="all", alpha=0)