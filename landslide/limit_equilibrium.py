import sys
import numpy as np
import scipy.optimize as optimize
import matplotlib.pyplot as plt

import porepy as pp

from topograhy_file import gamma, cohesion, phi, Ns, gamma_water, gravity_field, topography_func, bedrock_func, SlopeHeight, SlopeAngle, domain_extent_left, domain_extent_right, xx_plot

# Implementation of Morgenstern-Price method for the standard slope problem with the Simplex optimization algorithm 
# to find the failure surface that minimizes the Factor of Safety (FOS)

# plot topography
#ax = plt.gca()
#ax.plot(xx_plot, topography_func(xx_plot), '--')
#plt.axis('equal')
#plt.xlim([-10,10])
#plt.ylim([0,10])
#plt.show()
#sys.exit()


def MorgensternPrice(xa, ya, Ra, sds_in, sds_out, eta, psi, tree):
    """
    xa , ya , Ra = centro e raggio sds
    sds_in , sds_out, eta = ingresso e uscita sds (solo ascissa), eta: angolo 
    gamma = peso totale terreno
    cohesion , phi = coesione e angolo d'attrito
    """
    """
     SLICES
        Ns slices
    """

    # here the code is neglecting the pore water pressure contribution
    x_in = sds_in[0]
    #y_in = sds_in[1]

    x_out = sds_out[0]
    y_out = sds_out[1]
    

    theta_out = np.arctan((x_out - xa)/(ya - y_out))
    theta_in  = theta_out + 2.*eta; 

    theta = np.linspace(theta_out, theta_in, Ns+1)
    theta_center = (theta[1:] + theta[0:-1])*.5
    
    x_vect = Ra*np.array([np.sin(theta), -np.cos(theta)]) + np.array([xa*np.ones(np.size(theta)), ya*np.ones(np.size(theta))])
    x_vect_center = np.array([.5*(x_vect[0][0:-1]+x_vect[0][1:]), .5*(x_vect[1][0:-1]+x_vect[1][1:])])

    delta_x = x_vect[0][1:] - x_vect[0][0:-1]
    delta_y = x_vect[1][1:] - x_vect[1][0:-1]
    
    # patological cases, where multiple non-connected sets are present
    delta_z = topography_func(x_vect_center[0]) - x_vect_center[1]
    selector_func = delta_z<0
    index_c = 0
    if (np.prod(selector_func)>0):
        index_c = x_vect[0].tolist().index(np.max(x_vect_center[0][selector_func]))+1
  
    
    delta_x       = delta_x      [index_c:]
    delta_y       = delta_y      [index_c:]
    delta_z       = delta_z      [index_c:]
    theta_center  = theta_center [index_c:]
    x_vect_center = x_vect_center[index_c:]

    pore_press_slice = theta_center*0

    for i in np.arange(np.size(theta_center)):
        n_nodes = tree.search(pp.adtree.ADTNode(99, [x_vect_center[0][i], x_vect_center[1][i]] * 2))
        pore_press_slice[i] = np.mean(psi[n_nodes]) # qui forse mettere una media pesata

    # scale by density and gravity to get the dimension of pressure from \psi, the density in ton/m^3 is such that we obtain kN/m^2
    pore_press_slice = pore_press_slice*gamma_water #density_water*gravity_field

    print(pore_press_slice)
    
    weight_force = gamma * delta_z*delta_x
    driving_moment = weight_force*np.sin(theta_center)
 
    beta_coeff = np.sqrt(delta_x*delta_x + delta_y*delta_y)
    cohesion_tot = cohesion*beta_coeff
    resisting_moment = cohesion_tot + weight_force*np.cos(theta_center)*np.tan(phi)

    FoS = np.sum(resisting_moment)/np.sum(driving_moment) # \dfrac{KN \cdot m^{-1}}{kN \cdot m^{-1}}

     
    # def FoS_func_momentum_Bishp(x):
    #     m_alpha = np.cos(theta_center) + np.tan(phi)*np.sin(theta_center)/x
    #     resisting_moment = (cohesion_tot*np.cos(theta_center) + weight_force*np.tan(phi))/m_alpha
    #     return (np.sum(resisting_moment)/np.sum(driving_moment) - x)
    
    L = x_in - (x_vect_center[0][index_c]-delta_x[0]*.5)
    def fx(x):
      fx = np.sin(np.pi*x/L)
      return  fx
  
    f_x = fx(x_vect_center[0][index_c:]) #1.

    def normal_force(x): # this is the total force at the slip surface
        m_alpha = np.cos(theta_center) + np.tan(phi)*np.sin(theta_center)/x[0]
        return((weight_force - cohesion_tot/x[0]*(np.sin(theta_center) - np.cos(theta_center)*x[1]*f_x) + pore_press_slice*beta_coeff*np.tan(phi)/x[0]*(np.sin(theta_center)-x[1]*f_x*np.cos(theta_center)) )/(m_alpha + x[1]*f_x*np.cos(theta_center)*(np.tan(theta_center) - np.tan(phi)/x[0]) ))
    
    def FoS_func_momentum(x):
        resisting_moment = cohesion_tot + (normal_force(x)-pore_press_slice*beta_coeff)*np.tan(phi)
        return (np.sum(resisting_moment)/np.sum(driving_moment) - x[0])
    
    def FoS_func_force(x):
        resisting_moment = (cohesion_tot + (normal_force(x)-pore_press_slice*beta_coeff)*np.tan(phi))*np.cos(theta_center)
        return (np.sum(resisting_moment)/np.sum(normal_force(x)*np.sin(theta_center)) - x[0]) 
    
    def FoS_func(x):
        return([FoS_func_momentum(x), FoS_func_force(x)])
    
    #[FoS, q_par] = optimize.newton(FoS_func, [FoS, 0.])
    #FoS = optimize.newton(FoS_func_momentum_Bishp, FoS)
    sol = optimize.root(FoS_func, [FoS, 0.])
    FoS = sol.x[0]

    
    return FoS


def circ_2pts_tan(sds_in , sds_out , eta):
    """
    trova il centro, il raggio e l'arco di cerchio dati 
    i punti di ingresso e uscita della sds e la tangente all'ingresso

    Parameters
    ----------
    sds_in : ingresso superficie di scorrimento
    sds_out : uscita superficie di scorrimento
    slope_in : pendenza superficie di scorrimento

    Returns
    -------
    xc, yc, R : coord-x del centro, coord-y del centro, raggio arco di circ

    """

    x_in = sds_in[0]
    y_in = sds_in[1]

    x_out = sds_out[0]
    y_out = sds_out[1]

    x_mid_point = (x_in + x_out)*.5
    y_mid_point = (y_in + y_out)*.5
    
    delta_length = np.sqrt((x_mid_point-x_in)*(x_mid_point-x_in) + (y_mid_point-y_in)*(y_mid_point-y_in))
    
    alfa_angle = np.arctan((y_in-y_out)/(x_in-x_out))
    R = delta_length/np.sin(eta)
    height = R*np.cos(eta)
    
    xc = x_mid_point - height*np.sin(alfa_angle)
    yc = y_mid_point + height*np.cos(alfa_angle)
    
    delta = alfa_angle+eta
    
    plt.close()

    ax = plt.gca()
    ax.plot(xx_plot, topography_func(xx_plot), '--')
    circle=plt.Circle((xc,yc),R, fill=False)
    ax.add_patch(circle)
    ax.plot((xc), (yc), 'o', color='black')
    plt.axis('equal')
    #plt.xlim([sds_out[0],sds_in[0]])
    plt.xlim([domain_extent_left,domain_extent_right])
    plt.ylim([np.min(bedrock_func(xx_plot)),np.max(topography_func(xx_plot))])
    #plt.xlabel('longitude')
    #plt.ylabel('latitude')
    plt.show()
    
    #print((xc,yc))
    
    return xc, yc, R, delta



# def valid(sds_in , sds_out , eta):
#     if sds_out[0] >= A[0]:
#         eta_min = np.arctan((sds_in[1]-sds_out[1])/ (sds_in[0]-sds_out[0]))
#     else:
#         m_perp_in_0 =  - sds_in[0]/sds_in[1]
#         xc_min = .5*sds_out[0]
#         yc_min = sds_in[1] + m_perp_in_0 * (xc_min - sds_in[0]) # questo era leggermente diverso
#         eta_min = max(np.arctan(- (sds_in[0]  - xc_min ) / (sds_in[1]  - yc_min )), 
#                       np.arctan(sds_in[1]/sds_in[0]))
#     return eta>eta_min
    


def func(trial, psi_sol, tree):
    x_in, x_out, eta = trial[0], trial[1], trial[2]

    sds_in  = [x_in,  topography_func(x_in )]
    sds_out = [x_out, topography_func(x_out)]
    
    #delta_min = np.arctan((sds_in[1]-sds_out[1])/ (sds_in[0]-sds_out[0]))
    
    xa, ya, Ra, delta = circ_2pts_tan(sds_in, sds_out, eta)
    if ya>sds_in[1] and ya>sds_out[1]:# and delta>delta_min:
        ff = MorgensternPrice(xa, ya, Ra, sds_in, sds_out, eta, psi_sol, tree)
    else:
        ff = 30.
    print(ff, x_in, x_out, eta)
    return ff


def call_optimizer(trial, psi_sol, tree):
    zero = optimize.minimize(func, x0=trial, args=(psi_sol, tree), method='Nelder-Mead', 
                             bounds = ((domain_extent_left, SlopeHeight/np.tan(SlopeAngle)),
                                       (0, domain_extent_right), 
                                       (np.radians(5), .5*np.pi)), 
                             options = {'disp':True, 'fatol':0.001, 'maxiter':800, 'return_all':True})
    return zero







