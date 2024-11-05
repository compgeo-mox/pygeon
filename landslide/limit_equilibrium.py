#import sys
import numpy as np
import scipy.optimize as optimize
import matplotlib.pyplot as plt

import porepy as pp

from topograhy_file import topography_func, bedrock_func

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


def MorgensternPrice(xa, ya, Ra, sds_in, sds_out, eta, psi, tree, PERM):
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
    

    alpha_out = np.arctan((x_out - xa)/(ya - y_out))
    alpha_in  = alpha_out + 2.*eta; 

    alpha = np.linspace(alpha_out, alpha_in, PERM.Ns+1)
    alpha_center = (alpha[1:] + alpha[0:-1])*.5
    
    x_vect = Ra*np.array([np.sin(alpha), -np.cos(alpha)]) + np.array([xa*np.ones(np.size(alpha)), ya*np.ones(np.size(alpha))])
    x_vect_center = np.array([.5*(x_vect[0][0:-1]+x_vect[0][1:]), .5*(x_vect[1][0:-1]+x_vect[1][1:])])

    delta_x = x_vect[0][1:] - x_vect[0][0:-1]
    delta_y = x_vect[1][1:] - x_vect[1][0:-1]
    
    # patological cases, where multiple non-connected sets are present
    delta_z = topography_func(x_vect_center[0]) - x_vect_center[1]
    selector_func = delta_z<0
    index_c = 0
    if (np.sum(selector_func)>0):
        index_c = x_vect_center[0].tolist().index(np.max(x_vect_center[0][selector_func]))+1
  
    
    delta_x       = delta_x      [index_c:]
    delta_y       = delta_y      [index_c:]
    delta_z       = delta_z      [index_c:]
    alpha_center  = alpha_center [index_c:]
    #x_vect_center_n = np.array([x_vect_center[0][index_c:], x_vect_center[1][index_c:]])
    x_vect_center = np.array([x_vect_center[0][index_c:], x_vect_center[1][index_c:]]) #x_vect_center[index_c:]


    pore_press_slice = alpha_center*0
    pore_press_slice_gamma = alpha_center*0

    # computation of the base pore water pressure 
    for i in np.arange(np.size(alpha_center)):
        n_nodes = tree.search(pp.adtree.ADTNode(99, [x_vect_center[0][i], x_vect_center[1][i]] * 2))
        pore_press_slice[i] = np.mean(psi[n_nodes]) # qui forse mettere una media pesata

    # now compute a set of pwps inside each element to compute a mean unit weight
    for i in np.arange(np.size(alpha_center)):
        #left_base_rect = max(x_vect[1][i], x_vect[1][i+1])
        #n_nodes = tree.search(pp.adtree.ADTNode(99, [x_vect[0][i], x_vect[1][i], x_vect[0][i+1], x_vect[1][i+1]]))

        n_nodes = tree.search(pp.adtree.ADTNode(99, [x_vect_center[0][i]-.5*delta_x[i], x_vect_center[1][i], x_vect_center[0][i]+.5*delta_x[i], x_vect_center[1][i]+delta_z[i]]))
        pore_press_slice_gamma[i] = np.mean(psi[n_nodes])


    sat_deg_r = PERM.theta(pore_press_slice_gamma)/PERM.porosity
    sat_deg_e = (PERM.theta(pore_press_slice) - PERM.theta_r)/(PERM.theta_s - PERM.theta_r)
    
    # scale by density and gravity to get the dimension of pressure from \psi, the density in ton/m^3 is such that we obtain kN/m^2
    pore_press_slice = pore_press_slice*PERM.gamma_water #density_water*gravity_field
    #print(pore_press_slice)

    mois_term = PERM.porosity*sat_deg_r*(1 - PERM.porosity)*PERM.gamma/PERM.gamma_water
    gamma_mixture = (1 - PERM.porosity)*PERM.gamma*(1 + mois_term)
    weight_force = gamma_mixture * delta_z*delta_x
    driving_moment = weight_force*np.sin(alpha_center)
 
    beta_coeff = np.sqrt(delta_x*delta_x + delta_y*delta_y)
    cohesion_tot = PERM.cohesion*beta_coeff
    resisting_moment = cohesion_tot + weight_force*np.cos(alpha_center)*np.tan(PERM.phi)

    FoS_ini = np.sum(resisting_moment)/np.sum(driving_moment) # \dfrac{KN \cdot m^{-1}}{kN \cdot m^{-1}}

     
    # def FoS_func_momentum_Bishp(x):
    #     m_alpha = np.cos(alpha_center) + np.tan(phi)*np.sin(alpha_center)/x
    #     resisting_moment = (cohesion_tot*np.cos(alpha_center) + weight_force*np.tan(phi))/m_alpha
    #     return (np.sum(resisting_moment)/np.sum(driving_moment) - x)
    

    def fx(x):
      return 1
      #fx = np.sin(np.pi*(x-x_vect_center[0][0])/(x_vect_center[0][-1]-x_vect_center[0][0]))
      #return  fx
  
    f_x = fx(x_vect_center[0]) #1.

    def normal_force(x): # this is the total force at the slip surface
        m_alpha = np.cos(alpha_center) + np.tan(PERM.phi)*np.sin(alpha_center)/x[0]
        return((weight_force - cohesion_tot/x[0]*(np.sin(alpha_center) - np.cos(alpha_center)*x[1]*f_x) + sat_deg_e*pore_press_slice*beta_coeff*np.tan(PERM.phi)/x[0]*np.sin(alpha_center) )/(m_alpha + x[1]*f_x*np.cos(alpha_center)*(np.tan(alpha_center) - np.tan(PERM.phi)/x[0]) ))
        #return((weight_force - cohesion_tot/x[0]*(np.sin(alpha_center) - np.cos(alpha_center)*x[1]*f_x) + pore_press_slice*beta_coeff*np.tan(phi)/x[0]*(np.sin(alpha_center)-x[1]*f_x*np.cos(alpha_center)) )/(m_alpha + x[1]*f_x*np.cos(alpha_center)*(np.tan(alpha_center) - np.tan(phi)/x[0]) ))
    
    def FoS_func_momentum(x):
        resisting_moment = cohesion_tot + (normal_force(x)-sat_deg_e*pore_press_slice*beta_coeff)*np.tan(PERM.phi)
        return ((np.sum(resisting_moment)/np.sum(driving_moment)) - x[0])
    
    def FoS_func_force(x):
        resisting_moment = (cohesion_tot + (normal_force(x)-sat_deg_e*pore_press_slice*beta_coeff)*np.tan(PERM.phi))*np.cos(alpha_center)
        return ((np.sum(resisting_moment)/np.sum(normal_force(x)*np.sin(alpha_center))) - x[0]) 
    
    def FoS_func(x):
        return([FoS_func_momentum(x), FoS_func_force(x)])
    
    #def f_func(x): 
    #    return (np.abs(FoS_func_momentum(x) - FoS_func_force(x)) )

    #xv, yv = np.meshgrid(np.linspace(.1, 1, 100), np.linspace(0, 1, 100))
    #zv = (xv, yv)
    #index_res = np.argmin(f_func(zv))
    #FoS = xv.flatten()[index_res]
    #lambda_val = yv.flatten()[index_res]
    
    #[FoS, q_par] = optimize.newton(FoS_func, [FoS, 0.])
    #FoS = optimize.newton(FoS_func_momentum_Bishp, FoS)

    sol = optimize.root(FoS_func, [FoS_ini, 0])
    while sol.message!='The solution converged.' and np.abs(sol.x[0])<1e-4 and np.abs(sol.x[1])<1e-4:
        sol = optimize.root(FoS_func, [FoS_ini, -np.random.rand(1)[0]])

    FoS = sol.x[0]
    lambda_val = sol.x[1]

    #n_val = normal_force(sol.x)

    #lambda_val_cand = np.linspace(0, 1, 10)
    #FoS = FoS_ini
    #for i in np.arange(np.size(lambda_val_cand)):
    #    sol = optimize.root(FoS_func, [FoS, lambda_val_cand[i]])
    #    FoS = sol.x[0]
    #    lambda_val = sol.x[1]
    #    if FoS>0:
    #        break 

    
    return [FoS_ini, FoS, lambda_val]


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
    
    #plt.close()
    #ax = plt.gca()
    #ax.plot(xx_plot, topography_func(xx_plot), '--')
    #circle=plt.Circle((xc,yc),R, fill=False)
    #ax.add_patch(circle)
    #ax.plot((xc), (yc), 'o', color='black')
    #plt.axis('equal')
    ##plt.xlim([sds_out[0],sds_in[0]])
    #plt.xlim([domain_extent_left,domain_extent_right])
    #plt.ylim([np.min(bedrock_func(xx_plot)),np.max(topography_func(xx_plot))])
    ##plt.xlabel('longitude')
    ##plt.ylabel('latitude')
    #plt.show()
    
    #print((xc,yc))
    
    return xc, yc, R, delta

def plot_circ_2pts_tan(trial, domain_extent_left, domain_extent_right):
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

    x_in = trial[0]
    y_in = topography_func(x_in )

    x_out = trial[1]
    y_out = topography_func(x_out )

    eta = trial[2]

    x_mid_point = (x_in + x_out)*.5
    y_mid_point = (y_in + y_out)*.5
    
    delta_length = np.sqrt((x_mid_point-x_in)*(x_mid_point-x_in) + (y_mid_point-y_in)*(y_mid_point-y_in))
    
    alfa_angle = np.arctan((y_in-y_out)/(x_in-x_out))
    R = delta_length/np.sin(eta)
    height = R*np.cos(eta)
    
    xc = x_mid_point - height*np.sin(alfa_angle)
    yc = y_mid_point + height*np.cos(alfa_angle)
    
    delta = alfa_angle+eta

    xx_plot = np.linspace(domain_extent_left, domain_extent_right, 500)
    
    #plt.close()
    fig = plt.figure(figsize=(8, 6))
    ax = plt.gca()
    plt.plot(xx_plot, topography_func(xx_plot), '--')
    circle=plt.Circle((xc,yc),R, fill=False)
    ax.add_patch(circle)
    plt.plot((xc), (yc), 'o', color='black')
    plt.plot(x_in, y_in, 'go')
    plt.plot(x_out, y_out, 'ro')
    plt.axis('equal')
    ##plt.xlim([sds_out[0],sds_in[0]])
    plt.xlim([domain_extent_left,domain_extent_right])
    plt.ylim([np.min(bedrock_func(xx_plot)),np.max(topography_func(xx_plot))])
    ##plt.xlabel('longitude')
    ##plt.ylabel('latitude')
    #plt.show()

    return fig
    
    #print((xc,yc))
    



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
    


def func(trial, psi_sol, tree, PERM):
    x_in, x_out, eta = trial[0], trial[1], trial[2]

    sds_in  = [x_in,  topography_func(x_in )]
    sds_out = [x_out, topography_func(x_out)]
    
    #delta_min = np.arctan((sds_in[1]-sds_out[1])/ (sds_in[0]-sds_out[0]))
    
    xa, ya, Ra, delta = circ_2pts_tan(sds_in, sds_out, eta)
    if ya>sds_in[1] and ya>sds_out[1]:# and delta>delta_min:
        [ff_ini, ff, lambda_val] = MorgensternPrice(xa, ya, Ra, sds_in, sds_out, eta, psi_sol, tree, PERM)
    else:
        ff = 30.
        ff_ini = 30.
        lambda_val = 1.
    print(ff_ini, ff, lambda_val)#, x_in, x_out, eta)
    return ff


def call_optimizer(trial, psi_sol, tree, domain_extent_left, domain_extent_right, PERM):
    zero = optimize.minimize(func, x0=trial, args=(psi_sol, tree, PERM), method='Nelder-Mead', 
                             bounds = ((0, domain_extent_right), #(domain_extent_left, SlopeHeight/np.tan(SlopeAngle))
                                       (domain_extent_left, 3), #(0, domain_extent_right)
                                       (np.radians(5), .5*np.pi)), 
                             options = {'disp':True, 'fatol':0.001, 'maxiter':800, 'return_all':True})
    return zero








