
import numpy as np
import scipy as sp
import scipy.interpolate as interpolate
import porepy as pp
from matplotlib import pyplot as plt
import pdb
import os
from online_pod import *




class OnlineTest(OnlinePod):
    '''
    '''
    
    def __init__(self):
        super(OnlineTest, self).__init__()
        
        self.A = None
        self.b = None
        
        self.n = 62
        
        
        
    def compute_A_rhs(self):
        '''
        '''
        n_dof = self.n-2 # bcs of bc
        dx = 1/(self.n-1)
        mu_param = self.sample_parameters( np.random.default_rng() )
        
        # discretized solution:
        v1 = 1*np.ones(n_dof-1)
        v2 = -2*np.ones(n_dof)
        v3 = 1*np.ones(n_dof-1)
        self.A = np.diag(v1, k=1) + np.diag(v2) + np.diag(v3, k=-1) # bc already applied
        self.b = -1/(1+mu_param)*dx**2*np.ones(n_dof)
        # add bc:
        self.b[0] += -0 
        self.b[-1] += -1*1
        
        return self.A, self.b
        
    
    def compute_fom_solution(self, A, b):
        '''
        '''    
        n_dof = self.n-2 # bcs of bc
        dx = 1/(self.n-1)
        mu_param = self.sample_parameters( np.random.default_rng() )
        start = 0 + dx # snaps dont have to include bc!
        end = 1 - dx
        x = np.linspace(start, end, num=n_dof)
        sol_analytical = ( (3+2*mu_param)*x - x**2 ) / ( 2*(1+mu_param) )
        
        sol_fom = np.linalg.solve(A, b)
        
        return sol_analytical, sol_fom        
                
        
    def sample_parameters(self, bit_generator):
        '''
        '''
        min = 1e-3
        max = 1e1
        param = bit_generator.uniform(min, max)

        return param



    
