
import numpy as np
import scipy as sp
import scipy.interpolate as interpolate
import porepy as pp
from matplotlib import pyplot as plt
import pdb
import os
from offline import OfflineComputations
    


class OnlinePod: 
    """
    """
    
    def __init__(self):
        """
        """
        # # variable name:
        self.var_names = ['generic_var']
        
        # SVD decomposition and transition matrix
        self.U = 'U'
        self.S = 'S' # not used, can be useful for the determination of the number of snap to use
        self.Phi = 'Phi'
        self.data_keys = [self.U, self.S, self.Phi]
        self.data = {} 
        
        for var in self.var_names:                                             
            self.data[var] =  self.pod_data_dictionary(self.data_keys) 
        
        
        
    def pod_data_dictionary(self, keys): 
        """ initialize data
            input:
            - keys ([str]): list of dictionary keys 
            output:
            - dict (dict): dictionary with empty values
        """
        dict = {}
        for key in keys:
            dict[key] = None
            
        return dict

    
    
    def compute_A_rhs(self):
        """ method to be implemented in the child class by user. Calculate matrix and rrhs
        """
        A = None
        b = None
        
        return A, b
    
    

    def compute_reduced_solution(self, A, b, Phi):
        """ compute reduced solution
            input:
                - A, discretization matrix
                - b, rhs
                - Phi, transition matrix
            output:
                - solution reduced
        """
        
        if not np.allclose( np.diag(Phi.T@Phi), np.ones(Phi.shape[1]), rtol=1e-10, atol=1e-8 ):
            print('Phi.T@Phi is NOT diagonal\n')
            raise NotImplementedError

        # system projection and solution:
        A_reduced = Phi.T @ A @ Phi # full matrices, use sparse matrices if necessary
        b_reduced = Phi.T @ b
        solution_reduced = np.linalg.solve(A_reduced, b_reduced)
        
        return solution_reduced


