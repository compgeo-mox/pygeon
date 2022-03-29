
import numpy as np
import scipy as sp
import scipy.interpolate as interpolate
import porepy as pp
from porepy.numerics.fv.fvutils import compute_darcy_flux
from matplotlib import pyplot as plt
import pdb
import os



class OfflineComputations:
    """
        contains all offline functions for both POD and Neural Network approach
    """
    
    def __init__(self):
        """
        """
        
        # uncertain/stochastic parameters:
        self.bit_generator = np.random.default_rng()
        self.mu_param_data = None # ex: np.array([[min_1, ...], [ max_1, ...]]) of uniform distribution
        
        # variable name:
        self.var_names = ['generic_var']
        self.var_dof_indices = 'var_dof_indices'
        
        # snapshot matrix.
        self.snap_matrix = 'snap_matrix'
        self.deviation_matrix = 'deviation_matrix' # not used
        self.mean_snap = 'mean_snap' # not used
        
        # SVD decomposition:
        self.U = 'U'
        self.S = 'S'
        self.Vh = 'Vh'
        self.U_truncated = 'U_truncated'
        self.parameters = 'parameters' # uncertain parameters
        self.data_keys = [self.snap_matrix, self.deviation_matrix, self.mean_snap, self.U, self.S, self.Vh, self.U_truncated]
        self.data = {} # dictionary to store data as snap_matrix, and U, S, Vh from SVD 
        
        for var in self.var_names:                                              
            self.data[var] =  self._svd_data_dictionary(self.data_keys) 
        self.data[self.parameters] = None
        
        

    def _svd_data_dictionary(self, keys): 
        """ initialize data
            input:
            - keys ([str]): list of dictionary keys 
            output:
            - dict (dict): dictionary with empty values
        """
        dict = {}
        for key in keys:
            dict[key] = np.array([])
            
        return dict



    def generate_snapshots(self, n_snap_to_generate, vtu_filename=None, verbose=True):     
        """ generate snapshots saving them as numpy txt format
            input:
            - n_snap_to_generate (int): number of snapshots to generate
            - vtu_filename (str, optional): name of the vtu file
            - verbose (bool): toggle verbosity
            output: snapshots saved as ??? (np.savetxt fromat) files
            -
        """
        for index in range(n_snap_to_generate):
            
            if verbose == True:
                print(index)
                # and something else...    
            mu_param = self.sample_parameters(self.mu_param_data) # select parameters... TODO
            solution, other_outputs = self.solve_one_instance(index, mu_param)
            self.save_snapshot(solution, index)
            self.save_mu_parameter(mu_param, index)
            
            if vtu_filename != None: 
                gb = other_outputs[0]
                self.export_file_vtu(gb, solution, vtu_filename) 
                input("I'm waiting for you to have a look at the current snapshot...")
            
        return
                

    def solve_one_instance(self, index, mu_param):
        """ method to be implemented in the inherited class by user. Calculate one snapshot
        """
        solution = None
        return solution
        
        
        
    def sample_parameters(self, mu_param_data):
        """ sample parameters from distribution
            input: 
            - mu_param_data (list): parameters that define te distribution, 
              example for uniform distributin: np.array([[min_mu_1, min_mu_2, ...],  [max_mu_1, max_mu_2, ...])
            output:
            - mu_param (list): list of parameters
            
            TODO: add other distributions
        """
        
        min_val = -1
        max_val = 1
        mu_param = self.bit_generator.uniform(min_val, max_val, mu_param_data.shape[1])
        
        # linear transofrmation from [-1, 1] to [min_val, max_val]:
        for i in range(mu_param_data.shape[1]):
            mu_param[i] = mu_param_data[0][i] + ( mu_param[i] - min_val ) / ( max_val - min_val ) * ( mu_param_data[1][i]-mu_param_data[0][i] )
        
        return mu_param
        
        
        
    def save_snapshot(self, solution, index, format='numpy_savetxt'):
        """
        """
        if format == 'numpy_savetxt':
            for var in self.var_names:
                filename = './data/snap_' + var + '_' + str(index)
                np.savetxt(filename, solution[var]) 
        else:
            print('TODO: save snap in other format')
            
        return
    
    
    
    def save_mu_parameter(self, params, index, format='numpy_savetxt'):
        """
        """
        if format == 'numpy_savetxt':
            filename = './data/params_' + str(index)
            np.savetxt(filename, params)
        else:
            print('TODO: save param in other format')
            
        return


    def load_snapshots(self, n_snap_to_use, shuffle=False):
        """ fill snapshot matrices
            input:
            - n_snap_to_use (int): number of snapshots contained in the snapshot matrix 
            output:
            -
            
        """
        if shuffle == True:
            print('\nTODO: shuffle')

        # load snapshots:
        for var in self.var_names:
            for i_snap in range(n_snap_to_use):
                snap = np.loadtxt('./data/snap_' + var + '_' + str(i_snap))        
                if i_snap == 0:
                    self._matrix_initialization(var, snap, n_snap_to_use)
                    
                #self.data[var][self.snap_matrix][:, i_snap] = np.array( [ snap[i] for i in self.data[var][self.var_dof_indices] ] )
                self.data[var][self.snap_matrix][:, i_snap] = snap
                
        return
        
        
        
    # def fill_param_matrix(self, index, param):
    #     """
    #     """
    #     self.data[self.parameters][:, index] = param
    # 
    #     return

                            
                
    def _matrix_initialization(self, var, snap, n_snap_to_use):  
        """
        """
        self.data[var][self.snap_matrix] = np.zeros( [snap.size, n_snap_to_use] )
        self.data[var][self.deviation_matrix] = np.zeros( [snap.size, n_snap_to_use] ) # not used
    
        return    



    def compute_svd(self, do_monolithic):
        """ compute SVD decomposition
            input: 
            - do_monolithic (bool): compute "monolithic" or "block" SVD
            output:
            -
            TODO: I'm always adding the 'all' key in self.data, even in the case the problem has only one varible... improve it
        """
        n_snap_to_use = self.data[self.var_names[0]][self.snap_matrix].shape[1] # number of column of any snap matrix
        
        if do_monolithic == True:
            self.var_names.append('all')
            self.data[self.var_names[-1]] = self._svd_data_dictionary(self.data_keys) 
    
            # join the snapshots matrices:
            self.data[self.var_names[-1]][self.snap_matrix] = self.data[self.var_names[0]][self.snap_matrix] # problems with python...
            for var in self.var_names[1:-1]:
                self.data[self.var_names[-1]][self.snap_matrix] = np.append( self.data[self.var_names[-1]][self.snap_matrix], self.data[var][self.snap_matrix], axis=0 )
    
            U, S, Vh = np.linalg.svd( self.data[self.var_names[-1]][self.snap_matrix], full_matrices=False, compute_uv=True, hermitian=False )
            
            self.data[self.var_names[-1]][self.U] = U
            self.data[self.var_names[-1]][self.S] = S
            self.data[self.var_names[-1]][self.Vh] = Vh
    
    
        else:    
            # not used, deviation matrices and SVD computation:
            for var in self.var_names:
                self.data[var][self.mean_snap] = np.mean( self.data[var][self.snap_matrix], axis=1 )
        
                for i in range(n_snap_to_use):
                    self.data[var][self.deviation_matrix][:, i] = self.data[var][self.snap_matrix][:, i] - self.data[var][self.mean_snap] # not used
        
                # U, S, Vh = np.linalg.svd( self.data[var][self.deviation_matrix], full_matrices=False, compute_uv=True, hermitian=False ) 
                U, S, Vh = np.linalg.svd( self.data[var][self.snap_matrix], full_matrices=False, compute_uv=True, hermitian=False )
        
                print('var: ', var, 'U.shape = ', U.shape)
        
                self.data[var][self.U] = U
                self.data[var][self.S] = S
                self.data[var][self.Vh] = Vh
        
        return
        
        
        
    def truncate_U(self, n_modes_to_use):    
        """ truncate modes matrix U
            input:
            - n_modes_to_use (int): number of modes to use = number of column of U
            ouput:
            -
            TODO: improve it: n_modes_to_use = n_modes_to_use(var)
        """
        
        # variable n_modes_to_use:
        n_dofs = {}
        for var in self.var_names:
            if self.data[var][self.U].size != 0:
                n_dofs[var] = self.data[var][self.U].shape[0]
        
        for var in self.var_names:
            if self.data[var][self.U].size != 0:
                
                # truncation proportional to the number of dof of the specific variable:
                specific_n_modes_to_use = int( max( [np.floor(n_modes_to_use*n_dofs[var]/max(n_dofs.values())), 1] ) )
                
                # truncation s.t. n modes = n dofs (no truncation):
                # specific_n_modes_to_use = min( [n_dofs[var], self.data[var][self.U].shape[1]] )
                # equivalently:
                specific_n_modes_to_use = n_modes_to_use
                
                print('specific_n_modes_to_use = ', specific_n_modes_to_use)
                
                self.data[var][self.U_truncated] = self.data[var][self.U][:, 0:specific_n_modes_to_use]

        
        
    def save_svd_matrices(self):
        """
        """
        for var in self.var_names:
            np.savetxt('./data/U_' + var, self.data[var][self.U])
            np.savetxt('./data/S_' + var, self.data[var][self.S])
            np.savetxt('./data/Vh_' + var, self.data[var][self.Vh])
            np.savetxt('./data/U_truncated_' + var, self.data[var][self.U_truncated])
            
        return


        
    def assemble_phi(self):
        """ create transition matrix Phi
            input:
            -
            output:
            -
        """
        list_of_matrices = []
        
        for var in self.var_names:
            U_tr = self.data[var][self.U_truncated]
            # if type(U_tr) != type(None):
            if U_tr.size != 0:
                list_of_matrices.append( U_tr )
            else:
                pass
            
        Phi = sp.linalg.block_diag(*list_of_matrices) 
        
        # identity matrix check:
        if not np.allclose( np.diag(Phi.T@Phi), np.ones(Phi.shape[1]), rtol=1e-10, atol=1e-8 ):
            print('phi.T@phi is NOT diagonal\n')
            raise NotImplementedError
        
        np.savetxt('./data/Phi', Phi) 
        
        return Phi
    
    
    
    def load_phi(self):
        """
        """
        Phi = np.loadtxt('./data/Phi')
        
        return Phi 



    def save_data(self):
        """
        """
        np.savetxt('./data/input_parameters', self.data[self.parameters])
        np.savetxt('./data/snap_matrix', self.data[self.var_names[3]][self.snap_matrix]) # snapshots containing full solution...
        
        return
    


    def remove_old_data(self):
        """
        """
        import os
        
        os.system('rm ./data/*' )








