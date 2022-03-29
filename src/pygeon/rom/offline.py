
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
        self.mu_params = np.array([])
        self.mu_params_data = None # ex: np.array([[min_1, ...], [ max_1, ...]]) of uniform distribution
        
        # variable name:
        self.var_names = ['generic_var']
        self.var_dof_indices = 'var_dof_indices'
        
        # full order problem data:
        self.A_list = [] 
        self.b_list = []
        
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
        self.sample_parameters(self.mu_params_data, n_snap_to_generate, save=True)
        self.compute_full_order_matrices_rhss(save=True)
        
        for index in range(n_snap_to_generate):
            
            if verbose == True:
                print(index)
                # and something else...    
            solution, other_outputs = self.solve_one_instance(self.mu_params[index], save=True, index=index)
            
            if vtu_filename != None: 
                gb = other_outputs[0]
                self.export_file_vtu(gb, solution, vtu_filename) 
                input("I'm waiting for you to have a look at the current snapshot...")
            
        return
            
    
    
    def compute_full_order_matrices_rhss(self, save=True):
        """ method to be implemented in the child class by user.
            Suitable if affine parameter property holds.
        """
        self.A_list = []
        self.b_list = []
        
        if save:
            self._save_full_order_matrices_rhss()
            
        return 
             
             
    
    def _save_snapshot(solve_one_instance): 
        """
        """
        def inner(self, mu_params, save, index, format='numpy_savetxt'):
             """
             """
             solution, other = solve_one_instance(self, mu_params)
             
             if save:                                                           
                 if format == 'numpy_savetxt':
                     for var in self.var_names:
                         filename = './data/snap_' + var + '_' + str(index)
                         np.savetxt(filename, solution[var]) 
                 else:
                     print('TODO: save snap in other format')
                 
             return solution, other
         
        return inner
     
         
         
    @_save_snapshot # How much do we like decorators? we can use them more... and maybe we can create a unique and general "_save" method
    def solve_one_instance(self, mu_params):
        """ method to be implemented in the inherited class by user. Calculate one snapshot
        """
        # expeced operations:
        A_list = self.A_list
        b_list = self.b_list
        # assemble fom A and b
        # solve the system A*solution = b
        solution = None
        other = None
        
        return solution, other
        
        
        
    def sample_parameters(self, mu_params_data, n_snap_to_generate, save=True):
        """ sample parameters from distribution
            input: 
            - mu_params_data (list): parameters that define te distribution, 
              example for uniform distributin: np.array([[min_mu_1, min_mu_2, ...],  [max_mu_1, max_mu_2, ...])
            output:
            - mu_params (list): list of parameters
            
            TODO: add other distributions
        """
        
        min_val = -1
        max_val = 1
        self.mu_params = self.bit_generator.uniform(min_val, max_val, (n_snap_to_generate, mu_params_data.shape[1]))
        
        # linear transofrmation from [-1, 1] to [min_val, max_val]:
        for i in range(mu_params_data.shape[1]):
            self.mu_params[:,i] = mu_params_data[0][i] + ( self.mu_params[:,i] - min_val ) / ( max_val - min_val ) * ( mu_params_data[1][i]-mu_params_data[0][i] )
        
        if save:
            self._save_mu_parameter()
            
        return self.mu_params
    
    
    
    def _save_mu_parameter(self, format='numpy_savetxt'):
        """
        """
        if format == 'numpy_savetxt':
            filename = './data/mu_params'
            np.savetxt(filename, self.mu_params)
        else:
            print('TODO: save param in other format')
            
        return
        
        
    
    def _save_full_order_matrices_rhss(self, format='numpy_savetxt'):
        """
        """
        if format == 'numpy_savetxt':
            filename = './data/A_list'
            np.savetxt(filename, self.A_list)
            filename = './data/b_list'
            np.savetxt(filename, self.b_list)
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
        
                        
                
    def _matrix_initialization(self, var, snap, n_snap_to_use):  
        """
        """
        self.data[var][self.snap_matrix] = np.zeros( [snap.size, n_snap_to_use] )
        self.data[var][self.deviation_matrix] = np.zeros( [snap.size, n_snap_to_use] ) # not used
    
        return    



    def compute_svd(self, do_monolithic, save=True):
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
            
            for var in self.var_names[:-1]: ### if it works, improve it TODO #########################################
                del self.data[var]
            self.var_names = [self.var_names[-1]]
            
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
        
        if save:
            self._save_svd_matrices()
        
        
        
    def truncate_U(self, n_modes_to_use=None, threshold=1e-6, save_all_svd_matrices=True):    
        """ truncate modes matrix U
            input:
            - n_modes_to_use (int): number of modes to use = number of column of U ############################### NOT ANYMORE
            ouput:
            -
            TODO: improve it: n_modes_to_use = n_modes_to_use(var)
        """
    
        # if n_modes_to_use not given, create it according to singular values magnitude:
        if not n_modes_to_use :
            n_modes_to_use = {}
            for var in self.var_names:
                n_modes_to_use[var] = np.where( self.data[var][self.S] < threshold )[0]
                if len(n_modes_to_use[var]) != 0: # if not empty vector
                    n_modes_to_use[var] = n_modes_to_use[var][0] + 1
                else:
                    # threshold too low
                    raise NotImplementedError   
                     
        
        # data check:
        if len(n_modes_to_use) != len(self.var_names):
            # warning TODO 
            raise NotImplementedError 
            
        for var in self.var_names:          
            if n_modes_to_use[var] == 1:
                # warning: pay attention to the threshold: too high
                raise NotImplementedError
    
        # variable n_modes_to_use:                                              # please, do not touch these lines
        # n_dofs = {}
        # for var in self.var_names:
        #     if self.data[var][self.U].size != 0:
        #         n_dofs[var] = self.data[var][self.U].shape[0]
        
        specific_n_modes_to_use = {}
        for var in self.var_names:
            # if self.data[var][self.U].size != 0:                              # please, do not touch these lines
            # truncation proportional to the number of dof of the specific variable:
            # specific_n_modes_to_use = int( max( [np.floor(n_modes_to_use*n_dofs[var]/max(n_dofs.values())), 1] ) )
            
            # truncation s.t. n modes = n dofs (no truncation):
            # specific_n_modes_to_use = min( [n_dofs[var], self.data[var][self.U].shape[1]] )
            # equivalently:
            print(n_modes_to_use)
            specific_n_modes_to_use[var] = n_modes_to_use[var]
            print('specific_n_modes_to_use[var] = ', specific_n_modes_to_use[var])
            
            self.data[var][self.U_truncated] = self.data[var][self.U][:, 0:specific_n_modes_to_use[var]]
                
        if save_all_svd_matrices:
            self._save_svd_matrices()
    
        
        
    def _save_svd_matrices(self):
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
    


    def plot_singular_values(self):
        """
        """
        for var in self.var_names:
            fig, ax = plt.subplots()
            ax.plot(self.data[var][self.S], marker='o')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_ylabel('singular values of ' + var)
            ax.set_xlabel('index')
            plt.title(var)
        plt.show()
            
            

    def remove_old_data(self):
        """
        """
        import os
        
        os.system('rm ./data/*' )








