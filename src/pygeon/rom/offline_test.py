
from offline import *
from matplotlib import pyplot as plt



class OfflineTest(OfflineComputations):
    
    def __init__(self):
        super(OfflineTest, self).__init__()
        '''
            See Quarteroni, Manzoni, Negri p.138
        '''
        self.n = 62 # discretization parameter
        
        self.var_names = ['generic_variable']
        
        self.mu_params_data = np.array([[np.log(1e-3)], [np.log(1e1)]])
        
        self.data = {}
        for var in self.var_names:
            self.data[var] =  self._svd_data_dictionary(self.data_keys) 
        self.data[self.parameters] = None
        
        
    @OfflineComputations._save_snapshot
    def solve_one_instance(self, mu_params):
        '''
        '''
        n = self.n
        n_dof = n-2 # bcs of bc
        
        start = 0 + 1/(n-1) # snaps dont have to include bc!
        end = 1 - 1/(n-1)
        x = np.linspace(start, end, num=n_dof)
        sol = ( (3+2*mu_params)*x - x**2 ) / ( 2*(1+mu_params) )
        solution = {}
        
        for var in self.var_names: # useless loop ovv...
            solution[var] = sol
        
        return solution, None
        
            
    



