
import numpy as np
import scipy as sp
import scipy.interpolate as interpolate
import porepy as pp
from online_pod import *
from matplotlib import pyplot as plt
import pdb
import ast
import warnings



class OnlinePodPorePy(OnlinePod):
    '''
    '''
    def __init__(self):
        super(OnlinePodPorePy, self).__init__()
        '''
        '''
        # grid:
        self.gb = None
        
        # physical marametres:
        self.physical_param_key = 'flow'
        self.bit_generator = np.random.default_rng()
        
        # full order model:
        self.assembler = None
        
        # mean squared error: 
        self.mse_error = []    
        
        
        
    def generate_mesh(self):
        '''
            reads file created during the offline stage
            input:
            output:
        ''' 
        filename_2d = "fracture_2d.csv"
        with open("mesh_parameters_2d.txt", 'r') as file:
            domain_2d = ast.literal_eval(file.readline()) 
            mesh_args = ast.literal_eval(file.readline())

        filename_3d = "fracture_3d.csv"
        with open("mesh_parameters_3d.txt", 'r') as file:
            domain_3d = ast.literal_eval(file.readline())  #### non serve
            mesh_args = ast.literal_eval(file.readline())

        network_2d = pp.fracture_importer.network_2d_from_csv(filename_2d, domain=domain_2d)
        network_3d = pp.fracture_importer.network_3d_from_csv(filename_3d, has_domain=True)
        
        self.gb = network_2d.mesh(mesh_args)
        # self.gb = network_3d.mesh(mesh_args) ############################################ 3D
        
        
        
    def compute_A_rhs(self):
        '''
        '''
        gb = self.gb
        bit_generator = self.bit_generator
        physical_param_key = self.physical_param_key
        
        snap_g = gb.grids_of_dimension(2)[0] # I use the same grid          ### commentare?
        snap_d = gb.node_props(snap_g)

        for g, d in gb:
            
            if g.dim < gb.dim_max():
                kxx = np.ones(g.num_cells)
            else:
                min1 = np.log(1e-1) 
                max1 = np.log(1e0)
                min2 = np.log(1e2)
                max2 = np.log(1e3)
                min3 = np.log(1e-1)
                max3 = np.log(1e0)
                m1 = bit_generator.uniform(min1, max1) # uniform distribution has hihger variace wrt normal that is better for training
                m2 = bit_generator.uniform(min2, max2) # theese m are the parameters, input of NN
                m3 = bit_generator.uniform(min3, max3)
                k1 = np.exp(m1)
                k2 = np.exp(m2)
                k3 = np.exp(m3)
                
                x_centers = g.cell_centers[0]
                y_centers = g.cell_centers[1]
                bottom_strip_indices_1 = np.ravel( np.argwhere( y_centers<=0.33333-0.5 ) ) 
                bottom_strip_indices_2 = np.ravel( np.argwhere(np.logical_and(x_centers<0.33333-0.5, y_centers<0.66666-0.5)) )
                bottom_strip_indices_3 = np.ravel( np.argwhere(np.logical_and(x_centers>0.66666-0.5, y_centers<0.66666-0.5)) )
                bottom_strip_indices = np.unique( np.append(bottom_strip_indices_1, np.append(bottom_strip_indices_2, bottom_strip_indices_3)) )
                top_strip_indices = np.ravel( np.argwhere( y_centers>=0.66666-0.5 ) )
                middle_strip_indices = np.setdiff1d( np.arange(g.num_cells), np.append(bottom_strip_indices, top_strip_indices) ) # all - (bottom+top), i.e., what remains
                
                kxx = np.ones(g.num_cells)    
                kxx[bottom_strip_indices] *= k1 
                kxx[middle_strip_indices] *= k2
                kxx[top_strip_indices] *= k3
                
            permeability = pp.SecondOrderTensor(kxx) ##
            eps = 1e-6
            
            left_boundary_indices = np.ravel( np.argwhere(g.face_centers[0]<-0.5+eps) )
            right_boundary_indices = np.ravel( np.argwhere(g.face_centers[0]>0.5-eps) )
            dir_boundary_faces = np.append(left_boundary_indices, right_boundary_indices)
            # bottom_boundary_indices = np.ravel( np.argwhere(g.face_centers[1]<-0.5+eps) )
            # top_boundary_indices = np.ravel( np.argwhere(g.face_centers[1]>0.5-eps) )
            # dir_boundary_faces = np.append(bottom_boundary_indices, top_boundary_indices)
            
            # dir_boundary_faces = np.ravel( np.argwhere(np.logical_or(g.face_centers[0]<-0.5+eps, g.face_centers[0]>0.5-eps)) )
            
            labels = ['dir']*dir_boundary_faces.size
            bc = pp.BoundaryCondition(g, dir_boundary_faces, labels)
            
            bc_val = np.zeros(g.num_faces)        
            bc_val[left_boundary_indices] = np.ones(left_boundary_indices.size)
            # bc_val[top_boundary_indices] = np.ones(top_boundary_indices.size)
            
            f = 0*np.ones(g.num_cells)
            
            physical_param = {'second_order_tensor': permeability, 'bc': bc, 'bc_values': bc_val, 'source': f }

            # physical parameters:
            d[pp.PARAMETERS] = pp.Parameters(keywords=[physical_param_key], dictionaries=[physical_param])

            # mathematical parameters: 
            mpfa = pp.Mpfa(physical_param_key)
            source_discretization = pp.ScalarSource(physical_param_key)
            
            if g.dim == gb.dim_max():
                d[pp.PRIMARY_VARIABLES] = { 'ph': {'cells': 1, 'edges': 0} }
                d[pp.DISCRETIZATION] = { 'ph': {'diffusion': mpfa, 'source': source_discretization} }
            else:
                d[pp.PRIMARY_VARIABLES] = { 'pl': {'cells': 1, 'edges': 0} }
                d[pp.DISCRETIZATION] = { 'pl': {'diffusion': mpfa, 'source': source_discretization} }
            
            d[pp.DISCRETIZATION_MATRICES] = { physical_param_key: {} } 

        for e, d in gb.edges():
            g1, g2 = gb.nodes_of_edge(e)
            
            kn = 1e6
            physical_param = {'normal_diffusivity': kn}
            
            # phys params:
            d[pp.PARAMETERS] = pp.Parameters(keywords=[physical_param_key], dictionaries=[physical_param])
            
            # math params:
            d[pp.PRIMARY_VARIABLES] = { 'lmbda': {'cells': 1, 'edges': 0} }
            
            flow_coupling_discr = pp.RobinCoupling(physical_param_key, mpfa, mpfa)
            d[pp.COUPLING_DISCRETIZATION] = { 'lambda': {
                                        g1: ('pl', 'diffusion'),
                                        g2: ('ph', 'diffusion'),
                                        e: ('lmbda', flow_coupling_discr) } }
            
            d[pp.DISCRETIZATION_MATRICES] = { physical_param_key: {} }


        dof_manager = pp.DofManager(gb)
        print('dof_manager.full_dof = ', dof_manager.full_dof)
        print('sum(dof_manager.full_dof) = ', sum(dof_manager.full_dof))
        

        self.assembler = pp.Assembler(gb, dof_manager)
        self.assembler.discretize()
        A, b = self.assembler.assemble_matrix_rhs()
        
        return A, b
        
        
        
    def compute_fom_solution(self, A, b):
        '''
        '''
        sol_fom = sp.sparse.linalg.spsolve(A, b)
        self.assembler.distribute_variable(sol_fom)
        
        return sol_fom
        
        
    
    def plot_solution(self, grid, solution):
        '''
            TODO
        '''
        pp.plot_grid(grid, solution)
        
        
