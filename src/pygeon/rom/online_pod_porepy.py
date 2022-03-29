
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
    """
    """
    def __init__(self):
        super(OnlinePodPorePy, self).__init__()
        """
        """
        # grid:
        self.gb = None
        
        # physical marametres:
        self.physical_param_key = 'flow'
        self.bit_generator = np.random.default_rng()
        self.ymin = None
        self.ymax = None
        self.x_bottom = None
        self.x_top = None
        
        # schifo: non li vado a leggere dal file scritto durante offline. Dati copiati da OfflinePorePy
        self.xmin = -0.5
        self.xmax = 0.5
        self.ymin = -0.5
        self.ymax = 0.5
        self.beta = np.pi/6 # fault tilt angle
        height_domain = self.ymax-self.ymin
        self.x_bottom = -height_domain/2*np.tan(self.beta) # falut
        self.x_top = height_domain/2*np.tan(self.beta) # falut
        
        self.aperture = 1
        self.frac_permeability = 1e0
        
        # full order model:
        self.assembler = None
        
        # mean squared error: 
        self.mse_error = []    
        
        # data for uncertain parameter:
        self.mu_param_data = np.array([[np.log(1e-2), np.log(1e2), np.log(1e-1)], [np.log(1e-1), np.log(1e3), np.log(1e2)]]) # np.array([[min_1, ...], [ max_1, ...]]) of uniform distribution


        
    def generate_grid(self, add_fault=True):
        """ reads file created during the offline stage and genereate the mesh
            input:
            -
            output:
            -
        """ 
        filename_2d = "fracture_2d.csv"
        with open("mesh_parameters_2d.txt", 'r') as file:
            domain_2d = ast.literal_eval(file.readline()) 
            mesh_args = ast.literal_eval(file.readline())

        # filename_3d = "fracture_3d.csv"
        # with open("mesh_parameters_3d.txt", 'r') as file:
        #     domain_3d = ast.literal_eval(file.readline())  #### non serve
        #     mesh_args = ast.literal_eval(file.readline())
        
        # read x_bottom and x_top, required for boundary conditions             # AGGIUNTI NEGLI ATTRIBUTI
        # with open(filename_2d, 'r') as file:
        #     file.readline()                         
        #     coordinates = ast.literal_eval(file.readline())
        #     self.x_bottom = coordinates[1]
        #     self.x_top = coordinates[3]
        #     self.ymin = coordinates[2]
        #     self.ymax = coordinates[4]
    
        network_2d = pp.fracture_importer.network_2d_from_csv(filename_2d, domain=domain_2d)
        # network_3d = pp.fracture_importer.network_3d_from_csv(filename_3d, has_domain=True)
        
        if add_fault:
            self.gb = network_2d.mesh(mesh_args, constraints=np.array([1])) ### ...
        else:
            self.gb = network_2d.mesh(mesh_args)
            
        # self.gb = network_3d.mesh(mesh_args) ############################################ 3D
        
        
        
    def compute_A_rhs(self):
        """
        """
        bit_generator = self.bit_generator
        physical_param_key = self.physical_param_key
        
        snap_g = self.gb.grids_of_dimension(2)[0] # I use the same grid          ### commentare?
        snap_d = self.gb.node_props(snap_g)
        
        # mu_param: (not used)
        min_val = -1
        max_val = 1
        mu_param = self.bit_generator.uniform(min_val, max_val, self.mu_param_data.shape[1])
        for i in range(self.mu_param_data.shape[1]):
            mu_param[i] = self.mu_param_data[0][i] + ( mu_param[i] - min_val ) / ( max_val - min_val ) * ( self.mu_param_data[1][i]-self.mu_param_data[0][i] )

        for g, d in self.gb:
            specific_volumes = np.power(self.aperture, self.gb.dim_max()-g.dim)

            kxx = np.ones(g.num_cells) * specific_volumes
            
            if g.dim < self.gb.dim_max():
                kxx *= self.frac_permeability
            else:
                # k1 = np.exp( mu_param[0] )
                # k2 = np.exp( mu_param[1] )
                k1 = 5e-2                                                       ################## i rather prefer to fix them in the online comp
                k2 = 5e5                                                        #################
                
                x_centers = g.cell_centers[0]
                y_centers = g.cell_centers[1]
                top_right_indices = np.ravel( np.argwhere( np.logical_and(y_centers>=0, x_centers>=self.x_top/self.ymax*y_centers) ) ) 
                top_left_indices = np.ravel( np.argwhere( np.logical_and(y_centers>=0, x_centers<self.x_top/self.ymax*y_centers) ) )
                bottom_right_indices = np.ravel( np.argwhere( np.logical_and(y_centers<0, x_centers>=self.x_bottom/self.ymin*y_centers) ) )
                bottom_left_indices = np.ravel( np.argwhere( np.logical_and(y_centers<0, x_centers<=self.x_bottom/self.ymin*y_centers) ) )
                  
                kxx[top_right_indices] *= k1 
                kxx[top_left_indices] *= k2
                kxx[bottom_right_indices] *= k2
                kxx[bottom_left_indices] *= k1
                
            permeability = pp.SecondOrderTensor(kxx) 
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
            
            if g.dim == self.gb.dim_max():
                d[pp.PRIMARY_VARIABLES] = { 'ph': {'cells': 1, 'edges': 0} }
                d[pp.DISCRETIZATION] = { 'ph': {'diffusion': mpfa, 'source': source_discretization} }
            else:
                d[pp.PRIMARY_VARIABLES] = { 'pl': {'cells': 1, 'edges': 0} }
                d[pp.DISCRETIZATION] = { 'pl': {'diffusion': mpfa, 'source': source_discretization} }
            
            d[pp.DISCRETIZATION_MATRICES] = { physical_param_key: {} } 

        for e, d in self.gb.edges():
            g1, g2 = self.gb.nodes_of_edge(e)
            
            kn = self.frac_permeability / (self.aperture/2)  
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


        dof_manager = pp.DofManager(self.gb)
        print('dof_manager.full_dof = ', dof_manager.full_dof)
        print('sum(dof_manager.full_dof) = ', sum(dof_manager.full_dof))
        
        
        self.assembler = pp.Assembler(self.gb, dof_manager)
        self.assembler.discretize()
        A, b = self.assembler.assemble_matrix_rhs()
        
        return A, b
        
        
        
    def compute_fom_solution(self, A, b):
        """
        """
        sol_fom = sp.sparse.linalg.spsolve(A, b)
        self.assembler.distribute_variable(sol_fom)
        
        return sol_fom
        
        
    
    def write_vtu(self, sol, filename):
        """ write vtu file
            input: 
            - sol (np.array): solution
            - filename (string): name of the vtu file
            output: 
            - file .vtu
            TODO: varaible name is not clear and it is fixed as "ph": make it more flexible
        """
    
        self.assembler.distribute_variable(sol) # even in deformed grid case, we can use a unique assembler
        
        # change name for exporter: from pl to ph:
        for g, d in self.gb:
            if g.dim < self.gb.dim_max():
                d[pp.STATE]["ph"] = d[pp.STATE]["pl"] 
    
        exporter = pp.Exporter(self.gb, filename)
        exporter.write_vtu(["ph"])
        
        return
    
    
    
    def compute_darcy_velocity(self, sol): 
        """
        """
        self.assembler.distribute_variable(sol)
        g = self.gb.grids_of_dimension(2)[0]
        d = self.gb.node_props(g)
        pp.fvutils.compute_darcy_flux(g, keyword=self.physical_param_key, p_name="ph", data=d)
        darcy_flux = d[pp.PARAMETERS][self.physical_param_key]["darcy_flux"]

        discr_P0_flux = pp.MVEM(self.physical_param_key)
        discr_P0_flux.discretize(g, d)
        P0_flux_tpfa = discr_P0_flux.project_flux(g, darcy_flux, d)
        
        return P0_flux_tpfa
        
        
        
