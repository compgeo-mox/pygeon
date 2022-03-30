
import numpy as np
import scipy as sp
import scipy.interpolate as interpolate
import porepy as pp
from porepy.numerics.fv.fvutils import compute_darcy_flux
from matplotlib import pyplot as plt
import pdb
import offline
from offline import *

import os



class OfflinePorePy(OfflineComputations):
    """
    """

    def __init__(self):
        super(OfflinePorePy, self).__init__()
        """
        """
        # reference grid:
        self.xmin = -0.5
        self.xmax = 0.5
        self.ymin = -0.5
        self.ymax = 0.5
        self.beta = np.pi/6 # fault tilt angle
        height_domain = self.ymax-self.ymin
        self.x_bottom = -height_domain/2*np.tan(self.beta) # falut
        self.x_top = height_domain/2*np.tan(self.beta) # falut
        self.target_h_bound = 0.5 #
        self.target_h_fracture = self.target_h_min = 0.1 # attento ad usare griglie troopo lasce perche puoi ottenre piu snap di dof
        self.gb = None
        self.ref_g = None

        # physical marametres:
        self.physical_param_key = 'flow'
        self.aperture = 1
        self.frac_permeability = 1

        # data for svd:
        self.var_names = ['ph', 'pl', 'lmbda']
        #self.var_names = ['ph']

        self.snap_matrix = 'snap_matrix'
        self.deviation_matrix = 'deviation_matrix'
        self.mean_snap = 'mean_snap'

        self.data_keys = [self.snap_matrix, self.deviation_matrix, self.mean_snap, self.U, self.S, self.Vh, self.U_truncated]
        for var in self.var_names:
            self.data[var] =  self._svd_data_dictionary(self.data_keys)
        self.data[self.parameters] = None

        # data for uncertain parameter:
        self.mu_params_data = np.array([[np.log(1e-2), np.log(1e2), np.log(1e-1)], [np.log(1e-1), np.log(1e3), np.log(1e2)]]) # np.array([[min_1, ...], [ max_1, ...]]) of uniform distribution, last one is for the fault


    def generate_grid(self, add_fault):
        """
        """
        target_h_bound = self.target_h_bound
        target_h_fracture = self.target_h_fracture
        target_h_min = self.target_h_min

        pts_vect_frac = np.array( [0, self.x_bottom, self.ymin, self.x_top, self.ymax] )
        pts_vect_constr = np.array( [1, self.xmin, 0, self.xmax, 0] ) # constraint line

        pts = [ str(pts_vect_frac[i])+','  for i in range(pts_vect_frac.size-1) ] # create a csv string
        pts.append( str(pts_vect_frac[-1]) )                                    # TODO: find a better way...
        pts.append( '\n' )
        pts.append( str(pts_vect_constr[0])+',' )
        pts.append( str(pts_vect_constr[1])+',' )
        pts.append( str(pts_vect_constr[2])+',' )
        pts.append( str(pts_vect_constr[3])+',' )
        pts.append( str(pts_vect_constr[4]) )                                   # TODO: problems with python...

        # pts_3d_vect = np.array([-fracture_length/2*np.cos(beta), -fracture_length/2*np.cos(beta), -fracture_length/2, # x0, y0, z0
        #                         -fracture_length/2*np.cos(beta), -fracture_length/2*np.cos(beta), fracture_length/2, # x1, y1, z1
        #                         fracture_length/2*np.sin(beta), fracture_length/2*np.cos(beta), fracture_length/2,
        #                         fracture_length/2*np.sin(beta), fracture_length/2*np.cos(beta), -fracture_length/2])
        # pts_3d = [ str(pts_3d_vect[i])+','  for i in range(pts_3d_vect.size-1) ]
        # pts_3d.append( str(pts_3d_vect[-1]) ) # TODO: find a better way...

        fracture_filename_2d = "fracture_2d.csv"
        with open(fracture_filename_2d, 'w') as file:
            file.write('FID,START_X,START_Y,END_X,END_Y\n')
            for i in pts:
                if add_fault == True:
                    file.write(i)
                else:
                    file.write('') # empty file

        # fracture_filename_3d = "fracture_3d.csv"
        # with open(fracture_filename_3d, 'w') as file:
        #     file.write('-0.5, -0.5, -0.5, 0.5, 0.5, 0.5\n') # domain
        #     for i in pts_3d:
        #         if add_fault == True:
        #             file.write(i)
        #         else:
        #             file.write('') # empty file

        domain_2d = {"xmin": self.xmin, "xmax": self.xmax, "ymin": self.ymin, "ymax": self.ymax}
        network_2d = pp.fracture_importer.network_2d_from_csv(fracture_filename_2d, domain=domain_2d)
        # domain_3d = {"xmin": -0.5, "xmax": 0.5, "ymin": -0.5, "ymax": 0.5, "zmin": -0.5, "zmax": 0.5} ######################### already written in csv
        # network_3d = pp.fracture_importer.network_3d_from_csv(fracture_filename_3d, has_domain=True)

        mesh_args = {"mesh_size_bound": target_h_bound,
            "mesh_size_frac": target_h_fracture,
            "mesh_size_min": target_h_min}

        mesh_parameters_filename_2d = "mesh_parameters_2d.txt"
        with open(mesh_parameters_filename_2d, 'w') as file:
            file.write( str(domain_2d)+'\n' )
            file.write( str(mesh_args) )

        # mesh_parameters_filename_3d = "mesh_parameters_3d.txt"
        # with open(mesh_parameters_filename_3d, 'w') as file:
        #     file.write( str(domain_3d)+'\n' )
        #     file.write( str(mesh_args) )

        if add_fault:
            self.gb = network_2d.mesh(mesh_args, constraints=np.array([1])) ### ...
        else:
            self.gb = network_2d.mesh(mesh_args)
        self.ref_g = self.gb.grids_of_dimension(2)[0]

        # self.gb = network_3d.mesh(mesh_args) ############################################################################ 3D
        # self.ref_g = self.gb.grids_of_dimension(3)[0


    @OfflineComputations._save_snapshot
    def solve_one_instance(self, mu_params): 
        """ Calculate one snapshot
            input:
            - mu_params (list): list containig parameters (usually pointed as "mu" in books/papers)
            output:
            - solution (np.array): solution
            - data_out (list): other output (data_out = [gb])
        """
        gb = self.gb
        physical_param_key = self.physical_param_key

        for g, d in gb:
            specific_volumes = np.power(self.aperture, gb.dim_max()-g.dim)

            kxx = np.ones(g.num_cells) * specific_volumes
            self.frac_permeability *= mu_params[2]                               ### non manca un np.exp() ?

            if g.dim < gb.dim_max():
                kxx *= self.frac_permeability
            else:
                k1 = np.exp( mu_params[0] )
                k2 = np.exp( mu_params[1] )

                x_centers = g.cell_centers[0]
                y_centers = g.cell_centers[1]
                top_right_indices = np.ravel( np.argwhere( np.logical_and(y_centers>=0, x_centers>=self.x_top/self.ymax*y_centers) ) ) ####################################################
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
            # left_boundary_indices = np.ravel(                                   ############################## 3D
            # np.logical_and( np.argwhere(g.face_centers[0]<-0.5+eps),
            # np.argwhere(np.logical_and(g.face_centers[2]>-0.5+1e3*eps, g.face_centers[2]<0.5-1e3*eps)) ) )
            #
            # right_boundary_indices = np.ravel(
            # np.logical_and( np.argwhere(g.face_centers[0]>0.5-eps),
            # np.argwhere(np.logical_and(g.face_centers[2]>-0.5+1e3*eps, g.face_centers[2]<0.5-1e3*eps)) ) )
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
            d[pp.PARAMETERS]['vd'] = {'darcy_flux': {}}

            # mathematical parameters:
            mpfa = pp.Mpfa(physical_param_key)
            source_discretization = pp.ScalarSource(physical_param_key)

            if g.dim == gb.dim_max():
                d[pp.PRIMARY_VARIABLES] = { self.var_names[0]: {'cells': 1, 'edges': 0} }
                d[pp.DISCRETIZATION] = { self.var_names[0]: {'diffusion': mpfa, 'source': source_discretization} }
            else:
                d[pp.PRIMARY_VARIABLES] = { self.var_names[1]: {'cells': 1, 'edges': 0} }
                d[pp.DISCRETIZATION] = { self.var_names[1]: {'diffusion': mpfa, 'source': source_discretization} }

            d[pp.DISCRETIZATION_MATRICES] = { physical_param_key: {} }


        for e, d in gb.edges():
            g1, g2 = gb.nodes_of_edge(e)

            # kn = np.exp( mu_params[2] )
            kn = self.frac_permeability / (self.aperture/2) 
            physical_param = {'normal_diffusivity': kn}

            # phys params:
            d[pp.PARAMETERS] = pp.Parameters(keywords=[physical_param_key], dictionaries=[physical_param])

            # math params:
            d[pp.PRIMARY_VARIABLES] = { self.var_names[2]: {'cells': 1, 'edges': 0} }

            flow_coupling_discr = pp.RobinCoupling(physical_param_key, mpfa, mpfa)
            d[pp.COUPLING_DISCRETIZATION] = { 'lambda': {
                                        g1: (self.var_names[1], 'diffusion'),
                                        g2: (self.var_names[0], 'diffusion'),
                                        e: (self.var_names[2], flow_coupling_discr) } }

            d[pp.DISCRETIZATION_MATRICES] = { physical_param_key: {} }


        # solution:
        self.dof_manager = pp.DofManager(gb)

        assembler = pp.Assembler(gb, self.dof_manager)
        assembler.discretize()
        A, b = assembler.assemble_matrix_rhs()
        sol = sp.sparse.linalg.spsolve(A, b)
        solution = {}

        if len(self.var_names) != 1:
            for var in self.var_names:
                solution[var] = sol[self.dof_manager.dof_var(var)]
        else:
            solution[self.var_names[0]] = sol

        data_out = [gb]

        return solution, data_out



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



    def export_file_vtu(self, gb, solution, filename):
        """
        """
        grid = gb.grids_of_dimension(2)[0]
        data = gb.node_props(grid)
        exporter = pp.Exporter(grid, filename)
        ph = solution['ph']
        kxx = data[pp.PARAMETERS][self.physical_param_key]['second_order_tensor'].values[0, 0, ::]
        exporter.write_vtu({'pressure_h': ph, 'kxx': kxx})
        pp.plot_grid(grid, info='c', alpha=0.2)
        
        
