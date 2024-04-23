import time
import numpy as np
import scipy.sparse as sps
import porepy as pp 

def TicTocGenerator():
    # Generator that returns time differences
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti # returns the time difference

TicToc = TicTocGenerator() # create an instance of the TicTocGen generator

# This will be the main function through which we define both tic() and toc()
def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print( "Elapsed time: %f seconds.\n" %tempTimeInterval )

def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)


def assembe_inv_preconditioner(subdomain, RT0, p0p0_mass, data) -> sps.csc_matrix:
        """
        Assembles the lumped mass matrix L such that
        B^T L^{-1} B is a TPFA method.

        Args:
            sd (pg.Grid): Grid object or a subclass.
            data (Optional[dict]): Optional dictionary with physical parameters for scaling.

        Returns:
            sps.csc_matrix: The lumped mass matrix.
        """
        lumped_mass_u = RT0.assemble_lumped_matrix(subdomain, data)

        return sps.diags(np.concatenate([1./lumped_mass_u.data, 1./p0p0_mass.data])).tocsc()




class Nitsche_term:

    def __init__(self, subdomain, dof_q, dof_psi, gamma_Nitsche):
        self.dof_q = dof_q
        self.dof_psi = dof_psi
        self.matrix_Nitsche_qq   = sps.dok_matrix((dof_q, dof_q  ))
        self.matrix_Nitsche_qpsi = sps.dok_matrix((dof_q, dof_psi))

        self.flux_top_bc = np.zeros(dof_q)

        self.gamma_Nitsche = gamma_Nitsche

        self.subdomain = subdomain

        self.pressure_head_thr = 0

    def compute_Nitsche_contributions(self, gamma_topography, prec_proj, prev, normal_vector_top):

        Nitsche_argument = (prec_proj[gamma_topography] - prev[:self.dof_q][gamma_topography])*normal_vector_top/self.subdomain.face_areas[gamma_topography] - self.gamma_Nitsche*(prev[-self.dof_psi:][self.subdomain.cell_faces[gamma_topography].nonzero()[1]]/self.subdomain.cell_volumes[self.subdomain.cell_faces[gamma_topography].nonzero()[1]]-self.pressure_head_thr)
        
        self.matrix_Nitsche_qq  [gamma_topography, gamma_topography                                        ] = (Nitsche_argument>0)*1./self.gamma_Nitsche*normal_vector_top*normal_vector_top/self.subdomain.face_areas[gamma_topography]
        self.matrix_Nitsche_qpsi[gamma_topography, self.subdomain.cell_faces[gamma_topography].nonzero()[1]] = (Nitsche_argument>0)*normal_vector_top*1./self.subdomain.cell_volumes[self.subdomain.cell_faces[gamma_topography].nonzero()[1]]
        
        self.flux_top_bc[gamma_topography] = normal_vector_top*1./(self.gamma_Nitsche)*np.maximum(0, Nitsche_argument)

class Nitsche_term_hybrid:

    def __init__(self, subdomain, dof_q, dof_l, gamma_Nitsche):
        self.dof_q = dof_q
        self.dof_l = dof_l
        self.matrix_Nitsche_ll = sps.dok_matrix((dof_l, dof_l))
        self.matrix_Nitsche_lq = sps.dok_matrix((dof_l, dof_q))

        self.flux_top_bc = np.zeros(dof_l)

        self.gamma_Nitsche = gamma_Nitsche

        self.subdomain = subdomain

        self.pressure_head_thr = 0

    def compute_Nitsche_contributions(self, gamma_topography, prec_proj, prev, normal_vector_top):

        Nitsche_argument = (prev[-self.dof_l:]/self.subdomain.face_areas[gamma_topography]-self.pressure_head_thr) - self.gamma_Nitsche*(prec_proj[gamma_topography] - prev[:self.dof_q][gamma_topography])*normal_vector_top/self.subdomain.face_areas[gamma_topography]
        # 1./(gamma_Nitsche*1e7)*np.maximum(0,  ) 
        
        #self.matrix_Nitsche_ll[np.arange(self.dof_l), np.arange(self.dof_l)] = (Nitsche_argument>0)*(-1./self.subdomain.face_areas[gamma_topography])
        #self.matrix_Nitsche_lq[np.arange(self.dof_l), gamma_topography     ] = (Nitsche_argument>0)*(-normal_vector_top/self.subdomain.face_areas[gamma_topography])
        
        self.matrix_Nitsche_ll[np.arange(self.dof_l), np.arange(self.dof_l)] = (Nitsche_argument>0)*(-1./self.gamma_Nitsche/self.subdomain.face_areas[gamma_topography])
        self.matrix_Nitsche_lq[np.arange(self.dof_l), gamma_topography     ] = (Nitsche_argument>0)*(-normal_vector_top/self.subdomain.face_areas[gamma_topography])
        
        self.flux_top_bc = 1./(self.gamma_Nitsche)*np.maximum(0, Nitsche_argument)
    


class Matrix_Computer:
    """
    A simple class used to generate the various matrices required to perform the FEM method.
    """
    def __init__(self, sd, dof_q, dof_psi):
        self.dof_q   = dof_q
        self.dof_psi = dof_psi

        # compute mesh properties useful to compute the RT0 mass matrix
        cells_id = np.arange(0, dof_psi)
        self.faces_id = sd.cell_faces[:,cells_id].indices
        self.faces_id = np.reshape(self.faces_id, (dof_psi, 3))

        faces_val = sd.cell_faces[:,cells_id].data
        faces_val = np.reshape(faces_val, (dof_psi, 3))

        nodes_f0_id = np.reshape(sd.face_nodes[:,self.faces_id[:,0]].indices, (dof_psi, 2))
        nodes_f1_id = np.reshape(sd.face_nodes[:,self.faces_id[:,1]].indices, (dof_psi, 2))
        nodes_f2_id = np.reshape(sd.face_nodes[:,self.faces_id[:,2]].indices, (dof_psi, 2))

        node_p0_id = nodes_f1_id[(nodes_f0_id!=nodes_f1_id) * (np.flip(nodes_f0_id, 1)!=nodes_f1_id)] # node opposed to face 0
        node_p1_id = nodes_f0_id[(nodes_f1_id!=nodes_f0_id) * (np.flip(nodes_f1_id, 1)!=nodes_f0_id)] # node opposed to face 1
        node_p2_id = nodes_f0_id[(nodes_f2_id!=nodes_f0_id) * (np.flip(nodes_f2_id, 1)!=nodes_f0_id)] # node opposed to face 2
        

        nodes_coord_p0 = np.array([sd.nodes[0][node_p0_id], sd.nodes[1][node_p0_id]])
        nodes_coord_p1 = np.array([sd.nodes[0][node_p1_id], sd.nodes[1][node_p1_id]])
        nodes_coord_p2 = np.array([sd.nodes[0][node_p2_id], sd.nodes[1][node_p2_id]])

        nodes_coord_f0 = np.array([sd.nodes[0][nodes_f0_id], sd.nodes[1][nodes_f0_id]])

        # compute the edge mid-points
        x_m0 = .5*(nodes_coord_p0[0]      + nodes_coord_f0[0][:,0])
        x_m1 = .5*(nodes_coord_p0[0]      + nodes_coord_f0[0][:,1])
        x_m2 = .5*(nodes_coord_f0[0][:,0] + nodes_coord_f0[0][:,1])

        y_m0 = .5*(nodes_coord_p0[1]      + nodes_coord_f0[1][:,0])
        y_m1 = .5*(nodes_coord_p0[1]      + nodes_coord_f0[1][:,1])
        y_m2 = .5*(nodes_coord_f0[1][:,0] + nodes_coord_f0[1][:,1])

        self.xx_m0 = np.array([x_m0, y_m0])
        self.xx_m1 = np.array([x_m1, y_m1])
        self.xx_m2 = np.array([x_m2, y_m2])

        # RT= basis functions
        self.base_func_f0 = lambda x: .5*(x - nodes_coord_p0)*faces_val[:,0]/sd.cell_volumes
        self.base_func_f1 = lambda x: .5*(x - nodes_coord_p1)*faces_val[:,1]/sd.cell_volumes
        self.base_func_f2 = lambda x: .5*(x - nodes_coord_p2)*faces_val[:,2]/sd.cell_volumes

        # diagonal integrands
        f_00 = lambda x: self.base_func_f0(x)[0]*self.base_func_f0(x)[0] + self.base_func_f0(x)[1]*self.base_func_f0(x)[1]
        f_11 = lambda x: self.base_func_f1(x)[0]*self.base_func_f1(x)[0] + self.base_func_f1(x)[1]*self.base_func_f1(x)[1]
        f_22 = lambda x: self.base_func_f2(x)[0]*self.base_func_f2(x)[0] + self.base_func_f2(x)[1]*self.base_func_f2(x)[1]

        # off diagonal integrands
        f_01 = lambda x: self.base_func_f0(x)[0]*self.base_func_f1(x)[0] + self.base_func_f0(x)[1]*self.base_func_f1(x)[1]
        f_02 = lambda x: self.base_func_f0(x)[0]*self.base_func_f2(x)[0] + self.base_func_f0(x)[1]*self.base_func_f2(x)[1]
        f_12 = lambda x: self.base_func_f1(x)[0]*self.base_func_f2(x)[0] + self.base_func_f1(x)[1]*self.base_func_f2(x)[1]

        # computing integrals, diag contr
        self.contr_00 = (f_00(self.xx_m0) + f_00(self.xx_m1) + f_00(self.xx_m2))/3.*sd.cell_volumes
        self.contr_11 = (f_11(self.xx_m0) + f_11(self.xx_m1) + f_11(self.xx_m2))/3.*sd.cell_volumes
        self.contr_22 = (f_22(self.xx_m0) + f_22(self.xx_m1) + f_22(self.xx_m2))/3.*sd.cell_volumes

        # computing integrals, off diag contr
        self.contr_01 = (f_01(self.xx_m0) + f_01(self.xx_m1) + f_01(self.xx_m2))/3.*sd.cell_volumes
        self.contr_02 = (f_02(self.xx_m0) + f_02(self.xx_m1) + f_02(self.xx_m2))/3.*sd.cell_volumes
        self.contr_12 = (f_12(self.xx_m0) + f_12(self.xx_m1) + f_12(self.xx_m2))/3.*sd.cell_volumes

        self.row_index = np.concatenate((self.faces_id[:,0], self.faces_id[:,0], self.faces_id[:,0], self.faces_id[:,1], self.faces_id[:,1], self.faces_id[:,1], self.faces_id[:,2], self.faces_id[:,2], self.faces_id[:,2]))
        self.col_index = np.concatenate((self.faces_id[:,0], self.faces_id[:,1], self.faces_id[:,2], self.faces_id[:,0], self.faces_id[:,1], self.faces_id[:,2], self.faces_id[:,0], self.faces_id[:,1], self.faces_id[:,2]))

        self.row_index_dual_C = np.concatenate((self.faces_id[:,0], self.faces_id[:,1], self.faces_id[:,2]))
        self.col_index_dual_C = np.concatenate((cells_id, cells_id, cells_id))

        #sps.coo_matrix((np.ones(9*dof_psi), (self.row_index, self.col_index))).tocsc()
        #mass_matrix_values = np.concatenate((self.contr_00, self.contr_01, self.contr_02, self.contr_01, self.contr_11, self.contr_12, self.contr_02, self.contr_12, self.contr_22))
        #A = sps.coo_matrix((mass_matrix_values, (self.row_index, self.col_index)), shape=(self.dof_q, self.dof_q)).tocsc()
        #Mass_u = RT0.assemble_mass_matrix(subdomain, {pp.PARAMETERS: {key: {"second_order_tensor": pp.SecondOrderTensor(np.ones(dof_psi))}}, pp.DISCRETIZATION_MATRICES: {key: {}},})

        #res = 0
        #for i in np.arange(dof_q):
        #    cur = np.max(np.abs(A.toarray()[i,i]-Mass_u.toarray()[i,i]))
        #    if cur>res and cur>1e-4:
        #        res = cur
        #print(res)

        #np.max((A.toarray()-Mass_u.toarray())[0:10])
        #np.max(np.abs(A.toarray()-Mass_u.toarray()))
        # A.toarray()[63,63]
        # Mass_u.toarray()[63,63]
        # A.toarray()[63][A.toarray()[63]>0]
        # Mass_u.toarray()[63][Mass_u.toarray()[63]>0]


    def compute_RT0_mass_matrix(self, cond_coeff):
        mass_matrix_values = np.concatenate((self.contr_00/cond_coeff, self.contr_01/cond_coeff, self.contr_02/cond_coeff, self.contr_01/cond_coeff, self.contr_11/cond_coeff, self.contr_12/cond_coeff, self.contr_02/cond_coeff, self.contr_12/cond_coeff, self.contr_22/cond_coeff))
        return sps.coo_matrix((mass_matrix_values, (self.row_index, self.col_index)), shape=(self.dof_q, self.dof_q)).tocsc()

    def compute_dual_C(self, sd, q, cond_coeff):

        #
        q_0 = q[self.faces_id[:,0]]
        q_1 = q[self.faces_id[:,1]]
        q_2 = q[self.faces_id[:,2]]

        # diagonal integrands
        f_0 = lambda x: self.base_func_f0(x)[0]*(q_0*self.base_func_f0(x)[0] + q_1*self.base_func_f1(x)[0] + q_2*self.base_func_f2(x)[0]) + self.base_func_f0(x)[1]*(q_0*self.base_func_f0(x)[1] + q_1*self.base_func_f1(x)[1] + q_2*self.base_func_f2(x)[1])
        f_1 = lambda x: self.base_func_f1(x)[0]*(q_0*self.base_func_f0(x)[0] + q_1*self.base_func_f1(x)[0] + q_2*self.base_func_f2(x)[0]) + self.base_func_f1(x)[1]*(q_0*self.base_func_f0(x)[1] + q_1*self.base_func_f1(x)[1] + q_2*self.base_func_f2(x)[1])
        f_2 = lambda x: self.base_func_f2(x)[0]*(q_0*self.base_func_f0(x)[0] + q_1*self.base_func_f1(x)[0] + q_2*self.base_func_f2(x)[0]) + self.base_func_f2(x)[1]*(q_0*self.base_func_f0(x)[1] + q_1*self.base_func_f1(x)[1] + q_2*self.base_func_f2(x)[1])

        # computing integrals
        contr_0 = (f_0(self.xx_m0) + f_0(self.xx_m1) + f_0(self.xx_m2))/3.*sd.cell_volumes
        contr_1 = (f_1(self.xx_m0) + f_1(self.xx_m1) + f_1(self.xx_m2))/3.*sd.cell_volumes
        contr_2 = (f_2(self.xx_m0) + f_2(self.xx_m1) + f_2(self.xx_m2))/3.*sd.cell_volumes

        mass_matrix_values = np.concatenate((contr_0*cond_coeff/sd.cell_volumes, contr_1*cond_coeff/sd.cell_volumes, contr_2*cond_coeff/sd.cell_volumes))
        return sps.coo_matrix((mass_matrix_values, (self.row_index_dual_C, self.col_index_dual_C)), shape=(self.dof_q, self.dof_psi)).tocsc()
