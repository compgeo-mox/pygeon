import numpy as np
import scipy.sparse as sps
import time

import porepy as pp
from geometry import *

def reference_solution(data_key, g, data, discr):
    A, b_flow = discr.assemble_matrix_rhs(g, data)

    rhs_discr = pp.DualScalarSource(data_key)

    rhs_discr.discretize(g, data)
    _, b_rhs = rhs_discr.assemble_matrix_rhs(g, data)

    qp = sps.linalg.spsolve(A, b_flow+b_rhs)

    # Extract the flux and pressure from the solution
    q = discr.extract_flux(g, qp, data)
    p = discr.extract_pressure(g, qp, data)

    return q, p

def main(N=2):

    # 2D
    # g = pp.StructuredTriangleGrid([N]*2, [1]*2)
    # g = pp.CartGrid([N]*2, [1]*2)
    
    # 3D
    g = pp.StructuredTetrahedralGrid([N]*3, [1]*3)
    # g = pp.CartGrid([N]*3, [1]*3)
    
    g.compute_geometry()
    g.edge_nodes, g.face_edges = compute_edges(g)
    
    grad = g.edge_nodes.T
    curl = g.face_edges.T
    div  = g.cell_faces.T

    # Testing
    assert (curl * grad).nnz == 0
    assert (div * curl).nnz == 0
    if g.dim == 3:
        assert (abs(g.edge_nodes) * abs(g.face_edges) - 2 * g.face_nodes).nnz == 0, "Edges do not preserve connectivity"

    # Set up discretization
    perm = pp.SecondOrderTensor(kxx=4*np.ones(g.num_cells), kyy=np.ones(g.num_cells), kxy=np.ones(g.num_cells))
    b_faces = g.tags["domain_boundary_faces"].nonzero()[0]
    bc = pp.BoundaryCondition(g, b_faces, ["dir"]*b_faces.size)
    bc_val = np.zeros(g.num_faces)
    f = g.cell_volumes

    parameters = {"second_order_tensor": perm, "bc": bc, "bc_values": bc_val, "source": f}
    data_key = "flow"
    data = pp.initialize_default_data(g, {}, data_key, parameters)

    discr = pp.RT0(data_key)
    # discr = pp.MVEM(data_key)

    # step 1
    print("step 1")
    start_time = time.time()

    h_scaling = np.mean(g.cell_diameters())**(g.dim - 2)
    BBt = div*h_scaling*div.T

    p_f = sps.linalg.spsolve(BBt, f)
    q_f = h_scaling*div.T*p_f

    print("done in", time.time() - start_time)

    # step 2
    print("step 2")
    start_time = time.time()
    discr.discretize(g, data)

    M = data[pp.DISCRETIZATION_MATRICES][data_key]["mass"]
    A = curl.T*M*curl
    b = - curl.T*M*q_f

    if g.dim == 2:
        A = sps.bmat([[A, grad], [grad.T, 1]], format=A.getformat())
        b = np.append(b, [0.])
        sigma = sps.linalg.spsolve(A, b)[:-1]
    else:
        A += grad*h_scaling*grad.T
        sigma = sps.linalg.spsolve(A, b)

    print("done in", time.time() - start_time)

    # step 3
    print("step 3")
    start_time = time.time()
    q = q_f + curl*sigma
    p = sps.linalg.spsolve(BBt, h_scaling*div*M*q)
    print("done in", time.time() - start_time)

    # verification
    q_ref, p_ref = reference_solution(data_key, g, data, discr)

    print("Pressure error: {:.2E}".format(np.linalg.norm(p-p_ref)))
    print("Flux error:     {:.2E}".format(np.linalg.norm(q-q_ref)))
    print("Mass loss:      {:.2E}".format(np.linalg.norm(div*q - f)))


if __name__ == "__main__":
    np.set_printoptions(linewidth=9999)
    [main(N) for N in [2]]
