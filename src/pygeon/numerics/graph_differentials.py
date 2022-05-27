import numpy as np
import networkx as nx
import scipy.sparse as sps


def div(g):
    div_mat = nx.incidence_matrix(g.graph, oriented=True)
    return sps.sparse_matrix(div_mat)


def grad(g):
    return -div(g).T


def curl(g):
    cb = nx.cycle_basis(g)

    incidence = np.abs(div(g))

    n = np.concatenate(cb).size
    I = np.zeros(n, dtype=int)
    J = np.zeros(n, dtype=int)
    V = np.zeros(n)

    ind = 0
    for (i_c, cycle) in enumerate(cb):
        for i in np.arange(len(cycle)):
            start = cycle[i - 1]
            stop = cycle[i]

            vec = np.zeros(g.number_of_nodes())
            vec[start] = 1
            vec[stop] = 1

            out = incidence.T * vec

            J[ind] = i_c
            I[ind] = np.where(out == 2)[0]
            V[ind] = np.sign(stop - start)

            ind += 1

    return sps.csc_matrix((V, (I, J)))
