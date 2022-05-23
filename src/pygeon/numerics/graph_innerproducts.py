import networkx as nx
import scipy.sparse as sps

def P0_mass(g, label):
    return sps.diags(g.attr_to_array(label))

def hgrad_mass(g, label=None):
    return nx.laplacian_matrix(g.graph, weight=label)
