import numpy as np
import networkx as nx

def div(g):
    return nx.incidence_matrix(g.graph, oriented=True)

def grad(g):
    return div(g).T

def put_a_name(p, graph, line_graph, flux_label):
    # compute the flux
    flux = grad(line_graph)*p

    attrs = {}
    for e, f in zip(line_graph.graph.edges(), flux):
        nodes, counts = np.unique(e, return_counts=True)
        node = nodes[counts > 1][0]
        if flux_label in attrs.get(node, {}):
            attrs[node][flux_label] = np.abs(f)
        else:
            attrs[node] = {flux_label: np.abs(f)}

    # set the attributes to the graph
    nx.set_node_attributes(graph.graph, attrs)


