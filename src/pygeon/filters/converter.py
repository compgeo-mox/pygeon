import networkx as nx
import numpy as np
import porepy as pp


def fracture_network_to_graph(network):
    """Represent the fracture set as a graph, using the networkx data structure.

    Param

    TODO WE SHOULD RETURN A PG GRAPH HERE
    """
    if isinstance(network, pp.FractureNetwork2d):
        return _fracture_network_to_graph_2d(network)
    else:
        return _fracture_network_to_graph_3d(network)

def _fracture_network_to_graph_2d(network):
    """Represent the fracture set as a graph, using the networkx data structure.

    By default the fractures will first be split into non-intersecting branches.
    The former have graph id the fracture id while the latter the ordered list of
    fractures involved in the intersection. Both labels are converted into string.
    Each fracture is a graph node as well as each fracture intersection.
    The graph edges between two graph nodes are the connections (mortars).

    """
    # split the fractures in branches
    network = network.copy_with_split_intersections()
    edges = network.edges

    # define the intersection
    fracs = np.arange(edges.shape[1]) + np.amax(edges) + 1

    # create the edges
    edgelist_0 = np.vstack((fracs, edges[0, :]))
    edgelist_1 = np.vstack((fracs, edges[1, :]))

    # create the graph
    graph = nx.from_edgelist(np.hstack((edgelist_0, edgelist_1)).T)

    # attributes
    attrs = {i: {"dim": 0} for i in np.unique(edges)}
    attrs.update({i: {"dim": 1} for i in fracs})

    # set the attributes to the graph
    nx.set_node_attributes(graph, attrs)

    return graph

def _fracture_network_to_graph_2d(network):
    """Represent the fracture set as a graph, using the networkx data structure.

    By default the network will first calculate intersections.
    Each fracture is a graph node as well as each fracture intersection.
    The former have graph id the fracture id while the latter the ordered list of
    fractures involved in the intersection. Both labels are converted into string.
    Intersections of fracture intersections (0d points) are not represented.
    The graph edges between two graph nodes are the connections (mortars).

    """
    # Find intersections between fractures
    network.find_intersections()

    # attributes
    attrs = {}

    graph = nx.Graph()
    num_frac = network.num_frac()
    # first add the graph nodes from the fractures
    for e in np.arange(num_frac):
        graph.add_node(e)
        attrs[e] = {"dim": 2}

    # then add the intersection of fractures as graph node and set the graph edges
    for idx, (first, second) in enumerate(zip(network.intersections["first"], network.intersections["second"])):
        node_name = idx + num_frac
        # add the intersection as graph node
        graph.add_node(node_name)
        attrs[node_name] = {"dim": 1}
        # add the intersection -> first fracture as graph edge
        graph.add_edge(first.index, node_name)
        # add the intersection -> first fracture as graph edge
        graph.add_edge(second.index, node_name)

    # set the attributes to the graph
    nx.set_node_attributes(graph, attrs)

    return graph
