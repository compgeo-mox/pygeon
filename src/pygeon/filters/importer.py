import networkx as nx
import numpy as np
import scipy.sparse as sps

import pygeon as pg


def graph_from_file(**kwargs):
    # read the data I, J from the file
    # frac, intersect = np.loadtxt(kwargs.get("file_name"), dtype=int, unpack=True)
    intersect, frac = np.loadtxt(kwargs["file_name"], dtype=int, unpack=True)

    # shift the index if requested
    frac -= kwargs.get("index_from", 0)
    intersect -= kwargs.get("index_from", 0)

    # if the shape of the network is given read it otherwise guess it
    if kwargs.get("shape", None) is not None:
        shape = np.loadtxt(kwargs["shape"], dtype=int)
    else:
        shape = np.array([np.amax(frac), np.amax(intersect)]) + 1
    shape = np.flip(shape)

    # create the adjacency matrix representation of a graph
    frac_to_intersect = sps.coo_matrix(
        (np.ones(frac.size), (frac, intersect)), shape=shape
    )
    adj = sps.bmat([[None, frac_to_intersect], [frac_to_intersect.T, None]])

    # creates a new graph from an adjacency matrix given as a SciPy sparse matrix
    graph = nx.from_scipy_sparse_matrix(adj)

    # set the attribute dim
    max_dim = kwargs.get("max_dim", 2)
    num_frac = frac_to_intersect.shape[0]
    attrs = {i: {"dim": max_dim, "boundary_flag": 0} for i in np.unique(frac)}
    attrs.update(
        {
            j + num_frac: {"dim": max_dim - 1, "boundary_flag": 0}
            for j in np.unique(intersect)
        }
    )

    # load the centres if present and add them to be attributes
    if kwargs.get("centres", None) is not None:
        for idf, fc in enumerate(np.loadtxt(kwargs["centres"][0])):
            attrs[idf]["centre"] = fc
        for (
            idi,
            ic,
        ) in enumerate(np.loadtxt(kwargs["centres"][1])):
            attrs[idi + num_frac]["centre"] = ic

    # read the boundary flags: left 1, right 2, top 3, bottom 4, front 5, back 6, internal 0
    if kwargs.get("boundary_flag", None) is not None:
        for idi, flag in enumerate(np.loadtxt(kwargs["boundary_flag"], dtype=int)):
            attrs[idi + num_frac]["boundary_flag"] = flag

    # read the measure
    if kwargs.get("measures", None) is not None:
        for idf, fm in enumerate(np.loadtxt(kwargs["measures"][0])):
            attrs[idf]["measure"] = fm
        for (
            idi,
            im,
        ) in enumerate(np.loadtxt(kwargs["measures"][1])):
            attrs[idi + num_frac]["measure"] = im

    # set the attributes to the graph
    nx.set_node_attributes(graph, attrs)

    if kwargs.get("measures", None) is not None:
        for first, second, data in graph.edges(data=True):
            if graph.nodes[first]["dim"] > graph.nodes[second]["dim"]:
                data["measure"] = graph.nodes[second]["measure"]
            else:
                data["measure"] = graph.nodes[first]["measure"]

    if kwargs.get("domain", None) is not None:
        return pg.Graph(graph), np.loadtxt(kwargs["domain"])
    else:
        return pg.Graph(graph)
