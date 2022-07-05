import numpy as np
import networkx as nx
import scipy.sparse as sps

from itertools import combinations

import porepy as pp
import pygeon as pg

class Graph(pp.Grid):

    def __init__(self, graph):
        self.graph = graph

        self.dim = 2
        self.nodes = np.vstack([c for _, c in self.graph.nodes(data="centre")]).T

        # create the relation cell faces
        self.cell_faces = sps.csc_matrix(nx.incidence_matrix(self.graph, oriented=True).T)

        self.num_cells = self.cell_faces.shape[1]
        self.num_faces = self.cell_faces.shape[0]

        self.initiate_face_tags()
        self.update_boundary_face_tag()

    def compute_geometry(self):
        """Compute geometric quantities for the graph interpreted as a grid.

        This method initializes class variables describing the graph as grid
        geometry, see class documentation for details.

        The method could have been called from the constructor, however,
        in cases where the grid is modified after the initial construction (
        say, grid refinement), this may lead to costly, unnecessary
        computations.
        """

        self._compute_ridges()

        self._tag_tips()

    def _compute_ridges(self):
        cb = nx.cycle_basis(self.graph)

        incidence = np.abs(self.cell_faces.T)

        n = np.concatenate(cb).size
        I = np.zeros(n, dtype=int)
        J = np.zeros(n, dtype=int)
        V = np.zeros(n)

        ind = 0
        for (i_c, cycle) in enumerate(cb):
            for i in np.arange(len(cycle)):
                start = cycle[i - 1]
                stop = cycle[i]

                vec = np.zeros(self.graph.number_of_nodes())
                vec[start] = 1
                vec[stop] = 1

                out = incidence.T * vec

                I[ind] = i_c
                J[ind] = np.where(out == 2)[0]
                V[ind] = np.sign(stop - start)

                ind += 1

        self.num_ridges = len(cb)
        self.face_ridges = sps.csc_matrix((V, (I, J)), shape=(self.num_ridges, self.num_faces))

    def _tag_tips(self):
        """
        Tag the peaks and ridges of a grid bucket that are located on fracture tips.

        """

        self.tags["tip_ridges"] = np.zeros(self.num_ridges, dtype=np.bool)

    def line_graph(self):
        # construct the line graph associated with the original graph
        return Graph(graph=nx.line_graph(self.graph))

    def set_attribute(self, name, attrs, nodes=None):
        if nodes is None:
            nodes = self.graph.nodes
        # create the appropriate data structure
        data = {node: {name: attr} for node, attr in zip(nodes, attrs)}
        # set the attributes to the graph and get in the ordered way
        nx.set_node_attributes(self.graph, data)

    def attr_to_array(self, label, default=0):
        # get the attributes from the graph
        data = self.graph.nodes(data=label, default=default)
        # construct the rhs
        return np.fromiter(dict(data).values(), dtype=float)

    def edges_of_nodes(self, nodes):
        # return the sorted edges of input nodes
        return [tuple(sorted(e)) for e in self.graph.edges(np.atleast_1d(nodes))]

    def collapse(self, dim):
        to_remove = []
        # loop over all the nodes
        for node, data in self.graph.nodes(data=True):
            # select the nodes with dimension dim, remove them and redistribute the edges
            if data["dim"] == dim:
                # add the current node to the list of nodes that need to be removed
                to_remove.append(node)
                # get all the neighbouring nodes of the current node
                # NOTE by defaults all the neighbouring nodes have different dim than
                # the current node, so they are all kept
                neighbours = list(self.graph[node])
                # redistribute the connectivity by adding new edges
                for node1, node2 in combinations(neighbours, 2):
                    self.graph.add_edge(node1, node2)

        # remove all the nodes with dim given
        self.graph.remove_nodes_from(to_remove)

    def nodes_with_attributes(self, name, value):
        return np.array([n for n in self.graph.nodes if self.graph.nodes[n][name] == value])

    def draw(self, graph = None, node_label = None, edge_attr = None):
        import matplotlib.pyplot as plt
        if graph is None:
            graph = self.graph
        pos = nx.spring_layout(graph)
        nx.draw(graph, pos)

        if node_label is None:
            nx.draw_networkx_labels(graph, pos)
        else:
            data = graph.nodes(data = node_label, default = None)
            nx.draw_networkx_labels(graph, pos, labels = dict(data))
        if edge_attr is not None:
            nx.draw_networkx_edge_labels(graph, pos, edge_labels = edge_attr)

        plt.show()

    def all_paths(self, start, end, cutoff=None):
        # compute all the shortest and not shortest paths from the start to the end node
        sp = self.shortest_paths(start, end)
        nsp = self.not_shortest_paths(start, end, sp, cutoff)
        return sp, nsp

    def shortest_paths(self, start, end):
        # compute all the shortest paths from the start to the end node
        sp = nx.all_shortest_paths(self.graph, start, end)
        return np.array(list(sp), dtype=np.object)

    def not_shortest_paths(self, start, end, sp=None, cutoff=None):
        # compute all the shortest paths if are not given
        if sp is None:
            sp = self.shortest_paths(start, end)

        # compute all the paths from the start to the end node
        nsp = nx.all_simple_paths(self.graph, start, end, cutoff)
        nsp = np.array(list(nsp), dtype=np.object)

        # remove from the not shortest paths variables the shortest paths variable
        to_keep = np.ones(nsp.size, dtype=np.bool)
        for s in sp:
            for idx, ns in enumerate(nsp):
                if np.array_equal(np.sort(s), np.sort(ns)):
                    to_keep[idx] = False
        return nsp[to_keep]

    def all_backbone(self, sp, nsp, cond=None):
        # compute the primary (from the shortest paths) and secondary (from the not shortest paths)
        # backbones
        pb = self.primary_backbone(sp, cond)
        sb = self.secondary_backbone(nsp, pb, cond)
        return pb, sb

    def primary_backbone(self, sp, cond=None):
        # compute the primary back bone of the fracture network,
        # which is the list of all nodes in the shortest paths

        # consider a standard condition if not provided
        if cond is None:
            cond = lambda node: len(node.split()) == 1

        pb = []
        #loop on all the paths and add only the one that satisfy a condition
        for path in sp:
            [pb.append(int(node)) for node in path if cond(node)]
        return np.unique(pb)

    def secondary_backbone(self, nsp, pb, cond=None):
        # compute the secondary back bone of the fracture network,
        # which is the list of all nodes not in the shortest paths

        # consider a standard condition if not provided
        if cond is None:
            cond = lambda node: len(node.split()) == 1

        # apply the primary path algorithm to the not shortest paths to get
        # a first version of the secondary back bone
        sb = self.primary_backbone(nsp)

        # remove the elements that are already in the primary backbone from the
        # secondary one
        return np.setdiff1d(sb, pb, assume_unique=True)

    def to_file(self, file_name):
        # make sure that an edge is sorted by dimension
        sort = lambda e: e if self.graph.nodes[e[0]]["dim"] > self.graph.nodes[e[1]]["dim"] else np.flip(e)
        # collect all the edges
        data = np.array([sort(e) for e in self.graph.edges])
        # remap the values, we assume that are continuously divided into two separate sets
        data -= np.amin(data, axis=0)
        # save to file
        np.savetxt(file_name, data, fmt="%i")


