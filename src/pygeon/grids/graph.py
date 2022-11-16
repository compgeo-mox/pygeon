from itertools import combinations

import networkx as nx
import numpy as np
import porepy as pp
import scipy.sparse as sps


class Graph(pp.Grid):
    def __init__(self, graph, dim=2):
        """Construct a pp.Grid like representation of a graph.

        The following informations are stored
        dim: the dimension of the graph, always
        nodes: the coordinate of the nodes
        cell_faces: the map from the cells (graph nodes) to the faces (graph edges)
        num_cells: the number of cells (number of graph nodes)
        num_faces: the number of faces (number of graph edges)
        tags: tags to identify some geometrical quantities
        """
        self.graph = graph

        self.dim = dim
        self.nodes = np.vstack([c for _, c in self.graph.nodes(data="centre")]).T

        self.cell_faces = sps.csc_matrix(
            nx.incidence_matrix(self.graph, oriented=True).T
        )

        self.num_cells = self.cell_faces.shape[1]
        self.num_faces = self.cell_faces.shape[0]

        self.initiate_face_tags()
        self.update_boundary_face_tag()

    def compute_geometry(self):
        """Compute geometric quantities for the graph interpreted as a grid.

        This method initializes class variables describing the graph as grid
        geometry, see class documentation for details.

        The method could have been called from the constructor, however,
        in cases where the graph is modified after the initial construction,
        this may lead to costly, unnecessary computations.
        """
        self.cell_volumes = np.array(
            [vol for _, vol in self.graph.nodes(data="measure", default=1)]
        )
        self.face_areas = np.array(
            [area for _, _, area in self.graph.edges(data="measure", default=1)]
        )

        self.cell_centers = self.nodes
        self.face_centers = self.compute_face_centers()

        self.face_normals = self.cell_centers * self.cell_faces.T

        self.compute_ridges()
        self.tag_tips()
        self.tag_boundary()

    def compute_face_centers(self):
        """
        Compute the face centers on the graph.
        If the face_centers are given in the networkx graph, then they are inherited.
        Else, the face inherits the coordinates of the lower-dimensional neighbor cell.
        """

        face_centers = np.zeros((3, self.num_faces))

        for i, (cell_1, cell_2, c) in enumerate(
            self.graph.edges(data="center", default=None)
        ):
            if c is None:
                cells = np.array([cell_1, cell_2])
                dims = np.array([self.graph.nodes[cell]["dim"] for cell in cells])
                c = self.cell_centers[:, cells[dims == min(dims)][0]]

            face_centers[:, i] = c

        return face_centers

    def compute_ridges(self):
        """
        Computes the ridges and peaks and the corresponding connectivity matrices.
        The ridges in the graph correspond to its cycles and a graph has zero peaks.

        This function create mapping between the faces and the ridges in the graph,
        the following data are created
        num_ridges: the number of rideges (graph cicles)
        num_peaks: the number of peaks (always 0)
        face_ridges: the map from the faces to the ridges
        ridge_peaks: an empty matrix that maps the ridges to peaks
        """
        self.num_ridges = 0
        self.num_peaks = 0
        self.ridge_peaks = sps.csc_matrix((self.num_peaks, self.num_ridges), dtype=int)

        return

        cb = nx.cycle_basis(self.graph)

        incidence = np.abs(self.cell_faces.T)

        n = np.concatenate(cb).size if len(cb) else 0
        rows_I = np.zeros(n, dtype=int)
        cols_J = np.zeros(n, dtype=int)
        data_IJ = np.zeros(n)

        ind = 0
        for (i_c, cycle) in enumerate(cb):
            for i in np.arange(len(cycle)):
                start = cycle[i - 1]
                stop = cycle[i]

                vec = np.zeros(self.graph.number_of_nodes())
                vec[start] = 1
                vec[stop] = 1

                out = incidence.T * vec

                rows_I[ind] = i_c
                cols_J[ind] = np.where(out == 2)[0]
                data_IJ[ind] = np.sign(stop - start)

                ind += 1

        self.num_ridges = len(cb)
        self.face_ridges = sps.csc_matrix(
            (data_IJ, (rows_I, cols_J)), shape=(self.num_ridges, self.num_faces)
        )

        self.num_peaks = 0
        self.ridge_peaks = sps.csc_matrix((self.num_peaks, self.num_ridges), dtype=int)

    def tag_tips(self):
        """
        Dummy tags for the peaks and ridges.
        """

        self.tags["tip_ridges"] = np.zeros(self.num_ridges, dtype=np.bool)
        self.tags["tip_peaks"] = np.zeros(self.num_peaks, dtype=np.bool)

    def tag_boundary(self):
        """
        Tag the boundary cells (i.e. vertices) of the graph.
        """

        tag = np.array(
            [flag for _, flag in self.graph.nodes(data="boundary_flag", default=0)]
        )
        self.tags["domain_boundary_cells"] = tag

    def line_graph(self):
        """
        Construct the line graph associated with the original graph as a pygeon Graph
        """

        return Graph(graph=nx.line_graph(self.graph))

    def set_attribute(self, name, attrs, nodes=None):
        """Set an attribute to the underlying graph with a given name"""

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
        """Collapse the graph removing the nodes with a given dimension.
        From a bipartite graph it is possible to create a dim = 2 graph (only fracures)
        or a dim = 1 (only fracture intersections) graph.
        """

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
        return np.array(
            [n for n in self.graph.nodes if self.graph.nodes[n][name] == value]
        )

    def all_paths(self, start, end, cutoff=None):
        """Compute all the shortest and not shortest paths from the start to the end graph
        nodes."""

        sp = self.shortest_paths(start, end)
        nsp = self.not_shortest_paths(start, end, sp, cutoff)
        return sp, nsp

    def shortest_paths(self, start, end):
        """Compute all the shortest paths from the start to the end graph nodes."""

        sp = nx.all_shortest_paths(self.graph, start, end)
        return np.array(list(sp), dtype=np.object)

    def not_shortest_paths(self, start, end, sp=None, cutoff=None):
        """Compyte all the not shortest paths between two graph nodes.
        It is possible to set a cutoff to make the computation faster leaving out long paths.
        """

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
        """Compute the primary (from the shortest paths) and secondary (from the not shortest
        paths) backbones
        """

        pb = self.primary_backbone(sp, cond)
        sb = self.secondary_backbone(nsp, pb, cond)
        return pb, sb

    def primary_backbone(self, sp, cond=None):
        """Compute the primary back bone of the fracture network,
        which is the list of all nodes in the shortest paths
        """

        # consider a standard condition if not provided
        if cond is None:
            cond = lambda node: len(node.split()) == 1

        pb = []
        # loop on all the paths and add only the one that satisfy a condition
        for path in sp:
            [pb.append(int(node)) for node in path if cond(node)]
        return np.unique(pb)

    def secondary_backbone(self, nsp, pb, cond=None):
        """Compute the secondary back bone of the fracture network,
        which is the list of all nodes not in the shortest paths
        """

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
        """Save the graph to a file.
        TODO: To be improved.
        """

        # make sure that an edge is sorted by dimension
        sort = (
            lambda e: e
            if self.graph.nodes[e[0]]["dim"] > self.graph.nodes[e[1]]["dim"]
            else np.flip(e)
        )
        # collect all the edges
        data = np.array([sort(e) for e in self.graph.edges])
        # remap the values, we assume that are continuously divided into two separate sets
        data -= np.amin(data, axis=0)
        # save to file
        np.savetxt(file_name, data, fmt="%i")
