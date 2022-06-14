import numpy as np
import networkx as nx
import scipy.sparse as sps
import gmsh
import meshio

from itertools import combinations

import porepy as pp

class Graph():

    def __init__(self, **kwargs):
        self.graph = nx.Graph()

        if kwargs.get("graph", None):
            self.graph = kwargs.get("graph")
        elif kwargs.get("network", None):
            network = kwargs["network"].copy()
            if isinstance(network, pp.FractureNetwork2d):
                self._construct_2d(network)
            else:
                self._construct_3d(network)
        elif kwargs.get("file_name", None):
            self._from_file(**kwargs)

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

    def to_vtk(self, file_name, info=["dim"], radius=0.1, cylinder_radius=0.025):
        # create a single sphere grid
        sphere_cells, sphere_pts = self._sphere()

        # add all the nodes as sphere
        cells, pts, cell_data = [], [], {i: [] for i in info}
        num_cells = 0
        for n, d in self.graph.nodes(data=True):
            # add the cells, pts and cell_data
            cells.append(sphere_cells + num_cells)
            num_cells += sphere_pts.shape[0]
            pts.append(radius*sphere_pts + d["centre"])
            [cell_data[i].append(d[i]*np.ones(sphere_cells.shape[0])) for i in info]

        # create a single cylinder grid
        #cylinder_cells, cylinder_pts = self._cylinder()
        #for n0, n1 in self.graph.edges():
        #    # add the cells, pts and (zero) cell_data
        #    cells.append(cylinder_cells + num_cells)
        #    num_cells += cylinder_pts.shape[0]

        #    n0_centre = self.graph.nodes[n0]["centre"]
        #    n1_centre = self.graph.nodes[n1]["centre"]
        #    S = np.array([[cylinder_radius, 0, 0], [0, cylinder_radius, 0], [0, 0, n1_centre[2] - n0_centre[2]]])
        #    pts_loc = np.dot(S, cylinder_pts.T).T + n0_centre #############
        #    pts.append(pts_loc)
        #    [cell_data[i].append(np.zeros(cylinder_cells.shape[0])) for i in info]

        # group the cells and vertices
        cells = [("triangle", np.vstack(cells))]
        cell_data = {i: [np.hstack(v)] for i, v in cell_data.items()}

        # create the meshio grid
        meshio_grid = meshio.Mesh(np.vstack(pts), cells, cell_data=cell_data)
        meshio.write(file_name, meshio_grid, binary=True)

    def _construct_2d(self, network):
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
        self.graph = nx.from_edgelist(np.hstack((edgelist_0, edgelist_1)).T)

        # attributes
        attrs = {i: {"dim": 0} for i in np.unique(edges)}
        attrs.update({i: {"dim": 1} for i in fracs})

        # set the attributes to the graph
        nx.set_node_attributes(self.graph, attrs)

    def _construct_3d(self, network):
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

        # first add the graph nodes from the fractures
        for e in np.arange(network.num_frac()):
            node_name = self.name(e)
            self.graph.add_node(node_name)
            attrs[node_name] = {"dim": 2}

        # then add the intersection of fractures as graph node and set the graph edges
        for first, second in zip(network.intersections["first"], network.intersections["second"]):
            node_name = self.name(np.sort([first.index, second.index]))
            # add the intersection as graph node
            self.graph.add_node(node_name)
            attrs[node_name] = {"dim": 1}
            # add the intersection -> first fracture as graph edge
            self.graph.add_edge(self.name(first.index), node_name)
            # add the intersection -> first fracture as graph edge
            self.graph.add_edge(self.name(second.index), node_name)

        # set the attributes to the graph
        nx.set_node_attributes(self.graph, attrs)

    def _from_file(self, **kwargs):
        # read the data I, J from the file
        #frac, intersect = np.loadtxt(kwargs.get("file_name"), dtype=int, unpack=True)
        intersect, frac = np.loadtxt(kwargs["file_name"], dtype=int, unpack=True)

        # shift the index if requested
        frac -= kwargs.get("index_from", 0)
        intersect -= kwargs.get("index_from", 0)

        # if the shape of the network is given read it otherwise guess it
        if kwargs.get("shape", None) is not None:
            shape = np.loadtxt(kwargs["shape"], dtype=int)
        else:
            shape = np.array([np.amax(frac), np.amax(intersect)]) + 1

        # create the adjacency matrix representation of a graph
        frac_to_intersect = sps.coo_matrix((np.ones(frac.size), (frac, intersect)), shape=shape)
        adj = sps.bmat([[None, frac_to_intersect], [frac_to_intersect.T, None]])

        # creates a new graph from an adjacency matrix given as a SciPy sparse matrix
        self.graph = nx.from_scipy_sparse_matrix(adj)

        # set the attribute dim
        max_dim = kwargs.get("max_dim", 2)
        num_frac = frac_to_intersect.shape[0]
        attrs = {i: {"dim": max_dim, "boundary_flag": 0} for i in np.unique(frac)}
        attrs.update({j + num_frac: {"dim": max_dim-1, "boundary_flag": 0} for j in np.unique(intersect)})

        # load the centres if present and add them to be attributes
        if kwargs.get("centres", None) is not None:
            for idf, fc in enumerate(np.loadtxt(kwargs["centres"][0])):
                attrs[idf]["centre"] = fc
            for idi, ic, in enumerate(np.loadtxt(kwargs["centres"][1])):
                attrs[idi + num_frac]["centre"] = ic

        # read the boundary flags: left 1, right 2, top 3, bottom 4, front 5, back 6, internal 0
        if kwargs.get("boundary_flag", None) is not None:
            for idi, flag in enumerate(np.loadtxt(kwargs["boundary_flag"], dtype=int)):
                attrs[idi + num_frac]["boundary_flag"] = flag

        # set the attributes to the graph
        nx.set_node_attributes(self.graph, attrs)

    def _sphere(self):
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)

        # create a sphere object
        model = gmsh.model()
        model.occ.addSphere(0, 0, 0, 1)

        # generate mesh
        model.occ.synchronize()
        model.mesh.generate(2)

        # extract mesh data
        _, _, cells = model.mesh.getElements(dim=2)
        _, pts, _ = model.mesh.getNodes()
        gmsh.finalize()

        return cells[0].reshape(-1, 3) - 1, pts.reshape(-1, 3)

    def _cylinder(self):
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)

        # create a sphere object
        model = gmsh.model()
        model.occ.addCylinder(0, 0, 0, 0, 0, 1, 1)

        # generate mesh
        model.occ.synchronize()
        model.mesh.generate(2)

        # extract mesh data
        _, _, cells = model.mesh.getElements(dim=2)
        _, pts, _ = model.mesh.getNodes()
        gmsh.finalize()

        return cells[0].reshape(-1, 3) - 1, pts.reshape(-1, 3)
