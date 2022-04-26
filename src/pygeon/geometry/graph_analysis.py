import numpy as np
import networkx as nx
import pyamg

from itertools import combinations

import porepy as pp

class FracturesGraph():

    def __init__(self, network, **kwargs):
        self.network = network.copy()
        self.graph = nx.Graph()

        # set up a way of calling the elements in the graph
        self.name = kwargs.get("graph_name", lambda n: str(n))

        # set the injection and production
        self.injection = np.empty(0)
        self.production = np.empty(0)

        if isinstance(self.network, pp.FractureNetwork2d):
            self._construct_2d()
        else:
            self._construct_3d()

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
            import pdb; pdb.set_trace()
            nx.draw_networkx_labels(graph, pos, labels = dict(data))
        if edge_attr is not None:
            nx.draw_networkx_edge_labels(graph, pos, edge_labels = edge_attr)

        plt.show()

    #def _add_injection_production(self, injection, production, injection_name=None, production_name=None):

    def compute_flux(self, injection, production, weight_label="weight", flux_label="flux", **kwargs):
        # set the injection and production node name
        injection_name = kwargs.get("injection_name", lambda n: str(self.name(n)) + "i")
        production_name = kwargs.get("production_name", lambda n: str(self.name(n)) + "p")

        # set the node names
        injection = np.atleast_1d(injection)
        production = np.atleast_1d(production)

        # set the well contribution
        attrs = {}
        for n in injection:
            w = injection_name(n)
            self.graph.add_node(w)
            self.graph.add_edge(n, w)
            attrs[(n, w)] = {"rhs": 1/injection.size}

        for n in production:
            w = production_name(n)
            self.graph.add_node(w)
            self.graph.add_edge(n, w)
            attrs[(n, w)] = {"rhs": -1/production.size}

        # construct the line graph associated with the original graph
        line_graph = nx.line_graph(self.graph)
        # set the attributes to the graph and get in the ordered way
        nx.set_node_attributes(line_graph, attrs)
        data = line_graph.nodes(data="rhs", default=0)

        # construct the rhs
        rhs = np.fromiter(dict(data).values(), dtype=float)
        # construct the laplacian
        L = nx.laplacian_matrix(line_graph, weight=weight_label)
        # solve the problem
        p = pyamg.solve(L, rhs)

        # construct the incidence matrix and compute the flux
        I = nx.incidence_matrix(line_graph, oriented=True, weight=weight_label)
        flux = I.T*p

        attrs = {}
        for e, f in zip(line_graph.edges(), flux):
            nodes, counts = np.unique(e, return_counts=True)
            node = nodes[counts > 1][0]
            if flux_label in attrs.get(node, {}):
                attrs[node][flux_label] += np.abs(f)
            else:
                attrs[node] = {flux_label: np.abs(f)}

        # remove wells from the graph
        self.graph.remove_nodes_from([injection_name(n) for n in injection])
        self.graph.remove_nodes_from([production_name(n) for n in production])

        # set the attributes to the graph
        nx.set_node_attributes(self.graph, attrs)

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

    def _construct_2d(self):
        """Represent the fracture set as a graph, using the networkx data structure.

        By default the fractures will first be split into non-intersecting branches.
        The former have graph id the fracture id while the latter the ordered list of
        fractures involved in the intersection. Both labels are converted into string.
        Each fracture is a graph node as well as each fracture intersection.
        The graph edges between two graph nodes are the connections (mortars).

        """
        # split the fractures in branches
        self.network = self.network.copy_with_split_intersections()
        edges = self.network.edges

        # attributes
        attrs = {}

        # first add the graph nodes from the fractures
        for e in np.arange(edges.shape[1]):
            node_name = self.name(e)
            self.graph.add_node(node_name)
            attrs[node_name] = {"dim": 1}

        # then add the intersection of fractures as graph node and set the graph edges
        for e in np.arange(edges.shape[1]):
            for j in [0, 1]:
                # check the connections of the current fracture with others at ending point j
                _, conn = np.where(edges == edges[j, e])
                # uniquify and sort the list of connecting fractures
                conn = np.unique(conn)
                # do not consider the special case when the same fracture is found
                if np.any(conn != e):
                    # add the intersection as graph node
                    node_name = self.name(conn)
                    self.graph.add_node(node_name)
                    attrs[node_name] = {"dim": 0}

                    # add the intersection -> fracture as graph edge
                    self.graph.add_edge(self.name(e), self.name(conn))

        # set the attributes to the graph
        nx.set_node_attributes(self.graph, attrs)

    def _construct_3d(self):
        """Represent the fracture set as a graph, using the networkx data structure.

        By default the network will first calculate intersections.
        Each fracture is a graph node as well as each fracture intersection.
        The former have graph id the fracture id while the latter the ordered list of
        fractures involved in the intersection. Both labels are converted into string.
        Intersections of fracture intersections (0d points) are not represented.
        The graph edges between two graph nodes are the connections (mortars).

        """
        # Find intersections between fractures
        self.network.find_intersections()

        # attributes
        attrs = {}

        # first add the graph nodes from the fractures
        for e in np.arange(self.network.num_frac()):
            node_name = self.name(e)
            self.graph.add_node(node_name)
            attrs[node_name] = {"dim": 2}

        # then add the intersection of fractures as graph node and set the graph edges
        for first, second in zip(self.network.intersections["first"], self.network.intersections["second"]):
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
