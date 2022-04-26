import numpy as np
import porepy as pp

class BoundaryConditionNode(pp.params.bc.AbstractBoundaryCondition):

    """ Class to store information on boundary conditions for nodal numerical
        schemes.
    The BCs are specified by node number, and can have type Dirichlet or
    Neumann (Robin may be included later). For details on default values etc.,
    see constructor.
    Attributes:
        num_nodes (int): Number of nodes in the grid
        dim (int): Dimension of the boundary. One less than the dimension of
            the grid.
        is_neu (np.ndarray boolean, size g.num_nodes): Element i is true if
            node i has been assigned a Neumann condition. Tacitly assumes that
            the node is on the boundary. Should be false for internal nodes, as
            well as Dirichlet nodes.
        is_dir (np.ndarary, boolean, size g.num_nodes): Element i is true if
            node i has been assigned a Neumann condition.
    """

    def __init__(self, g, nodes=None, cond=None):
        """Constructor for BoundaryConditionsNode.
        The conditions are specified by node numbers. Nodes that do not get an
        explicit condition will have Neumann conditions assigned.
        Parameters:
            g (grid): For which boundary conditions are set.
            nodes (np.ndarray): Nodes for which conditions are assigned.
            cond (list of str): Conditions on the nodes, in the same order as
                used in nodes. Should be as long as nodes.
        """

        self.num_nodes = g.num_nodes
        self.dim = g.dim - 1

        self.bc_type = "scalar"

        # Find boundary nodes
        bn = g.get_all_boundary_nodes()

        # Keep track of internal boundaries
        self.is_internal = g.tags["fracture_nodes"]

        self.is_neu = np.zeros(self.num_nodes, dtype=np.bool)
        self.is_dir = np.zeros(self.num_nodes, dtype=np.bool)

        # By default, all nodes are Neumann.
        self.is_neu[bn] = True

        if nodes is not None:
            # Validate arguments
            assert cond is not None
            if nodes.dtype == bool:
                if nodes.size != self.num_nodes:
                    raise ValueError(
                        """When giving logical nodes, the size of
                                        array must match number of nodes"""
                    )
                nodes = np.argwhere(nodes)
            if not np.all(np.in1d(nodes, bn)):
                raise ValueError(
                    "Give boundary condition only on the \
                                 boundary"
                )
            domain_boundary_and_tips = np.argwhere(
                np.logical_or(g.tags["domain_boundary_nodes"], g.tags["tip_nodes"])
            )
            if not np.all(np.in1d(nodes, domain_boundary_and_tips)):
                warnings.warn(
                    "You are now specifying conditions on internal \
                              boundaries. Be very careful!"
                )
            if isinstance(cond, str):
                cond = [cond] * nodes.size
            if nodes.size != len(cond):
                raise ValueError("One BC per node")

            for l in np.arange(nodes.size):
                s = cond[l]
                if s.lower() == "neu":
                    pass  # Neumann is already default
                elif s.lower() == "dir":
                    self.is_dir[nodes[l]] = True
                    self.is_neu[nodes[l]] = False
                else:
                    raise ValueError("Boundary should be Dirichlet or Neumann")
