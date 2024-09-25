""" Module for spanning tree computation. """

from typing import Optional, Union

import numpy as np
import porepy as pp
import scipy.sparse as sps

import pygeon as pg


class SpanningTree:
    """
    Class that can perform a spanning tree solve, aka a "grid sweep".
    Useful to rapidly compute a flux field that balances a mass source.

    Attributes:
        system (sps.csc_matrix): The matrix used in the solve,
            which is triangular up to row/column permutations.
        expand (sps.csc_matrix): Expansion matrix from tree to global ordering.

        div (sps.csc_matrix): the divergence operator on the associated mdg.
        starting_cells (np.ndarray): the first cells of the spanning tree.
        starting_faces (np.ndarray): the first faces of the spanning tree.
        tree (sps.csc_array): The incidence matrix of the spanning tree.

    Methods:
        setup_system(mdg: pg.MixedDimensionalGrid, flagged_faces: np.ndarray) -> None:
            Sets up the linear system for solving the spanning tree problem.

        find_starting_faces(mdg: pg.MixedDimensionalGrid) -> int:
            Find the starting face for the spanning tree if None is provided.

        find_starting_cells() -> int:
            Find the starting cell for the spanning tree.

        compute_tree() -> sps.csc_array:
            Construct a spanning tree of the elements.

        flag_tree_faces() -> np.ndarray:
            Flag the faces in the mesh that correspond to edges of the tree.

        solve(f: np.ndarray) -> np.ndarray:
            Perform a spanning tree solve to compute a conservative flux field
            for given mass source.

        solve_transpose(rhs: np.ndarray) -> np.ndarray:
            Post-process the pressure by performing a transposed solve.

        visualize_2d(mdg: pg.MixedDimensionalGrid, fig_name: Optional[str] = None) -> None:
            Create a graphical illustration of the spanning tree superimposed on the grid.
    """

    def __init__(
        self,
        mdg: pg.MixedDimensionalGrid,
        starting_faces: Optional[Union[str, np.ndarray, int]] = "first_bdry",
    ) -> None:
        """
        Initializes a SpanningTree object.

        Args:
            mdg (pg.MixedDimensionalGrid): The mixed-dimensional grid.
            starting_faces (Union[np.ndarray, int, str], optional):
                - "first_bdry" (default): Choose the first boundary face.
                - "all_bdry": Choose all boundary faces.
                - np.array or int: Indices of the starting faces.
        """
        self.div = pg.div(mdg)

        self.starting_faces = self.find_starting_faces(mdg, starting_faces)

        self.add_outside_cell()

        self.tree = self.compute_tree()
        self.flagged_faces = self.flag_tree_faces()

        self.remove_outside_cell()

        self.starting_cells = self.find_starting_cells()
        self.setup_system(mdg, self.flagged_faces)

    def setup_system(
        self, mdg: pg.MixedDimensionalGrid, flagged_faces: np.ndarray
    ) -> None:
        """
        Sets up the linear system for solving the spanning tree problem.

        Args:
            mdg (pg.MixedDimensionalGrid): The mixed-dimensional grid.
            flagged_faces (np.ndarray): Array of flagged faces.

        Returns:
            None
        """
        self.expand = pg.numerics.linear_system.create_restriction(
            flagged_faces
        ).T.tocsc()

        # Save the sparse LU decomposition of the system
        self.system = pg.cell_mass(mdg) @ self.div @ self.expand
        self.system_splu = sps.linalg.splu(self.system)

    def find_starting_faces(
        self,
        mdg: pg.MixedDimensionalGrid,
        starting_faces: Union[str, np.ndarray, int],
    ) -> np.ndarray:
        """
        Find the starting face for the spanning tree.
        By default, this method returns the indices of the boundary faces of the mesh.

        Args:
            mdg (pg.MixedDimensionalGrid): The mixed-dimensional grid object.
            starting_faces:
                - "first_bdry" (default): Choose the first boundary face.
                - "all_bdry": Choose all boundary faces.
                - np.array or int: Indices of the starting faces.

        Returns:
            np.ndarray or int: The indices of the starting faces for the spanning tree.

        Raises:
            KeyError: if starting_faces does not have the right type
        """

        if isinstance(starting_faces, str):
            # Extract top-dimensional domain
            sd = mdg.subdomains(dim=mdg.dim_max())[0]
            bdry_faces = np.where(sd.tags["domain_boundary_faces"])[0]

            # The default case
            if starting_faces == "first_bdry":
                return bdry_faces[[0]]

            elif starting_faces == "all_bdry":
                return bdry_faces

            else:
                raise KeyError(
                    "Not a supported string. Input must be first_bdry or all_bdry"
                )

        # Scalars and arrays
        else:
            return np.atleast_1d(starting_faces)

    def find_starting_cells(self) -> np.ndarray:
        """
        Find the starting cell for the spanning tree.

        Returns:
            np.ndarray: The indices of the starting cells.
        """
        return self.div.tocsc()[:, self.starting_faces].indices

    def add_outside_cell(self) -> None:
        outside_cell = np.zeros((1, self.div.shape[1]))
        outside_cell[0, self.starting_faces] = -np.sum(
            self.div[:, self.starting_faces], axis=0
        )

        self.div = sps.vstack([self.div, outside_cell], format="csc")

    def remove_outside_cell(self) -> None:
        self.div = self.div[:-1, :]
        self.tree = self.tree[:-1, :-1]

    def compute_tree(self) -> sps.csc_array:
        """
        Construct a spanning tree of the elements.

        Returns:
            sps.csc_array: The computed spanning tree as a compressed sparse column matrix.
        """
        incidence = self.div @ self.div.T

        tree = sps.csgraph.breadth_first_tree(
            incidence, incidence.shape[0] - 1, directed=False
        )

        return sps.csc_array(tree)

    def flag_tree_faces(self) -> np.ndarray:
        """
        Flag the faces in the mesh that correspond to edges of the tree.

        Returns:
            np.ndarray: A boolean array indicating the flagged faces in the mesh.
        """
        # Extract start/end cells for each edge of the tree
        c_start, c_end, _ = sps.find(self.tree)

        # Find the mesh faces that correspond to tree edges
        rows = np.hstack((c_start, c_end))
        cols = np.hstack([np.arange(c_start.size)] * 2)
        vals = np.ones_like(rows)

        face_finder = sps.csc_array(
            (vals, (rows, cols)), shape=(self.div.shape[0], self.tree.nnz)
        )
        face_finder = np.abs(self.div.T) @ face_finder
        I, J, V = sps.find(face_finder)

        _, index = np.unique(J[V == 2], return_index=True)
        tree_faces = I[V == 2][index]

        # Flag the relevant mesh faces in the grid
        flagged_faces = np.zeros(self.div.shape[1], dtype=bool)
        flagged_faces[tree_faces] = True

        # Update the starting faces by removing the ones that are unnecessary
        self.starting_faces = self.starting_faces[flagged_faces[self.starting_faces]]

        return flagged_faces

    def solve(self, f: np.ndarray) -> np.ndarray:
        """
        Perform a spanning tree solve to compute a conservative flux field
        for given mass source.

        Args:
            f (np.ndarray): Mass source, integrated against PwConstants.

        Returns:
            np.ndarray: the post-processed flux field
        """
        return self.expand @ self.system_splu.solve(f)

    def assemble_SI(self) -> sps.sparray:
        """
        Assembles the operator S_I as a sparse array.
        NOTE: This will be slow for large systems.
        If you only need the action of S_I, consider using self.solve() instead.

        Returns:
            sps.sparray: S_I, a right inverse of the B-operator
        """
        identity = np.eye(self.system.shape[0])
        inv_system = self.system_splu.solve(identity)

        return self.expand @ sps.csc_array(inv_system)

    def solve_transpose(self, rhs: np.ndarray) -> np.ndarray:
        """
        Post-process the pressure by performing a transposed solve.

        Args:
            rhs (np.ndarray): Right-hand side, usually the mass matrix times the flux
                              minus boundary terms.

        Returns:
            np.ndarray: the post-processed pressure field
        """
        return self.system_splu.solve(self.expand.T.tocsc() @ rhs, "T")

    def visualize_2d(
        self,
        mdg: pg.MixedDimensionalGrid,
        fig_name: Optional[str] = None,
        draw_grid=True,
        draw_tree=True,
        draw_complement=False,
    ):
        """
        Create a graphical illustration of the spanning tree superimposed on the grid.

        Args:
            mdg (pg.MixedDimensionalGrid) The object representing the grid.
            fig_name (Optional[str], optional). The name of the figure file to save the
                visualization.
        """
        import matplotlib.pyplot as plt
        import networkx as nx

        assert mdg.dim_max() == 2

        fig_num = 1

        # Draw grid
        if draw_grid:
            pp.plot_grid(
                mdg,
                alpha=0,
                fig_num=fig_num,
                plot_2d=True,
                if_plot=False,
            )

        # Define the figure and axes
        fig = plt.figure(fig_num)

        if draw_grid:
            pp_ax = fig.gca()
            ax = fig.add_subplot(111)
            ax.set_xlim(pp_ax.get_xlim())
            ax.set_ylim(pp_ax.get_ylim())
        else:
            ax = fig.gca()

        # Draw the tree that spans all cells
        if draw_tree:
            graph = nx.from_scipy_sparse_array(self.tree)
            cell_centers = np.hstack([sd.cell_centers for sd in mdg.subdomains()])
            node_color = ["blue"] * cell_centers.shape[1]
            for sc in self.starting_cells:
                node_color[sc] = "green"

            nx.draw(
                graph,
                cell_centers[: mdg.dim_max(), :].T,
                node_color=node_color,
                node_size=40,
                edge_color="red",
                ax=ax,
            )

            # Add connections from the roots to the starting faces
            num_bdry = len(self.starting_faces)
            bdry_graph = sps.diags(
                np.ones(num_bdry),
                num_bdry,
                shape=(2 * num_bdry, 2 * num_bdry),
            )
            graph = nx.from_scipy_sparse_array(bdry_graph)
            sd = mdg.subdomains()[0]
            node_centers = np.hstack(
                (
                    sd.face_centers[: mdg.dim_max(), self.starting_faces],
                    sd.cell_centers[: mdg.dim_max(), self.starting_cells],
                )
            ).T

            nx.draw(
                graph,
                node_centers,
                node_size=0,
                edge_color="red",
                ax=ax,
            )

        # Draw the tree that spans all nodes
        if draw_complement:
            curl = pg.curl(mdg)[~self.flagged_faces, :]
            incidence = curl.T @ curl
            incidence -= sps.triu(incidence)

            graph = nx.from_scipy_sparse_array(incidence)
            nodes = np.hstack([sd.nodes for sd in mdg.subdomains()])

            node_color = ["black"] * nodes.shape[1]

            nx.draw(
                graph,
                nodes[: mdg.dim_max(), :].T,
                node_color=node_color,
                node_size=30,
                edge_color="purple",
                width=1.5,
                ax=ax,
            )

        plt.draw()
        if fig_name is not None:
            plt.savefig(fig_name, bbox_inches="tight", pad_inches=0.1)

        plt.close()


class SpanningTreeElasticity(SpanningTree):
    """
    Represents a class for computing the spanning tree for the elastic problem.

    Attributes:
        expand (sps.csc_matrix): The expanded matrix for spanning tree computation.
        system (sps.csc_matrix): The computed system matrix.

    Methods:
        setup_system(self, mdg: pg.MixedDimensionalGrid, flagged_faces: np.ndarray) -> None:
            Set up the system for the spanning tree algorithm.

        compute_expand(self, sd: pg.Grid, flagged_faces: np.ndarray) -> sps.csc_matrix:
            Compute the expanded matrix for spanning tree computation.

        compute_system(self, sd: pg.Grid) -> sps.csc_matrix:
            Computes the system matrix for the given grid.
    """

    def setup_system(
        self, mdg: Union[pg.MixedDimensionalGrid, pg.Grid], flagged_faces: np.ndarray
    ) -> None:
        """
        Set up the system for the spanning tree algorithm.

        Args:
            mdg (pg.MixedDimensionalGrid): The mixed-dimensional grid.
            flagged_faces (np.ndarray): Array of flagged faces.

        Returns:
            None
        """
        # Extract top-dimensional domain
        sd = mdg.subdomains(dim=mdg.dim_max())[0]
        self.expand = self.compute_expand(sd, flagged_faces)

        # Save the sparse LU decomposition of the system
        self.system = self.compute_system(sd)
        self.system_splu = sps.linalg.splu(self.system)

    def compute_expand(self, sd: pg.Grid, flagged_faces: np.ndarray) -> sps.csc_matrix:
        """
        Compute the expanded matrix for spanning tree computation.

        Args:
            sd (pg.Grid): The grid object.
            flagged_faces (np.ndarray): Array of flagged faces.

        Returns:
            sps.csc_matrix: The expanded matrix for spanning tree computation.
        """
        key = "tree"
        bdm1 = pg.BDM1(key)

        # this operator fix one dof per face of the bdm space to capture displacement
        P_div = bdm1.proj_from_RT0(sd)

        # Normalize the face normals and split into xyz-components
        fn = sd.face_normals.copy()
        fn /= np.linalg.norm(fn, axis=0)
        fn_xyz = np.split(fn.ravel(), 3)

        if sd.dim == 2:
            # This operator maps to div-free functions to capture the scalar rotation
            # We take the difference between the first basis func
            # on a face and the second.
            P_tn = sps.csc_array(P_div, copy=True)
            P_tn.data[1::2] = -1

            # scale with the face normals to obtain tensor-valued functions
            n_times_P_tn = [P_tn * fn_xyz[i] for i in np.arange(sd.dim)]
            P_asym = sps.vstack(n_times_P_tn)

        elif sd.dim == 3:
            # Given an orthonormal basis (s, t) of the face,
            # we generate three matrices that capture asymmetries
            # in (t, n), (n, s), and (s, t).
            P_asym = np.empty(3, dtype=sps.csc_array)

            # (t, n) Take the difference between
            # the first dof on a face and the second dof
            P_tn = sps.csc_array(P_div, copy=True)
            # P_tn.data[0::3] = 1
            P_tn.data[1::3] = -1
            P_tn.data[2::3] = 0

            # scale with the face normals
            n_times_P_tn = [P_tn * fn_i for fn_i in fn_xyz]
            P_asym[0] = sps.vstack(n_times_P_tn)

            # (s, n) Take the difference between
            # the first dof on a face and the third dof
            P_sn = sps.csc_array(P_div, copy=True)
            # P_sn.data[0::3] = 1
            P_sn.data[1::3] = 0
            P_sn.data[2::3] = -1

            # scale with the face normals
            n_times_P_sn = [P_sn * fn_i for fn_i in fn_xyz]
            P_asym[1] = sps.vstack(n_times_P_sn)

            # (s, t) This one is more complicated
            # Extract the vectors s and t
            dof_loc = [sd.nodes[:, sd.face_nodes.indices[i::3]] for i in np.arange(3)]
            s = dof_loc[1] - dof_loc[0]
            t = np.cross(fn, s, axisa=0, axisb=0, axisc=0)

            # Normalize such that both scale as 1/h
            s /= np.sum(np.square(s), axis=0)
            t /= 2 * sd.face_areas

            # Generate functions in the s-direction
            P_s = sps.csc_array(P_div, copy=True)
            for ind in np.arange(3):
                P_s.data[ind::3] = np.sum((dof_loc[ind] - sd.face_centers) * s, axis=0)
            # Scale in the t-direction
            t_times_P_s = [P_s * t_i for t_i in t]
            P_asym[2] = sps.vstack(t_times_P_s)

            # Generate functions in the t-direction
            P_t = sps.csc_array(P_div, copy=True)
            for ind in np.arange(3):
                P_t.data[ind::3] = np.sum((dof_loc[ind] - sd.face_centers) * t, axis=0)
            # Scale in the s-direction
            s_times_P_t = [P_t * s_i for s_i in s]
            P_asym[2] -= sps.vstack(s_times_P_t)

            P_asym = sps.hstack(P_asym)
        else:
            raise NotImplementedError("Grid must be 2D or 3D.")

        # combine all the P
        P_div = sps.block_diag([P_div] * sd.dim)
        P = sps.hstack((P_div, P_asym), format="csc")

        # restriction to the flagged faces and restrict P to them
        expand = pg.numerics.linear_system.create_restriction(flagged_faces).T.tocsc()
        # 3 dof in 2D, 6 dof in 3D
        dofs_per_face = sd.dim * (sd.dim + 1) // 2

        return P @ sps.block_diag([expand] * dofs_per_face, format="csc")

    def compute_system(self, sd: pg.Grid) -> sps.csc_matrix:
        """
        Computes the system matrix for the given grid.

        Args:
            sd (pg.Grid): The grid object representing the domain.

        Returns:
            sps.csc_matrix: The computed system matrix.
        """
        # first we assemble the B matrix
        key = "tree"
        vec_bdm1 = pg.VecBDM1(key)
        vec_p0 = pg.VecPwConstants(key)

        M_div = vec_p0.assemble_mass_matrix(sd)

        if sd.dim == 2:
            p0 = pg.PwConstants(key)
            M_asym = p0.assemble_mass_matrix(sd)
        elif sd.dim == 3:
            M_asym = M_div
        else:
            raise NotImplementedError("Grid must be 2D or 3D.")

        div = M_div @ vec_bdm1.assemble_diff_matrix(sd)
        asym = M_asym @ vec_bdm1.assemble_asym_matrix(sd)

        # create the solution operator
        return sps.vstack((-div, -asym)) @ self.expand


class SpanningWeightedTrees:
    """
    Class that can perform a spanning weighted trees solve, based on one of
    the previously introduced classes SpanningTree and SpanningTreeElasticity.
    It works very similarly to the previous one by considering multiple
    trees instead.

    Attributes:
        sptrs (list): List of SpanningTree objects.
        avg (function): Function to compute the average of a given array.

    Methods:
        solve(f: np.ndarray) -> np.ndarray:
            Perform a spanning weighted trees solve to compute a conservative flux field
            for given mass source.

        solve_transpose(rhs: np.ndarray) -> np.ndarray:
            Post-process the pressure by performing a transposed solve.

        find_starting_faces(mdg: pg.MixedDimensionalGrid, num: int) -> np.ndarray:
            Find the starting faces for each spanning tree if None is provided in the
            constructor.
            By default, an equidistribution of boundary faces is constructed.

    """

    def __init__(
        self,
        mdg: pg.MixedDimensionalGrid,
        spt: object,
        weights: np.ndarray,
        starting_faces: Optional[np.ndarray] = None,
    ) -> None:
        """Constructor of the class

        Args:
            mdg (pg.MixedDimensionalGrid): The mixed dimensional grid.
            spt (object): The spanning tree object to use.
            weights (np.ndarray): The weights to impose for each spanning tree, they need to
                sum to 1.
            starting_faces (Optional[np.ndarray]): The set of starting faces, if not specified
                equi-distributed boundary faces are selected.
        """
        if starting_faces is None:
            num = np.asarray(weights).size
            starting_faces = self.find_starting_faces(mdg, num)

        self.sptrs = [spt(mdg, f) for f in starting_faces]
        self.avg = lambda v: np.average(v, axis=0, weights=weights)

    def solve(self, f: np.ndarray) -> np.ndarray:
        """
        Perform a spanning weighted trees solve to compute a conservative flux field
        for given mass source.

        Args:
            f (np.ndarray): Mass source, integrated against PwConstants.

        Returns:
            np.ndarray: The post-processed flux field.
        """
        return self.avg([st.solve(f) for st in self.sptrs])

    def assemble_SI(self) -> sps.sparray:
        """
        Assembles the operator S_I as a sparse array.
        NOTE: This will be slow for large systems.
        If you only need the action of S_I, consider using self.solve() instead.

        Returns:
            sps.sparray: S_I, a right inverse of the B-operator
        """
        return self.avg([st.assemble_SI() for st in self.sptrs])

    def solve_transpose(self, rhs: np.ndarray) -> np.ndarray:
        """
        Post-process the pressure by performing a transposed solve.

        Args:
            rhs (np.ndarray): Right-hand side, usually the mass matrix times the flux
                              minus boundary terms.

        Returns:
            np.ndarray: The post-processed pressure field.
        """
        return self.avg([st.solve_transpose(rhs) for st in self.sptrs])

    def find_starting_faces(self, mdg: pg.MixedDimensionalGrid, num: int) -> np.ndarray:
        """
        Find the starting faces for each spanning tree if None is provided in the
        constructor.
        By default, an equidistribution of boundary faces is constructed.

        Args:
            mdg (pg.MixedDimensionalGrid): The (mixed dimensional) grid.
            num (int): Number of faces to be selected.

        Returns:
            np.ndarray: The selected faces at the boundary.
        """

        # Extract top-dimensional domain
        sd = mdg.subdomains(dim=mdg.dim_max())[0]
        faces = np.where(sd.tags["domain_boundary_faces"])[0]

        return faces[np.linspace(0, faces.size, num, endpoint=False, dtype=int)]
