import xml.etree.ElementTree as ET
from typing import Dict, List

import numpy as np
import porepy as pp
import scipy.sparse as sps

import pygeon as pg


class EinSteinGrid(pg.Grid):
    """Build the EinStein grid from an svg file that has been generated from
    https://cs.uwaterloo.ca/~csk/hat/app.html
    """

    def __init__(self, file_name: str) -> None:
        """
        Initialize the EinSteinGrid object.

        Args:
            file_name (str): The name of the file containing the grid data.

        Returns:
            None
        """
        # read all the data from the file
        self.poly, self.trans, self.root = self.from_file(file_name)
        # adding the hanging nodes to fix T-junctions
        self.add_hanging_node()

        # create the list of polygons
        self.poly_list: List[np.ndarray] = []
        self.poly_adder(*self.root)

        # build the connectivity of the grid
        coords, cell_faces, face_nodes = self.build_connectivity()

        # build the grid
        super(EinSteinGrid, self).__init__(
            2, coords, face_nodes, cell_faces, "EinSteinGrid"
        )

    def add_hanging_node(self) -> None:
        """
        Function to add the hanging nodes for each polygons to fix
        T-junctions.

        This function adds a hanging node to each polygon in the grid
        to fix T-junctions. The hanging node is inserted at the midpoint
        of the third and fourth vertices of each polygon.

        Args:
            None

        Returns:
            None
        """
        for key, p in self.poly.items():
            self.poly[key] = np.insert(p, 3, 0.5 * (p[:, 2] + p[:, 3]), axis=1)

    def build_connectivity(self) -> tuple[np.ndarray, sps.csc_matrix, sps.csc_matrix]:
        """
        Build the connectivity of the grid.

        Args:
            None

        Returns:
            Tuple[np.ndarray, sps.csc_matrix, sps.csc_matrix]:
            A tuple containing the following:
                - coords (np.ndarray):
                    The rescaled points of the polygons in the unit square.
                - cell_faces (sps.csc_matrix):
                    The sparse matrix representing the cell-face relations.
                - face_nodes (sps.csc_matrix):
                    The sparse matrix representing the face-nodes relations.
        """
        # rescale the points of the polygons to be in the unit square
        all_pts = self.rescale()

        n = 14
        # uniquify the points coordinate
        coords, _, cell_nodes = pp.utils.setmembership.unique_columns_tol(all_pts)
        # cell node map after uniquify
        cell_nodes = cell_nodes.reshape((-1, n))

        # build the face-nodes relations and fix the polygon orientation
        face_nodes = np.zeros((2, cell_nodes.size), dtype=int)
        for cell, nodes in enumerate(cell_nodes):
            pos = slice(n * cell, n * (cell + 1))
            # check and fix the polygon orientation
            ccw = pp.geometry.geometry_property_checks.is_ccw_polygon(coords[:, nodes])
            if not ccw:
                nodes = nodes[::-1]

            # construct the face nodes relations for the current cell
            face_nodes[0, pos] = nodes
            face_nodes[1, pos] = np.roll(nodes, -1)

        # cell-face orientation for the normal
        cf_data = 2 * (face_nodes[1] > face_nodes[0]) - 1
        cf_indptr = np.arange(0, cf_data.size + 1, n)

        face_nodes = np.sort(face_nodes, axis=0)
        face_nodes, cf_indices = np.unique(face_nodes, axis=1, return_inverse=True)

        # build the data structure for the face_nodes of
        fn_indices = face_nodes.ravel(order="F")
        fn_indptr = np.arange(0, fn_indices.size + 1, 2)
        fn_data = np.ones_like(fn_indices)

        # build the sparse matrices
        cell_faces = sps.csc_matrix((cf_data, cf_indices, cf_indptr))
        face_nodes = sps.csc_matrix((fn_data, fn_indices, fn_indptr))

        return coords, cell_faces, face_nodes

    def rescale(self) -> np.ndarray:
        """
        Rescale the coordinates of the polygons in the unit square.

        Returns:
            np.ndarray: The rescaled coordinates of the polygons.
        """
        all_pts = np.hstack(self.poly_list)
        all_pts[2] = 0
        all_pts -= np.atleast_2d(np.min(all_pts, axis=1)).T
        all_pts /= np.max(all_pts)
        return all_pts

    def poly_adder(self, input_str: str, transform: np.ndarray) -> None:
        """
        Recursive function to build all the polygons based on the transformation matrices.

        Args:
            input_str (str): The input string representing the current tag.
            transform (np.ndarray): The transformation matrix.

        Returns:
            None
        """
        if input_str in self.poly:
            # if the current tag is a polygon then build it and add it to the list
            self.poly_list.append(transform @ self.poly[input_str])
        else:
            # if it is a transformation, or a list of transformation, then call again
            # the function with the updated transformation matrix
            for sub in self.trans[input_str]:
                self.poly_adder(sub[0], transform @ sub[1])

    def from_file(self, file_name: str) -> tuple[dict, dict, tuple]:
        """
        Read an SVG file and create the first data structure.

        Args:
            file_name (str): The path to the SVG file.

        Returns:
            tuple[dict, dict, tuple]: A tuple containing three elements:
                - poly_dict: A dictionary mapping polygon IDs to their corresponding polygons.
                - trans_dict: A dictionary mapping transformation IDs to a
                    list of transformations.
                - use_info: A tuple containing the ID and transformation matrix of
                    the root use element.
        """
        root = ET.parse(file_name).getroot()[0]
        tag_str = r"{http://www.w3.org/1999/xlink}href"

        poly_dict = {}
        trans_dict: Dict[str, List] = {}
        for elem in root:
            id = elem.attrib["id"]
            if id[-1] == "f":
                # we consider only the lines ending with "f"
                if "polygon" in elem[0].tag:
                    # if it's a polygon then save it in the list of polygons
                    pts = elem[0].attrib["points"]
                    poly_dict[id] = self.as_polygon(pts)
                else:
                    trans_dict[id] = []
                    for sub_elem in elem:
                        if "use" in sub_elem.tag:
                            # save the transformations
                            mat = sub_elem.attrib["transform"]
                            ref = sub_elem.attrib[tag_str][1:]
                            # save the reference element and the transformation matrix
                            trans_dict[id].append((ref, self.as_matrix(mat)))

        # At the end of the file we know from which tag to start
        root_use = ET.parse(file_name).getroot()[1]

        use_id = root_use.attrib[tag_str][1:]
        use_matrix = self.as_matrix(root_use.attrib["transform"])

        return poly_dict, trans_dict, (use_id, use_matrix)

    def as_polygon(self, pts: str) -> np.ndarray:
        """
        Convert a string to a 2d polygon in homogeneous coordinate

        Args:
            pts (str): The string representing the polygon points

        Returns:
            np.ndarray: The 2d polygon in homogeneous coordinate
        """
        coords = np.fromstring(pts.replace(",", " "), sep=" ")
        coords = coords.reshape((2, -1), order="F")
        # add the homogeneous coordinate and return the pts
        return np.vstack((coords, np.ones(coords.shape[1])))

    def as_matrix(self, mat: str) -> np.ndarray:
        """
        Convert a string to a 2d matrix in homogeneous coordinate

        Args:
            mat (str): The string representation of the matrix

        Returns:
            np.ndarray: The 2d matrix in homogeneous coordinate
        """
        matrix = np.fromstring(mat[7:-1], sep=" ")
        matrix = matrix.reshape((2, -1), order="F")
        # add the homogeneous coordinate and return the matrix
        return np.vstack((matrix, [0, 0, 1]))
