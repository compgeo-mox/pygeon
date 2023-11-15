import numpy as np
import scipy.sparse as sps
import xml.etree.ElementTree as ET

import pygeon as pg
import porepy as pp

# Download from https://cs.uwaterloo.ca/~csk/hat/app.html without the supertiles


class EinsteinGrid(pg.Grid):
    def __init__(self, file_name):
        self.poly, self.trans, self.root = self.from_file(file_name)

        self.poly_list = []
        self.poly_adder(*self.root)

        coords, cell_faces, face_nodes = self.build_grid()

        super(EinsteinGrid, self).__init__(
            2, coords, face_nodes, cell_faces, "EinsteinGrid"
        )

    def build_grid(self):
        all_pts = np.hstack(self.poly_list)
        all_pts[2] = 0
        all_pts -= np.atleast_2d(np.min(all_pts, axis=1)).T
        all_pts /= np.max(all_pts)

        n = 13
        coords, _, cell_nodes = pp.utils.setmembership.unique_columns_tol(all_pts)
        cell_nodes = cell_nodes.reshape((-1, n))

        face_nodes = np.zeros((2, cell_nodes.size), dtype=int)
        for cell, nodes in enumerate(cell_nodes):
            pos = slice(n * cell, n * (cell + 1))
            ccw = pp.geometry.geometry_property_checks.is_ccw_polygon(coords[:, nodes])
            if not ccw:
                nodes = nodes[::-1]

            face_nodes[0, pos] = nodes
            face_nodes[1, pos] = np.roll(nodes, -1)

        cf_data = 2 * (face_nodes[1] > face_nodes[0]) - 1
        face_nodes = np.sort(face_nodes, axis=0)

        face_nodes, cf_indices = np.unique(face_nodes, axis=1, return_inverse=True)
        cf_indptr = np.arange(0, cf_data.size + 1, n)

        fn_indices = face_nodes.ravel(order="F")
        fn_indptr = np.arange(0, fn_indices.size + 1, 2)
        fn_data = np.ones_like(fn_indices)

        cell_faces = sps.csc_matrix((cf_data, cf_indices, cf_indptr))
        face_nodes = sps.csc_matrix((fn_data, fn_indices, fn_indptr))

        return coords, cell_faces, face_nodes

    def poly_adder(self, input_str, transform):
        if input_str in self.poly:
            self.poly_list.append(transform @ self.poly[input_str])
        else:
            for sub in self.trans[input_str]:
                self.poly_adder(sub[0], transform @ sub[1])

    def from_file(self, file_name):
        root = ET.parse(file_name).getroot()[0]
        tag_str = r"{http://www.w3.org/1999/xlink}href"

        poly, trans = {}, {}
        for elem in root:
            id = elem.attrib["id"]
            if id[-1] == "f":
                if "polygon" in elem[0].tag:
                    s_poly = elem[0].attrib["points"]
                    poly[id] = self.as_polygon(s_poly)
                else:
                    trans[id] = []
                    for j in elem:
                        if "use" in j.tag:
                            s_mat = j.attrib["transform"]
                            ref = j.attrib[tag_str]
                            trans[id].append((ref[1:], self.as_matrix(s_mat)))

        root_use = ET.parse(file_name).getroot()[1]

        use_id = root_use.attrib[tag_str][1:]
        use_matrix = self.as_matrix(root_use.attrib["transform"])

        return poly, trans, (use_id, use_matrix)

    def as_polygon(self, s):
        """Convert a string to a 2d polygon in homogeneous coordinate"""
        pts = np.fromstring(s.replace(",", " "), sep=" ")
        pts = pts.reshape((2, -1), order="F")
        # add the homogeneous coordinate and return the pts
        return np.vstack((pts, np.ones(pts.shape[1])))

    def as_matrix(self, s):
        """Convert a string to a 2d matrix in homogeneous coordinate"""
        mat = np.fromstring(s[7:-1], sep=" ")
        mat = mat.reshape((2, -1), order="F")
        # add the homogeneous coordinate and return the matrix
        return np.vstack((mat, [0, 0, 1]))


if __name__ == "__main__":
    eg = EinsteinGrid("/home/elle/output100.svg")
    eg.compute_geometry()

    pp.plot_grid(eg)
