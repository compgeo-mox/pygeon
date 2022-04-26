import numpy as np
import meshio

import porepy as pp

class P1Exporter(pp.Exporter):

    # ------------------------------------------------------------------------------#

    def __init__(
        self,
        grid,
        file_name,
        folder_name = None,
        **kwargs,
    ):
        super(P1Exporter, self).__init__(grid, file_name, folder_name, **kwargs)

    # ------------------------------------------------------------------------------#

    def _write(self, fields, file_name, meshio_geom):

        point_data = {}

        # we need to split the data for each group of geometrically uniform cells
        point_id = meshio_geom[0]
        num_block = point_id.shape[0]

        for field in fields:
            if field.values is None:
                continue

            # for each field create a sub-vector for each geometrically uniform group of cells
            point_data[field.name] = field.values

        # remove the grid_dim field created in the pp.Exporter
        point_data.pop("grid_dim", None)

        # create the meshio object
        meshio_grid_to_export = meshio.Mesh(
            meshio_geom[0], meshio_geom[1], point_data=point_data
        )
        meshio.write(file_name, meshio_grid_to_export, binary=self.binary)

