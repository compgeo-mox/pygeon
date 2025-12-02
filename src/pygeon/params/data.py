import numpy as np
import porepy as pp

import pygeon as pg

data_default = {
    pg.LAME_LAMBDA: 0,
}


def get_cell_data(sd, data, key, param, tensor_order=pg.SCALAR):
    if data is None:
        data_key = {}
    else:
        data_key = data[pp.PARAMETERS][key]

    value = data_key.get(param, data_default[param])

    if isinstance(value, np.ScalarType):
        value = np.full(sd.num_cells, value)

        if tensor_order == pg.VECTOR:
            value = pp.SecondOrderTensor(value)

    return value
