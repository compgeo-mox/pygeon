from typing import overload

import numpy as np
import porepy as pp

import pygeon as pg

# Default parameter values used as fallback when parameters are not provided
# in the data dictionary
data_default = {
    pg.SECOND_ORDER_TENSOR: 1,  # Default permeability/conductivity tensor value
    pg.LAME_LAMBDA: 0,  # First Lamé parameter (bulk modulus related)
    pg.LAME_MU: 0.5,  # Second Lamé parameter (shear modulus)
    pg.LAME_MU_COSSERAT: 0.5,  # Cosserat shear modulus for micropolar materials
    pg.WEIGHT: 1,  # Default weight/scaling factor
    pg.NORMAL_DIFFUSIVITY: 1,  # Default normal diffusivity across interfaces
}


@overload
def get_cell_data(
    sd: pg.Grid,
    data: dict | None,
    keyword: str,
    param: str,
) -> np.ndarray: ...


@overload
def get_cell_data(
    sd: pg.Grid,
    data: dict | None,
    keyword: str,
    param: str,
    tensor_order: int,
) -> pp.SecondOrderTensor: ...


def get_cell_data(
    sd: pg.Grid,
    data: dict | None,
    keyword: str,
    param: str,
    tensor_order: int = pg.SCALAR,
) -> np.ndarray | pp.SecondOrderTensor:
    """
    Retrieve cell data from a grid with appropriate formatting based on tensor order.

    Args:
        sd (pg.Grid): The grid object containing cell information.
        data (dict): Dictionary containing parameter data. If None, default values from
            data_default dictionary is used.
        keyword (str): The key to access the parameter dictionary within
            data[pp.PARAMETERS].
        param (str): The parameter name to retrieve from the data dictionary.
        tensor_order (int): The tensor order of the data. Default is pg.SCALAR.
            If pg.VECTOR, the result is wrapped in a pp.SecondOrderTensor.

    Returns:
        np.ndarray | pp.SecondOrderTensor: The parameter values for all cells.
            Returns a numpy array for scalar data, or a SecondOrderTensor for
            vector discretization.

    Notes:
        If the parameter is not found in the data dictionary, it falls back to
        data_default[param]. Scalar values are automatically broadcast to match the
        number of cells in the grid.
    """
    # Handle empty or None data by using an empty dictionary
    if not data:
        data_key = {}
    elif pp.PARAMETERS not in data or keyword not in data[pp.PARAMETERS]:
        # Handle missing PARAMETERS or keyword by using an empty dictionary
        data_key = {}
    else:
        # Extract the parameter dictionary for the given key
        data_key = data[pp.PARAMETERS][keyword]

    # Get the parameter value, falling back to default if not found
    value = data_key.get(param, data_default[param])

    # If value is a scalar, broadcast it to all cells
    if isinstance(value, np.ScalarType):
        value = np.full(sd.num_cells, value)

        # Wrap vector tensor orders in a SecondOrderTensor
        if tensor_order == pg.VECTOR:
            value = pp.SecondOrderTensor(value)

    return value
