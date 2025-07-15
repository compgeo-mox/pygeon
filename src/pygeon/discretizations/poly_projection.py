from typing import Type
import pygeon as pg


def get_PwPolynomials(poly_order: int, tensor_order: int) -> Type[pg.Discretization]:
    """
    Returns the piecewise polynomial discretization class based on the polynomial order.

    Args:
        poly_order (int): The polynomial order.
        tensor_order (int): The tensor order.

    Returns:
        Type[pg.Discretization]: The corresponding piecewise polynomial discretization
            class.
    """
    match (poly_order, tensor_order):
        case (0, pg.SCALAR):
            return pg.PwConstants
        case (1, pg.SCALAR):
            return pg.PwLinears
        case (2, pg.SCALAR):
            return pg.PwQuadratics
        case (0, pg.VECTOR):
            return pg.VecPwConstants
        case (1, pg.VECTOR):
            return pg.VecPwLinears
        case (2, pg.VECTOR):
            return pg.VecPwQuadratics
        case (0, pg.MATRIX):
            return pg.MatPwConstants
        case (1, pg.MATRIX):
            return pg.MatPwLinears
        case (2, pg.MATRIX):
            return pg.MatPwQuadratics
        case _:
            raise ValueError(
                f"Unsupported polynomial order {poly_order} and tensor order"
                "{tensor_order}."
            )
