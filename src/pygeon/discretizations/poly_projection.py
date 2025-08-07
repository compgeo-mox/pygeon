from typing import Type, Union

import scipy.sparse as sps

import pygeon as pg


def get_PwPolynomials(
    poly_order: int, tensor_order: int
) -> Union[Type[pg.PwPolynomials], Type[pg.VecPwPolynomials]]:
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


def proj_to_PwPolynomials(
    discr: pg.Discretization, sd: pg.Grid, poly_order: int
) -> sps.csc_array:
    """
    Constructs a projection operator to piecewise polynomial spaces of a specified
    order.

    Args:
        discr (pg.Discretization): The current discretization object.
        sd (pg.Grid): The grid on which the projection is performed.
        poly_order (int): The target polynomial order for the projection.

    Returns:
        sps.csc_array: A sparse matrix representing the projection operator to the
            specified piecewise polynomial space.
    """
    pi = discr.proj_to_PwPolynomials(sd)

    current_poly_order = discr.poly_order

    while current_poly_order < poly_order:
        poly_discr = get_PwPolynomials(current_poly_order, discr.tensor_order)(
            discr.keyword
        )
        pi = poly_discr.proj_to_higher_PwPolynomials(sd) @ pi
        current_poly_order += 1

    while current_poly_order > poly_order:
        poly_discr = get_PwPolynomials(current_poly_order, discr.tensor_order)(
            discr.keyword
        )
        pi = poly_discr.proj_to_lower_PwPolynomials(sd) @ pi
        current_poly_order -= 1

    return pi
