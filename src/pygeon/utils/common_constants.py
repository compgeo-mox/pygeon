from typing import Final, Literal

# Common constants for PyGeoN

AMBIENT_DIM: Final[Literal[3]] = 3
"""The ambient dimension of the space in which PyGeoN operates. This is a fixed constant
equal to 3, that defines the maximum dimension of the problems that can be handled."""

UNITARY_DATA: Final[str] = "unitary_data"
"""Keyword used to identify unitary (identity) data in discretizations"""

# Tensor order constants

SCALAR: Final[Literal[0]] = 0
"""0-order tensor (scalar field)"""

VECTOR: Final[Literal[1]] = 1
"""1-order tensor (vector field)"""

MATRIX: Final[Literal[2]] = 2
"""2-order tensor (matrix/tensor field)"""

# Keywords for common physical parameters

LAME_LAMBDA: Final[str] = "lame_lambda"
"""Lamé's first parameter (lambda)"""

LAME_MU: Final[str] = "lame_mu"
"""Lamé's second parameter (mu)"""

LAME_MU_COSSERAT: Final[str] = "lame_mu_cosserat"
"""Lamé's second parameter (mu_c) for Cosserat media"""

SECOND_ORDER_TENSOR: Final[str] = "second_order_tensor"
"""Keyword used to identify second-order tensor data in discretizations, e.g. for
permeability"""

VECTOR_FIELD: Final[str] = "vector_field"
"""Keyword used to identify vector field data in discretizations, e.g. for advection"""

WEIGHT: Final[str] = "weight"
"""Keyword used to identify weight data in discretizations, not covered by other 
keywords"""

NORMAL_DIFFUSIVITY: Final[str] = "normal_diffusivity"
"""Keyword used to identify normal diffusivity data across interfaces in 
mixed-dimensional discretizations"""
