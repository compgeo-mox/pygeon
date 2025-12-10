UNITARY_DATA = "unitary_data"
"""Keyword used to identify unitary (identity) data in discretizations"""

# Tensor order constants

SCALAR = 0
"""0-order tensor (scalar field)"""

VECTOR = 1
"""1-order tensor (vector field)"""

MATRIX = 2
"""2-order tensor (matrix/tensor field)"""

# Keywords for common physical parameters

LAME_LAMBDA = "lame_lambda"
"""Lamé's first parameter (lambda)"""

LAME_MU = "lame_mu"
"""Lamé's second parameter (mu)"""

LAME_MU_COSSERAT = "lame_mu_cosserat"
"""Lamé's second parameter (mu_c) for Cosserat media"""

SECOND_ORDER_TENSOR = "second_order_tensor"
"""Keyword used to identify second-order tensor data in discretizations, e.g. for
permeability"""

WEIGHT = "weight"
"""Keyword used to identify weight data in discretizations, not covered by other 
keywords"""

NORMAL_DIFFUSIVITY = "normal_diffusivity"
"""Keyword used to identify normal diffusivity data across interfaces in 
mixed-dimensional discretizations"""
