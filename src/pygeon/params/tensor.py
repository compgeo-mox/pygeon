"""
This tensor module provides custom Second Order Tensor class.

This class overwrite PorePy's base functionality to support cases where
tensors may not be positive definite (e.g., systems deviating from
Onsager's principle).
"""

import numpy as np
import porepy as pp
from typing import Optional


class SecondOrderTensor(pp.SecondOrderTensor):
    """
    Cell-wise representation of a second-order tensor.

    While the underlying geometry is 3D, 1D and 2D problems are supported
    by assigning unit values to the unused diagonal components and
    zeroing out cross-terms.
    """

    def __init__(
        self,
        kxx: np.ndarray,
        kyy: Optional[np.ndarray] = None,
        kzz: Optional[np.ndarray] = None,
        kxy: Optional[np.ndarray] = None,
        kxz: Optional[np.ndarray] = None,
        kyz: Optional[np.ndarray] = None,
    ):
        """
        Initialize the tensor with cell-wise values.

        Parameters:
            kxx: Array of values for the xx-component.
            kyy: Array for the yy-component. Defaults to kxx.
            kzz: Array for the zz-component. Defaults to kxx.
            kxy, kxz, kyz: Arrays for off-diagonal components. Defaults to zero.
        """
        Nc = kxx.size
        # Initialize a 3x3 tensor for each of the Nc cells
        tensor = np.zeros((3, 3, Nc))

        # Assign values to the
        # Diagonal
        tensor[0, 0] = kxx
        tensor[1, 1] = kyy if kyy is not None else kxx
        tensor[2, 2] = kzz if kzz is not None else kxx

        # Off-diagonal
        if kxy is not None:
            tensor[0, 1] = tensor[1, 0] = kxy
        if kxz is not None:
            tensor[0, 2] = tensor[2, 0] = kxz
        if kyz is not None:
            tensor[1, 2] = tensor[2, 1] = kyz

        self.values = tensor
