"""Module for the discretizations of the matrix L2 space."""

import pygeon as pg


class MatPwConstants(pg.VecPwConstants):
    """
    A class representing the discretization using matrix piecewise constant functions.

    Attributes:
        keyword (str): The keyword for the matrix discretization class.
        base_discr (pg.Discretization): The base discretization class.

    Methods:
        error_l2(sd: pg.Grid, num_sol: np.ndarray, ana_sol: Callable[[np.ndarray],
            np.ndarray], relative: Optional[bool] = True, etype:
            Optional[str] = "specific") -> float:
            Returns the l2 error computed against an analytical solution given as a
            function.
    """

    def __init__(self, keyword: str = pg.UNITARY_DATA) -> None:
        """
        Initialize the matrix discretization class.
        The base discretization class is pg.PwConstants.

        Args:
            keyword (str): The keyword for the matrix discretization class.

        Returns:
            None
        """
        super().__init__(keyword)
        self.base_discr = pg.VecPwConstants(keyword)


class MatPwLinears(pg.VecPwLinears):
    """
    A class representing the discretization using matrix piecewise linear functions.

    Attributes:
        keyword (str): The keyword for the matrix discretization class.
        base_discr (pg.Discretization): The base discretization class.

    """

    def __init__(self, keyword: str = pg.UNITARY_DATA) -> None:
        """
        Initialize the matrix discretization class.
        The base discretization class is pg.PwLinears.

        Args:
            keyword (str): The keyword for the matrix discretization class.

        Returns:
            None
        """
        super().__init__(keyword)
        self.base_discr = pg.VecPwLinears(keyword)


class MatPwQuadratics(pg.VecPwQuadratics):
    """
    A class representing the discretization using matrix piecewise quadratic functions.

    Attributes:
        keyword (str): The keyword for the matrix discretization class.
        base_discr (pg.Discretization): The base discretization class.

    """

    def __init__(self, keyword: str = pg.UNITARY_DATA) -> None:
        """
        Initialize the matrix discretization class.
        The base discretization class is pg.PwQuadratics.

        Args:
            keyword (str): The keyword for the matrix discretization class.

        Returns:
            None
        """
        super().__init__(keyword)
        self.base_discr = pg.VecPwQuadratics(keyword)
