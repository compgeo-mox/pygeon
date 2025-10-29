import pytest

import pygeon as pg


def test_string_repr():
    discr = pg.PwConstants("test")
    repr = str(discr)
    known = "Discretization of type PwConstants with keyword test"

    assert repr == known
