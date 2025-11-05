import numpy as np
import porepy as pp
import pytest

import pygeon as pg


@pytest.fixture
def discr():
    return pg.VecRT1("test")


def test_ndof(discr, ref_sd):
    known = [0, 3, 16, 45]
    assert discr.ndof(ref_sd) == known[ref_sd.dim]
