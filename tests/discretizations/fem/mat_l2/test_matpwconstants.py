import pytest

import pygeon as pg


def test_ndof(ref_sd):
    discr = pg.MatPwConstants()
    assert discr.ndof(ref_sd) == ref_sd.dim**2
