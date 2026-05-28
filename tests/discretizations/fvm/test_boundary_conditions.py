"""Tests for finite-volume boundary condition objects."""

import numpy as np
import porepy as pp
import pytest

import pygeon as pg


def test_double_bdry_condition(unit_sd_3d):
    """Check that setting both primary and dual BCs emits a warning."""
    data = pp.initialize_data({}, pg.UNITARY_DATA)
    bcs = pg.FlowBC(unit_sd_3d, data)

    bdry_faces = unit_sd_3d.tags["domain_boundary_faces"]
    bcs.set_flux_bcs(bdry_faces, np.ones(unit_sd_3d.num_faces))

    with pytest.warns():
        bcs.set_pressure_bcs(bdry_faces, np.ones(unit_sd_3d.num_faces))
