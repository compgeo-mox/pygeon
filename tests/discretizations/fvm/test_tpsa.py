import porepy as pp
import pytest

import pygeon as pg

sd = pp.CartGrid([3] * 3)
sd = pg.convert_from_pp(sd)
sd.compute_geometry()
data = pp.initialize_data({}, pg.UNITARY_DATA, {pg.LAME_LAMBDA: 1.0, pg.LAME_MU: 1.0})

tpsa = pg.discretizations.fvm.tpsa.TPSA()

M = tpsa.assemble_elasticity_matrix(sd, data)
