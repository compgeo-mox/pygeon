"""isort:skip_file"""

from pygeon.utils.common_constants import *

import pygeon.geometry.transformation as transformation
from pygeon.grids.grid import Grid
from pygeon.grids.md_grid import MixedDimensionalGrid
from pygeon.filters.convert_from_pp import convert_from_pp, as_mdg
from pygeon.grids.mortar_grid import MortarGrid
from pygeon.grids.octagon import OctagonGrid
from pygeon.grids.voronoi import VoronoiGrid
from pygeon.grids.einstein import EinSteinGrid
from pygeon.grids.levelset_remesh import levelset_remesh
from pygeon.grids.create_grid import (
    grid_from_domain,
    grid_from_boundary_pts,
    unit_grid,
    reference_element,
)

from pygeon.discretizations.discretization import Discretization
from pygeon.discretizations.vec_discretization import VecDiscretization

from pygeon.discretizations.fem.hcurl import Nedelec0, Nedelec1
from pygeon.discretizations.fem.hdiv import RT0, BDM1, RT1
from pygeon.discretizations.fem.h1 import Lagrange1, Lagrange2
from pygeon.discretizations.fem.l2 import PwConstants, PwLinears, PwQuadratics
from pygeon.discretizations.fem.vec_hdiv import VecBDM1, VecRT0
from pygeon.discretizations.fem.vec_h1 import VecLagrange1, VecLagrange2
from pygeon.discretizations.fem.vec_l2 import VecPwConstants, VecPwLinears

from pygeon.discretizations.vem.hdiv import VRT0, VBDM1
from pygeon.discretizations.vem.h1 import VLagrange1
from pygeon.discretizations.vem.vec_hdiv import VecVRT0
from pygeon.discretizations.vem.vec_h1 import VecVLagrange1

from pygeon.numerics.differentials import grad, curl, div
from pygeon.numerics.innerproducts import (
    cell_mass,
    face_mass,
    ridge_mass,
    peak_mass,
    lumped_cell_mass,
    lumped_face_mass,
    lumped_ridge_mass,
    lumped_peak_mass,
)
from pygeon.numerics.stiffness import cell_stiff, face_stiff, ridge_stiff, peak_stiff
from pygeon.numerics.restrictions import remove_tip_dofs
from pygeon.numerics.linear_system import LinearSystem
from pygeon.numerics.projections import eval_at_cell_centers
from pygeon.numerics.spanningtree import (
    SpanningTree,
    SpanningWeightedTrees,
    SpanningTreeElasticity,
    SpanningTreeCosserat,
)
from pygeon.numerics.poincare import Poincare
from pygeon.numerics.block_diag_solver import (
    assemble_inverse,
    block_diag_solver,
    block_diag_solver_dense,
)

import pygeon.utils.bmat as bmat
import pygeon.utils.sort_points as sort_points
