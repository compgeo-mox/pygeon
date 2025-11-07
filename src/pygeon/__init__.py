# isort:skip_file

"""
PyGeoN: a Python package for Geo-Numerics
==========================================

PyGeoN is a comprehensive Python library for structure-preserving numerical methods
in geoscience applications, with a focus on mixed-dimensional modeling and porous
media flow simulation.

Main Features
-------------
* **Grids**: Create and manipulate computational meshes including simplicial, Voronoi,
  octagonal, and mixed-dimensional grids with support for fracture networks.

* **Discretizations**: Structure-preserving finite element (FEM) and virtual element (VEM)
  methods for H1, Hdiv, Hcurl, and L2 spaces with both scalar, vector, and matrix formulations.

* **Differential Operators**: Discrete gradient, divergence, and curl operators that
  preserve the de Rham complex structure.

* **Numerics**: Advanced solvers including spanning tree, Poincaré operators,
  and mass/stiffness matrix assembly for mixed-dimensional problems.

* **PorePy Integration**: Seamless conversion between PyGeoN and PorePy data structures
  for enhanced compatibility with the PorePy ecosystem.
"""

from pygeon.utils.common_constants import *

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
from pygeon.grids.refinement import barycentric_split

from pygeon.discretizations.discretization import Discretization
from pygeon.discretizations.vec_discretization import VecDiscretization

from pygeon.discretizations.fem.hcurl import Nedelec0, Nedelec1
from pygeon.discretizations.fem.hdiv import RT0, BDM1, RT1
from pygeon.discretizations.fem.h1 import Lagrange1, Lagrange2
from pygeon.discretizations.fem.l2 import (
    PwPolynomials,
    PwConstants,
    PwLinears,
    PwQuadratics,
)
from pygeon.discretizations.fem.vec_hdiv import VecBDM1, VecRT0, VecRT1
from pygeon.discretizations.fem.vec_h1 import VecLagrange1, VecLagrange2
from pygeon.discretizations.fem.vec_l2 import (
    VecPwPolynomials,
    VecPwConstants,
    VecPwLinears,
    VecPwQuadratics,
)
from pygeon.discretizations.fem.mat_l2 import (
    MatPwConstants,
    MatPwLinears,
    MatPwQuadratics,
)

from pygeon.discretizations.vem.hdiv import VRT0, VBDM1
from pygeon.discretizations.vem.h1 import VLagrange1
from pygeon.discretizations.vem.vec_hdiv import VecVRT0
from pygeon.discretizations.vem.vec_h1 import VecVLagrange1

from pygeon.discretizations.poly_projection import (
    get_PwPolynomials,
    proj_to_PwPolynomials,
)

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
