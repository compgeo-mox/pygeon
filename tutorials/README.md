# Tutorials

This folder contains several examples to guide you in using PyGeoN. Each tutorial demonstrates how to solve different PDEs using finite element methods.

## PyGeoN Tutorials

### Fluid Flow
* [Darcy equation](./fluid_flow/darcy.ipynb) — solve the Darcy equation in mixed form (flux and pressure) on a single domain
* [Darcy equation (TPFA)](./fluid_flow/darcy_tpfa.ipynb) — solve the Darcy equation using the two-point flux approximation with cell-centre pressures
* [Stokes equation](./fluid_flow/stokes.ipynb) — solve the Stokes equation in mixed form (vorticity, velocity and pressure) for viscous flow
* [Stokes equation (stabilized)](./fluid_flow/stokes_stabilized.ipynb) — solve the Stokes equation using an equal-order velocity-pressure pair with Brezzi-Pitkäranta pressure stabilization

### Solid Mechanics
* [Elasticity (primal)](./solid_mechanics/elasticity.ipynb) — solve the linear elasticity equation for displacement using Lagrange elements
* [Elasticity (mixed)](./solid_mechanics/elasticity_mixed.ipynb) — solve elasticity in mixed form with stress, displacement, and rotation as unknowns
* [Elasticity (finite volume + stress reconstruction)](./solid_mechanics/elasticity_stress_reconstruction.ipynb) — solve elasticity using PorePy's finite volume method with PyGeoN post-processing
* [Elasticity (TPSA)](./solid_mechanics/elasticity_tpsa.ipynb) — solve elasticity using the two-point stress approximation with displacement, rotation, and solid pressure unknowns
* [Cosserat equation](./solid_mechanics/cosserat.ipynb) — solve the Cosserat continuum model with micro-rotations in mixed form

### Coupled Problems
* [Biot equation](./coupled_problems/biot.ipynb) — solve the static Biot poroelasticity problem coupling fluid flow and solid deformation

### Advanced Topics
* [Poincaré operators](./advanced_topics/poincare_operators.ipynb) — efficient solution of Hodge-Laplace problems using Poincaré operators and subspace decomposition