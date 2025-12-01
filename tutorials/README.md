# Tutorials

This folder contains several examples to guide you in using PyGeoN. Each tutorial demonstrates how to solve different PDEs using finite element methods.

## PyGeoN Tutorials

### Fluid Flow
* [Darcy equation](./darcy.ipynb) — solve the Darcy equation in mixed form (flux and pressure) on a single domain
* [Stokes equation](./stokes.ipynb) — solve the Stokes equation in mixed form (velocity and pressure) for viscous flow

### Solid Mechanics
* [Elasticity (primal)](./elasticity.ipynb) — solve the linear elasticity equation for displacement using Lagrange elements
* [Elasticity (mixed)](./elasticity_mixed.ipynb) — solve elasticity in mixed form with stress, displacement, and rotation as unknowns
* [Elasticity (finite volume)](./elasticity_fv.ipynb) — solve elasticity using PorePy's finite volume method with PyGeoN post-processing

### Coupled Problems
* [Biot equation](./biot.ipynb) — solve the static Biot poroelasticity problem coupling fluid flow and solid deformation
* [Cosserat equation](./cosserat.ipynb) — solve the Cosserat continuum model with micro-rotations in mixed form

### Advanced Topics
* [Poincaré operators](./poincare_operators.ipynb) — efficient solution of Hodge-Laplace problems using Poincaré operators and subspace decomposition