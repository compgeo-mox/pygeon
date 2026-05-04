# Degrees of Freedom

This document describes the degrees of freedom (DOFs) for each finite element space
available in PyGeoN, and how they are laid out in the global solution vector.



## Scalar spaces

### `Lagrange1` — $H^1$, piecewise linear ([source](https://github.com/compgeo-mox/pygeon/blob/main/src/pygeon/discretizations/fem/h1.py#L13))

One DOF per mesh **node**.

$$N_\text{dof} = N_\text{nodes}$$

**DOF meaning.** Each DOF is the function value at the corresponding node.

**DOF ordering.** DOF $n$ has global index $n$ (directly indexed by node number).

### `Lagrange2` — $H^1$, piecewise quadratic ([source](https://github.com/compgeo-mox/pygeon/blob/main/src/pygeon/discretizations/fem/h1.py#L309))

One DOF per mesh **node** and one DOF per mesh **edge**.

$$N_\text{dof} = N_\text{nodes} + N_\text{edges}$$

**DOF meaning.** The two sets are heterogeneous: nodal DOFs are function values at nodes; edge DOFs are function values at edge midpoints.

**DOF ordering.** All nodal DOFs come first (index = node number), followed by all edge DOFs (global index of edge $e$ = $N_\text{nodes} + e$).

### `PwConstants` — $L^2$, piecewise constant ([source](https://github.com/compgeo-mox/pygeon/blob/main/src/pygeon/discretizations/fem/l2.py#L248))

One DOF per mesh **cell**.

$$N_\text{dof} = N_\text{cells}$$

**DOF meaning.** Each DOF is the cell-volume-weighted average of the function over the cell.

**DOF ordering.** DOF $c$ has global index $c$ (directly indexed by cell number).

### `PwLinears` — $L^2$, piecewise linear (broken) ([source](https://github.com/compgeo-mox/pygeon/blob/main/src/pygeon/discretizations/fem/l2.py#L386))

$(d+1)$ DOFs per cell (one per local node), totalling:

$$N_\text{dof} = (d+1) \, N_\text{cells}$$

**DOF meaning.** Each DOF is the coefficient of the corresponding local linear basis function, i.e. the function value at a local node of the cell.

**DOF ordering.** Cell index varies fastest: the $k$-th local basis function of cell $c$ has global index $k \cdot N_\text{cells} + c$.

### `PwQuadratics` — $L^2$, piecewise quadratic (broken) ([source](https://github.com/compgeo-mox/pygeon/blob/main/src/pygeon/discretizations/fem/l2.py#L585))

Piecewise degree-2 polynomials, discontinuous across cell boundaries. The local basis
consists of all monomials of degree $\leq 2$, giving $\tfrac{(d+1)(d+2)}{2}$ DOFs per
cell: one per vertex plus one per edge (midpoint) of each simplex. The total count is:

$$N_\text{dof} = \frac{(d+1)(d+2)}{2}\,N_\text{cells}$$

**DOF meaning.** The local DOFs are heterogeneous within each cell: the first $(d+1)$ are function values at the cell vertices, and the remaining $\tfrac{d(d+1)}{2}$ are function values at the midpoints of the cell edges.

**DOF ordering.** Cell index varies fastest: the $k$-th local basis function of cell $c$ has global index $k \cdot N_\text{cells} + c$.

### `RT0` — $H(\text{div})$, lowest-order Raviart–Thomas ([source](https://github.com/compgeo-mox/pygeon/blob/main/src/pygeon/discretizations/fem/hdiv.py#L13))

One DOF per mesh **face**.

$$N_\text{dof} = N_\text{faces}$$

**DOF meaning.** Each DOF is the integral of the normal component of the field across the face: $\int_f \mathbf{u} \cdot \mathbf{n}_f \, ds$. 

**DOF ordering.** DOF $f$ has global index $f$ (directly indexed by face number).

**Note.** Even if `RT0` is a space for vector-valued functions, the degrees of freedom have a scalar meaning.

### `BDM1` — $H(\text{div})$, first-order Brezzi–Douglas–Marini ([source](https://github.com/compgeo-mox/pygeon/blob/main/src/pygeon/discretizations/fem/hdiv.py#L257))

Like RT0 the normal flux across each face is the key quantity, but here the flux is
allowed to vary *linearly* over the face. This requires one DOF per (face, node) pair.
Since each face of a $d$-simplex has exactly $d$ nodes:

$$N_\text{dof} = d \cdot N_\text{faces}$$

**DOF meaning.** Each DOF is a weighted normal-flux moment $\int_f (\mathbf{u} \cdot \mathbf{n}_f)\,\varphi_i^f \, ds$, where $\varphi_i^f$ is the Lagrange basis function of node $i$ restricted to face $f$.

**DOF ordering.** The DOFs are grouped by node-position within the face: the first
$N_\text{faces}$ entries correspond to the first node of each face, the next
$N_\text{faces}$ entries to the second node, and so on. The global index of DOF $(f,i)$
is $f + i\,N_\text{faces}$, $i = 0,\ldots,d-1$.

**Note.** Even if `BDM1` is a space for vector-valued functions, the degrees of freedom have a scalar meaning.

### `RT1` — $H(\text{div})$, first-order Raviart–Thomas ([source](https://github.com/compgeo-mox/pygeon/blob/main/src/pygeon/discretizations/fem/hdiv.py#L480))

$d$ DOFs per face and $d$ DOFs per cell.

$$N_\text{dof} = d\,(N_\text{faces} + N_\text{cells})$$

**DOF meaning.** The two sets are heterogeneous: face DOFs are directional normal-flux moments (one per spatial direction per face); cell DOFs are the $d$ components of the cell-average of the field.

**DOF ordering.** The global vector is split into a face block followed by a cell
block, each further divided into $d$ sub-blocks of uniform size.

Global index of face DOF $(f, k)$: $f + k\,N_\text{faces}$; global index of cell DOF
$(c, k)$: $d\,N_\text{faces} + c + k\,N_\text{cells}$.

**Note.** Even if `RT1` is a space for vector-valued functions, the degrees of freedom have a scalar meaning.

### `Nedelec0` — $H(\text{curl})$, lowest-order Nédélec ([source](https://github.com/compgeo-mox/pygeon/blob/main/src/pygeon/discretizations/fem/hcurl.py#L11))

One DOF per mesh **edge**.

$$N_\text{dof} = N_\text{edges}$$

**DOF meaning.** Each DOF is the integral of the tangential component of the field along the edge: $\int_e \mathbf{u} \cdot \boldsymbol{\tau}_e \, ds$.

**DOF ordering.** DOF $e$ has global index $e$ (directly indexed by edge number).

**Note.** Even if `Nedelec0` is a space for vector-valued functions, the degrees of freedom have a scalar meaning.

### `Nedelec1` — $H(\text{curl})$, first-order Nédélec ([source](https://github.com/compgeo-mox/pygeon/blob/main/src/pygeon/discretizations/fem/hcurl.py#L161))

Analogous to BDM1 but for tangential circulations: the tangential component along each
edge is allowed to vary linearly, requiring one DOF per (edge, node) pair. Each edge
has exactly 2 endpoint nodes in any dimension, giving:

$$N_\text{dof} = 2\,N_\text{edges}$$

**DOF meaning.** Each DOF is a weighted tangential-circulation moment $\int_e (\mathbf{u} \cdot \boldsymbol{\tau}_e)\,\varphi_i^e \, ds$, where $\varphi_i^e$ is the Lagrange basis function of node $i$ on edge $e$.

**DOF ordering.** Two blocks of $N_\text{edges}$ each (one per endpoint): the global
index of DOF $(e, i)$ is $e + i\,N_\text{edges}$, $i \in \{0, 1\}$.

**Note.** Even if `Nedelec1` is a space for vector-valued functions, the degrees of freedom have a scalar meaning.

## Vector-valued spaces

A vector-valued space `Vec<X>` wraps the corresponding scalar space `<X>` and
replicates its DOFs for each spatial component. The total DOF count is:

$$N_\text{dof} = d \cdot N_\text{dof}^{\text{scalar}}$$

### DOF layout

All vector spaces use a **component-block** ordering: first all DOFs of the
$x$-component, then all DOFs of the $y$-component, and so on.  



## Matrix-valued spaces

The matrix-valued $L^2$ spaces `MatPwConstants`, `MatPwLinears`, and
`MatPwQuadratics` replicate the DOFs of their underlying scalar piecewise
polynomial spaces for each entry of the $d \times d$ tensor.
replicate their DOFs for each entry of the $d \times d$ tensor.

$$N_\text{dof} = d^2 \cdot N_\text{dof}^{\text{scalar}}$$

### DOF layout

The tensor is **unrolled row-wise** and stored in $d^2$ consecutive blocks of
size $N_\text{dof}^{\text{scalar}}$. 

Equivalently, the global DOF index for component $(i,j)$ of scalar DOF $k$ is:

$$\text{global index} = (i \cdot d + j) \cdot N_\text{dof}^{\text{scalar}} + k$$

Note that `VecBDM1` / `VecRT0` classes serve as matrix-valued $H(\text{div})$ spaces
for tensors but their degrees of freedom are vector-valued; 
their DOF layout follows the same row-major block structure.

## Symmetrizing a matrix-valued space

The matrix-valued spaces (`MatPwConstants`, `MatPwLinears`, `MatPwQuadratics`)
always store all $d^2$ components per scalar DOF and do not use a reduced-DOF
representation for symmetric tensors. Instead, symmetry is enforced via the
`assemble_symmetrizing_matrix` method of `MatPwPolynomials`, which returns the
linear operator that maps a full $d^2$-block DOF vector to its symmetric part by
averaging off-diagonal pairs: $\sigma_{ij} \mapsto \tfrac{1}{2}(\sigma_{ij} +
\sigma_{ji})$ for $i \neq j$, leaving diagonal entries unchanged.
