# Matrix Assembly via Projection to Piecewise Polynomial Spaces

## Overview

In PyGeoN, inner-product matrices (mass and lumped mass) for any finite element space
$V_h$ are assembled by factoring through a piecewise polynomial space $P_h$ of matching
polynomial degree and tensor order. This avoids rewriting quadrature routines for each
space and yields a uniform assembly pipeline.

## Mass Matrix

Let $\Pi : V_h \to P_h$ be the projection from the finite element space to the
corresponding piecewise polynomial space (implemented as `proj_to_PwPolynomials`). Then

$$M_{V_h} = \Pi^\top M_{P_h} \Pi,$$

where $M_{P_h}$ is the (cell-local) mass matrix of $P_h$. 

The same pattern is used for the lumped mass matrix.

## Stiffness Matrix

The stiffness matrix uses the differential operator $B$ (gradient, curl, or divergence) and the mass matrix $A$ of the *range* space:

$$K = B^\top A B.$$

The range-space mass matrix $A$ is itself assembled via the same projection procedure
above.

## Projection matrices

Each finite element space implements `proj_to_PwPolynomials` according to its degrees of
freedom. For vector- and matrix-valued spaces (`VecHDiv`, etc.) the same formula applies.
