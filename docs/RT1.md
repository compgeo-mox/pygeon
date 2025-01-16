# Raviart-Thomas 1 using barycentric coordinates

## 2D
Consider $d = 2$, a triangle with nodes $\{x_0, x_1, x_2\}$, and let the tangents be given by

$$
\tau_{ij} = x_j - x_i
$$

Following Eggers and Radu (2020), we consider the following basis functions

$$
\begin{align}
    \varphi_0 &\coloneqq \lambda_0 \lambda_2 \tau_{02} + \lambda_0 \lambda_1 \tau_{01} \\
    \varphi_1 &\coloneqq \lambda_1 \lambda_0 \tau_{10} + \lambda_1 \lambda_2 \tau_{12} \\
    \varphi_2 &\coloneqq \lambda_2 \lambda_1 \tau_{21} + \lambda_2 \lambda_0 \tau_{20}
\end{align}
$$

At the cell center, each $\lambda_i = \frac1{d + 1}$, so we have the following identity:

$$
\begin{align}
    (\varphi_0 - \varphi_1)(x_c) &= \frac{1}{(d + 1)^2}(\tau_{02} + \tau_{01} - \tau_{10} - \tau_{12}) \\
    &= \frac{1}{(d + 1)^2}(3 \tau_{01} - (\tau_{01} + \tau_{12} + \tau_{20}))
    = \frac{3}{(d + 1)^2}\tau_{01}
    = \frac{1}{d + 1}\tau_{01}
\end{align}
$$

Let $\phi_i^j$ denote the basis function for the the degree of freedom at face $f_j$ and node $x_i$. Letting $x^j$ be the node opposite face $f_j$,
we propose the following basis function

$$
\begin{align}
    \phi_1^0 &\coloneqq \pm \frac{1}{d |\Delta|} \left(\lambda_1 \tau_{01} - (\varphi_0 - \varphi_1) \right) 
    % &= \pm  \frac{1}{d |\Delta|} \left(\lambda_1 \tau_{01} -( 2 \lambda_0 \lambda_1 \tau_{01} + \lambda_0 \lambda_2 \tau_{02} - \lambda_1 \lambda_2 \tau_{12}) \right)
\end{align}
$$

More generally, we define

$$
\begin{align}
    \phi_i^j &\coloneqq \pm \frac{1}{d |\Delta|} \left(\lambda_1 \tau_{ji} - (\varphi_j - \varphi_i) \right)
\end{align}
$$

where the sign is determined by whether the normal of face $j$ is outward with respect to the element. 

For the cell degrees of freedom, we define

$$
\begin{align}
    \phi_k &\coloneqq \frac{1}{d |\Delta|} \varphi_k
\end{align}
$$

Note that $\phi_0 + \phi_1 + \phi_2 = 0$, so we can span all three with $\{\phi_0, \phi_1\}$. 

### Properties

#### Degrees of freedom
We now show that these basis functions have the desired properties. First, we have 

$$
    n \cdot \phi_k = 0
$$

on each face because either $\phi_k = 0$ or it is tangential to the face. In turn, we have $\phi_k = 0$ at the nodes.

Second, we have at the cell-centers

$$
    \phi_i^j(x_c) = \pm \frac{1}{d |\Delta|} \left(\frac1{d + 1} \tau_{ji} - \frac{1}{d + 1}\tau_{ji} \right) = 0
$$

This means that we can evaluate the degrees of freedom by considering the nodal and cell center values of a function.

#### The divergence

The divergence of the cell basis functions can be calculated as:

$$
\begin{align}
    \nabla \cdot \varphi_k 
    &= \nabla \cdot(\lambda_k \lambda_{k - 1} \tau_{k,k - 1} + \lambda_k \lambda_{k + 1} \tau_{k, k + 1}) \\
    &= (\lambda_k (\nabla \lambda_{k - 1}) + (\nabla \lambda_k) \lambda_{k - 1}) \cdot \tau_{k,k - 1} + (\lambda_k (\nabla \lambda_{k + 1}) + (\nabla \lambda_k) \lambda_{k + 1}) \cdot \tau_{k, k + 1} \\
    &= \lambda_k - \lambda_{k - 1} + \lambda_k - \lambda_{k + 1} \\
    &= 3\lambda_k - (\lambda_{k - 1} + \lambda_k + \lambda_{k + 1}) \\
    &= (d + 1) \lambda_k - 1,
\end{align}
$$

where $k$ is understood modulo 3. This has mean zero. For the face degrees of freedom, we compute

$$
\begin{align}
    \nabla \cdot \phi_i^j 
    &= \pm \frac{1}{d |\Delta|} \left(\nabla\lambda_i \cdot \tau_{ji} - \nabla \cdot(\varphi_j - \varphi_i) \right) \\
    &= \pm \frac{1}{d |\Delta|}\left( 1 - (d + 1) (\lambda_j - \lambda_i) \right)
\end{align}
$$

From which we easily deduce that $\int_\Delta \nabla \cdot \phi_i^j = \pm \frac1d$.

### Implementation

We consider the following spatial basis functions (in order)

$$
    \{ 
        \lambda_0, \ \lambda_1, \ \lambda_2, \ \lambda_0 \lambda_1,  \lambda_0 \lambda_2,  \lambda_1 \lambda_2 \}
$$

Since these are the basis functions used for Lagrange2, we can fetch the inner products from there. After applying a Kronecker product, this local mass matrix is adapted to the vector-valued setting.

As is common in PyGeoN, we then create an array $\Psi$ whose rows contain the coefficents for each basis function. As an example, let us compute the row for $\phi_1^0$. We first introduce the helper functions

$$
\begin{align}
    \psi_0 &= 
    \begin{bmatrix} 
        0 \\\ 0 \\\ 0 \\\ \tau_{01} \\\ \tau_{02} \\\ 0 
    \end{bmatrix}
    &
    \psi_1 &= 
    \begin{bmatrix} 
        0 \\\ 0 \\\ 0 \\\ \tau_{10} \\\ 0 \\\ \tau_{12}
    \end{bmatrix}
    &
    \psi_2 &= 
    \begin{bmatrix} 
        0 \\\ 0 \\\ 0 \\\ 0 \\\ \tau_{20} \\\ \tau_{21} 
    \end{bmatrix}
\end{align}
$$

These are computed by looping over the edges of the element.
Using these, we can rapidly compute, for example

$$
    \Psi_1^0 \coloneqq \pm \frac1{d |\Delta|}\left(
    \begin{bmatrix} 
        0 \\\ \tau_{01} \\\ 0 \\\ 0 \\\ 0 \\\ 0 
    \end{bmatrix}
    - (\psi_0 - \psi_1)
    \right).\text{ravel}()
$$

and for the cell-based degrees of freedom:

$$
    \Psi_0 \coloneqq \frac{1}{d |\Delta|} \psi_0
$$

## 3D

The three-dimensional generalization of $\varphi$ is given by

$$
\begin{align}
    \varphi_k &\coloneqq
    \sum_{i \ne k} \lambda_k \lambda_i \tau_{ki}
\end{align}
$$

A similar calculation as in 2D shows us that

$$
\begin{align}
    (\varphi_0 - \varphi_1)(x_c)
    &= \frac{1}{(d + 1)^2}(\tau_{01} + \tau_{02} + \tau_{03} - \tau_{10} - \tau_{12} - \tau_{13}) \\
    &= \frac{1}{(d + 1)^2}(4 \tau_{01} - (\tau_{01} + \tau_{12} + \tau_{20}) - (\tau_{01} + \tau_{13} + \tau_{30}))
    = \frac{4\tau_{01}}{(d + 1)^2}
    = \frac{\tau_{01}}{d + 1}
\end{align}
$$

so we again propose the basis functions

$$
\begin{align}
    \phi_i^j &\coloneqq \pm \frac{1}{d |\Delta|} \left(\lambda_1 \tau_{ji} - (\varphi_j - \varphi_i) \right) \\
    \phi_k &\coloneqq \frac{1}{d |\Delta|} \varphi_k
\end{align}
$$

The properties concerning the degrees of freedom follow immediately. All we need to double check is the divergence

### The divergence

$$
\begin{align}
    \nabla \cdot \varphi_k &=
    \sum_{i \ne k} \nabla \cdot \lambda_k \lambda_i \tau_{ki} \\
    &= \sum_{i \ne k} \lambda_k - \lambda_i \\
    &= (d + 1) \lambda_k - 1
\end{align}
$$

this gives us

$$
\begin{align}
    \nabla \cdot (\varphi_j - \varphi_i) &=
    (d + 1) (\lambda_j - \lambda_i)
\end{align}
$$