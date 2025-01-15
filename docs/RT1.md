# Raviart-Thomas 1 using barycentric coordinates

## 2D
Consider $d = 2$, a triangle with nodes $\{x_0, x_1, x_2\}$, and let the tangents be given by
$$
\tau_{ij} = x_j - x_i
$$

Following Eggers and Radu (2020), we consider the following basis functions for the cell degrees of freedom

$$
\begin{align}
    \phi_0 &\coloneqq \frac{(d + 1)^2}{d |\Delta|} (\lambda_0 \lambda_2 \tau_{02} + \lambda_0 \lambda_1 \tau_{01})  \\
    \phi_1 &\coloneqq \frac{(d + 1)^2}{d |\Delta|} (\lambda_1 \lambda_0 \tau_{10} + \lambda_1 \lambda_2 \tau_{12}) \\
    \phi_2 &\coloneqq \frac{(d + 1)^2}{d |\Delta|} (\lambda_2 \lambda_1 \tau_{21} + \lambda_2 \lambda_0 \tau_{20})
\end{align}
$$
Note that $\phi_0 + \phi_1 + \phi_2 = 0$, so we can span all three with $\{\phi_0, \phi_1\}$. 
At the cell center, each $\lambda_i = \frac1{d + 1}$, so we have the following identity:
$$
\begin{align}
    (\phi_0 - \phi_1)|_{c} 
    &= \frac{1}{d |\Delta|}(\tau_{02} + \tau_{01} - \tau_{10} - \tau_{12}) \\
    &= \frac{1}{d |\Delta|}(3 \tau_{01} - (\tau_{01} + \tau_{12} + \tau_{20}))
    = \frac{3\tau_{01}}{d |\Delta|}
\end{align}
$$

Let $\phi_i^j$ denote the basis function for the the degree of freedom at face $f_j$ and node $x_i$. Letting $x^j$ be the node opposite face $f_j$,
we propose the following basis function
$$
\begin{align}
    \phi_1^0 &\coloneqq \pm \left(\lambda_1 \frac{\tau_{01}}{d |\Delta|} - \frac1{(d + 1)^2} (\phi_0 - \phi_1) \right) 
    % &= \pm  \frac{1}{d |\Delta|} \left(\lambda_1 \tau_{01} -( 2 \lambda_0 \lambda_1 \tau_{01} + \lambda_0 \lambda_2 \tau_{02} - \lambda_1 \lambda_2 \tau_{12}) \right)
\end{align}
$$
More generally, we define
$$
\begin{align}
    \phi_i^j &\coloneqq \pm \left(\lambda_i \frac{\tau_{ji}}{d |\Delta|} - \frac1{(d + 1)^2} (\phi_j - \phi_i) \right) 
    % \\
    % &= \pm \left(\lambda_i \frac{\tau_{ji}}{d |\Delta|} - \frac1{(d + 1)^2} (\phi_j - \phi_i) \right) \\
\end{align}
$$
where the sign is determined by whether the normal of face $j$ is outward with respect to the element. 


### Properties

#### Degrees of freedom
We now show that these basis functions have the desired properties. First, we have 
$$
    n \cdot \phi_k = 0
$$
on each face because either $\phi_k = 0$ or it is tangential to the face. In turn, we have $\phi_k = 0$ at the nodes.

Second, we have at the cell-centers
$$
    \phi_i^j(x_c) = \pm \left(\frac1{d + 1} \frac{\tau_{ji}}{d |\Delta|} - \frac1{(d + 1)^2}  \frac{3\tau_{ji}}{d |\Delta|} \right) = 0
$$

This means that we can evaluate the degrees of freedom by considering the nodal and cell center values of a function.

#### The divergence

The divergence of the basis functions can easily be calculated:
$$
\begin{align}
    \nabla \cdot \phi_k 
    &= \frac{(d + 1)^2}{d |\Delta|} \nabla \cdot(\lambda_k \lambda_{k - 1} \tau_{k,k - 1} + \lambda_k \lambda_{k + 1} \tau_{k, k + 1}) \\
    &= \frac{(d + 1)^2}{d |\Delta|} ((\lambda_k (\nabla \lambda_{k - 1}) + (\nabla \lambda_k) \lambda_{k - 1}) \cdot \tau_{k,k - 1} + (\lambda_k (\nabla \lambda_{k + 1}) + (\nabla \lambda_k) \lambda_{k + 1}) \cdot \tau_{k, k + 1}) \\
    &= \frac{(d + 1)^2}{d |\Delta|} (\lambda_k - \lambda_{k - 1} + \lambda_k - \lambda_{k + 1}) \\
    &= \frac{(d + 1)^2}{d |\Delta|} (2\lambda_k - \lambda_{k - 1} - \lambda_{k + 1}).
\end{align}
$$
This has mean zero. For the cell degrees of freedom, we compute
$$
\begin{align}
    \nabla \cdot \phi_i^j 
    &= \pm \left(\nabla\lambda_i \cdot \frac{\tau_{ji}}{d |\Delta|} - \frac1{(d + 1)^2} \nabla \cdot(\phi_j - \phi_i) \right) \\
    &= \pm \frac{1}{d |\Delta|}\left( 1 - 3 (\lambda_j - \lambda_i) \right)
\end{align}
$$
From which we easily deduce that $\int_\Delta \nabla \cdot \phi_i^j = \pm \frac1d$.

### Implementation

We consider the following spatial basis functions (in order)
$$
    \left \{ 
        \lambda_0, \  
        \lambda_1, \  
        \lambda_2, \  
        \lambda_0 \lambda_1, \ 
        \lambda_0 \lambda_2, \ 
        \lambda_1 \lambda_2. 
    \right \}
$$
Since these are the basis functions used for Lagrange2, we can fetch the inner products from there. After applying a Kronecker product, this local mass matrix is adapted to the vector-valued setting.

As is common in PyGeoN, we then create an array $\Psi$ whose rows contain the coefficents for each basis function. As an example, let us compute the row for $\phi_1^0$. We first introduce the helper functions
$$
\begin{align}
    \psi_0 &= 
    \begin{bmatrix} 
        0 \\ 0 \\ 0 \\
        \tau_{01} \\ \tau_{02} \\ 0 
    \end{bmatrix}
    &
    \psi_1 &= 
    \begin{bmatrix} 
        0 \\ 0 \\ 0 \\
        \tau_{10} \\ 0 \\ \tau_{12}
    \end{bmatrix}
    &
    \psi_2 &= 
    \begin{bmatrix} 
        0 \\ 0 \\ 0 \\
        0 \\ \tau_{20} \\ \tau_{21} 
    \end{bmatrix}
\end{align}
$$

Using these, we can rapidly compute, for example
$$
    \Psi_1^0 \coloneqq \pm \frac1{d |\Delta|}\left(
    \begin{bmatrix} 
        0 \\ \tau_{01} \\ 0 \\
        0 \\ 0 \\ 0 
    \end{bmatrix}
    - (\psi_0 - \psi_1)
    \right).\text{ravel}()
$$

and for the cell-based degrees of freedom:
$$
    \Psi_0 \coloneqq \frac{(d + 1)^2}{d |\Delta|} \psi_0
$$

## 3D (work in progress)
