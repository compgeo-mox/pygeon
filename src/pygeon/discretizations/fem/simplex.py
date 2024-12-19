import numpy as np
import math


def compute_coefficient(vec: np.ndarray) -> float:
    dim = len(vec) - 1
    fac_a = [factorial(a_i) for a_i in vec]
    return factorial(dim) * np.prod(fac_a) / factorial(dim + np.sum(vec))


def assemble_coeff_mat(A: np.ndarray) -> np.ndarray:

    n_monomials = A.shape[0]
    C = np.zeros((n_monomials, n_monomials))

    for i in np.arange(n_monomials):
        for j in np.arange(n_monomials):
            C[i, j] = compute_coefficient(A[i] + A[j])

    return C


factorial = lambda x: math.factorial(int(x))

# 2D
dim = 2
I = np.eye(dim + 1)
O = np.zeros((dim + 1, dim + 1))
A = np.vstack((I, 1 - I, 2 * I))

C = assemble_coeff_mat(A)

basis_nodes = np.vstack((-I, O, 2 * I))
basis_edges = np.vstack((O, 4 * I, O))
basis = np.hstack((basis_nodes, basis_edges))

M = basis.T @ C @ basis

print(180 * M)

# 3D
dim = 3
I = np.eye(dim + 1)
E_0 = np.hstack((np.zeros((3, 1)), 1 - np.eye(3)))
E_1 = np.hstack((np.ones((3, 1)), np.eye(3)[::-1]))
E = np.vstack((E_0, E_1))
A = np.vstack((I, E, 2 * I))

C = assemble_coeff_mat(A)

O = np.zeros((6, 4))
basis_nodes = np.vstack((-I, O, 2 * I))
basis_edges = np.vstack((O.T, 4 * np.eye(6), O.T))
basis = np.hstack((basis_nodes, basis_edges))

M = basis.T @ C @ basis

print(420 * M)
