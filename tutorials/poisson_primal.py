import numpy as np
import porepy as pp
import pygeon as pg


def exact_sol(x, dim):
    return np.prod(np.cos(8 * x[:dim])) + np.exp(x[0])


def source(x, dim):
    return dim * (8**2) * np.prod(np.cos(8 * x[:dim])) - np.exp(x[0])


def create_grid(dim, h):
    if dim == 3:
        sd = pp.StructuredTetrahedralGrid([int(1 / h)] * dim, [1] * dim)
    elif dim == 2:
        sd = pp.StructuredTriangleGrid([int(1 / h)] * dim, [1] * dim)
    else:
        sd = pp.CartGrid([int(1 / h)], 1)
    pg.convert_from_pp(sd)
    sd.compute_geometry()
    h = np.mean(sd.cell_diameters())

    return sd, h


def solve_problem_P2(dim=2, h=0.5):
    sol_func = lambda x: exact_sol(x, dim)
    source_func = lambda x: source(x, dim)

    disc = pg.Lagrange2()
    sd, h = create_grid(dim, h)

    A = disc.assemble_stiff_matrix(sd, None)
    f = disc.source_term(sd, source_func)

    if sd.dim == 1:
        bdry_edges = np.zeros(sd.num_cells, dtype=bool)
    elif sd.dim == 2:
        bdry_edges = sd.tags["domain_boundary_faces"]
    elif sd.dim == 3:
        bdry_edges = sd.tags["domain_boundary_ridges"]
    ess_bc = np.hstack((sd.tags["domain_boundary_nodes"], bdry_edges), dtype=bool)
    ess_vals = np.zeros_like(ess_bc, dtype=float)

    interp_sol = disc.interpolate(sd, sol_func)
    ess_vals[ess_bc] = interp_sol[ess_bc]

    LS = pg.LinearSystem(A, f)
    LS.flag_ess_bc(ess_bc, ess_vals)

    u = LS.solve()

    return h, disc.ndof(sd), disc.error_l2(sd, u, sol_func, relative=False)


def solve_problem_P1(dim=2, h=0.5):
    sol_func = lambda x: exact_sol(x, dim)
    source_func = lambda x: source(x, dim)

    disc = pg.Lagrange1()
    sd, h = create_grid(dim, h)

    A = disc.assemble_stiff_matrix(sd, None)
    f = disc.source_term(sd, source_func)

    ess_bc = sd.tags["domain_boundary_nodes"]
    ess_vals = np.zeros_like(ess_bc, dtype=float)

    interp_sol = disc.interpolate(sd, sol_func)
    ess_vals[ess_bc] = interp_sol[ess_bc]

    LS = pg.LinearSystem(A, f)
    LS.flag_ess_bc(ess_bc, ess_vals)

    u = LS.solve()
    error = disc.error_l2(sd, u, sol_func, relative=False)

    return h, disc.ndof(sd), error


def convergence_test(solver=solve_problem_P1, dim=3, n_grids=4):
    h_list = 0.5 ** np.arange(1, n_grids + 1)
    errors = np.zeros_like(h_list)
    dofs = np.zeros_like(h_list)

    for ind, h in enumerate(h_list):
        h_list[ind], dofs[ind], errors[ind] = solver(dim, h)

    rates = np.zeros_like(errors)
    rates[1:] = np.log(errors[1:] / errors[:-1]) / np.log(h_list[1:] / h_list[:-1])

    conv_table = np.vstack((h_list, dofs, errors, rates)).T

    print("Solver = {}".format(solver.__name__))
    print("Dimension = {}".format(dim))
    print("    h         dof    error     rate")
    np.set_printoptions(formatter={"float": lambda x: format(x, ".2e")})
    print(conv_table)


for dim in [1, 2, 3]:
    for solver in [solve_problem_P1, solve_problem_P2]:
        convergence_test(solver, dim, 5)
