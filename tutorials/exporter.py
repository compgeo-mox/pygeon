import porepy as pp
import pygeon as pg

import os

output_directory = os.path.join(os.path.dirname(__file__), "sol")

mdg = pg.unit_grid(2, 0.2)
sd = mdg.subdomains()[0]

P1 = pg.Lagrange1()
proj_u = P1.eval_at_cell_centers(sd)

T = 10

save = pp.Exporter(mdg, "dumb_solution", folder_name=output_directory)

for t in range(T):
    for sd, data in mdg.subdomains(return_data=True):
        u_func = lambda x: t * x[0]
        u = P1.interpolate(sd, u_func)

        # post process variables
        cell_u = proj_u @ u

        pp.set_solution_values("mass", cell_u, data, time_step_index=0)
        save.write_vtu(["mass"], time_step=t)

save.write_pvd(range(T))
