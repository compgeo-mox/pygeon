import porepy as pp

def hdiv_mass(grid, discr, data, data_key="flow"):
    if isinstance(grid, pp.Grid):
        discr.discretize(grid, data)
        return data[pp.DISCRETIZATION_MATRICES][data_key]["mass"]
    elif isinstance(grid, pp.GridBucket):
        return _gb_hdiv_mass(grid, discr, data, data_key)

def _gb_hdiv_mass(gb, discr, data, data_key):
    gb_hdiv_mass = np.empty(shape=(gb.size(), gb.size()), dtype=sps.spmatrix)

    # Local mass
    for g, d_g in gb:
        nn_g = d_g["node_number"]
        gb_hdiv_mass[nn_g, nn_g] = hdiv_mass(g, discr, d_g, data_key)

    # mortar contributions
    for e, d_e in gb.edges():
        # Get adjacent grids and mortar_grid
        g_down, g_up = gb.nodes_of_edge(e)
        mg = d_e['mortar_grid']

        # Get indices in grid_bucket
        nn_g_d = gb.node_props(g_down, 'node_number')
        nn_g_u = gb.node_props(g_up, 'node_number')
        nn_mg = d_e['edge_number'] + gb.num_graph_nodes()

        # Place in the matrix
        gb_div[nn_g_d, nn_mg] = - mg.mortar_to_secondary_int()
        gb_div[nn_g_u, nn_mg] = div(g_up) * signed_mortar_to_primary(gb, e)

    return sps.bmat(gb_div, format='csc')




