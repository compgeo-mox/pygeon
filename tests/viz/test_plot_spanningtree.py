"""Module contains Spanning Tree visualizer tests."""

import tempfile
from pathlib import Path

import matplotlib

import pygeon as pg

# Use non-interactive backend for testing
matplotlib.use("Agg")


def test_plot_spanningtree_sd(unit_sd_2d):
    mdg = pg.as_mdg(unit_sd_2d)
    spt = pg.SpanningTree(mdg)
    pg.plot_spanningtree(spt, mdg)

    pg.plot_spanningtree(spt, mdg, draw_cotree=True)

    pg.plot_spanningtree(spt, mdg, draw_grid=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        fig_path = Path(tmpdir) / "spanning_tree.png"

        pg.plot_spanningtree(spt, mdg, fig_name=str(fig_path))

        # Check that file was created
        assert fig_path.exists()
        assert fig_path.stat().st_size > 0

    assert True  # If no exceptions, the test passes


def test_plot_spanningtree_mdg(mdg_embedded_frac_2d):
    spt = pg.SpanningTree(mdg_embedded_frac_2d)
    pg.plot_spanningtree(spt, mdg_embedded_frac_2d)

    pg.plot_spanningtree(spt, mdg_embedded_frac_2d, draw_cotree=True)

    pg.plot_spanningtree(spt, mdg_embedded_frac_2d, draw_grid=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        fig_path = Path(tmpdir) / "spanning_tree.png"

        pg.plot_spanningtree(spt, mdg_embedded_frac_2d, fig_name=str(fig_path))

        # Check that file was created
        assert fig_path.exists()
        assert fig_path.stat().st_size > 0

    assert True  # If no exceptions, the test passes
