"""Module contains Spanning Tree visualizer tests."""

import tempfile
from pathlib import Path

import matplotlib
import pytest

import pygeon as pg

# Use non-interactive backend for testing
matplotlib.use("Agg")


@pytest.fixture(
    params=["unit_sd_2d", "mdg_embedded_frac_2d", "octagon_sd_2d", "cart_sd_2d"]
)
def grid_2d(request: pytest.FixtureRequest):
    # resolve the underlying fixture by name
    return request.getfixturevalue(request.param)


def test_plot_spanningtree(grid_2d):
    spt = pg.SpanningTree(grid_2d)
    pg.plot_spanningtree(spt, grid_2d)


def test_plot_spanningtree_cotree(grid_2d):
    spt = pg.SpanningTree(grid_2d)
    pg.plot_spanningtree(spt, grid_2d, draw_cotree=True)


def test_plot_spanningtree_option(grid_2d):
    spt = pg.SpanningTree(grid_2d)
    pg.plot_spanningtree(spt, grid_2d, draw_grid=False)


def test_plot_spanningtree_save_image(grid_2d):
    spt = pg.SpanningTree(grid_2d)
    with tempfile.TemporaryDirectory() as tmpdir:
        fig_path = Path(tmpdir) / "spanning_tree.png"

        pg.plot_spanningtree(spt, grid_2d, fig_name=str(fig_path))

        # Check that file was created
        assert fig_path.exists()
        assert fig_path.stat().st_size > 0
