"""Module contains Spanning Tree visualizer tests."""

import tempfile
from pathlib import Path

import matplotlib
import pytest

import pygeon as pg

# Use non-interactive backend for testing
matplotlib.use("Agg")


@pytest.fixture(
    scope="module",
    params=["unit_sd_2d", "mdg_embedded_frac_2d", "octagon_sd_2d", "cart_sd_2d"],
)
def spt_sd_pair(request: pytest.FixtureRequest):
    # resolve the underlying fixture by name
    sd = request.getfixturevalue(request.param)
    spt = pg.SpanningTree(sd)
    return spt, sd


def test_plot_spanningtree(spt_sd_pair):
    pg.plot_spanningtree(*spt_sd_pair)


def test_plot_spanningtree_cotree(spt_sd_pair):
    pg.plot_spanningtree(*spt_sd_pair, draw_cotree=True)


def test_plot_spanningtree_option(spt_sd_pair):
    pg.plot_spanningtree(*spt_sd_pair, draw_grid=False)


def test_plot_spanningtree_save_image(spt_sd_pair):
    with tempfile.TemporaryDirectory() as tmpdir:
        fig_path = Path(tmpdir) / "spanning_tree.png"

        pg.plot_spanningtree(*spt_sd_pair, fig_name=str(fig_path))

        # Check that file was created
        assert fig_path.exists()
        assert fig_path.stat().st_size > 0
