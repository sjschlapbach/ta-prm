import pytest
from .plot_rrt_tree import plot_rrt_tree


class TestGraphPlotting:
    def test_plot_rrt_trees(self):
        plot_rrt_tree(plotting=False)
