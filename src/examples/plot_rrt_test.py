import pytest
from .plot_rrt import plot_rrt


class TestGraphPlotting:
    def test_plot_rrt(self):
        plot_rrt(plotting=False)
