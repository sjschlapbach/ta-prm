import pytest
from .plot_rrt import plot_rrt
from .plot_rrt_star import plot_rrt_star


class TestGraphPlotting:
    def test_plot_rrt(self):
        plot_rrt(plotting=False)

    def test_plot_rrt_star(self):
        plot_rrt_star(plotting=False)
