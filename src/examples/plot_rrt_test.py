import pytest
from .plot_rrt import plot_rrt
from .plot_rrt_star import plot_rrt_star
from .plot_rrt_replanning import plot_rrt_replanning


class TestGraphPlotting:
    def test_plot_rrt(self):
        plot_rrt(plotting=False)

    def test_plot_rrt_star(self):
        plot_rrt_star(plotting=False)

    def test_replanning_rrt(self):
        plot_rrt_replanning(plotting=False)
