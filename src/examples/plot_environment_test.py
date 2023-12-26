import pytest
from .plot_environment import plot_environment


class TestEnvironmentPlotting:
    def test_plot_environment(self):
        plot_environment(plotting=False)
