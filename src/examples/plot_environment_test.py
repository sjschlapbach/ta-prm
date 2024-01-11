import pytest

from src.examples.plot_environment import plot_environment
from src.examples.plot_environment_instance import plot_environment_instance
from src.examples.plot_random_environment import plot_random_environment


class TestEnvironmentPlotting:
    def test_plot_environment(self):
        plot_environment(plotting=False)

    def test_plot_environment_instance(self):
        plot_environment_instance(plotting=False)

    def test_plot_random_environment(self):
        plot_random_environment(plotting=False)
