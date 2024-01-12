import pytest
from .plot_graph import plot_graph


class TestGraphPlotting:
    def test_plot_graphs(self):
        plot_graph(plotting=False)
