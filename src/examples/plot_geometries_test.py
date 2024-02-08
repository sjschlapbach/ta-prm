import pytest
from .plot_geometries import plot_geometries


class TestGeometriesPlotting:
    def test_plot_geometries(self):
        plot_geometries(plotting=False, query_time=10.5)
