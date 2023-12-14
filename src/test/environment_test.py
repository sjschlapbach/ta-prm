import pytest
import os
from shapely.geometry import Polygon, Point, LineString

from src.environment import Environment


class TestEnvironment:
    def test_create_environment(self):
        empty_env = Environment()
        assert empty_env is not None

    def test_create_env_polygon(self):
        polygon = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
        env_poly = Environment([polygon])
        assert env_poly is not None
        assert env_poly.polygons[0] == polygon
        assert len(env_poly.polygons) == 1

    def test_create_env_file(self):
        pass

    def test_point_distance(self):
        poly1 = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
        poly2 = Polygon([(10, 10), (10, 11), (11, 11), (11, 10)])
        env = Environment([poly1, poly2])

        point = Point(0, 0)
        assert env.closest_polygon_distance(point) == 0
        point = Point(0.5, 0.5)
        assert env.closest_polygon_distance(point) == 0
        point = Point(5, 5)
        assert abs(env.closest_polygon_distance(point) - 5.656854249492381) < 1e-10
        point = Point(10, 10)
        assert env.closest_polygon_distance(point) == 0
        point = Point(10.5, 10.5)
        assert env.closest_polygon_distance(point) == 0

    def test_line_distance(self):
        poly1 = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
        poly2 = Polygon([(10, 10), (10, 11), (11, 11), (11, 10)])
        env = Environment([poly1, poly2])

        line = LineString([(0, 0), (1, 1)])
        assert env.closest_line_distance(line) == 0
        line = LineString([(2, 2), (9, 9)])
        assert abs(env.closest_line_distance(line) - 1.4142135623730951) < 1e-10
        line = LineString([(10, 10), (11, 11)])
        assert env.closest_line_distance(line) == 0
        line = LineString([(5, 5), (15, 15)])
        assert env.closest_line_distance(line) == 0

    def test_change_polygons(self):
        poly1 = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
        poly2 = Polygon([(10, 10), (10, 11), (11, 11), (11, 10)])
        env = Environment([poly1, poly2])

        poly3 = Polygon([(5, 5), (5, 6), (6, 6), (6, 5)])
        poly4 = Polygon([(15, 15), (15, 16), (16, 16), (16, 15)])
        env.change_polygons([poly3, poly4])
        assert env.polygons == [poly3, poly4]
        assert len(env.polygons) == 2

    def test_file_safe_loading(self):
        poly1 = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
        poly2 = Polygon([(10, 10), (10, 11), (11, 11), (11, 10)])
        env = Environment([poly1, poly2])
        env.save("test_env.txt")
        env2 = Environment(filepath="test_env.txt")
        assert env.polygons == env2.polygons

        poly3 = Polygon([(5, 5), (5, 6), (6, 6), (6, 5)])
        poly4 = Polygon([(15, 15), (15, 16), (16, 16), (16, 15)])
        env.change_polygons([poly3, poly4])
        assert env.polygons != env2.polygons

        env3 = Environment()
        env3.load("test_env.txt")
        assert env2.polygons == env3.polygons
        os.remove("test_env.txt")
