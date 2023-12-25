import pytest
import os
from pandas import Interval
from shapely.geometry import (
    Polygon as ShapelyPolygon,
    Point as ShapelyPoint,
    LineString as ShapelyLine,
)

from src.environment import Environment
from src.obstacles.point import Point
from src.obstacles.line import Line
from src.obstacles.polygon import Polygon


class TestEnvironment:
    def test_create_environment(self):
        empty_env = Environment()
        assert empty_env is not None

        sh_pt1 = ShapelyPoint(0, 0)
        sh_pt1_copy = ShapelyPoint(0, 0)
        sh_pt2 = ShapelyPoint(1, 1)
        sh_pt2_copy = ShapelyPoint(1, 1)
        pt1 = Point(sh_pt1)
        pt2 = Point(sh_pt2, Interval(10, 20))

        sh_line1 = ShapelyLine([(2, 2), (3, 3)])
        sh_line1_copy = ShapelyLine([(2, 2), (3, 3)])
        sh_line2 = ShapelyLine([(4, 4), (5, 5)])
        sh_line2_copy = ShapelyLine([(4, 4), (5, 5)])
        line1 = Line(sh_line1)
        line2 = Line(sh_line2, Interval(10, 25), radius=0.5)

        sh_poly1 = ShapelyPolygon([(6, 6), (7, 7), (8, 8)])
        sh_poly1_copy = ShapelyPolygon([(6, 6), (7, 7), (8, 8)])
        sh_poly2 = ShapelyPolygon([(9, 9), (10, 10), (11, 11)])
        sh_poly2_copy = ShapelyPolygon([(9, 9), (10, 10), (11, 11)])
        poly1 = Polygon(sh_poly1)
        poly2 = Polygon(sh_poly2, Interval(10, 30), radius=3)

        # create an environment with point obstacles only
        env = Environment(obstacles=[pt1, pt2])
        assert env.obstacles[0].geometry == sh_pt1_copy
        assert env.obstacles[1].geometry == sh_pt2_copy

        assert env.obstacles[0].time_interval == None
        assert env.obstacles[1].time_interval == Interval(10, 20)

        assert env.obstacles[0].radius == 0
        assert env.obstacles[1].radius == 0
        assert env is not None

        # create an environment with line obstacles only
        env = Environment(obstacles=[line1, line2])
        assert env.obstacles[0].geometry == sh_line1_copy
        assert env.obstacles[1].geometry == sh_line2_copy

        assert env.obstacles[0].time_interval == None
        assert env.obstacles[1].time_interval == Interval(10, 25)

        assert env.obstacles[0].radius == 0
        assert env.obstacles[1].radius == 0.5
        assert env is not None

        # create an environment with polygon obstacles only
        env = Environment(obstacles=[poly1, poly2])
        assert env.obstacles[0].geometry == sh_poly1_copy
        assert env.obstacles[1].geometry == sh_poly2_copy

        assert env.obstacles[0].time_interval == None
        assert env.obstacles[1].time_interval == Interval(10, 30)

        assert env.obstacles[0].radius == 0
        assert env.obstacles[1].radius == 3
        assert env is not None

        # create an environment with all types of obstacles
        env = Environment(obstacles=[pt1, pt2, line1, line2, poly1, poly2])
        assert env.obstacles[0].geometry == sh_pt1_copy
        assert env.obstacles[1].geometry == sh_pt2_copy
        assert env.obstacles[2].geometry == sh_line1_copy
        assert env.obstacles[3].geometry == sh_line2_copy
        assert env.obstacles[4].geometry == sh_poly1_copy
        assert env.obstacles[5].geometry == sh_poly2_copy

        assert env.obstacles[0].time_interval == None
        assert env.obstacles[1].time_interval == Interval(10, 20)
        assert env.obstacles[2].time_interval == None
        assert env.obstacles[3].time_interval == Interval(10, 25)
        assert env.obstacles[4].time_interval == None
        assert env.obstacles[5].time_interval == Interval(10, 30)
        assert env is not None

    def test_reset_add_obstacles(self):
        sh_pt = ShapelyPoint(0, 0)
        pt = Point(sh_pt)

        sh_line = ShapelyLine([(2, 2), (3, 3)])
        line = Line(sh_line, Interval(10, 25), radius=0.5)

        sh_poly = ShapelyPolygon([(6, 6), (7, 7), (8, 8)])
        poly = Polygon(sh_poly, Interval(10, 30), radius=3)

        env = Environment(obstacles=[pt, line, poly])
        assert env.obstacles[0].geometry == sh_pt
        assert env.obstacles[1].geometry == sh_line
        assert env.obstacles[2].geometry == sh_poly

        env.reset()
        assert env.obstacles == []

        env.add_obstacles([pt, line, poly])
        assert env.obstacles[0].geometry == sh_pt
        assert env.obstacles[1].geometry == sh_line
        assert env.obstacles[2].geometry == sh_poly

    # def test_create_env_polygon(self):
    #     polygon = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
    #     env_poly = Environment([polygon])
    #     assert env_poly is not None
    #     assert env_poly.polygons[0] == polygon
    #     assert len(env_poly.polygons) == 1

    # def test_create_env_file(self):
    #     pass

    # def test_point_distance(self):
    #     poly1 = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
    #     poly2 = Polygon([(10, 10), (10, 11), (11, 11), (11, 10)])
    #     env = Environment([poly1, poly2])

    #     point = Point(0, 0)
    #     assert env.closest_polygon_distance(point) == 0
    #     point = Point(0.5, 0.5)
    #     assert env.closest_polygon_distance(point) == 0
    #     point = Point(5, 5)
    #     assert abs(env.closest_polygon_distance(point) - 5.656854249492381) < 1e-10
    #     point = Point(10, 10)
    #     assert env.closest_polygon_distance(point) == 0
    #     point = Point(10.5, 10.5)
    #     assert env.closest_polygon_distance(point) == 0

    # def test_line_distance(self):
    #     poly1 = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
    #     poly2 = Polygon([(10, 10), (10, 11), (11, 11), (11, 10)])
    #     env = Environment([poly1, poly2])

    #     line = LineString([(0, 0), (1, 1)])
    #     assert env.closest_line_distance(line) == 0
    #     line = LineString([(2, 2), (9, 9)])
    #     assert abs(env.closest_line_distance(line) - 1.4142135623730951) < 1e-10
    #     line = LineString([(10, 10), (11, 11)])
    #     assert env.closest_line_distance(line) == 0
    #     line = LineString([(5, 5), (15, 15)])
    #     assert env.closest_line_distance(line) == 0

    # def test_change_polygons(self):
    #     poly1 = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
    #     poly2 = Polygon([(10, 10), (10, 11), (11, 11), (11, 10)])
    #     env = Environment([poly1, poly2])

    #     poly3 = Polygon([(5, 5), (5, 6), (6, 6), (6, 5)])
    #     poly4 = Polygon([(15, 15), (15, 16), (16, 16), (16, 15)])
    #     env.change_polygons([poly3, poly4])
    #     assert env.polygons == [poly3, poly4]
    #     assert len(env.polygons) == 2

    # def test_file_safe_loading(self):
    #     poly1 = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
    #     poly2 = Polygon([(10, 10), (10, 11), (11, 11), (11, 10)])
    #     env = Environment([poly1, poly2])
    #     env.save("test_env.txt")
    #     env2 = Environment(filepath="test_env.txt")
    #     assert env.polygons == env2.polygons

    #     poly3 = Polygon([(5, 5), (5, 6), (6, 6), (6, 5)])
    #     poly4 = Polygon([(15, 15), (15, 16), (16, 16), (16, 15)])
    #     env.change_polygons([poly3, poly4])
    #     assert env.polygons != env2.polygons

    #     env3 = Environment()
    #     env3.load("test_env.txt")
    #     assert env2.polygons == env3.polygons
    #     os.remove("test_env.txt")
