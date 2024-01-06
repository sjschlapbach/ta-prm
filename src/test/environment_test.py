import pytest
import os
from pandas import Interval
from shapely.geometry import (
    Polygon as ShapelyPolygon,
    Point as ShapelyPoint,
    LineString as ShapelyLine,
)

from src.envs.environment import Environment
from src.obstacles.point import Point
from src.obstacles.line import Line
from src.obstacles.polygon import Polygon
from src.util.recurrence import Recurrence


class TestEnvironment:
    def test_create_environment(self):
        empty_env = Environment()
        assert empty_env is not None
        assert empty_env.dim_x is None
        assert empty_env.dim_y is None

        sh_pt1 = ShapelyPoint(0, 0)
        sh_pt1_copy = ShapelyPoint(0, 0)
        sh_pt2 = ShapelyPoint(1, 1)
        sh_pt2_copy = ShapelyPoint(1, 1)

        sh_line1 = ShapelyLine([(2, 2), (3, 3)])
        sh_line1_copy = ShapelyLine([(2, 2), (3, 3)])
        sh_line2 = ShapelyLine([(4, 4), (5, 5)])
        sh_line2_copy = ShapelyLine([(4, 4), (5, 5)])

        sh_poly1 = ShapelyPolygon([(6, 6), (7, 7), (8, 8)])
        sh_poly1_copy = ShapelyPolygon([(6, 6), (7, 7), (8, 8)])
        sh_poly2 = ShapelyPolygon([(9, 9), (10, 10), (11, 11)])
        sh_poly2_copy = ShapelyPolygon([(9, 9), (10, 10), (11, 11)])

        # test that dimensionality is saved correctly
        env = Environment(dimension_x=(10, 20), dimension_y=(20, 30))
        assert env.dim_x == [10, 20]
        assert env.dim_y == [20, 30]

        # create an environment with point obstacles only
        pt1 = Point(geometry=sh_pt1)
        pt2 = Point(
            geometry=sh_pt2,
            time_interval=Interval(10, 20),
            recurrence=Recurrence.MINUTELY,
        )
        env = Environment(obstacles=[pt1, pt2])
        assert env.obstacles[0].recurrence == Recurrence.NONE
        assert env.obstacles[0].geometry == sh_pt1_copy
        assert env.obstacles[1].recurrence == Recurrence.MINUTELY
        assert env.obstacles[1].geometry == sh_pt2_copy

        assert env.obstacles[0].time_interval == None
        assert env.obstacles[1].time_interval == Interval(10, 20)

        assert env.obstacles[0].radius == 0
        assert env.obstacles[1].radius == 0
        assert env is not None

        # create an environment with line obstacles only
        line1 = Line(geometry=sh_line1, recurrence=Recurrence.MINUTELY)
        line2 = Line(
            geometry=sh_line2,
            time_interval=Interval(10, 25),
            radius=0.5,
            recurrence=Recurrence.HOURLY,
        )
        env = Environment(obstacles=[line1, line2])
        assert env.obstacles[0].recurrence == Recurrence.MINUTELY
        assert env.obstacles[0].geometry == sh_line1_copy
        assert env.obstacles[1].recurrence == Recurrence.HOURLY
        assert env.obstacles[1].geometry == sh_line2_copy

        assert env.obstacles[0].time_interval == None
        assert env.obstacles[1].time_interval == Interval(10, 25)

        assert env.obstacles[0].radius == 0
        assert env.obstacles[1].radius == 0.5
        assert env is not None

        # create an environment with polygon obstacles only
        poly1 = Polygon(geometry=sh_poly1, recurrence=Recurrence.HOURLY)
        poly2 = Polygon(
            geometry=sh_poly2,
            time_interval=Interval(10, 30),
            radius=3,
            recurrence=Recurrence.DAILY,
        )
        env = Environment(obstacles=[poly1, poly2])

        assert env.obstacles[0].recurrence == Recurrence.HOURLY
        assert env.obstacles[0].geometry == sh_poly1_copy
        assert env.obstacles[1].recurrence == Recurrence.DAILY
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

    def test_set_dimensions(self):
        env = Environment()
        assert env.dim_x is None
        assert env.dim_y is None

        env.set_dimensions((10, 20), (20, 30))
        assert env.dim_x == [10, 20]
        assert env.dim_y == [20, 30]

        env.set_dimensions((30, 40), (40, 50))
        assert env.dim_x == [30, 40]
        assert env.dim_y == [40, 50]

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

    def test_save_load(self):
        # initialize two point obstacles
        sh_pt = ShapelyPoint(0, 0)
        sh_pt2 = ShapelyPoint(1, 1)
        pt = Point(geometry=sh_pt, recurrence=Recurrence.MINUTELY)
        pt2 = Point(geometry=sh_pt2, time_interval=Interval(10, 20), radius=1)

        # initialize two line obstacles
        sh_line = ShapelyLine([(2, 2), (3, 3)])
        sh_line2 = ShapelyLine([(4, 4), (5, 5)])
        line = Line(
            geometry=sh_line,
            time_interval=Interval(10, 25),
            radius=0.5,
            recurrence=Recurrence.HOURLY,
        )
        line2 = Line(geometry=sh_line2, time_interval=Interval(10, 25), radius=0.5)

        # initialize two polygon obstacles
        sh_poly = ShapelyPolygon([(6, 6), (7, 7), (8, 8)])
        sh_poly2 = ShapelyPolygon([(9, 9), (10, 10), (11, 11)])
        poly = Polygon(
            geometry=sh_poly,
            time_interval=Interval(10, 30),
            radius=3,
            recurrence=Recurrence.DAILY,
        )
        poly2 = Polygon(geometry=sh_poly2, time_interval=Interval(10, 30), radius=3)

        # create an environment with all types of obstacles
        env = Environment(obstacles=[pt, line, poly, pt2, line2, poly2])

        assert env.obstacles[0].recurrence == Recurrence.MINUTELY
        assert env.obstacles[0].geometry == sh_pt
        assert env.obstacles[1].recurrence == Recurrence.HOURLY
        assert env.obstacles[1].geometry == sh_line
        assert env.obstacles[2].recurrence == Recurrence.DAILY
        assert env.obstacles[2].geometry == sh_poly
        assert env.obstacles[3].recurrence == Recurrence.NONE
        assert env.obstacles[3].geometry == sh_pt2
        assert env.obstacles[4].recurrence == Recurrence.NONE
        assert env.obstacles[4].geometry == sh_line2
        assert env.obstacles[5].recurrence == Recurrence.NONE
        assert env.obstacles[5].geometry == sh_poly2

        # save the environment to a file
        env.save("test_env.txt")

        # load the environment from the file
        env2 = Environment(filepath="test_env.txt")

        # check if the content of the environment is still the same (loaded in type order)
        assert env2.obstacles[0].recurrence == Recurrence.MINUTELY
        assert env2.obstacles[1].recurrence == Recurrence.NONE
        assert env2.obstacles[2].recurrence == Recurrence.HOURLY
        assert env2.obstacles[3].recurrence == Recurrence.NONE
        assert env2.obstacles[4].recurrence == Recurrence.DAILY
        assert env2.obstacles[5].recurrence == Recurrence.NONE
        assert env2.obstacles[0].geometry == sh_pt
        assert env2.obstacles[1].geometry == sh_pt2
        assert env2.obstacles[2].geometry == sh_line
        assert env2.obstacles[3].geometry == sh_line2
        assert env2.obstacles[4].geometry == sh_poly
        assert env2.obstacles[5].geometry == sh_poly2

        # check if the additional parameters (interval and radius are also correct)
        # if no interval or radius were specified, they default to None and 0 respectively
        assert env2.obstacles[0].time_interval == None
        assert env2.obstacles[0].radius == 0
        assert env2.obstacles[1].time_interval == Interval(10, 20)
        assert env2.obstacles[1].radius == 1
        assert env2.obstacles[2].time_interval == Interval(10, 25)
        assert env2.obstacles[2].radius == 0.5
        assert env2.obstacles[3].time_interval == Interval(10, 25)
        assert env2.obstacles[3].radius == 0.5
        assert env2.obstacles[4].time_interval == Interval(10, 30)
        assert env2.obstacles[4].radius == 3
        assert env2.obstacles[5].time_interval == Interval(10, 30)
        assert env2.obstacles[5].radius == 3

        # remove the file
        os.remove("test_env.txt")
