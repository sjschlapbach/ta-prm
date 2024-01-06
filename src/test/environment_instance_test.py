import pytest
import os
from pandas import Interval
from shapely.geometry import (
    Polygon as ShapelyPolygon,
    Point as ShapelyPoint,
    LineString as ShapelyLine,
)

from src.envs.environment import Environment
from src.envs.environment_instance import EnvironmentInstance
from src.obstacles.point import Point
from src.obstacles.line import Line
from src.obstacles.polygon import Polygon
from src.util.recurrence import Recurrence as Rec


class TestEnvironmentInstance:
    def test_create_environment_instance_intervals(self):
        # create shapely points and copies for later comparison
        spt_static = ShapelyPoint(0, 0)
        spt_static_copy = ShapelyPoint(0, 0)
        spt_limited = ShapelyPoint(1, 1)
        spt_limited_copy = ShapelyPoint(1, 1)
        spt_minute = ShapelyPoint(2, 2)
        spt_minute_copy = ShapelyPoint(2, 2)
        spt_hour = ShapelyPoint(3, 3)
        spt_hour_copy = ShapelyPoint(3, 3)
        spt_day = ShapelyPoint(4, 4)
        spt_day_copy = ShapelyPoint(4, 4)

        # create point obstacles with different recurrence and intervals
        interval = Interval(10, 20, closed="both")
        pt_static = Point(geometry=spt_static, radius=1.0)
        pt_limited = Point(
            geometry=spt_limited,
            time_interval=interval,
            radius=2.0,
        )
        pt_minute = Point(
            geometry=spt_minute,
            time_interval=interval,
            recurrence=Rec.MINUTELY,
            radius=3.0,
        )
        pt_hour = Point(
            geometry=spt_hour, time_interval=interval, recurrence=Rec.HOURLY, radius=4.0
        )
        pt_day = Point(
            geometry=spt_day, time_interval=interval, recurrence=Rec.DAILY, radius=5.0
        )

        # Test case 0: no obstacles - should create empty obstacles
        env = Environment(obstacles=[])
        env_instance = EnvironmentInstance(env, Interval(10, 30))
        assert len(env_instance.static_obstacles) == 0
        assert len(env_instance.dynamic_obstacles) == 0

        # Test case 1: single static obstacle - should be added to static obstacles
        env = Environment(obstacles=[pt_static])
        env_instance = EnvironmentInstance(env, Interval(10, 30))
        assert len(env_instance.static_obstacles) == 1
        assert len(env_instance.dynamic_obstacles) == 0
        saved_obstacle = env_instance.static_obstacles[1]
        assert saved_obstacle.geometry == spt_static_copy
        assert saved_obstacle.time_interval is None
        assert saved_obstacle.recurrence == Rec.NONE
        assert saved_obstacle.radius == 1.0

        # Test case 2: single static limited obstacle, overlapping with start of query interval
        # should be added to dynamic obstacles
        env = Environment(obstacles=[pt_limited])
        env_instance = EnvironmentInstance(env, Interval(15, 30))
        assert len(env_instance.static_obstacles) == 0
        assert len(env_instance.dynamic_obstacles) == 1
        saved_obstacle = env_instance.dynamic_obstacles[1]
        assert saved_obstacle.geometry == spt_limited_copy
        assert saved_obstacle.time_interval == interval
        assert saved_obstacle.recurrence == Rec.NONE
        assert saved_obstacle.radius == 2.0

        # Test case 3: single static limited obstacle, lying inside query interval
        # should be added to dynamic obstacles
        env = Environment(obstacles=[pt_limited])
        env_instance = EnvironmentInstance(env, Interval(5, 30))
        assert len(env_instance.static_obstacles) == 0
        assert len(env_instance.dynamic_obstacles) == 1
        saved_obstacle = env_instance.dynamic_obstacles[1]
        assert saved_obstacle.geometry == spt_limited_copy
        assert saved_obstacle.time_interval == interval
        assert saved_obstacle.recurrence == Rec.NONE
        assert saved_obstacle.radius == 2.0

        # Test case 4: single static limited obstacle, overlapping with end of query interval
        # should be added to dynamic obstacles
        env = Environment(obstacles=[pt_limited])
        env_instance = EnvironmentInstance(env, Interval(5, 15))
        assert len(env_instance.static_obstacles) == 0
        assert len(env_instance.dynamic_obstacles) == 1
        saved_obstacle = env_instance.dynamic_obstacles[1]
        assert saved_obstacle.geometry == spt_limited_copy
        assert saved_obstacle.time_interval == interval
        assert saved_obstacle.recurrence == Rec.NONE
        assert saved_obstacle.radius == 2.0

        # Test case 5: single static limited obstacle, overlapping with start and end of query interval
        # should be added to static obstacles
        env = Environment(obstacles=[pt_limited])
        env_instance = EnvironmentInstance(env, Interval(12, 15))
        assert len(env_instance.static_obstacles) == 1
        assert len(env_instance.dynamic_obstacles) == 0
        saved_obstacle = env_instance.static_obstacles[1]
        assert saved_obstacle.geometry == spt_limited_copy
        assert saved_obstacle.time_interval is None
        assert saved_obstacle.recurrence == Rec.NONE
        assert saved_obstacle.radius == 2.0

        # Test case 6: recurring obstacle with first occurence overlapping with start of query interval
        # should be added to dynamic obstacles
        env1 = Environment(obstacles=[pt_minute])
        env_instance1 = EnvironmentInstance(env1, Interval(15, 30))
        assert len(env_instance1.static_obstacles) == 0
        assert len(env_instance1.dynamic_obstacles) == 1
        saved_obstacle = env_instance1.dynamic_obstacles[1]
        assert saved_obstacle.geometry == spt_minute_copy
        assert saved_obstacle.time_interval == interval
        assert saved_obstacle.recurrence == Rec.MINUTELY
        assert saved_obstacle.radius == 3.0

        env2 = Environment(obstacles=[pt_hour])
        env_instance2 = EnvironmentInstance(env2, Interval(15, 30))
        assert len(env_instance2.static_obstacles) == 0
        assert len(env_instance2.dynamic_obstacles) == 1
        saved_obstacle = env_instance2.dynamic_obstacles[1]
        assert saved_obstacle.geometry == spt_hour_copy
        assert saved_obstacle.time_interval == interval
        assert saved_obstacle.recurrence == Rec.HOURLY
        assert saved_obstacle.radius == 4.0

        env3 = Environment(obstacles=[pt_day])
        env_instance3 = EnvironmentInstance(env3, Interval(15, 30))
        assert len(env_instance3.static_obstacles) == 0
        assert len(env_instance3.dynamic_obstacles) == 1
        saved_obstacle = env_instance3.dynamic_obstacles[1]
        assert saved_obstacle.geometry == spt_day_copy
        assert saved_obstacle.time_interval == interval
        assert saved_obstacle.recurrence == Rec.DAILY
        assert saved_obstacle.radius == 5.0

        # Test case 7: recurring obstacle with first occurrence lying inside query interval
        # should be added to dynamic obstacles
        env1 = Environment(obstacles=[pt_minute])
        env_instance1 = EnvironmentInstance(env1, Interval(5, 30))
        assert len(env_instance1.static_obstacles) == 0
        assert len(env_instance1.dynamic_obstacles) == 1
        saved_obstacle = env_instance1.dynamic_obstacles[1]
        assert saved_obstacle.geometry == spt_minute_copy
        assert saved_obstacle.time_interval == interval
        assert saved_obstacle.recurrence == Rec.MINUTELY
        assert saved_obstacle.radius == 3.0

        env2 = Environment(obstacles=[pt_hour])
        env_instance2 = EnvironmentInstance(env2, Interval(5, 30))
        assert len(env_instance2.static_obstacles) == 0
        assert len(env_instance2.dynamic_obstacles) == 1
        saved_obstacle = env_instance2.dynamic_obstacles[1]
        assert saved_obstacle.geometry == spt_hour_copy
        assert saved_obstacle.time_interval == interval
        assert saved_obstacle.recurrence == Rec.HOURLY
        assert saved_obstacle.radius == 4.0

        env3 = Environment(obstacles=[pt_day])
        env_instance3 = EnvironmentInstance(env3, Interval(5, 30))
        assert len(env_instance3.static_obstacles) == 0
        assert len(env_instance3.dynamic_obstacles) == 1
        saved_obstacle = env_instance3.dynamic_obstacles[1]
        assert saved_obstacle.geometry == spt_day_copy
        assert saved_obstacle.time_interval == interval
        assert saved_obstacle.recurrence == Rec.DAILY
        assert saved_obstacle.radius == 5.0

        # Test case 8: recurring obstacle with first occurrence overlapping with end of query interval
        # should be added to dynamic obstacles
        env1 = Environment(obstacles=[pt_minute])
        env_instance1 = EnvironmentInstance(env1, Interval(5, 15))
        assert len(env_instance1.static_obstacles) == 0
        assert len(env_instance1.dynamic_obstacles) == 1
        saved_obstacle = env_instance1.dynamic_obstacles[1]
        assert saved_obstacle.geometry == spt_minute_copy
        assert saved_obstacle.time_interval == interval
        assert saved_obstacle.recurrence == Rec.MINUTELY
        assert saved_obstacle.radius == 3.0

        env2 = Environment(obstacles=[pt_hour])
        env_instance2 = EnvironmentInstance(env2, Interval(5, 15))
        assert len(env_instance2.static_obstacles) == 0
        assert len(env_instance2.dynamic_obstacles) == 1
        saved_obstacle = env_instance2.dynamic_obstacles[1]
        assert saved_obstacle.geometry == spt_hour_copy
        assert saved_obstacle.time_interval == interval
        assert saved_obstacle.recurrence == Rec.HOURLY
        assert saved_obstacle.radius == 4.0

        env3 = Environment(obstacles=[pt_day])
        env_instance3 = EnvironmentInstance(env3, Interval(5, 15))
        assert len(env_instance3.static_obstacles) == 0
        assert len(env_instance3.dynamic_obstacles) == 1
        saved_obstacle = env_instance3.dynamic_obstacles[1]
        assert saved_obstacle.geometry == spt_day_copy
        assert saved_obstacle.time_interval == interval
        assert saved_obstacle.recurrence == Rec.DAILY
        assert saved_obstacle.radius == 5.0

        # Test case 9: recurring obstacle with first occurrence overlapping with start and end of query interval
        # should be added to static obstacles
        env1 = Environment(obstacles=[pt_minute])
        env_instance1 = EnvironmentInstance(env1, Interval(12, 15))
        assert len(env_instance1.static_obstacles) == 1
        assert len(env_instance1.dynamic_obstacles) == 0
        saved_obstacle = env_instance1.static_obstacles[1]
        assert saved_obstacle.geometry == spt_minute_copy
        assert saved_obstacle.time_interval is None
        assert saved_obstacle.recurrence == Rec.NONE
        assert saved_obstacle.radius == 3.0

        env2 = Environment(obstacles=[pt_hour])
        env_instance2 = EnvironmentInstance(env2, Interval(12, 15))
        assert len(env_instance2.static_obstacles) == 1
        assert len(env_instance2.dynamic_obstacles) == 0
        saved_obstacle = env_instance2.static_obstacles[1]
        assert saved_obstacle.geometry == spt_hour_copy
        assert saved_obstacle.time_interval is None
        assert saved_obstacle.recurrence == Rec.NONE
        assert saved_obstacle.radius == 4.0

        env3 = Environment(obstacles=[pt_day])
        env_instance3 = EnvironmentInstance(env3, Interval(12, 15))
        assert len(env_instance3.static_obstacles) == 1
        assert len(env_instance3.dynamic_obstacles) == 0
        saved_obstacle = env_instance3.static_obstacles[1]
        assert saved_obstacle.geometry == spt_day_copy
        assert saved_obstacle.time_interval is None
        assert saved_obstacle.recurrence == Rec.NONE
        assert saved_obstacle.radius == 5.0

        # Test case 10: recurring obstacle with third occurence overlapping with start of query interval
        # should be added to dynamic obstacles
        env1 = Environment(obstacles=[pt_minute])
        env_instance1 = EnvironmentInstance(env1, Interval(135, 150))
        assert len(env_instance1.static_obstacles) == 0
        assert len(env_instance1.dynamic_obstacles) == 1
        saved_obstacle = env_instance1.dynamic_obstacles[1]
        assert saved_obstacle.geometry == spt_minute_copy
        assert saved_obstacle.time_interval == interval
        assert saved_obstacle.recurrence == Rec.MINUTELY
        assert saved_obstacle.radius == 3.0

        env2 = Environment(obstacles=[pt_hour])
        env_instance2_wrong = EnvironmentInstance(env2, Interval(135, 150))
        assert len(env_instance2_wrong.static_obstacles) == 0
        assert len(env_instance2_wrong.dynamic_obstacles) == 0
        env_instance2 = EnvironmentInstance(env2, Interval(7215, 7230))
        assert len(env_instance2.static_obstacles) == 0
        assert len(env_instance2.dynamic_obstacles) == 1
        saved_obstacle = env_instance2.dynamic_obstacles[1]
        assert saved_obstacle.geometry == spt_hour_copy
        assert saved_obstacle.time_interval == interval
        assert saved_obstacle.recurrence == Rec.HOURLY
        assert saved_obstacle.radius == 4.0

        env3 = Environment(obstacles=[pt_day])
        env_instance3_wrong = EnvironmentInstance(env2, Interval(135, 150))
        assert len(env_instance3_wrong.static_obstacles) == 0
        assert len(env_instance3_wrong.dynamic_obstacles) == 0
        env_instance3 = EnvironmentInstance(env3, Interval(172815, 172830))
        assert len(env_instance3.static_obstacles) == 0
        assert len(env_instance3.dynamic_obstacles) == 1
        saved_obstacle = env_instance3.dynamic_obstacles[1]
        assert saved_obstacle.geometry == spt_day_copy
        assert saved_obstacle.time_interval == interval
        assert saved_obstacle.recurrence == Rec.DAILY
        assert saved_obstacle.radius == 5.0

        # Test case 11: recurring obstacle with third occurrence lying inside query interval
        # should be added to dynamic obstacles
        env1 = Environment(obstacles=[pt_minute])
        env_instance1 = EnvironmentInstance(env1, Interval(125, 150))
        assert len(env_instance1.static_obstacles) == 0
        assert len(env_instance1.dynamic_obstacles) == 1
        saved_obstacle = env_instance1.dynamic_obstacles[1]
        assert saved_obstacle.geometry == spt_minute_copy
        assert saved_obstacle.time_interval == interval
        assert saved_obstacle.recurrence == Rec.MINUTELY
        assert saved_obstacle.radius == 3.0

        env2 = Environment(obstacles=[pt_hour])
        env_instance2 = EnvironmentInstance(env2, Interval(7205, 7230))
        assert len(env_instance2.static_obstacles) == 0
        assert len(env_instance2.dynamic_obstacles) == 1
        saved_obstacle = env_instance2.dynamic_obstacles[1]
        assert saved_obstacle.geometry == spt_hour_copy
        assert saved_obstacle.time_interval == interval
        assert saved_obstacle.recurrence == Rec.HOURLY
        assert saved_obstacle.radius == 4.0

        env3 = Environment(obstacles=[pt_day])
        env_instance3 = EnvironmentInstance(env3, Interval(172805, 172830))
        assert len(env_instance3.static_obstacles) == 0
        assert len(env_instance3.dynamic_obstacles) == 1
        saved_obstacle = env_instance3.dynamic_obstacles[1]
        assert saved_obstacle.geometry == spt_day_copy
        assert saved_obstacle.time_interval == interval
        assert saved_obstacle.recurrence == Rec.DAILY
        assert saved_obstacle.radius == 5.0

        # Test case 12: recurring obstacle with third occurrence overlapping with end of query interval
        # should be added to dynamic obstacles
        env1 = Environment(obstacles=[pt_minute])
        env_instance1 = EnvironmentInstance(env1, Interval(125, 135))
        assert len(env_instance1.static_obstacles) == 0
        assert len(env_instance1.dynamic_obstacles) == 1
        saved_obstacle = env_instance1.dynamic_obstacles[1]
        assert saved_obstacle.geometry == spt_minute_copy
        assert saved_obstacle.time_interval == interval
        assert saved_obstacle.recurrence == Rec.MINUTELY
        assert saved_obstacle.radius == 3.0

        env2 = Environment(obstacles=[pt_hour])
        env_instance2 = EnvironmentInstance(env2, Interval(7205, 7215))
        assert len(env_instance2.static_obstacles) == 0
        assert len(env_instance2.dynamic_obstacles) == 1
        saved_obstacle = env_instance2.dynamic_obstacles[1]
        assert saved_obstacle.geometry == spt_hour_copy
        assert saved_obstacle.time_interval == interval
        assert saved_obstacle.recurrence == Rec.HOURLY
        assert saved_obstacle.radius == 4.0

        env3 = Environment(obstacles=[pt_day])
        env_instance3 = EnvironmentInstance(env3, Interval(172805, 172815))
        assert len(env_instance3.static_obstacles) == 0
        assert len(env_instance3.dynamic_obstacles) == 1
        saved_obstacle = env_instance3.dynamic_obstacles[1]
        assert saved_obstacle.geometry == spt_day_copy
        assert saved_obstacle.time_interval == interval
        assert saved_obstacle.recurrence == Rec.DAILY
        assert saved_obstacle.radius == 5.0

        # Test case 13: recurring obstacle with third occurrence overlapping with start and end of query interval
        # should be added to static obstacles
        env1 = Environment(obstacles=[pt_minute])
        env_instance1 = EnvironmentInstance(env1, Interval(132, 135))
        assert len(env_instance1.static_obstacles) == 1
        assert len(env_instance1.dynamic_obstacles) == 0
        saved_obstacle = env_instance1.static_obstacles[1]
        assert saved_obstacle.geometry == spt_minute_copy
        assert saved_obstacle.time_interval is None
        assert saved_obstacle.recurrence == Rec.NONE
        assert saved_obstacle.radius == 3.0

        env2 = Environment(obstacles=[pt_hour])
        env_instance2 = EnvironmentInstance(env2, Interval(7212, 7215))
        assert len(env_instance2.static_obstacles) == 1
        assert len(env_instance2.dynamic_obstacles) == 0
        saved_obstacle = env_instance2.static_obstacles[1]
        assert saved_obstacle.geometry == spt_hour_copy
        assert saved_obstacle.time_interval is None
        assert saved_obstacle.recurrence == Rec.NONE
        assert saved_obstacle.radius == 4.0

        env3 = Environment(obstacles=[pt_day])
        env_instance3 = EnvironmentInstance(env3, Interval(172812, 172815))
        assert len(env_instance3.static_obstacles) == 1
        assert len(env_instance3.dynamic_obstacles) == 0
        saved_obstacle = env_instance3.static_obstacles[1]
        assert saved_obstacle.geometry == spt_day_copy
        assert saved_obstacle.time_interval is None
        assert saved_obstacle.recurrence == Rec.NONE
        assert saved_obstacle.radius == 5.0

        # Test cast 14: Query interval spanning over multiple occurences
        # obstacle should be added to dynamic obstacles
        env1 = Environment(obstacles=[pt_minute])
        env_instance1 = EnvironmentInstance(env1, Interval(5, 145))
        assert len(env_instance1.static_obstacles) == 0
        assert len(env_instance1.dynamic_obstacles) == 1
        saved_obstacle = env_instance1.dynamic_obstacles[1]
        assert saved_obstacle.geometry == spt_minute_copy
        assert saved_obstacle.time_interval == interval
        assert saved_obstacle.recurrence == Rec.MINUTELY
        assert saved_obstacle.radius == 3.0

        env2 = Environment(obstacles=[pt_minute])
        env_instance2 = EnvironmentInstance(env2, Interval(125, 265))
        assert len(env_instance2.static_obstacles) == 0
        assert len(env_instance2.dynamic_obstacles) == 1
        saved_obstacle1 = env_instance2.dynamic_obstacles[1]
        assert saved_obstacle1.geometry == spt_minute_copy
        assert saved_obstacle1.time_interval == interval
        assert saved_obstacle1.recurrence == Rec.MINUTELY
        assert saved_obstacle1.radius == 3.0

        env3 = Environment(obstacles=[pt_hour])
        env_instance3 = EnvironmentInstance(env3, Interval(5, 7225))
        assert len(env_instance3.static_obstacles) == 0
        assert len(env_instance3.dynamic_obstacles) == 1
        saved_obstacle = env_instance3.dynamic_obstacles[1]
        assert saved_obstacle.geometry == spt_hour_copy
        assert saved_obstacle.time_interval == interval
        assert saved_obstacle.recurrence == Rec.HOURLY
        assert saved_obstacle.radius == 4.0

        env4 = Environment(obstacles=[pt_hour])
        env_instance4 = EnvironmentInstance(env4, Interval(7205, 14425))
        assert len(env_instance4.static_obstacles) == 0
        assert len(env_instance4.dynamic_obstacles) == 1
        saved_obstacle1 = env_instance4.dynamic_obstacles[1]
        assert saved_obstacle1.geometry == spt_hour_copy
        assert saved_obstacle1.time_interval == interval
        assert saved_obstacle1.recurrence == Rec.HOURLY
        assert saved_obstacle1.radius == 4.0

        env5 = Environment(obstacles=[pt_day])
        env_instance5 = EnvironmentInstance(env5, Interval(5, 172825))
        assert len(env_instance5.static_obstacles) == 0
        assert len(env_instance5.dynamic_obstacles) == 1
        saved_obstacle = env_instance5.dynamic_obstacles[1]
        assert saved_obstacle.geometry == spt_day_copy
        assert saved_obstacle.time_interval == interval
        assert saved_obstacle.recurrence == Rec.DAILY
        assert saved_obstacle.radius == 5.0

        env6 = Environment(obstacles=[pt_day])
        env_instance6 = EnvironmentInstance(env6, Interval(172805, 345625))
        assert len(env_instance6.static_obstacles) == 0
        assert len(env_instance6.dynamic_obstacles) == 1
        saved_obstacle1 = env_instance6.dynamic_obstacles[1]
        assert saved_obstacle1.geometry == spt_day_copy
        assert saved_obstacle1.time_interval == interval
        assert saved_obstacle1.recurrence == Rec.DAILY
        assert saved_obstacle1.radius == 5.0

        # Test case 15: Test that obstacles starting after query interval should not be added
        env = Environment(obstacles=[pt_limited, pt_minute, pt_hour, pt_day])
        env_instance = EnvironmentInstance(env, Interval(5, 9))
        assert len(env_instance.static_obstacles) == 0
        assert len(env_instance.dynamic_obstacles) == 0

        # Test case 16: Test that obstacle ending before query interval should not be added
        env = Environment(obstacles=[pt_limited])
        env_instance = EnvironmentInstance(env, Interval(25, 30))
        assert len(env_instance.static_obstacles) == 0
        assert len(env_instance.dynamic_obstacles) == 0

        # Test case 17: Test that query intervals falling in between obstacles should cause obstacles to be added
        env = Environment(obstacles=[pt_minute, pt_hour, pt_day])
        env_instance = EnvironmentInstance(env, Interval(25, 40))
        assert len(env_instance.static_obstacles) == 0
        assert len(env_instance.dynamic_obstacles) == 0

        # Test case 18: Test that query intervals falling in between repetitions of obstacles
        # should cause obstacles to be added
        env1 = Environment(obstacles=[pt_minute])
        env_instance1 = EnvironmentInstance(env1, Interval(145, 160))
        assert len(env_instance1.static_obstacles) == 0
        assert len(env_instance1.dynamic_obstacles) == 0

        env2 = Environment(obstacles=[pt_hour])
        env_instance2 = EnvironmentInstance(env2, Interval(7225, 7240))
        assert len(env_instance2.static_obstacles) == 0
        assert len(env_instance2.dynamic_obstacles) == 0

        env3 = Environment(obstacles=[pt_day])
        env_instance3 = EnvironmentInstance(env3, Interval(172825, 172840))
        assert len(env_instance3.static_obstacles) == 0
        assert len(env_instance3.dynamic_obstacles) == 0

        # Test case 19: add multiple objects and check if they are all added correctly
        env1 = Environment(obstacles=[pt_minute, pt_hour, pt_day])
        env_instance1 = EnvironmentInstance(env1, Interval(172805, 172815))
        assert len(env_instance1.static_obstacles) == 0
        assert len(env_instance1.dynamic_obstacles) == 3

        env2 = Environment(obstacles=[pt_minute, pt_hour, pt_day])
        env_instance2 = EnvironmentInstance(env2, Interval(172815, 172817))
        assert len(env_instance2.static_obstacles) == 3
        assert len(env_instance2.dynamic_obstacles) == 0

        env3 = Environment(obstacles=[pt_limited, pt_minute, pt_hour, pt_day])
        env_instance3 = EnvironmentInstance(env3, Interval(5, 15))
        assert len(env_instance3.static_obstacles) == 0
        assert len(env_instance3.dynamic_obstacles) == 4

        env4 = Environment(obstacles=[pt_limited, pt_minute, pt_hour, pt_day])
        env_instance4 = EnvironmentInstance(env4, Interval(15, 17))
        assert len(env_instance4.static_obstacles) == 4
        assert len(env_instance4.dynamic_obstacles) == 0

    def test_create_environment_instance_overloads(self):
        ## Re-reun test cases 1-6 for line and polygon obstacles again to ensure that overloads work the same
        # create shapely lines and copies for later comparison
        sln_static = ShapelyLine([(1, 1), (2, 2)])
        sln_static_copy = ShapelyLine([(1, 1), (2, 2)])
        sln_limited = ShapelyLine([(3, 3), (4, 4)])
        sln_limited_copy = ShapelyLine([(3, 3), (4, 4)])
        sln_minute = ShapelyLine([(5, 5), (6, 6)])
        sln_minute_copy = ShapelyLine([(5, 5), (6, 6)])
        sln_hour = ShapelyLine([(7, 7), (8, 8)])
        sln_hour_copy = ShapelyLine([(7, 7), (8, 8)])
        sln_day = ShapelyLine([(9, 9), (10, 10)])
        sln_day_copy = ShapelyLine([(9, 9), (10, 10)])

        # create shapely polygons and copies for later comparison
        spoly_static = ShapelyPolygon([(1, 1), (2, 2), (3, 3)])
        spoly_static_copy = ShapelyPolygon([(1, 1), (2, 2), (3, 3)])
        spoly_limited = ShapelyPolygon([(4, 4), (5, 5), (6, 6)])
        spoly_limited_copy = ShapelyPolygon([(4, 4), (5, 5), (6, 6)])
        spoly_minute = ShapelyPolygon([(7, 7), (8, 8), (9, 9)])
        spoly_minute_copy = ShapelyPolygon([(7, 7), (8, 8), (9, 9)])
        spoly_hour = ShapelyPolygon([(10, 10), (11, 11), (12, 12)])
        spoly_hour_copy = ShapelyPolygon([(10, 10), (11, 11), (12, 12)])
        spoly_day = ShapelyPolygon([(13, 13), (14, 14), (15, 15)])
        spoly_day_copy = ShapelyPolygon([(13, 13), (14, 14), (15, 15)])

        # create line obstacles with different recurrence and intervals
        interval = Interval(10, 20, closed="both")
        ln_static = Line(geometry=sln_static, radius=1.0)
        ln_limited = Line(
            geometry=sln_limited,
            time_interval=interval,
            radius=2.0,
        )
        ln_minute = Line(
            geometry=sln_minute,
            time_interval=interval,
            recurrence=Rec.MINUTELY,
            radius=3.0,
        )
        ln_hour = Line(
            geometry=sln_hour, time_interval=interval, recurrence=Rec.HOURLY, radius=4.0
        )
        ln_day = Line(
            geometry=sln_day, time_interval=interval, recurrence=Rec.DAILY, radius=5.0
        )

        # create polygon obstacles with different recurrence and intervals
        interval = Interval(10, 20, closed="both")
        poly_static = Polygon(geometry=spoly_static, radius=1.0)
        poly_limited = Polygon(
            geometry=spoly_limited,
            time_interval=interval,
            radius=2.0,
        )
        poly_minute = Polygon(
            geometry=spoly_minute,
            time_interval=interval,
            recurrence=Rec.MINUTELY,
            radius=3.0,
        )
        poly_hour = Polygon(
            geometry=spoly_hour,
            time_interval=interval,
            recurrence=Rec.HOURLY,
            radius=4.0,
        )
        poly_day = Polygon(
            geometry=spoly_day, time_interval=interval, recurrence=Rec.DAILY, radius=5.0
        )

        # Test case 1: single static obstacle - should be added to static obstacles
        env = Environment(obstacles=[ln_static])
        env_instance = EnvironmentInstance(env, Interval(10, 30))
        assert len(env_instance.static_obstacles) == 1
        assert len(env_instance.dynamic_obstacles) == 0
        saved_obstacle = env_instance.static_obstacles[1]
        assert saved_obstacle.geometry == sln_static_copy
        assert saved_obstacle.time_interval is None
        assert saved_obstacle.recurrence == Rec.NONE
        assert saved_obstacle.radius == 1.0

        env = Environment(obstacles=[poly_static])
        env_instance = EnvironmentInstance(env, Interval(10, 30))
        assert len(env_instance.static_obstacles) == 1
        assert len(env_instance.dynamic_obstacles) == 0
        saved_obstacle = env_instance.static_obstacles[1]
        assert saved_obstacle.geometry == spoly_static_copy
        assert saved_obstacle.time_interval is None
        assert saved_obstacle.recurrence == Rec.NONE
        assert saved_obstacle.radius == 1.0

        # Test case 2: single static limited obstacle, overlapping with start of query interval
        # should be added to dynamic obstacles
        env = Environment(obstacles=[ln_limited])
        env_instance = EnvironmentInstance(env, Interval(15, 30))
        assert len(env_instance.static_obstacles) == 0
        assert len(env_instance.dynamic_obstacles) == 1
        saved_obstacle = env_instance.dynamic_obstacles[1]
        assert saved_obstacle.geometry == sln_limited_copy
        assert saved_obstacle.time_interval == interval
        assert saved_obstacle.recurrence == Rec.NONE
        assert saved_obstacle.radius == 2.0

        env = Environment(obstacles=[poly_limited])
        env_instance = EnvironmentInstance(env, Interval(15, 30))
        assert len(env_instance.static_obstacles) == 0
        assert len(env_instance.dynamic_obstacles) == 1
        saved_obstacle = env_instance.dynamic_obstacles[1]
        assert saved_obstacle.geometry == spoly_limited_copy
        assert saved_obstacle.time_interval == interval
        assert saved_obstacle.recurrence == Rec.NONE
        assert saved_obstacle.radius == 2.0

        # Test case 3: single static limited obstacle, lying inside query interval
        # should be added to dynamic obstacles
        env = Environment(obstacles=[ln_limited])
        env_instance = EnvironmentInstance(env, Interval(5, 30))
        assert len(env_instance.static_obstacles) == 0
        assert len(env_instance.dynamic_obstacles) == 1
        saved_obstacle = env_instance.dynamic_obstacles[1]
        assert saved_obstacle.geometry == sln_limited_copy
        assert saved_obstacle.time_interval == interval
        assert saved_obstacle.recurrence == Rec.NONE
        assert saved_obstacle.radius == 2.0

        env = Environment(obstacles=[poly_limited])
        env_instance = EnvironmentInstance(env, Interval(5, 30))
        assert len(env_instance.static_obstacles) == 0
        assert len(env_instance.dynamic_obstacles) == 1
        saved_obstacle = env_instance.dynamic_obstacles[1]
        assert saved_obstacle.geometry == spoly_limited_copy
        assert saved_obstacle.time_interval == interval
        assert saved_obstacle.recurrence == Rec.NONE
        assert saved_obstacle.radius == 2.0

        # Test case 4: single static limited obstacle, overlapping with end of query interval
        # should be added to dynamic obstacles
        env = Environment(obstacles=[ln_limited])
        env_instance = EnvironmentInstance(env, Interval(5, 15))
        assert len(env_instance.static_obstacles) == 0
        assert len(env_instance.dynamic_obstacles) == 1
        saved_obstacle = env_instance.dynamic_obstacles[1]
        assert saved_obstacle.geometry == sln_limited_copy
        assert saved_obstacle.time_interval == interval
        assert saved_obstacle.recurrence == Rec.NONE
        assert saved_obstacle.radius == 2.0

        env = Environment(obstacles=[poly_limited])
        env_instance = EnvironmentInstance(env, Interval(5, 15))
        assert len(env_instance.static_obstacles) == 0
        assert len(env_instance.dynamic_obstacles) == 1
        saved_obstacle = env_instance.dynamic_obstacles[1]
        assert saved_obstacle.geometry == spoly_limited_copy
        assert saved_obstacle.time_interval == interval
        assert saved_obstacle.recurrence == Rec.NONE
        assert saved_obstacle.radius == 2.0

        # Test case 5: single static limited obstacle, overlapping with start and end of query interval
        # should be added to static obstacles
        env = Environment(obstacles=[ln_limited])
        env_instance = EnvironmentInstance(env, Interval(12, 15))
        assert len(env_instance.static_obstacles) == 1
        assert len(env_instance.dynamic_obstacles) == 0
        saved_obstacle = env_instance.static_obstacles[1]
        assert saved_obstacle.geometry == sln_limited_copy
        assert saved_obstacle.time_interval is None
        assert saved_obstacle.recurrence == Rec.NONE
        assert saved_obstacle.radius == 2.0

        env = Environment(obstacles=[poly_limited])
        env_instance = EnvironmentInstance(env, Interval(12, 15))
        assert len(env_instance.static_obstacles) == 1
        assert len(env_instance.dynamic_obstacles) == 0
        saved_obstacle = env_instance.static_obstacles[1]
        assert saved_obstacle.geometry == spoly_limited_copy
        assert saved_obstacle.time_interval is None
        assert saved_obstacle.recurrence == Rec.NONE
        assert saved_obstacle.radius == 2.0

        # Test case 6: recurring obstacle with first occurence overlapping with start of query interval
        # should be added to dynamic obstacles
        env1 = Environment(obstacles=[ln_minute])
        env_instance1 = EnvironmentInstance(env1, Interval(15, 30))
        assert len(env_instance1.static_obstacles) == 0
        assert len(env_instance1.dynamic_obstacles) == 1
        saved_obstacle = env_instance1.dynamic_obstacles[1]
        assert saved_obstacle.geometry == sln_minute_copy
        assert saved_obstacle.time_interval == interval
        assert saved_obstacle.recurrence == Rec.MINUTELY
        assert saved_obstacle.radius == 3.0

        env2 = Environment(obstacles=[ln_hour])
        env_instance2 = EnvironmentInstance(env2, Interval(15, 30))
        assert len(env_instance2.static_obstacles) == 0
        assert len(env_instance2.dynamic_obstacles) == 1
        saved_obstacle = env_instance2.dynamic_obstacles[1]
        assert saved_obstacle.geometry == sln_hour_copy
        assert saved_obstacle.time_interval == interval
        assert saved_obstacle.recurrence == Rec.HOURLY
        assert saved_obstacle.radius == 4.0

        env3 = Environment(obstacles=[ln_day])
        env_instance3 = EnvironmentInstance(env3, Interval(15, 30))
        assert len(env_instance3.static_obstacles) == 0
        assert len(env_instance3.dynamic_obstacles) == 1
        saved_obstacle = env_instance3.dynamic_obstacles[1]
        assert saved_obstacle.geometry == sln_day_copy
        assert saved_obstacle.time_interval == interval
        assert saved_obstacle.recurrence == Rec.DAILY
        assert saved_obstacle.radius == 5.0

        env1 = Environment(obstacles=[poly_minute])
        env_instance1 = EnvironmentInstance(env1, Interval(15, 30))
        assert len(env_instance1.static_obstacles) == 0
        assert len(env_instance1.dynamic_obstacles) == 1
        saved_obstacle = env_instance1.dynamic_obstacles[1]
        assert saved_obstacle.geometry == spoly_minute_copy
        assert saved_obstacle.time_interval == interval
        assert saved_obstacle.recurrence == Rec.MINUTELY
        assert saved_obstacle.radius == 3.0

        env2 = Environment(obstacles=[poly_hour])
        env_instance2 = EnvironmentInstance(env2, Interval(15, 30))
        assert len(env_instance2.static_obstacles) == 0
        assert len(env_instance2.dynamic_obstacles) == 1
        saved_obstacle = env_instance2.dynamic_obstacles[1]
        assert saved_obstacle.geometry == spoly_hour_copy
        assert saved_obstacle.time_interval == interval
        assert saved_obstacle.recurrence == Rec.HOURLY
        assert saved_obstacle.radius == 4.0

        env3 = Environment(obstacles=[poly_day])
        env_instance3 = EnvironmentInstance(env3, Interval(15, 30))
        assert len(env_instance3.static_obstacles) == 0
        assert len(env_instance3.dynamic_obstacles) == 1
        saved_obstacle = env_instance3.dynamic_obstacles[1]
        assert saved_obstacle.geometry == spoly_day_copy
        assert saved_obstacle.time_interval == interval
        assert saved_obstacle.recurrence == Rec.DAILY
        assert saved_obstacle.radius == 5.0

    # TODO - implement once spatial index is available
    def test_create_environment_instance_idx(self):
        pass
