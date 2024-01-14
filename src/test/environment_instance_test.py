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
        # specify scenario dimensions, which include all points
        range_x = (-100, 100)
        range_y = (-100, 100)

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
        env_instance = EnvironmentInstance(env, Interval(10, 30), range_x, range_y)
        assert len(env_instance.static_obstacles) == 0
        assert len(env_instance.dynamic_obstacles) == 0

        # Test case 1: single static obstacle - should be added to static obstacles
        env = Environment(obstacles=[pt_static])
        env_instance = EnvironmentInstance(env, Interval(10, 30), range_x, range_y)
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
        env_instance = EnvironmentInstance(env, Interval(15, 30), range_x, range_y)
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
        env_instance = EnvironmentInstance(env, Interval(5, 30), range_x, range_y)
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
        env_instance = EnvironmentInstance(env, Interval(5, 15), range_x, range_y)
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
        env_instance = EnvironmentInstance(env, Interval(12, 15), range_x, range_y)
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
        env_instance1 = EnvironmentInstance(env1, Interval(15, 30), range_x, range_y)
        assert len(env_instance1.static_obstacles) == 0
        assert len(env_instance1.dynamic_obstacles) == 1
        saved_obstacle = env_instance1.dynamic_obstacles[1]
        assert saved_obstacle.geometry == spt_minute_copy
        assert saved_obstacle.time_interval == interval
        assert saved_obstacle.recurrence == Rec.MINUTELY
        assert saved_obstacle.radius == 3.0

        env2 = Environment(obstacles=[pt_hour])
        env_instance2 = EnvironmentInstance(env2, Interval(15, 30), range_x, range_y)
        assert len(env_instance2.static_obstacles) == 0
        assert len(env_instance2.dynamic_obstacles) == 1
        saved_obstacle = env_instance2.dynamic_obstacles[1]
        assert saved_obstacle.geometry == spt_hour_copy
        assert saved_obstacle.time_interval == interval
        assert saved_obstacle.recurrence == Rec.HOURLY
        assert saved_obstacle.radius == 4.0

        env3 = Environment(obstacles=[pt_day])
        env_instance3 = EnvironmentInstance(env3, Interval(15, 30), range_x, range_y)
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
        env_instance1 = EnvironmentInstance(env1, Interval(5, 30), range_x, range_y)
        assert len(env_instance1.static_obstacles) == 0
        assert len(env_instance1.dynamic_obstacles) == 1
        saved_obstacle = env_instance1.dynamic_obstacles[1]
        assert saved_obstacle.geometry == spt_minute_copy
        assert saved_obstacle.time_interval == interval
        assert saved_obstacle.recurrence == Rec.MINUTELY
        assert saved_obstacle.radius == 3.0

        env2 = Environment(obstacles=[pt_hour])
        env_instance2 = EnvironmentInstance(env2, Interval(5, 30), range_x, range_y)
        assert len(env_instance2.static_obstacles) == 0
        assert len(env_instance2.dynamic_obstacles) == 1
        saved_obstacle = env_instance2.dynamic_obstacles[1]
        assert saved_obstacle.geometry == spt_hour_copy
        assert saved_obstacle.time_interval == interval
        assert saved_obstacle.recurrence == Rec.HOURLY
        assert saved_obstacle.radius == 4.0

        env3 = Environment(obstacles=[pt_day])
        env_instance3 = EnvironmentInstance(env3, Interval(5, 30), range_x, range_y)
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
        env_instance1 = EnvironmentInstance(env1, Interval(5, 15), range_x, range_y)
        assert len(env_instance1.static_obstacles) == 0
        assert len(env_instance1.dynamic_obstacles) == 1
        saved_obstacle = env_instance1.dynamic_obstacles[1]
        assert saved_obstacle.geometry == spt_minute_copy
        assert saved_obstacle.time_interval == interval
        assert saved_obstacle.recurrence == Rec.MINUTELY
        assert saved_obstacle.radius == 3.0

        env2 = Environment(obstacles=[pt_hour])
        env_instance2 = EnvironmentInstance(env2, Interval(5, 15), range_x, range_y)
        assert len(env_instance2.static_obstacles) == 0
        assert len(env_instance2.dynamic_obstacles) == 1
        saved_obstacle = env_instance2.dynamic_obstacles[1]
        assert saved_obstacle.geometry == spt_hour_copy
        assert saved_obstacle.time_interval == interval
        assert saved_obstacle.recurrence == Rec.HOURLY
        assert saved_obstacle.radius == 4.0

        env3 = Environment(obstacles=[pt_day])
        env_instance3 = EnvironmentInstance(env3, Interval(5, 15), range_x, range_y)
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
        env_instance1 = EnvironmentInstance(env1, Interval(12, 15), range_x, range_y)
        assert len(env_instance1.static_obstacles) == 1
        assert len(env_instance1.dynamic_obstacles) == 0
        saved_obstacle = env_instance1.static_obstacles[1]
        assert saved_obstacle.geometry == spt_minute_copy
        assert saved_obstacle.time_interval is None
        assert saved_obstacle.recurrence == Rec.NONE
        assert saved_obstacle.radius == 3.0

        env2 = Environment(obstacles=[pt_hour])
        env_instance2 = EnvironmentInstance(env2, Interval(12, 15), range_x, range_y)
        assert len(env_instance2.static_obstacles) == 1
        assert len(env_instance2.dynamic_obstacles) == 0
        saved_obstacle = env_instance2.static_obstacles[1]
        assert saved_obstacle.geometry == spt_hour_copy
        assert saved_obstacle.time_interval is None
        assert saved_obstacle.recurrence == Rec.NONE
        assert saved_obstacle.radius == 4.0

        env3 = Environment(obstacles=[pt_day])
        env_instance3 = EnvironmentInstance(env3, Interval(12, 15), range_x, range_y)
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
        env_instance1 = EnvironmentInstance(env1, Interval(135, 150), range_x, range_y)
        assert len(env_instance1.static_obstacles) == 0
        assert len(env_instance1.dynamic_obstacles) == 1
        saved_obstacle = env_instance1.dynamic_obstacles[1]
        assert saved_obstacle.geometry == spt_minute_copy
        assert saved_obstacle.time_interval == interval
        assert saved_obstacle.recurrence == Rec.MINUTELY
        assert saved_obstacle.radius == 3.0

        env2 = Environment(obstacles=[pt_hour])
        env_instance2_wrong = EnvironmentInstance(
            env2, Interval(135, 150), range_x, range_y
        )
        assert len(env_instance2_wrong.static_obstacles) == 0
        assert len(env_instance2_wrong.dynamic_obstacles) == 0
        env_instance2 = EnvironmentInstance(
            env2, Interval(7215, 7230), range_x, range_y
        )
        assert len(env_instance2.static_obstacles) == 0
        assert len(env_instance2.dynamic_obstacles) == 1
        saved_obstacle = env_instance2.dynamic_obstacles[1]
        assert saved_obstacle.geometry == spt_hour_copy
        assert saved_obstacle.time_interval == interval
        assert saved_obstacle.recurrence == Rec.HOURLY
        assert saved_obstacle.radius == 4.0

        env3 = Environment(obstacles=[pt_day])
        env_instance3_wrong = EnvironmentInstance(
            env2, Interval(135, 150), range_x, range_y
        )
        assert len(env_instance3_wrong.static_obstacles) == 0
        assert len(env_instance3_wrong.dynamic_obstacles) == 0
        env_instance3 = EnvironmentInstance(
            env3, Interval(172815, 172830), range_x, range_y
        )
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
        env_instance1 = EnvironmentInstance(env1, Interval(125, 150), range_x, range_y)
        assert len(env_instance1.static_obstacles) == 0
        assert len(env_instance1.dynamic_obstacles) == 1
        saved_obstacle = env_instance1.dynamic_obstacles[1]
        assert saved_obstacle.geometry == spt_minute_copy
        assert saved_obstacle.time_interval == interval
        assert saved_obstacle.recurrence == Rec.MINUTELY
        assert saved_obstacle.radius == 3.0

        env2 = Environment(obstacles=[pt_hour])
        env_instance2 = EnvironmentInstance(
            env2, Interval(7205, 7230), range_x, range_y
        )
        assert len(env_instance2.static_obstacles) == 0
        assert len(env_instance2.dynamic_obstacles) == 1
        saved_obstacle = env_instance2.dynamic_obstacles[1]
        assert saved_obstacle.geometry == spt_hour_copy
        assert saved_obstacle.time_interval == interval
        assert saved_obstacle.recurrence == Rec.HOURLY
        assert saved_obstacle.radius == 4.0

        env3 = Environment(obstacles=[pt_day])
        env_instance3 = EnvironmentInstance(
            env3, Interval(172805, 172830), range_x, range_y
        )
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
        env_instance1 = EnvironmentInstance(env1, Interval(125, 135), range_x, range_y)
        assert len(env_instance1.static_obstacles) == 0
        assert len(env_instance1.dynamic_obstacles) == 1
        saved_obstacle = env_instance1.dynamic_obstacles[1]
        assert saved_obstacle.geometry == spt_minute_copy
        assert saved_obstacle.time_interval == interval
        assert saved_obstacle.recurrence == Rec.MINUTELY
        assert saved_obstacle.radius == 3.0

        env2 = Environment(obstacles=[pt_hour])
        env_instance2 = EnvironmentInstance(
            env2, Interval(7205, 7215), range_x, range_y
        )
        assert len(env_instance2.static_obstacles) == 0
        assert len(env_instance2.dynamic_obstacles) == 1
        saved_obstacle = env_instance2.dynamic_obstacles[1]
        assert saved_obstacle.geometry == spt_hour_copy
        assert saved_obstacle.time_interval == interval
        assert saved_obstacle.recurrence == Rec.HOURLY
        assert saved_obstacle.radius == 4.0

        env3 = Environment(obstacles=[pt_day])
        env_instance3 = EnvironmentInstance(
            env3, Interval(172805, 172815), range_x, range_y
        )
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
        env_instance1 = EnvironmentInstance(env1, Interval(132, 135), range_x, range_y)
        assert len(env_instance1.static_obstacles) == 1
        assert len(env_instance1.dynamic_obstacles) == 0
        saved_obstacle = env_instance1.static_obstacles[1]
        assert saved_obstacle.geometry == spt_minute_copy
        assert saved_obstacle.time_interval is None
        assert saved_obstacle.recurrence == Rec.NONE
        assert saved_obstacle.radius == 3.0

        env2 = Environment(obstacles=[pt_hour])
        env_instance2 = EnvironmentInstance(
            env2, Interval(7212, 7215), range_x, range_y
        )
        assert len(env_instance2.static_obstacles) == 1
        assert len(env_instance2.dynamic_obstacles) == 0
        saved_obstacle = env_instance2.static_obstacles[1]
        assert saved_obstacle.geometry == spt_hour_copy
        assert saved_obstacle.time_interval is None
        assert saved_obstacle.recurrence == Rec.NONE
        assert saved_obstacle.radius == 4.0

        env3 = Environment(obstacles=[pt_day])
        env_instance3 = EnvironmentInstance(
            env3, Interval(172812, 172815), range_x, range_y
        )
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
        env_instance1 = EnvironmentInstance(env1, Interval(5, 145), range_x, range_y)
        assert len(env_instance1.static_obstacles) == 0
        assert len(env_instance1.dynamic_obstacles) == 1
        saved_obstacle = env_instance1.dynamic_obstacles[1]
        assert saved_obstacle.geometry == spt_minute_copy
        assert saved_obstacle.time_interval == interval
        assert saved_obstacle.recurrence == Rec.MINUTELY
        assert saved_obstacle.radius == 3.0

        env2 = Environment(obstacles=[pt_minute])
        env_instance2 = EnvironmentInstance(env2, Interval(125, 265), range_x, range_y)
        assert len(env_instance2.static_obstacles) == 0
        assert len(env_instance2.dynamic_obstacles) == 1
        saved_obstacle1 = env_instance2.dynamic_obstacles[1]
        assert saved_obstacle1.geometry == spt_minute_copy
        assert saved_obstacle1.time_interval == interval
        assert saved_obstacle1.recurrence == Rec.MINUTELY
        assert saved_obstacle1.radius == 3.0

        env3 = Environment(obstacles=[pt_hour])
        env_instance3 = EnvironmentInstance(env3, Interval(5, 7225), range_x, range_y)
        assert len(env_instance3.static_obstacles) == 0
        assert len(env_instance3.dynamic_obstacles) == 1
        saved_obstacle = env_instance3.dynamic_obstacles[1]
        assert saved_obstacle.geometry == spt_hour_copy
        assert saved_obstacle.time_interval == interval
        assert saved_obstacle.recurrence == Rec.HOURLY
        assert saved_obstacle.radius == 4.0

        env4 = Environment(obstacles=[pt_hour])
        env_instance4 = EnvironmentInstance(
            env4, Interval(7205, 14425), range_x, range_y
        )
        assert len(env_instance4.static_obstacles) == 0
        assert len(env_instance4.dynamic_obstacles) == 1
        saved_obstacle1 = env_instance4.dynamic_obstacles[1]
        assert saved_obstacle1.geometry == spt_hour_copy
        assert saved_obstacle1.time_interval == interval
        assert saved_obstacle1.recurrence == Rec.HOURLY
        assert saved_obstacle1.radius == 4.0

        env5 = Environment(obstacles=[pt_day])
        env_instance5 = EnvironmentInstance(env5, Interval(5, 172825), range_x, range_y)
        assert len(env_instance5.static_obstacles) == 0
        assert len(env_instance5.dynamic_obstacles) == 1
        saved_obstacle = env_instance5.dynamic_obstacles[1]
        assert saved_obstacle.geometry == spt_day_copy
        assert saved_obstacle.time_interval == interval
        assert saved_obstacle.recurrence == Rec.DAILY
        assert saved_obstacle.radius == 5.0

        env6 = Environment(obstacles=[pt_day])
        env_instance6 = EnvironmentInstance(
            env6, Interval(172805, 345625), range_x, range_y
        )
        assert len(env_instance6.static_obstacles) == 0
        assert len(env_instance6.dynamic_obstacles) == 1
        saved_obstacle1 = env_instance6.dynamic_obstacles[1]
        assert saved_obstacle1.geometry == spt_day_copy
        assert saved_obstacle1.time_interval == interval
        assert saved_obstacle1.recurrence == Rec.DAILY
        assert saved_obstacle1.radius == 5.0

        # Test case 15: Test that obstacles starting after query interval should not be added
        env = Environment(obstacles=[pt_limited, pt_minute, pt_hour, pt_day])
        env_instance = EnvironmentInstance(env, Interval(5, 9), range_x, range_y)
        assert len(env_instance.static_obstacles) == 0
        assert len(env_instance.dynamic_obstacles) == 0

        # Test case 16: Test that obstacle ending before query interval should not be added
        env = Environment(obstacles=[pt_limited])
        env_instance = EnvironmentInstance(env, Interval(25, 30), range_x, range_y)
        assert len(env_instance.static_obstacles) == 0
        assert len(env_instance.dynamic_obstacles) == 0

        # Test case 17: Test that query intervals falling in between obstacles should cause obstacles to be added
        env = Environment(obstacles=[pt_minute, pt_hour, pt_day])
        env_instance = EnvironmentInstance(env, Interval(25, 40), range_x, range_y)
        assert len(env_instance.static_obstacles) == 0
        assert len(env_instance.dynamic_obstacles) == 0

        # Test case 18: Test that query intervals falling in between repetitions of obstacles
        # should cause obstacles to be added
        env1 = Environment(obstacles=[pt_minute])
        env_instance1 = EnvironmentInstance(env1, Interval(145, 160), range_x, range_y)
        assert len(env_instance1.static_obstacles) == 0
        assert len(env_instance1.dynamic_obstacles) == 0

        env2 = Environment(obstacles=[pt_hour])
        env_instance2 = EnvironmentInstance(
            env2, Interval(7225, 7240), range_x, range_y
        )
        assert len(env_instance2.static_obstacles) == 0
        assert len(env_instance2.dynamic_obstacles) == 0

        env3 = Environment(obstacles=[pt_day])
        env_instance3 = EnvironmentInstance(
            env3, Interval(172825, 172840), range_x, range_y
        )
        assert len(env_instance3.static_obstacles) == 0
        assert len(env_instance3.dynamic_obstacles) == 0

        # Test case 19: add multiple objects and check if they are all added correctly
        env1 = Environment(obstacles=[pt_minute, pt_hour, pt_day])
        env_instance1 = EnvironmentInstance(
            env1, Interval(172805, 172815), range_x, range_y
        )
        assert len(env_instance1.static_obstacles) == 0
        assert len(env_instance1.dynamic_obstacles) == 3

        env2 = Environment(obstacles=[pt_minute, pt_hour, pt_day])
        env_instance2 = EnvironmentInstance(
            env2, Interval(172815, 172817), range_x, range_y
        )
        assert len(env_instance2.static_obstacles) == 3
        assert len(env_instance2.dynamic_obstacles) == 0

        env3 = Environment(obstacles=[pt_limited, pt_minute, pt_hour, pt_day])
        env_instance3 = EnvironmentInstance(env3, Interval(5, 15), range_x, range_y)
        assert len(env_instance3.static_obstacles) == 0
        assert len(env_instance3.dynamic_obstacles) == 4

        env4 = Environment(obstacles=[pt_limited, pt_minute, pt_hour, pt_day])
        env_instance4 = EnvironmentInstance(env4, Interval(15, 17), range_x, range_y)
        assert len(env_instance4.static_obstacles) == 4
        assert len(env_instance4.dynamic_obstacles) == 0

        # Test creation of environment instance with combined static and dynamic obstacles
        pt1 = ShapelyPoint(0, 0)
        interval_dyn = Interval(10, 20)
        rad_dyn = 0.5
        pt_dynamic_obs = Point(
            geometry=pt1,
            time_interval=interval_dyn,
            radius=rad_dyn,
            recurrence=Rec.MINUTELY,
        )

        pt2 = ShapelyPoint(1, 1)
        interval_static = Interval(0, 100)
        rad_static = 1.0
        pt_static_obs = Point(
            geometry=pt2,
            time_interval=interval_static,
            radius=rad_static,
            recurrence=Rec.DAILY,
        )

        env = Environment(obstacles=[pt_dynamic_obs, pt_static_obs])
        env_instance = EnvironmentInstance(env, Interval(5, 25), range_x, range_y)
        assert len(env_instance.static_obstacles) == 1
        assert len(env_instance.dynamic_obstacles) == 1

        saved_obstacle1 = env_instance.dynamic_obstacles[1]
        saved_obstacle2 = env_instance.static_obstacles[2]

        assert saved_obstacle1.geometry == pt1
        assert saved_obstacle1.time_interval == interval_dyn
        assert saved_obstacle1.recurrence == Rec.MINUTELY
        assert saved_obstacle1.radius == rad_dyn

        assert saved_obstacle2.geometry == pt2
        assert saved_obstacle2.time_interval == None
        assert saved_obstacle2.recurrence == Rec.NONE
        assert saved_obstacle2.radius == rad_static

    def test_create_environment_instance_overloads(self):
        ## Re-reun test cases 1-6 for line and polygon obstacles again to ensure that overloads work the same
        # specify scenario dimensions, which include all points
        range_x = (-100, 100)
        range_y = (-100, 100)

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
        env_instance = EnvironmentInstance(env, Interval(10, 30), range_x, range_y)
        assert len(env_instance.static_obstacles) == 1
        assert len(env_instance.dynamic_obstacles) == 0
        saved_obstacle = env_instance.static_obstacles[1]
        assert saved_obstacle.geometry == sln_static_copy
        assert saved_obstacle.time_interval is None
        assert saved_obstacle.recurrence == Rec.NONE
        assert saved_obstacle.radius == 1.0

        env = Environment(obstacles=[poly_static])
        env_instance = EnvironmentInstance(env, Interval(10, 30), range_x, range_y)
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
        env_instance = EnvironmentInstance(env, Interval(15, 30), range_x, range_y)
        assert len(env_instance.static_obstacles) == 0
        assert len(env_instance.dynamic_obstacles) == 1
        saved_obstacle = env_instance.dynamic_obstacles[1]
        assert saved_obstacle.geometry == sln_limited_copy
        assert saved_obstacle.time_interval == interval
        assert saved_obstacle.recurrence == Rec.NONE
        assert saved_obstacle.radius == 2.0

        env = Environment(obstacles=[poly_limited])
        env_instance = EnvironmentInstance(env, Interval(15, 30), range_x, range_y)
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
        env_instance = EnvironmentInstance(env, Interval(5, 30), range_x, range_y)
        assert len(env_instance.static_obstacles) == 0
        assert len(env_instance.dynamic_obstacles) == 1
        saved_obstacle = env_instance.dynamic_obstacles[1]
        assert saved_obstacle.geometry == sln_limited_copy
        assert saved_obstacle.time_interval == interval
        assert saved_obstacle.recurrence == Rec.NONE
        assert saved_obstacle.radius == 2.0

        env = Environment(obstacles=[poly_limited])
        env_instance = EnvironmentInstance(env, Interval(5, 30), range_x, range_y)
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
        env_instance = EnvironmentInstance(env, Interval(5, 15), range_x, range_y)
        assert len(env_instance.static_obstacles) == 0
        assert len(env_instance.dynamic_obstacles) == 1
        saved_obstacle = env_instance.dynamic_obstacles[1]
        assert saved_obstacle.geometry == sln_limited_copy
        assert saved_obstacle.time_interval == interval
        assert saved_obstacle.recurrence == Rec.NONE
        assert saved_obstacle.radius == 2.0

        env = Environment(obstacles=[poly_limited])
        env_instance = EnvironmentInstance(env, Interval(5, 15), range_x, range_y)
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
        env_instance = EnvironmentInstance(env, Interval(12, 15), range_x, range_y)
        assert len(env_instance.static_obstacles) == 1
        assert len(env_instance.dynamic_obstacles) == 0
        saved_obstacle = env_instance.static_obstacles[1]
        assert saved_obstacle.geometry == sln_limited_copy
        assert saved_obstacle.time_interval is None
        assert saved_obstacle.recurrence == Rec.NONE
        assert saved_obstacle.radius == 2.0

        env = Environment(obstacles=[poly_limited])
        env_instance = EnvironmentInstance(env, Interval(12, 15), range_x, range_y)
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
        env_instance1 = EnvironmentInstance(env1, Interval(15, 30), range_x, range_y)
        assert len(env_instance1.static_obstacles) == 0
        assert len(env_instance1.dynamic_obstacles) == 1
        saved_obstacle = env_instance1.dynamic_obstacles[1]
        assert saved_obstacle.geometry == sln_minute_copy
        assert saved_obstacle.time_interval == interval
        assert saved_obstacle.recurrence == Rec.MINUTELY
        assert saved_obstacle.radius == 3.0

        env2 = Environment(obstacles=[ln_hour])
        env_instance2 = EnvironmentInstance(env2, Interval(15, 30), range_x, range_y)
        assert len(env_instance2.static_obstacles) == 0
        assert len(env_instance2.dynamic_obstacles) == 1
        saved_obstacle = env_instance2.dynamic_obstacles[1]
        assert saved_obstacle.geometry == sln_hour_copy
        assert saved_obstacle.time_interval == interval
        assert saved_obstacle.recurrence == Rec.HOURLY
        assert saved_obstacle.radius == 4.0

        env3 = Environment(obstacles=[ln_day])
        env_instance3 = EnvironmentInstance(env3, Interval(15, 30), range_x, range_y)
        assert len(env_instance3.static_obstacles) == 0
        assert len(env_instance3.dynamic_obstacles) == 1
        saved_obstacle = env_instance3.dynamic_obstacles[1]
        assert saved_obstacle.geometry == sln_day_copy
        assert saved_obstacle.time_interval == interval
        assert saved_obstacle.recurrence == Rec.DAILY
        assert saved_obstacle.radius == 5.0

        env1 = Environment(obstacles=[poly_minute])
        env_instance1 = EnvironmentInstance(env1, Interval(15, 30), range_x, range_y)
        assert len(env_instance1.static_obstacles) == 0
        assert len(env_instance1.dynamic_obstacles) == 1
        saved_obstacle = env_instance1.dynamic_obstacles[1]
        assert saved_obstacle.geometry == spoly_minute_copy
        assert saved_obstacle.time_interval == interval
        assert saved_obstacle.recurrence == Rec.MINUTELY
        assert saved_obstacle.radius == 3.0

        env2 = Environment(obstacles=[poly_hour])
        env_instance2 = EnvironmentInstance(env2, Interval(15, 30), range_x, range_y)
        assert len(env_instance2.static_obstacles) == 0
        assert len(env_instance2.dynamic_obstacles) == 1
        saved_obstacle = env_instance2.dynamic_obstacles[1]
        assert saved_obstacle.geometry == spoly_hour_copy
        assert saved_obstacle.time_interval == interval
        assert saved_obstacle.recurrence == Rec.HOURLY
        assert saved_obstacle.radius == 4.0

        env3 = Environment(obstacles=[poly_day])
        env_instance3 = EnvironmentInstance(env3, Interval(15, 30), range_x, range_y)
        assert len(env_instance3.static_obstacles) == 0
        assert len(env_instance3.dynamic_obstacles) == 1
        saved_obstacle = env_instance3.dynamic_obstacles[1]
        assert saved_obstacle.geometry == spoly_day_copy
        assert saved_obstacle.time_interval == interval
        assert saved_obstacle.recurrence == Rec.DAILY
        assert saved_obstacle.radius == 5.0

    def test_create_environment_dimensionality_check(self):
        # initialize points and query interval matching intervals and mismatching intervals
        pt_inside = ShapelyPoint(0, 0)
        pt_inside_copy = ShapelyPoint(0, 0)
        pt_outside = ShapelyPoint(100, 100)
        pt_outside_copy = ShapelyPoint(100, 100)
        pt_intersecting = ShapelyPoint(11, 11)
        pt_intersecting_copy = ShapelyPoint(11, 11)

        interval_included = Interval(10, 20, closed="both")
        interval_outside = Interval(100, 200, closed="both")
        query_interval = Interval(5, 25, closed="both")

        # create point obstacles categorized into matching and mismatching
        pt1_match = Point(geometry=pt_inside, radius=0.0)
        pt2_match = Point(
            geometry=pt_inside, time_interval=interval_included, radius=1.0
        )
        pt3_match = Point(geometry=pt_intersecting, radius=2.0)

        pt1_mismatch = Point(geometry=pt_outside, radius=5.0)
        pt2_mismatch = Point(
            geometry=pt_inside, time_interval=interval_outside, radius=0.0
        )
        pt3_mismatch = Point(geometry=pt_intersecting, radius=0.0)
        pt4_mismatch = Point(
            geometry=pt_intersecting, time_interval=interval_outside, radius=5.0
        )

        # create environment with all obstacles in random order
        env = Environment(
            obstacles=[
                pt1_match,
                pt1_mismatch,
                pt2_match,
                pt2_mismatch,
                pt3_mismatch,
                pt3_match,
                pt4_mismatch,
            ],
        )

        # create environment instance and check that only matching obstacles are added
        env_instance = EnvironmentInstance(
            environment=env,
            query_interval=query_interval,
            scenario_range_x=(0, 10),
            scenario_range_y=(0, 10),
        )
        assert len(env_instance.static_obstacles) == 2
        assert len(env_instance.dynamic_obstacles) == 1

        saved_obstacle1 = env_instance.static_obstacles[1]
        saved_obstacle2 = env_instance.static_obstacles[3]
        saved_obstacle3 = env_instance.dynamic_obstacles[2]

        assert saved_obstacle1.geometry == pt_inside_copy
        assert saved_obstacle1.time_interval is None
        assert saved_obstacle1.recurrence == Rec.NONE
        assert saved_obstacle1.radius == 0.0

        assert saved_obstacle2.geometry == pt_intersecting_copy
        assert saved_obstacle2.time_interval is None
        assert saved_obstacle2.recurrence == Rec.NONE
        assert saved_obstacle2.radius == 2.0

        assert saved_obstacle3.geometry == pt_inside_copy
        assert saved_obstacle3.time_interval == interval_included
        assert saved_obstacle3.recurrence == Rec.NONE
        assert saved_obstacle3.radius == 1.0

    def test_create_environment_instance_idx(self):
        resolution = 5
        range_x = (0, 10)
        range_y = (0, 10)

        # create point, which is part of 4 cells (both static and dynamic)
        shapely_pt = ShapelyPoint(2, 2)
        pt1 = Point(geometry=shapely_pt, radius=1.0)
        pt2 = Point(
            geometry=shapely_pt,
            time_interval=Interval(10, 20, closed="both"),
            radius=1.0,
        )

        # create a line, which is part of 6 static cells and 3 dynamic cells
        shapely_line = ShapelyLine([(3, 4.5), (7, 4.5)])
        line1 = Line(geometry=shapely_line, radius=0.6)
        line2 = Line(
            geometry=shapely_line,
            time_interval=Interval(10, 20, closed="both"),
            radius=0.1,
        )

        # create a polygon, which is part of 5 static and dynamic cells
        shapely_poly = ShapelyPolygon([(1, 1), (3, 1), (3, 3), (1, 4.5)])
        poly1 = Polygon(geometry=shapely_poly, radius=0.01)
        poly2 = Polygon(
            geometry=shapely_poly,
            time_interval=Interval(10, 20, closed="both"),
            radius=0.01,
        )

        # create environment with all obstacles
        env = Environment(obstacles=[pt1, pt2, line1, line2, poly1, poly2])

        # create environment instance and check that all obstacles are added
        env_instance = EnvironmentInstance(
            environment=env,
            query_interval=Interval(5, 25, closed="both"),
            scenario_range_x=range_x,
            scenario_range_y=range_y,
            resolution=resolution,
        )

        assert len(env_instance.static_obstacles) == 3
        assert len(env_instance.dynamic_obstacles) == 3

        # check cell content of the two spatial indices
        static_index = env_instance.static_idx
        dynamic_index = env_instance.dynamic_idx

        st_pt_ix = 1
        st_line_ix = 3
        st_poly_ix = 5
        dyn_pt_ix = 2
        dyn_line_ix = 4
        dyn_poly_ix = 6

        assert len(static_index) == resolution
        assert len(dynamic_index) == resolution
        assert len(static_index[0]) == resolution
        assert len(dynamic_index[0]) == resolution

        # check static index content
        assert len(static_index[0][0]) == 2
        assert static_index[0][0] == [st_pt_ix, st_poly_ix]
        assert len(static_index[0][1]) == 2
        assert static_index[0][1] == [st_pt_ix, st_poly_ix]
        assert len(static_index[0][2]) == 1
        assert static_index[0][2] == [st_poly_ix]
        assert len(static_index[0][3]) == 0
        assert len(static_index[0][4]) == 0

        assert len(static_index[1][0]) == 2
        assert static_index[1][0] == [st_pt_ix, st_poly_ix]
        assert len(static_index[1][1]) == 3
        assert static_index[1][1] == [st_pt_ix, st_line_ix, st_poly_ix]
        assert len(static_index[1][2]) == 1
        assert static_index[1][2] == [st_line_ix]
        assert len(static_index[1][3]) == 0
        assert len(static_index[1][4]) == 0

        assert len(static_index[2][0]) == 0
        assert len(static_index[2][1]) == 1
        assert static_index[2][1] == [st_line_ix]
        assert len(static_index[2][2]) == 1
        assert static_index[2][2] == [st_line_ix]
        assert len(static_index[2][3]) == 0
        assert len(static_index[2][4]) == 0

        assert len(static_index[3][0]) == 0
        assert len(static_index[3][1]) == 1
        assert static_index[3][1] == [st_line_ix]
        assert len(static_index[3][2]) == 1
        assert static_index[3][2] == [st_line_ix]
        assert len(static_index[3][3]) == 0
        assert len(static_index[3][4]) == 0

        assert len(static_index[4][0]) == 0
        assert len(static_index[4][1]) == 0
        assert len(static_index[4][2]) == 0
        assert len(static_index[4][3]) == 0
        assert len(static_index[4][4]) == 0

        # check dynamic index content
        assert len(dynamic_index[0][0]) == 2
        assert dynamic_index[0][0] == [dyn_pt_ix, dyn_poly_ix]
        assert len(dynamic_index[0][1]) == 2
        assert dynamic_index[0][1] == [dyn_pt_ix, dyn_poly_ix]
        assert len(dynamic_index[0][2]) == 1
        assert dynamic_index[0][2] == [dyn_poly_ix]
        assert len(dynamic_index[0][3]) == 0
        assert len(dynamic_index[0][4]) == 0

        assert len(dynamic_index[1][0]) == 2
        assert dynamic_index[1][0] == [dyn_pt_ix, dyn_poly_ix]
        assert len(dynamic_index[1][1]) == 2
        assert dynamic_index[1][1] == [dyn_pt_ix, dyn_poly_ix]
        assert len(dynamic_index[1][2]) == 1
        assert dynamic_index[1][2] == [dyn_line_ix]
        assert len(dynamic_index[1][3]) == 0
        assert len(dynamic_index[1][4]) == 0

        assert len(dynamic_index[2][0]) == 0
        assert len(dynamic_index[2][1]) == 0
        assert len(dynamic_index[2][2]) == 1
        assert dynamic_index[2][2] == [dyn_line_ix]
        assert len(dynamic_index[2][3]) == 0
        assert len(dynamic_index[2][4]) == 0

        assert len(dynamic_index[3][0]) == 0
        assert len(dynamic_index[3][1]) == 0
        assert len(dynamic_index[3][2]) == 1
        assert dynamic_index[3][2] == [dyn_line_ix]
        assert len(dynamic_index[3][3]) == 0
        assert len(dynamic_index[3][4]) == 0

        assert len(dynamic_index[4][0]) == 0
        assert len(dynamic_index[4][1]) == 0
        assert len(dynamic_index[4][2]) == 0
        assert len(dynamic_index[4][3]) == 0
        assert len(dynamic_index[4][4]) == 0

    def test_sample(self):
        # create environment with random obstacles
        min_x = 0
        max_x = 200
        min_y = 0
        max_y = 300
        env = Environment()
        env.add_random_obstacles(
            num_points=100,
            num_lines=100,
            num_polygons=100,
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
            min_radius=0,
            max_radius=4,
        )

        # create environment instance
        env_instance = EnvironmentInstance(
            environment=env,
            query_interval=Interval(0, 100, closed="both"),
            scenario_range_x=(0, 10),
            scenario_range_y=(0, 10),
        )

        # check that the random sample generation creates different points within the environment range
        sample = env_instance.sample_point()
        assert sample.x >= min_x and sample.x <= max_x
        assert sample.y >= min_y and sample.y <= max_y

        sample2 = env_instance.sample_point()
        assert sample.x != sample2.x and sample.y != sample2.y
        assert sample.x >= min_x and sample.x <= max_x
        assert sample.y >= min_y and sample.y <= max_y

        sample3 = env_instance.sample_point()
        assert (
            sample.x != sample3.x
            and sample.y != sample3.y
            and sample2.x != sample3.x
            and sample2.y != sample3.y
        )
        assert sample.x >= min_x and sample.x <= max_x
        assert sample.y >= min_y and sample.y <= max_y

    def test_static_collision(self):
        # create first static obstacle
        sh_poly1 = ShapelyPolygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        poly1 = Polygon(geometry=sh_poly1, radius=0.0)

        # create second static obstacle
        sh_poly2 = ShapelyPolygon([(3, 3), (5, 3), (5, 5), (3, 5)])
        poly2 = Polygon(geometry=sh_poly2, radius=0.0)

        # create first dynamic obstacle
        sh_poly3 = ShapelyPolygon([(1, 1), (4, 1), (4, 4), (1, 4)])
        poly3 = Polygon(
            geometry=sh_poly3, radius=0.0, time_interval=Interval(0, 10, closed="both")
        )

        # create second dynamic obstacle
        sh_poly4 = ShapelyPolygon([(4, 4), (6, 4), (6, 6), (4, 6)])
        poly4 = Polygon(
            geometry=sh_poly4, radius=0.0, time_interval=Interval(0, 10, closed="both")
        )

        # create point in static collision
        pt1 = ShapelyPoint(0.5, 0.5)

        # create second point in static collision (and dynamic collision)
        pt2 = ShapelyPoint(3.5, 3.5)

        # create point in dynamic collision only
        pt3 = ShapelyPoint(1.5, 1.5)

        # create second point in dynamic collision only
        pt4 = ShapelyPoint(5.5, 5.5)

        # create environment with all obstacles
        env = Environment(obstacles=[poly1, poly2, poly3, poly4])

        # create environment instance
        env_instance = EnvironmentInstance(
            environment=env,
            query_interval=Interval(0, 100, closed="both"),
            scenario_range_x=(0, 10),
            scenario_range_y=(0, 10),
        )

        # test static obstacle collision test function
        assert env_instance.static_collision_free(pt1) == False
        assert env_instance.static_collision_free(pt2) == False
        assert env_instance.static_collision_free(pt3) == True
        assert env_instance.static_collision_free(pt4) == True

    def test_static_collision_ln(self):
        # set time interval parameters and scenario size
        interval_min = 0
        interval_max = 100
        range_x = (0, 9)
        range_y = (0, 6)
        resolution = 3

        # create environment for all test cases
        env = Environment()

        # create first static obstacle
        sh_poly1 = ShapelyPolygon([(2, 1), (4, 1), (4, 3), (2, 3)])
        poly1 = Polygon(geometry=sh_poly1, radius=0.0)

        # create second static obstacle
        sh_poly2 = ShapelyPolygon([(3, 4), (6, 4), (6, 6), (3, 6)])
        poly2 = Polygon(geometry=sh_poly2, radius=0.0)

        # create third static obstacle
        sh_poly3 = ShapelyPolygon([(7, 3), (8, 3), (8, 5), (7, 5)])
        poly3 = Polygon(geometry=sh_poly3, radius=0.0)

        # add static obstacle to environment
        env.add_obstacles([poly1, poly2, poly3])

        # add 15 random dynamic obstacles to environment
        env.add_random_obstacles(
            num_points=5,
            num_lines=5,
            num_polygons=5,
            min_x=range_x[0],
            max_x=range_x[1],
            min_y=range_y[0],
            max_y=range_y[1],
            min_radius=0,
            max_radius=4,
            min_interval=interval_min,
            max_interval=interval_max,
            only_dynamic=True,
            random_recurrence=True,
        )

        # create environment instance from environment
        env_inst = EnvironmentInstance(
            environment=env,
            query_interval=Interval(interval_min, interval_max, closed="both"),
            scenario_range_x=range_x,
            scenario_range_y=range_y,
            resolution=resolution,
        )

        # check that the number of obstalces is correct
        assert len(env_inst.static_obstacles) == 3
        assert len(env_inst.dynamic_obstacles) == 15

        # Test case 1 - test line in empty cell
        sh_line = ShapelyLine([(7, 1), (8, 1)])
        free1, cells1 = env_inst.static_collision_free_ln(sh_line)
        assert free1 == True
        assert len(cells1) == 1
        assert cells1[0] == (2, 0)

        # Test case 2 - test line in cell with static obstacle
        sh_line = ShapelyLine([(5, 3), (5.5, 3.5)])
        free2, cells2 = env_inst.static_collision_free_ln(sh_line)
        assert free2 == True
        assert len(cells2) == 1
        assert cells2[0] == (1, 1)

        # Test case 3 - test line in collision with obstacle in cell
        sh_line = ShapelyLine([(4, 5), (5, 5)])
        free3, cells3 = env_inst.static_collision_free_ln(sh_line)
        assert free3 == False
        assert len(cells3) == 0

        # Test case 4 - test line over multiple cells with no collision
        sh_line = ShapelyLine([(1, 3.5), (5, 3.5)])
        free4, cells4 = env_inst.static_collision_free_ln(sh_line)
        assert free4 == True
        assert len(cells4) == 2
        assert (0, 1) in cells4
        assert (1, 1) in cells4

        # Test case 5 - test line over multiple cells with close collision
        sh_line = ShapelyLine([(1.5, 0.5), (7.5, 2.5)])
        free5, cells5 = env_inst.static_collision_free_ln(sh_line)
        assert free5 == False
        assert len(cells5) == 0

        # Test case 6 - test line over multiple cells with collision
        sh_line = ShapelyLine([(2, 5), (8.5, 3.5)])
        free6, cells6 = env_inst.static_collision_free_ln(sh_line)
        assert free6 == False
        assert len(cells6) == 0

        # Test case 7 - test line over multiple cells without collision
        sh_line = ShapelyLine([(0.5, 4.5), (6.5, 2.5)])
        free7, cells7 = env_inst.static_collision_free_ln(sh_line)
        assert free7 == True
        assert len(cells7) == 4
        assert (0, 1) in cells7
        assert (0, 2) in cells7
        assert (1, 1) in cells7
        assert (2, 1) in cells7

    def test_dynamic_line_free_intervals(self):
        ## PART 1 - scenarios with recurrence free intervals
        env = Environment()
        collision_poly = ShapelyPolygon([(1, 1), (10, 1), (10, 10), (1, 10)])
        range_x = (0, 11)
        range_y = (0, 11)
        resolution = 2
        obs_interval1 = Interval(10, 20, closed="both")

        # add the obstacle to the environment
        poly_no_recurrence = Polygon(
            geometry=collision_poly,
            time_interval=obs_interval1,
            recurrence=Rec.NONE,
            radius=0.0,
        )
        env.add_obstacles([poly_no_recurrence])

        # colliding query line
        collision_line = ShapelyLine([(0, 0), (11, 11)])
        collision_cells = [(0, 0), (0, 1), (1, 0), (1, 1)]

        # collision-free query line
        free_line = ShapelyLine([(0.5, 0.5), (0.5, 11)])
        free_cells = [(0, 0), (0, 1)]

        # Test case 1: query interval overlaps with dynamic obstacle in the end
        query_in1 = Interval(15, 30, closed="both")
        env_inst1 = EnvironmentInstance(
            environment=env,
            query_interval=query_in1,
            scenario_range_x=range_x,
            scenario_range_y=range_y,
            resolution=resolution,
        )

        avail, blocked, in1 = env_inst1.collision_free_intervals_ln(
            line=collision_line, cells=collision_cells
        )
        assert len(in1) == 1
        assert in1[0] == Interval(20, 30, closed="both")
        assert avail == False
        assert blocked == False

        # Non-colliding line should not result in collision
        avail, blocked, in1_free = env_inst1.collision_free_intervals_ln(
            line=free_line, cells=free_cells
        )
        assert len(in1_free) == 1
        assert in1_free[0] == query_in1
        assert avail == True
        assert blocked == False

        avail, blocked, in1_free2 = env_inst1.collision_free_intervals_ln(
            line=free_line, cells=collision_cells
        )
        assert len(in1_free2) == 1
        assert in1_free2[0] == query_in1
        assert avail == True
        assert blocked == False

        # Test case 2: query interval overlaps with entire dynamic obstacle
        env_inst2 = EnvironmentInstance(
            environment=env,
            query_interval=Interval(10, 30, closed="both"),
            scenario_range_x=range_x,
            scenario_range_y=range_y,
            resolution=resolution,
        )

        avail, blocked, in2 = env_inst2.collision_free_intervals_ln(
            line=collision_line, cells=collision_cells
        )
        assert len(in2) == 1
        assert in2[0] == Interval(20, 30, closed="both")
        assert avail == False
        assert blocked == False

        # Test case 3: query interval overlaps with entire dynamic obstacle
        env_inst3 = EnvironmentInstance(
            environment=env,
            query_interval=Interval(5, 30, closed="both"),
            scenario_range_x=range_x,
            scenario_range_y=range_y,
            resolution=resolution,
        )

        avail, blocked, in3 = env_inst3.collision_free_intervals_ln(
            line=collision_line, cells=collision_cells
        )
        assert len(in3) == 2
        assert in3[0] == Interval(5, 10, closed="both")
        assert in3[1] == Interval(20, 30, closed="both")
        assert avail == False
        assert blocked == False

        # Test case 4: query interval overlaps with dynamic obstacle in the beginning
        env_inst4 = EnvironmentInstance(
            environment=env,
            query_interval=Interval(5, 15, closed="both"),
            scenario_range_x=range_x,
            scenario_range_y=range_y,
            resolution=resolution,
        )

        avail, blocked, in4 = env_inst4.collision_free_intervals_ln(
            line=collision_line, cells=collision_cells
        )
        assert len(in4) == 1
        assert in4[0] == Interval(5, 10, closed="both")
        assert avail == False
        assert blocked == False

        ## PART 2 - scenarios with recurrence intervals
        env2 = Environment()
        poly_recurrence = Polygon(
            geometry=collision_poly,
            time_interval=obs_interval1,
            recurrence=Rec.MINUTELY,
            radius=0.0,
        )
        env2.add_obstacles([poly_recurrence])

        # Test case 5: query interval overlaps with dynamic obstacle in the end
        env_inst5 = EnvironmentInstance(
            environment=env2,
            query_interval=Interval(135, 145, closed="both"),
            scenario_range_x=range_x,
            scenario_range_y=range_y,
            resolution=resolution,
        )

        avail, blocked, in5 = env_inst5.collision_free_intervals_ln(
            line=collision_line, cells=collision_cells
        )
        assert len(in5) == 1
        assert in5[0] == Interval(140, 145, closed="both")
        assert avail == False
        assert blocked == False

        # Test case 6: query interval overlaps with entire dynamic obstacle
        env_inst6 = EnvironmentInstance(
            environment=env2,
            query_interval=Interval(130, 150, closed="both"),
            scenario_range_x=range_x,
            scenario_range_y=range_y,
            resolution=resolution,
        )

        avail, blocked, in6 = env_inst6.collision_free_intervals_ln(
            line=collision_line, cells=collision_cells
        )
        assert len(in6) == 1
        assert in6[0] == Interval(140, 150, closed="both")
        assert avail == False
        assert blocked == False

        # Test case 7: query interval overlaps with entire dynamic obstacle
        env_inst7 = EnvironmentInstance(
            environment=env2,
            query_interval=Interval(125, 155, closed="both"),
            scenario_range_x=range_x,
            scenario_range_y=range_y,
            resolution=resolution,
        )

        avail, blocked, in7 = env_inst7.collision_free_intervals_ln(
            line=collision_line, cells=collision_cells
        )
        assert len(in7) == 2
        assert in7[0] == Interval(125, 130, closed="both")
        assert in7[1] == Interval(140, 155, closed="both")
        assert avail == False
        assert blocked == False

        # Test case 8: query interval overlaps with dynamic obstacle in the beginning
        env_inst8 = EnvironmentInstance(
            environment=env2,
            query_interval=Interval(125, 135, closed="both"),
            scenario_range_x=range_x,
            scenario_range_y=range_y,
            resolution=resolution,
        )

        avail, blocked, in8 = env_inst8.collision_free_intervals_ln(
            line=collision_line, cells=collision_cells
        )
        assert len(in8) == 1
        assert in8[0] == Interval(125, 130, closed="both")
        assert avail == False
        assert blocked == False

        # Test case 9: query interval spans over multiple occurences
        env_inst9 = EnvironmentInstance(
            environment=env2,
            query_interval=Interval(60, 160, closed="both"),
            scenario_range_x=range_x,
            scenario_range_y=range_y,
            resolution=resolution,
        )

        avail, blocked, in9 = env_inst9.collision_free_intervals_ln(
            line=collision_line, cells=collision_cells
        )
        assert len(in9) == 3
        assert in9[0] == Interval(60, 70, closed="both")
        assert in9[1] == Interval(80, 130, closed="both")
        assert in9[2] == Interval(140, 160, closed="both")
        assert avail == False
        assert blocked == False

        # Test case 10: query interval spans over multiple occurences
        env_inst10 = EnvironmentInstance(
            environment=env2,
            query_interval=Interval(70, 135, closed="both"),
            scenario_range_x=range_x,
            scenario_range_y=range_y,
            resolution=resolution,
        )

        avail, blocked, in10 = env_inst10.collision_free_intervals_ln(
            line=collision_line, cells=collision_cells
        )
        assert len(in10) == 1
        assert in10[0] == Interval(80, 130, closed="both")
        assert avail == False
        assert blocked == False

        # Test case 11: query interval spans over multiple occurences
        env_inst11 = EnvironmentInstance(
            environment=env2,
            query_interval=Interval(75, 135, closed="both"),
            scenario_range_x=range_x,
            scenario_range_y=range_y,
            resolution=resolution,
        )

        avail, blocked, in11 = env_inst11.collision_free_intervals_ln(
            line=collision_line, cells=collision_cells
        )
        assert len(in11) == 1
        assert in11[0] == Interval(80, 130, closed="both")
        assert avail == False
        assert blocked == False

        ## PART 3 - combinations of dynamic obstacles with and without recurrence
        env3 = Environment()
        poly_rec1 = Polygon(
            geometry=collision_poly,
            time_interval=Interval(10, 20, closed="both"),
            recurrence=Rec.MINUTELY,
            radius=0.0,
        )
        poly_rec2 = Polygon(
            geometry=collision_poly,
            time_interval=Interval(135, 195, closed="both"),
            recurrence=Rec.HOURLY,
            radius=0.0,
        )
        poly_norec = Polygon(
            geometry=collision_poly,
            time_interval=Interval(15, 25, closed="both"),
            radius=0.0,
        )
        env3.add_obstacles([poly_rec1, poly_rec2, poly_norec])

        # Test case 12: query interval spans over multiple occurences
        env_inst12 = EnvironmentInstance(
            environment=env3,
            query_interval=Interval(130, 195, closed="both"),
            scenario_range_x=range_x,
            scenario_range_y=range_y,
            resolution=resolution,
        )

        avail, blocked, in12 = env_inst12.collision_free_intervals_ln(
            line=collision_line, cells=collision_cells
        )
        assert len(in12) == 0
        assert avail == False
        assert blocked == True

        # Test case 13: query interval spans over multiple occurences
        env_inst13 = EnvironmentInstance(
            environment=env3,
            query_interval=Interval(10, 25, closed="both"),
            scenario_range_x=range_x,
            scenario_range_y=range_y,
            resolution=resolution,
        )

        avail, blocked, in13 = env_inst13.collision_free_intervals_ln(
            line=collision_line, cells=collision_cells
        )
        assert len(in13) == 0
        assert avail == False
        assert blocked == True

        # Test case 14: query interval spans over multiple occurences
        env_inst14 = EnvironmentInstance(
            environment=env3,
            query_interval=Interval(15, 120, closed="both"),
            scenario_range_x=range_x,
            scenario_range_y=range_y,
            resolution=resolution,
        )

        avail, blocked, in14 = env_inst14.collision_free_intervals_ln(
            line=collision_line, cells=collision_cells
        )
        assert len(in14) == 2
        assert in14[0] == Interval(25, 70, closed="both")
        assert in14[1] == Interval(80, 120, closed="both")
        assert avail == False
        assert blocked == False

        ## PART 4 - edge cases like empty dynamic obstacles
        env4 = Environment()
        query_in4 = Interval(0, 100, closed="both")
        env_inst4 = EnvironmentInstance(
            environment=env4,
            query_interval=query_in4,
            scenario_range_x=range_x,
            scenario_range_y=range_y,
            resolution=resolution,
        )

        avail, blocked, in15 = env_inst4.collision_free_intervals_ln(
            line=collision_line, cells=collision_cells
        )
        assert len(in15) == 1
        assert in15[0] == query_in4
        assert avail == True
        assert blocked == False

        avail, blocked, in16 = env_inst4.collision_free_intervals_ln(
            line=free_line, cells=free_cells
        )
        assert len(in16) == 1
        assert in16[0] == query_in4
        assert avail == True
        assert blocked == False

        avail, blocked, in17 = env_inst4.collision_free_intervals_ln(
            line=collision_line, cells=collision_cells
        )
        assert len(in17) == 1
        assert in17[0] == query_in4
        assert avail == True
        assert blocked == False
