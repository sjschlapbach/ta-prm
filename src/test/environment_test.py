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
from src.util.recurrence import Recurrence


class TestEnvironment:
    def test_create_environment(self):
        empty_env = Environment()
        assert empty_env is not None

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

    def test_add_random_obstacles(self):
        # Test case 1: only add obstacles of one type and check number of obstacles
        env1 = Environment()
        env1.add_random_obstacles(
            num_points=100,
            num_lines=0,
            num_polygons=0,
            min_x=0,
            max_x=300,
            min_y=0,
            max_y=300,
            min_radius=0,
            max_radius=4,
            min_interval=0,
            max_interval=100,
            max_size=20,
            only_static=False,
            only_dynamic=False,
            random_recurrence=True,
        )
        assert len(env1.obstacles) == 100

        env2 = Environment()
        env2.add_random_obstacles(
            num_points=0,
            num_lines=200,
            num_polygons=0,
            min_x=0,
            max_x=300,
            min_y=0,
            max_y=300,
            min_radius=0,
            max_radius=4,
            min_interval=0,
            max_interval=100,
            max_size=20,
            only_static=False,
            only_dynamic=False,
            random_recurrence=True,
        )
        assert len(env2.obstacles) == 200

        env3 = Environment()
        env3.add_random_obstacles(
            num_points=0,
            num_lines=0,
            num_polygons=300,
            min_x=0,
            max_x=300,
            min_y=0,
            max_y=300,
            min_radius=0,
            max_radius=4,
            min_interval=0,
            max_interval=100,
            max_size=20,
            only_static=False,
            only_dynamic=False,
            random_recurrence=True,
        )
        assert len(env3.obstacles) == 300

        # Test case 2: only add static obstacles and check the none has a time interval or recurrence parameter specified
        env4 = Environment()
        env4.add_random_obstacles(
            num_points=100,
            num_lines=100,
            num_polygons=100,
            min_x=0,
            max_x=300,
            min_y=0,
            max_y=300,
            min_radius=0,
            max_radius=4,
            min_interval=0,
            max_interval=100,
            max_size=20,
            only_static=True,
        )
        for obstacle in env4.obstacles:
            assert obstacle.time_interval == None
            assert obstacle.recurrence == Recurrence.NONE

        # Test case 3: only add dynamic obstacles and check the all have a time interval and possibly recurrence parameter specified
        env5 = Environment()
        env5.add_random_obstacles(
            num_points=100,
            num_lines=100,
            num_polygons=100,
            min_x=0,
            max_x=300,
            min_y=0,
            max_y=300,
            min_radius=0,
            max_radius=4,
            min_interval=0,
            max_interval=100,
            max_size=20,
            only_dynamic=True,
            random_recurrence=True,
        )
        for obstacle in env5.obstacles:
            assert obstacle.time_interval != None
            assert obstacle.recurrence in [
                Recurrence.NONE,
                Recurrence.MINUTELY,
                Recurrence.HOURLY,
                Recurrence.DAILY,
            ]

        # Test case 4: generate random environment with only static, only dynamic and combined obstacles and generate instances from them
        env6 = Environment()
        interval6 = Interval(0, 100)
        range_x6 = (0, 300)
        range_y6 = (0, 300)
        env6.add_random_obstacles(
            num_points=100,
            num_lines=100,
            num_polygons=100,
            min_x=range_x6[0],
            max_x=range_x6[1],
            min_y=range_y6[0],
            max_y=range_y6[1],
            min_radius=0,
            max_radius=4,
            min_interval=interval6.left,
            max_interval=interval6.right,
            max_size=20,
            only_static=True,
        )
        env6_instance = EnvironmentInstance(
            environment=env6,
            query_interval=interval6,
            scenario_range_x=range_x6,
            scenario_range_y=range_y6,
        )

        # Test case 5: check that the number of obstacles in the instance can be smaller, equal or larger depending on the query interval
        env7 = Environment()
        interval5 = Interval(0, 100)
        range_x5 = (0, 250)
        range_y5 = (0, 250)
        env7.add_random_obstacles(
            num_points=100,
            num_lines=100,
            num_polygons=100,
            min_x=range_x5[0],
            max_x=range_x5[1],
            min_y=range_y5[0],
            max_y=range_y5[1],
            min_radius=0,
            max_radius=4,
            min_interval=interval5.left,
            max_interval=interval5.right,
            max_size=20,
            only_static=True,
        )
        env7_instance = EnvironmentInstance(
            environment=env7,
            query_interval=interval5,
            scenario_range_x=range_x5,
            scenario_range_y=range_y5,
        )
        assert len(env7_instance.static_obstacles) == 300
        assert len(env7_instance.dynamic_obstacles) == 0

        env8 = Environment()
        interval8 = Interval(0, 100)
        range_x8 = (0, 350)
        range_y8 = (0, 350)
        env8.add_random_obstacles(
            num_points=100,
            num_lines=100,
            num_polygons=100,
            min_x=range_x8[0],
            max_x=range_x8[1],
            min_y=range_y8[0],
            max_y=range_y8[1],
            min_radius=0,
            max_radius=4,
            min_interval=interval8.left,
            max_interval=interval8.right,
            max_size=20,
            only_dynamic=True,
        )
        env8_instance = EnvironmentInstance(
            environment=env8,
            query_interval=interval8,
            scenario_range_x=range_x8,
            scenario_range_y=range_y8,
        )
        assert len(env8_instance.static_obstacles) == 0
        assert len(env8_instance.dynamic_obstacles) == 300

        env9 = Environment()
        interval9 = Interval(0, 100)
        interval9_shorter = Interval(20, 50)
        range_x9 = (0, 400)
        range_y9 = (0, 400)
        env9.add_random_obstacles(
            num_points=100,
            num_lines=100,
            num_polygons=100,
            min_x=range_x9[0],
            max_x=range_x9[1],
            min_y=range_y9[0],
            max_y=range_y9[1],
            min_radius=0,
            max_radius=4,
            min_interval=interval9.left,
            max_interval=interval9.right,
            max_size=20,
        )
        env9_instance = EnvironmentInstance(
            environment=env9,
            query_interval=interval9_shorter,
            scenario_range_x=range_x9,
            scenario_range_y=range_y9,
        )
        assert len(env9_instance.static_obstacles) <= 300
        assert len(env9_instance.dynamic_obstacles) <= 300
        assert len(env9_instance.static_obstacles) >= 0
        assert len(env9_instance.dynamic_obstacles) >= 0

        env10 = Environment()
        interval10 = Interval(0, 100)
        interval10_shorter = Interval(20, 50)
        range_x10 = (0, 450)
        range_y10 = (0, 450)
        env10.add_random_obstacles(
            num_points=100,
            num_lines=100,
            num_polygons=100,
            min_x=range_x10[0],
            max_x=range_x10[1],
            min_y=range_y10[0],
            max_y=range_y10[1],
            min_radius=0,
            max_radius=4,
            min_interval=interval10.left,
            max_interval=interval10.right,
            max_size=20,
            only_dynamic=True,
        )
        env10_instance = EnvironmentInstance(
            environment=env10,
            query_interval=interval10_shorter,
            scenario_range_x=range_x10,
            scenario_range_y=range_y10,
        )

        assert len(env10_instance.static_obstacles) <= 300
        assert len(env10_instance.dynamic_obstacles) <= 300
        assert len(env10_instance.static_obstacles) >= 0
        assert len(env10_instance.dynamic_obstacles) >= 0

        env11 = Environment()
        interval11 = Interval(0, 100)
        interval11_longer = Interval(0, 10000)
        range_x11 = (0, 500)
        range_y11 = (0, 500)
        env11.add_random_obstacles(
            num_points=100,
            num_lines=100,
            num_polygons=100,
            min_x=range_x11[0],
            max_x=range_x11[1],
            min_y=range_y11[0],
            max_y=range_y11[1],
            min_radius=0,
            max_radius=4,
            min_interval=interval11.left,
            max_interval=interval11.right,
            max_size=20,
            only_dynamic=True,
        )
        env11_instance = EnvironmentInstance(
            environment=env11,
            query_interval=interval11_longer,
            scenario_range_x=range_x11,
            scenario_range_y=range_y11,
        )
        assert len(env11_instance.static_obstacles) == 0
        assert len(env11_instance.dynamic_obstacles) >= 300

        # Test case 7: check that the number of obstacles in the instance is smaller or equal for smaller scenario sizes
        env12 = Environment()
        interval12 = Interval(0, 100)
        range_x12 = (0, 550)
        range_y12 = (0, 550)
        env12.add_random_obstacles(
            num_points=100,
            num_lines=100,
            num_polygons=100,
            min_x=range_x12[0],
            max_x=range_x12[1],
            min_y=range_y12[0],
            max_y=range_y12[1],
            min_radius=0,
            max_radius=4,
            min_interval=interval12.left,
            max_interval=interval12.right,
            max_size=20,
        )
        env12_instance = EnvironmentInstance(
            environment=env12,
            query_interval=interval12,
            scenario_range_x=range_x12,
            scenario_range_y=range_y12,
        )
        assert len(env12_instance.static_obstacles) <= 300
        assert len(env12_instance.dynamic_obstacles) <= 300
        assert len(env12_instance.static_obstacles) >= 0
        assert len(env12_instance.dynamic_obstacles) >= 0

        env13 = Environment()
        interval13 = Interval(0, 100)
        range_x13 = (0, 600)
        range_y13 = (0, 600)
        range_x13_smaller = (0, 300)
        range_y13_smaller = (0, 300)
        env13.add_random_obstacles(
            num_points=1000,
            num_lines=1000,
            num_polygons=1000,
            min_x=range_x13[0],
            max_x=range_x13[1],
            min_y=range_y13[0],
            max_y=range_y13[1],
            min_radius=0,
            max_radius=4,
            min_interval=interval13.left,
            max_interval=interval13.right,
            max_size=20,
        )
        assert len(env13.obstacles) == 3000

        env13_instance = EnvironmentInstance(
            environment=env13,
            query_interval=interval13,
            scenario_range_x=range_x13_smaller,
            scenario_range_y=range_y13_smaller,
        )
        assert len(env13_instance.static_obstacles) < 3000
