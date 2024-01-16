from pandas import Interval
from shapely.geometry import LineString as ShapelyLine

from src.algorithm.graph import Graph
from src.envs.environment import Environment
from src.envs.environment_instance import EnvironmentInstance
from src.algorithm.timed_edge import TimedEdge


class TestGraph:
    def test_graph_creation(self):
        # time interval
        interval_start = 0
        interval_end = 100
        x_range = (0, 300)
        y_range = (0, 300)

        # create environment with random obstacles
        env = Environment()
        env.add_random_obstacles(
            num_points=100,
            num_lines=100,
            num_polygons=100,
            min_x=x_range[0],
            max_x=x_range[1],
            min_y=y_range[0],
            max_y=y_range[1],
            min_interval=interval_start,
            max_interval=interval_end,
            random_recurrence=True,
        )

        # create environment instance
        env_inst = EnvironmentInstance(
            environment=env,
            query_interval=Interval(interval_start, interval_end, closed="both"),
            scenario_range_x=x_range,
            scenario_range_y=y_range,
        )

        # default parameters
        default_samples = 1000
        default_max_distance = 10.0

        # create graph
        graph = Graph(
            num_samples=default_samples,
            neighbour_distance=default_max_distance,
            max_connections=10,
            env=env_inst,
        )

        # check if graph vertices are within specified range and collision-free
        assert len(graph.vertices) == default_samples
        assert graph.num_vertices == default_samples
        assert graph.neighbour_distance == default_max_distance
        assert graph.max_connections == 10
        for vertex in graph.vertices.values():
            assert vertex.x >= x_range[0] and vertex.x <= x_range[1]
            assert vertex.y >= y_range[0] and vertex.y <= y_range[1]
            assert env_inst.static_collision_free(vertex)

    def test_timed_line_availability(self):
        # create shapely line objects
        ln = ShapelyLine([(0, 0), (100, 0)])

        # create intervals
        interval1 = Interval(10, 20, closed="both")
        interval2 = Interval(40, 45, closed="both")
        interval3 = Interval(60, 100, closed="both")
        interval4 = Interval(120, 130, closed="both")
        interval5 = Interval(170, 190, closed="both")
        interval6 = Interval(200, 240, closed="both")

        # create timed edges with different number of availabilities
        line1 = TimedEdge(geometry=ln, always_available=True, availability=[])
        line2 = TimedEdge(geometry=ln, availability=[interval1])
        line3 = TimedEdge(geometry=ln, availability=[interval1, interval2])
        line4 = TimedEdge(
            geometry=ln,
            availability=[
                interval1,
                interval2,
                interval3,
                interval4,
                interval5,
                interval6,
            ],
        )

        # Test case 1: line is always available
        line1 = TimedEdge(geometry=ln, always_available=True, availability=[])
        test_in1 = Interval(0, 100, closed="both")
        test_in1_alt = Interval(100, 200, closed="both")
        assert line1.is_available(test_in1)
        assert line1.is_available(test_in1_alt)

        # Test case 2: query interval ends before first availability
        line2 = TimedEdge(geometry=ln, availability=[interval1])
        test_in2 = Interval(0, 5, closed="both")
        assert not line2.is_available(test_in2)

        # Test case 3: query interval intersects with beginning of availability
        test_in3 = Interval(5, 15, closed="both")
        assert not line2.is_available(test_in2)

        # Test case 4: query interval lies inside first availability
        test_in4 = Interval(15, 17, closed="both")
        assert line2.is_available(test_in4)

        # Test case 5: query interval is identical to first availability
        test_in5 = Interval(10, 20, closed="both")
        assert line2.is_available(test_in5)

        # Test case 6: query interval intersects with end of availability
        test_in6 = Interval(18, 22, closed="both")
        assert not line2.is_available(test_in6)

        # Test case 7: query interval starts after first availability
        test_in7 = Interval(22, 25, closed="both")
        assert not line2.is_available(test_in7)

        # Test case 8: query interval is longer than first availability
        test_in8 = Interval(5, 25, closed="both")
        assert not line2.is_available(test_in8)

        # Test case 9: query interval lies before first availability
        test_in9 = Interval(0, 7, closed="both")
        assert not line3.is_available(test_in9)

        # Test case 10: query interval intersects with beginning of first availability
        test_in10 = Interval(7, 12, closed="both")
        assert not line3.is_available(test_in10)

        # Test case 11: query interval lies inside first availability
        test_in11 = Interval(12, 15, closed="both")
        assert line3.is_available(test_in11)

        # Test case 12: query interval intersects with end of first availability
        test_in12 = Interval(17, 22, closed="both")
        assert not line3.is_available(test_in12)

        # Test case 13: query interval lies in between availabilities
        test_in13 = Interval(25, 35, closed="both")
        assert not line3.is_available(test_in13)

        # Test case 14: query interval intersects with beginning of second availability
        test_in14 = Interval(35, 45, closed="both")
        assert not line3.is_available(test_in14)

        # Test case 15: query interval lies inside second availability
        test_in15 = Interval(41, 43, closed="both")
        assert line3.is_available(test_in15)

        # Test case 16: query interval intersects with end of second availability
        test_in16 = Interval(43, 47, closed="both")
        assert not line3.is_available(test_in16)

        # Test case 17: query interval is equal to first availability
        test_in17 = Interval(10, 20, closed="both")
        assert line3.is_available(test_in17)

        # Test case 18: query interval is equal to second availability
        test_in18 = Interval(40, 45, closed="both")
        assert line3.is_available(test_in18)

        # Test case 19: query interval is longer than first availability
        test_in19 = Interval(5, 25, closed="both")
        assert not line3.is_available(test_in19)

        # Test case 20: query interval is longer than second availability
        test_in20 = Interval(35, 50, closed="both")
        assert not line3.is_available(test_in20)

        # Test case 21: query interval bridges gap between availabilities
        test_in21 = Interval(15, 42, closed="both")
        assert not line3.is_available(test_in21)

        # Test case 22: query interval is longer than both availabilities
        test_in22 = Interval(5, 50, closed="both")
        assert not line3.is_available(test_in22)

        # Test case 23: query interval lies before first availability
        test_in23 = Interval(0, 7, closed="both")
        assert not line4.is_available(test_in23)

        # Test case 24: query interval intersects with beginning of first availability
        test_in24 = Interval(7, 12, closed="both")
        assert not line4.is_available(test_in24)

        # Test case 25: query interval lies inside first availability
        test_in25 = Interval(12, 15, closed="both")
        assert line4.is_available(test_in25)

        # Test case 26: query interval is equal to first availability
        test_in26 = Interval(10, 20, closed="both")
        assert line4.is_available(test_in26)

        # Test case 27: query interval intersects with end of first availability
        test_in27 = Interval(17, 22, closed="both")
        assert not line4.is_available(test_in27)

        # Test case 28: query interval lies in between availabilities
        test_in28 = Interval(25, 35, closed="both")
        assert not line4.is_available(test_in28)

        # Test case 29: query interval intersects with beginning of second availability
        test_in29 = Interval(35, 42, closed="both")
        assert not line4.is_available(test_in29)

        # Test case 30: query interval bridges gap between availabilities
        test_in30 = Interval(15, 42, closed="both")
        assert not line4.is_available(test_in30)

        # Test case 31: query interval covers first two availabilities
        test_in31 = Interval(5, 50, closed="both")
        assert not line4.is_available(test_in31)

        # Test case 32: query interval lies inside second availability
        test_in32 = Interval(41, 43, closed="both")
        assert line4.is_available(test_in32)

        # Test case 33: query interval is equal to second availability
        test_in33 = Interval(40, 45, closed="both")
        assert line4.is_available(test_in33)

        # Test case 34: query interval intersects with end of second availability
        test_in34 = Interval(43, 47, closed="both")
        assert not line4.is_available(test_in34)

        # Test case 35: query interval bridges gap between second and third availability
        test_in35 = Interval(45, 60, closed="both")
        assert not line4.is_available(test_in35)

        # Test case 36: query interval overlaps second and partially third availability
        test_in36 = Interval(35, 65, closed="both")
        assert not line4.is_available(test_in36)

        # Test case 37: query interval overlaps first 2 availabilities and partially third availability
        test_in37 = Interval(5, 65, closed="both")
        assert not line4.is_available(test_in37)

        # Test case 38: query interval overlaps first start of third availability
        test_in38 = Interval(55, 65, closed="both")
        assert not line4.is_available(test_in38)

        # Test case 39: query interval lies inside thrid availability
        test_in39 = Interval(65, 95, closed="both")
        assert line4.is_available(test_in39)

        # Test case 40: query interval is equal to third availability
        test_in40 = Interval(60, 100, closed="both")
        assert line4.is_available(test_in40)

        # Test case 41: query interval overlaps with end of third availability
        test_in41 = Interval(95, 105, closed="both")
        assert not line4.is_available(test_in41)

        # Test case 42: query interval overlaps with entire third availability
        test_in42 = Interval(55, 105, closed="both")
        assert not line4.is_available(test_in42)

        # Test case 43: query interval lies after between third and fourth availability
        test_in43 = Interval(105, 115, closed="both")
        assert not line4.is_available(test_in43)

        # Test case 44: query interval intersects with start of fourth availability
        test_in44 = Interval(115, 125, closed="both")
        assert not line4.is_available(test_in44)

        # Test case 45: query interval lies inside fourth availability
        test_in45 = Interval(125, 128, closed="both")
        assert line4.is_available(test_in45)

        # Test case 46: query interval is equal to fourth availability
        test_in46 = Interval(120, 130, closed="both")
        assert line4.is_available(test_in46)

        # Test case 47: query interval intersects with end of fourth availability
        test_in47 = Interval(125, 135, closed="both")
        assert not line4.is_available(test_in47)

        # Test case 48: query interval overlaps fourth and fifth availability
        test_in48 = Interval(115, 200, closed="both")
        assert not line4.is_available(test_in48)

        # Test case 49: query interval overlaps with start of fifth availability
        test_in49 = Interval(135, 175, closed="both")
        assert not line4.is_available(test_in49)

        # Test case 50: query interval lies inside fifth availability
        test_in50 = Interval(175, 185, closed="both")
        assert line4.is_available(test_in50)

        # Test case 51: query interval is equal to fifth availability
        test_in51 = Interval(170, 190, closed="both")
        assert line4.is_available(test_in51)

        # Test case 52: query interval overlaps with end of fifth availability
        test_in52 = Interval(185, 195, closed="both")
        assert not line4.is_available(test_in52)

        # Test case 53: query interval overlaps with more than entire fifth availability
        test_in53 = Interval(165, 195, closed="both")
        assert not line4.is_available(test_in53)

        # Test case 54: query interval lies between fifth and sixth availability
        test_in54 = Interval(192, 197, closed="both")
        assert not line4.is_available(test_in54)

        # Test case 55: query interval intersects with start of sixth availability
        test_in55 = Interval(197, 205, closed="both")
        assert not line4.is_available(test_in55)

        # Test case 56: query interval lies inside sixth availability
        test_in56 = Interval(210, 230, closed="both")
        assert line4.is_available(test_in56)

        # Test case 57: query interval is equal to sixth availability
        test_in57 = Interval(200, 240, closed="both")
        assert line4.is_available(test_in57)

        # Test case 58: query interval intersects with end of sixth availability
        test_in58 = Interval(235, 245, closed="both")
        assert not line4.is_available(test_in58)

        # Test case 59: query interval overlaps with more than entire sixth availability
        test_in59 = Interval(195, 245, closed="both")
        assert not line4.is_available(test_in59)

        # Test case 60: query interval lies after sixth availability
        test_in60 = Interval(245, 255, closed="both")
        assert not line4.is_available(test_in60)

        # Test case 61: query interval covers the last three availabilities
        test_in61 = Interval(110, 250, closed="both")
        assert not line4.is_available(test_in61)

        # Test case 62: query interval covers the second and third availabilities
        test_in62 = Interval(35, 105, closed="both")
        assert not line4.is_available(test_in62)

        # Test case 63: query interval intersects with all availabilities
        test_in63 = Interval(15, 230, closed="both")
        assert not line4.is_available(test_in63)

        # Test case 64: query interval is longer than all availabilities
        test_in64 = Interval(5, 250, closed="both")
        assert not line4.is_available(test_in64)
