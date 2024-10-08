from pandas import Interval
from shapely.geometry import LineString as ShapelyLine
import numpy as np
import json
import os

from src.algorithms.graph import Graph
from src.envs.environment import Environment
from src.envs.environment_instance import EnvironmentInstance
from src.algorithms.timed_edge import TimedEdge


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
            seed=0,
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

        # create graph
        graph = Graph(
            num_samples=default_samples,
            seed=0,
            env=env_inst,
        )

        # check if graph vertices are within specified range and collision-free
        assert len(graph.vertices) == default_samples
        assert graph.num_vertices == default_samples
        for vertex in graph.vertices.values():
            assert vertex.x >= x_range[0] and vertex.x <= x_range[1]
            assert vertex.y >= y_range[0] and vertex.y <= y_range[1]
            assert env_inst.static_collision_free(vertex)

        # connect start and goal node
        start_coords = (x_range[0] + 2, y_range[0] + 2)
        goal_coords = (x_range[1] - 2, y_range[1] - 2)
        graph.connect_start(coords=start_coords)
        graph.connect_goal(coords=goal_coords)

        # check if start and goal node are connected
        assert len(graph.vertices) == default_samples + 2
        assert graph.num_vertices == default_samples
        assert graph.start == default_samples
        assert graph.goal == default_samples + 1
        assert graph.vertices[graph.start] is not None
        assert graph.vertices[graph.goal] is not None
        assert graph.vertices[graph.start].x == start_coords[0]
        assert graph.vertices[graph.start].y == start_coords[1]
        assert graph.vertices[graph.goal].x == goal_coords[0]
        assert graph.vertices[graph.goal].y == goal_coords[1]
        assert len(graph.connections[graph.start]) > 0
        assert len(graph.connections[graph.goal]) > 0

        # check that the heuristic does not contain any inf values anymore
        for value in graph.heuristic.values():
            assert np.isfinite(value)

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
        cost = 20
        line1 = TimedEdge(
            geometry=ln, always_available=True, cost=cost, availability=[]
        )
        line2 = TimedEdge(geometry=ln, availability=[interval1], cost=cost)
        line3 = TimedEdge(geometry=ln, availability=[interval1, interval2], cost=cost)
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
            cost=cost,
        )

        # Test case 1: line is always available
        line1 = TimedEdge(
            geometry=ln, always_available=True, availability=[], cost=cost
        )
        test_in1 = Interval(0, 100, closed="both")
        test_in1_alt = Interval(100, 200, closed="both")
        assert line1.get_cost(test_in1) == cost
        assert line1.get_cost(test_in1_alt) == cost

        # Test case 2: query interval ends before first availability
        line2 = TimedEdge(geometry=ln, availability=[interval1], cost=cost)
        test_in2 = Interval(0, 5, closed="both")
        assert np.isinf(line2.get_cost(test_in2))

        # Test case 3: query interval intersects with beginning of availability
        test_in3 = Interval(5, 15, closed="both")
        assert np.isinf(line2.get_cost(test_in3))

        # Test case 4: query interval lies inside first availability
        test_in4 = Interval(15, 17, closed="both")
        assert line2.get_cost(test_in4) == cost

        # Test case 5: query interval is identical to first availability
        test_in5 = Interval(10, 20, closed="both")
        assert line2.get_cost(test_in5) == cost

        # Test case 6: query interval intersects with end of availability
        test_in6 = Interval(18, 22, closed="both")
        assert np.isinf(line2.get_cost(test_in6))

        # Test case 7: query interval starts after first availability
        test_in7 = Interval(22, 25, closed="both")
        assert np.isinf(line2.get_cost(test_in7))

        # Test case 8: query interval is longer than first availability
        test_in8 = Interval(5, 25, closed="both")
        assert np.isinf(line2.get_cost(test_in8))

        # Test case 9: query interval lies before first availability
        test_in9 = Interval(0, 7, closed="both")
        assert np.isinf(line3.get_cost(test_in9))

        # Test case 10: query interval intersects with beginning of first availability
        test_in10 = Interval(7, 12, closed="both")
        assert np.isinf(line3.get_cost(test_in10))

        # Test case 11: query interval lies inside first availability
        test_in11 = Interval(12, 15, closed="both")
        assert line3.get_cost(test_in11) == cost

        # Test case 12: query interval intersects with end of first availability
        test_in12 = Interval(17, 22, closed="both")
        assert np.isinf(line3.get_cost(test_in12))

        # Test case 13: query interval lies in between availabilities
        test_in13 = Interval(25, 35, closed="both")
        assert np.isinf(line3.get_cost(test_in13))

        # Test case 14: query interval intersects with beginning of second availability
        test_in14 = Interval(35, 45, closed="both")
        assert np.isinf(line3.get_cost(test_in14))

        # Test case 15: query interval lies inside second availability
        test_in15 = Interval(41, 43, closed="both")
        assert line3.get_cost(test_in15) == cost

        # Test case 16: query interval intersects with end of second availability
        test_in16 = Interval(43, 47, closed="both")
        assert np.isinf(line3.get_cost(test_in16))

        # Test case 17: query interval is equal to first availability
        test_in17 = Interval(10, 20, closed="both")
        assert line3.get_cost(test_in17) == cost

        # Test case 18: query interval is equal to second availability
        test_in18 = Interval(40, 45, closed="both")
        assert line3.get_cost(test_in18) == cost

        # Test case 19: query interval is longer than first availability
        test_in19 = Interval(5, 25, closed="both")
        assert np.isinf(line3.get_cost(test_in19))

        # Test case 20: query interval is longer than second availability
        test_in20 = Interval(35, 50, closed="both")
        assert np.isinf(line3.get_cost(test_in20))

        # Test case 21: query interval bridges gap between availabilities
        test_in21 = Interval(15, 42, closed="both")
        assert np.isinf(line3.get_cost(test_in21))

        # Test case 22: query interval is longer than both availabilities
        test_in22 = Interval(5, 50, closed="both")
        assert np.isinf(line3.get_cost(test_in22))

        # Test case 23: query interval lies before first availability
        test_in23 = Interval(0, 7, closed="both")
        assert np.isinf(line4.get_cost(test_in23))

        # Test case 24: query interval intersects with beginning of first availability
        test_in24 = Interval(7, 12, closed="both")
        assert np.isinf(line4.get_cost(test_in24))

        # Test case 25: query interval lies inside first availability
        test_in25 = Interval(12, 15, closed="both")
        assert line4.get_cost(test_in25) == cost

        # Test case 26: query interval is equal to first availability
        test_in26 = Interval(10, 20, closed="both")
        assert line4.get_cost(test_in26) == cost

        # Test case 27: query interval intersects with end of first availability
        test_in27 = Interval(17, 22, closed="both")
        assert np.isinf(line4.get_cost(test_in27))

        # Test case 28: query interval lies in between availabilities
        test_in28 = Interval(25, 35, closed="both")
        assert np.isinf(line4.get_cost(test_in28))

        # Test case 29: query interval intersects with beginning of second availability
        test_in29 = Interval(35, 42, closed="both")
        assert np.isinf(line4.get_cost(test_in29))

        # Test case 30: query interval bridges gap between availabilities
        test_in30 = Interval(15, 42, closed="both")
        assert np.isinf(line4.get_cost(test_in30))

        # Test case 31: query interval covers first two availabilities
        test_in31 = Interval(5, 50, closed="both")
        assert np.isinf(line4.get_cost(test_in31))

        # Test case 32: query interval lies inside second availability
        test_in32 = Interval(41, 43, closed="both")
        assert line4.get_cost(test_in32) == cost

        # Test case 33: query interval is equal to second availability
        test_in33 = Interval(40, 45, closed="both")
        assert line4.get_cost(test_in33) == cost

        # Test case 34: query interval intersects with end of second availability
        test_in34 = Interval(43, 47, closed="both")
        assert np.isinf(line4.get_cost(test_in34))

        # Test case 35: query interval bridges gap between second and third availability
        test_in35 = Interval(45, 60, closed="both")
        assert np.isinf(line4.get_cost(test_in35))

        # Test case 36: query interval overlaps second and partially third availability
        test_in36 = Interval(35, 65, closed="both")
        assert np.isinf(line4.get_cost(test_in36))

        # Test case 37: query interval overlaps first 2 availabilities and partially third availability
        test_in37 = Interval(5, 65, closed="both")
        assert np.isinf(line4.get_cost(test_in37))

        # Test case 38: query interval overlaps first start of third availability
        test_in38 = Interval(55, 65, closed="both")
        assert np.isinf(line4.get_cost(test_in38))

        # Test case 39: query interval lies inside thrid availability
        test_in39 = Interval(65, 95, closed="both")
        assert line4.get_cost(test_in39) == cost

        # Test case 40: query interval is equal to third availability
        test_in40 = Interval(60, 100, closed="both")
        assert line4.get_cost(test_in40) == cost

        # Test case 41: query interval overlaps with end of third availability
        test_in41 = Interval(95, 105, closed="both")
        assert np.isinf(line4.get_cost(test_in41))

        # Test case 42: query interval overlaps with entire third availability
        test_in42 = Interval(55, 105, closed="both")
        assert np.isinf(line4.get_cost(test_in42))

        # Test case 43: query interval lies after between third and fourth availability
        test_in43 = Interval(105, 115, closed="both")
        assert np.isinf(line4.get_cost(test_in43))

        # Test case 44: query interval intersects with start of fourth availability
        test_in44 = Interval(115, 125, closed="both")
        assert np.isinf(line4.get_cost(test_in44))

        # Test case 45: query interval lies inside fourth availability
        test_in45 = Interval(125, 128, closed="both")
        assert line4.get_cost(test_in45) == cost

        # Test case 46: query interval is equal to fourth availability
        test_in46 = Interval(120, 130, closed="both")
        assert line4.get_cost(test_in46) == cost

        # Test case 47: query interval intersects with end of fourth availability
        test_in47 = Interval(125, 135, closed="both")
        assert np.isinf(line4.get_cost(test_in47))

        # Test case 48: query interval overlaps fourth and fifth availability
        test_in48 = Interval(115, 200, closed="both")
        assert np.isinf(line4.get_cost(test_in48))

        # Test case 49: query interval overlaps with start of fifth availability
        test_in49 = Interval(135, 175, closed="both")
        assert np.isinf(line4.get_cost(test_in49))

        # Test case 50: query interval lies inside fifth availability
        test_in50 = Interval(175, 185, closed="both")
        assert line4.get_cost(test_in50) == cost

        # Test case 51: query interval is equal to fifth availability
        test_in51 = Interval(170, 190, closed="both")
        assert line4.get_cost(test_in51) == cost

        # Test case 52: query interval overlaps with end of fifth availability
        test_in52 = Interval(185, 195, closed="both")
        assert np.isinf(line4.get_cost(test_in52))

        # Test case 53: query interval overlaps with more than entire fifth availability
        test_in53 = Interval(165, 195, closed="both")
        assert np.isinf(line4.get_cost(test_in53))

        # Test case 54: query interval lies between fifth and sixth availability
        test_in54 = Interval(192, 197, closed="both")
        assert np.isinf(line4.get_cost(test_in54))

        # Test case 55: query interval intersects with start of sixth availability
        test_in55 = Interval(197, 205, closed="both")
        assert np.isinf(line4.get_cost(test_in55))

        # Test case 56: query interval lies inside sixth availability
        test_in56 = Interval(210, 230, closed="both")
        assert line4.get_cost(test_in56) == cost

        # Test case 57: query interval is equal to sixth availability
        test_in57 = Interval(200, 240, closed="both")
        assert line4.get_cost(test_in57) == cost

        # Test case 58: query interval intersects with end of sixth availability
        test_in58 = Interval(235, 245, closed="both")
        assert np.isinf(line4.get_cost(test_in58))

        # Test case 59: query interval overlaps with more than entire sixth availability
        test_in59 = Interval(195, 245, closed="both")
        assert np.isinf(line4.get_cost(test_in59))

        # Test case 60: query interval lies after sixth availability
        test_in60 = Interval(245, 255, closed="both")
        assert np.isinf(line4.get_cost(test_in60))

        # Test case 61: query interval covers the last three availabilities
        test_in61 = Interval(110, 250, closed="both")
        assert np.isinf(line4.get_cost(test_in61))

        # Test case 62: query interval covers the second and third availabilities
        test_in62 = Interval(35, 105, closed="both")
        assert np.isinf(line4.get_cost(test_in62))

        # Test case 63: query interval intersects with all availabilities
        test_in63 = Interval(15, 230, closed="both")
        assert np.isinf(line4.get_cost(test_in63))

        # Test case 64: query interval is longer than all availabilities
        test_in64 = Interval(5, 250, closed="both")
        assert np.isinf(line4.get_cost(test_in64))

    def test_save_load_timed_edge(self):
        # Test case 1: timed edge, which is temporarily restricted
        geometry = ShapelyLine([(0, 0), (100, 0)])
        length = geometry.length
        cost = 20
        interval1 = Interval(10, 20, closed="both")
        interval2 = Interval(40, 45, closed="both")
        interval3 = Interval(60, 100, closed="both")
        always_available = False

        tedge1 = TimedEdge(
            geometry=geometry,
            availability=[interval1, interval2, interval3],
            always_available=always_available,
            cost=cost,
        )

        assert tedge1.geometry == geometry
        assert tedge1.always_available == always_available
        assert tedge1.cost == cost
        assert tedge1.length == length
        assert len(tedge1.availability) == 3
        assert tedge1.availability[0] == interval1
        assert tedge1.availability[1] == interval2
        assert tedge1.availability[2] == interval3

        json_edge = tedge1.export_to_json()

        with open("test_edge_1.txt", "w") as f:
            json.dump(json_edge, f)
        with open("test_edge_1.txt", "r") as f:
            tedge1_input = json.load(f)
        os.remove("test_edge_1.txt")

        tedge1_loaded = TimedEdge(geometry=None, availability=[], json_obj=tedge1_input)

        assert tedge1_loaded.geometry == tedge1.geometry
        assert len(tedge1_loaded.availability) == len(tedge1.availability)
        assert tedge1_loaded.always_available == tedge1.always_available
        assert tedge1_loaded.cost == tedge1.cost
        assert tedge1_loaded.length == tedge1.length

        for i in range(len(tedge1_loaded.availability)):
            assert tedge1_loaded.availability[i] == tedge1.availability[i]

        # Test case 2: timed edge, which is always available
        geometry = ShapelyLine([(0, 0), (100, 0)])
        length = geometry.length
        cost = 20
        always_available = True

        tedge2 = TimedEdge(
            geometry=geometry,
            availability=[],
            always_available=always_available,
            cost=cost,
        )

        assert tedge2.geometry == geometry
        assert tedge2.always_available == always_available
        assert tedge2.cost == cost
        assert tedge2.length == length
        assert len(tedge2.availability) == 0

        json_edge = tedge2.export_to_json()

        with open("test_edge_2.json", "w") as f:
            json.dump(json_edge, f)
        with open("test_edge_2.json", "r") as f:
            tedge2_input = json.load(f)
        os.remove("test_edge_2.json")

        tedge2_loaded = TimedEdge(geometry=None, availability=[], json_obj=tedge2_input)

        assert tedge2_loaded.geometry == tedge2.geometry
        assert len(tedge2_loaded.availability) == len(tedge2.availability)
        assert tedge2_loaded.always_available == tedge2.always_available
        assert tedge2_loaded.cost == tedge2.cost
        assert tedge2_loaded.length == tedge2.length

    def test_save_load_graph(self):
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
            seed=0,
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

        # create graph
        graph = Graph(
            num_samples=default_samples,
            seed=0,
            env=env_inst,
        )

        # connect start and goal node
        start_coords = (x_range[0] + 2, y_range[0] + 2)
        goal_coords = (x_range[1] - 2, y_range[1] - 2)
        graph.connect_start(coords=start_coords)
        graph.connect_goal(coords=goal_coords)

        # save graph
        json_graph = graph.save("test_graph.json")

        # load graph again
        graph_loaded = Graph(env=env_inst, filename="test_graph.json")

        # check that the content of the saved and loaded graph are the same
        assert graph_loaded.num_vertices == graph.num_vertices
        assert graph_loaded.start == graph.start
        assert graph_loaded.goal == graph.goal

        for key in graph.vertices.keys():
            original_vertex = graph.vertices[key]
            loaded_vertex = graph_loaded.vertices[key]
            assert abs(original_vertex.x - loaded_vertex.x) < 1e-10
            assert abs(original_vertex.y - loaded_vertex.y) < 1e-10

        for key in graph.connections.keys():
            std_connections = graph.connections[key]
            loaded_connections = graph_loaded.connections[key]
            assert len(std_connections) == len(loaded_connections)
            for i in range(len(std_connections)):
                assert std_connections[i] == (
                    loaded_connections[i][0],
                    loaded_connections[i][1],
                )

        for key in graph.heuristic.keys():
            assert graph.heuristic[key] == graph_loaded.heuristic[key]

        for key in graph.edges.keys():
            assert graph.edges[key].geometry.equals_exact(
                graph_loaded.edges[key].geometry, tolerance=1e-10
            )
            assert (
                graph.edges[key].always_available
                == graph_loaded.edges[key].always_available
            )
            assert graph.edges[key].cost == graph_loaded.edges[key].cost
            assert graph.edges[key].length == graph_loaded.edges[key].length
            assert len(graph.edges[key].availability) == len(
                graph_loaded.edges[key].availability
            )
            for i in range(len(graph.edges[key].availability)):
                assert (
                    graph.edges[key].availability[i]
                    == graph_loaded.edges[key].availability[i]
                )

        # remove saved graph
        os.remove("test_graph.json")
