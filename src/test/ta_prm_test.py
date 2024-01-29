import pytest
from pandas import Interval
from shapely.geometry import Point as ShapelyPoint

from src.algorithm.graph import Graph
from src.envs.environment import Environment
from src.envs.environment_instance import EnvironmentInstance
from src.algorithm.ta_prm import TAPRM
from src.obstacles.point import Point


class TestTAPRM:
    def __create_random_graph(self):
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
        default_max_distance = 10.0

        # create graph
        graph = Graph(
            num_samples=default_samples,
            neighbour_distance=default_max_distance,
            max_connections=10,
            seed=0,
            env=env_inst,
        )

        return graph

    def __add_start_goal(self, graph: Graph):
        x_range = (0, 300)
        y_range = (0, 300)

        # connect start and goal node
        start_coords = (x_range[0] + 2, y_range[0] + 2)
        goal_coords = (x_range[1] - 2, y_range[1] - 2)
        graph.connect_start(coords=start_coords)
        graph.connect_goal(coords=goal_coords)

        return graph

    def test_init(self):
        # create random graph
        graph = self.__create_random_graph()

        # initialize algorithm with manually connected start and goal
        start_coords = (2, 2)
        goal_coords = (298, 298)
        ta_prm = TAPRM(graph=graph, start=start_coords, goal=goal_coords)

        # initialize algorithm with completed graph
        graph2 = self.__create_random_graph()
        graph2 = self.__add_start_goal(graph2)
        ta_prm = TAPRM(graph=graph2)

    def test_plan(self):
        x_range = (-5, 105)
        y_range = (-5, 105)
        query_interval = Interval(0, 20, closed="both")

        # create obstacles, blocking the correct lines
        obstacle1 = Point(
            geometry=ShapelyPoint(10, 10),
            time_interval=Interval(0, 2, closed="left"),
            radius=2,
        )
        obstacle2 = Point(
            geometry=ShapelyPoint(100, 50),
            time_interval=Interval(0, 3, closed="left"),
            radius=5,
        )
        obstacle3 = Point(
            geometry=ShapelyPoint(50, 100),
            time_interval=Interval(0, 3, closed="left"),
            radius=5,
        )

        # create environment with obstacles
        env = Environment(obstacles=[obstacle1, obstacle2, obstacle3])

        # create environment instance
        env_inst = EnvironmentInstance(
            environment=env,
            query_interval=query_interval,
            scenario_range_x=x_range,
            scenario_range_y=y_range,
        )

        ## create path, override the sampling and place samples manually, connect nodes manually
        graph = Graph(
            env=env_inst,
            num_samples=2,
            max_connections=10,
            neighbour_distance=200,
            seed=0,
        )

        # overwrite the graphs vertices and add start and goal node
        graph.vertices[0] = ShapelyPoint(100, 0)
        graph.vertices[1] = ShapelyPoint(0, 100)
        graph.connect_vertices()

        # check that the graph is correctly initialized
        assert len(graph.vertices) == 2
        assert len(graph.connections) == 2
        assert len(graph.edges) == 1
        assert graph.connections[0] == [(1, 0)]
        assert graph.connections[1] == [(0, 0)]

        # connect start and goal nodes
        start_coords = (0, 0)
        goal_coords = (100, 100)
        graph.connect_start(start_coords)
        graph.connect_goal(goal_coords)

        # check that start and goal node are correctly connected to the graph
        assert len(graph.vertices) == 4
        assert len(graph.edges) == 6
        for value in graph.connections.values():
            assert len(value) == 3

        # override generated heuristic
        graph.heuristic[0] = 10
        graph.heuristic[1] = 15
        graph.heuristic[2] = 20
        assert graph.heuristic == {0: 10, 1: 15, 2: 20, 3: 0}

        # override edge costs and flight time lengths to 2
        for edge_id in graph.edges:
            graph.edges[edge_id].cost = 2
            graph.edges[edge_id].length = 2

        # run TA-PRM and check debugging output
        algo = TAPRM(graph=graph)
        success, path, max_open = algo.plan(start_time=0)
        assert success == True
        assert path == [2, 1, 0, 3]
        assert max_open > 0

        # run TA-PRM with temporal pruning and check debugging output
        success2, path2, max_open2 = algo.plan_temporal(start_time=0, temporal_res=0)
        assert success2 == True
        assert path2 == [2, 1, 0, 3]
        assert max_open2 > 0

        success3, path3, max_open3 = algo.plan_temporal(start_time=0, temporal_res=10)
        assert success3 == True
        assert path3 == [2, 1, 0, 3]
        assert max_open3 > 0
