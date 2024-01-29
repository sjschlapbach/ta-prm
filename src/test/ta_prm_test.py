import pytest
from pandas import Interval
from shapely.geometry import Point as ShapelyPoint, LineString as ShapelyLine

from src.algorithm.graph import Graph
from src.envs.environment import Environment
from src.envs.environment_instance import EnvironmentInstance
from src.algorithm.ta_prm import TAPRM
from src.obstacles.point import Point
from src.algorithm.timed_edge import TimedEdge


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
        success, path, max_open, expansions = algo.plan(start_time=0)
        assert success == True
        assert path == [2, 1, 0, 3]
        assert max_open > 0

        # run TA-PRM with temporal pruning and check debugging output
        success2, path2, max_open2, expansions2 = algo.plan_temporal(
            start_time=0, temporal_precision=0
        )
        assert success2 == True
        assert path2 == [2, 1, 0, 3]
        assert max_open2 > 0

        success3, path3, max_open3, expansions3 = algo.plan_temporal(
            start_time=0, temporal_precision=10
        )
        assert success3 == True
        assert path3 == [2, 1, 0, 3]
        assert max_open3 > 0

    def test_limited_precision1(self):
        # Test case 1: arriving at the same node at the same time (lower cost)
        # -> update node in open list
        env = Environment(obstacles=[])
        env_inst = EnvironmentInstance(
            environment=env,
            query_interval=Interval(0, 200, closed="both"),
            scenario_range_x=(0, 100),
            scenario_range_y=(0, 100),
        )
        graph = Graph(env=env_inst, num_samples=0, quiet=True)

        start = ShapelyPoint(10, 10)
        pt0 = ShapelyPoint(15, 5)
        pt1 = ShapelyPoint(20, 10)
        goal = ShapelyPoint(30, 10)

        graph.vertices = {0: start, 1: pt0, 2: pt1, 3: goal}
        graph.start = 0
        graph.goal = 3

        # edge from start to pt0
        tmp_ln = ShapelyLine([(start.x, start.y), (pt0.x, pt0.y)])
        edge_start_pt0 = TimedEdge(
            geometry=tmp_ln, availability=[], always_available=True, cost=5
        )
        edge_start_pt0.length = 5

        # edge from start to pt1
        tmp_ln = ShapelyLine([(start.x, start.y), (pt1.x, pt1.y)])
        edge_start_pt1 = TimedEdge(
            geometry=tmp_ln, availability=[], always_available=True, cost=20
        )
        edge_start_pt1.length = 10

        # edge from pt0 to pt1
        tmp_ln = ShapelyLine([(pt0.x, pt0.y), (pt1.x, pt1.y)])
        edge_pt0_pt1 = TimedEdge(
            geometry=tmp_ln, availability=[], always_available=True, cost=5
        )
        edge_pt0_pt1.length = 5

        # edge from pt1 to goal
        tmp_ln = ShapelyLine([(pt1.x, pt1.y), (goal.x, goal.y)])
        edge_pt1_goal = TimedEdge(
            geometry=tmp_ln, availability=[], always_available=True, cost=20
        )
        edge_pt1_goal.length = 5

        # add the edges to the graph and index connections
        graph.edges = {
            0: edge_start_pt0,
            1: edge_start_pt1,
            2: edge_pt0_pt1,
            3: edge_pt1_goal,
        }
        graph.connections = {
            0: [(1, 0), (2, 1)],
            1: [(2, 2)],
            2: [(3, 3)],
            3: [],
        }
        graph.heuristic = {0: 12, 1: 11, 2: 10, 3: 0}
        algo = TAPRM(graph=graph)

        # applying the standard algorithm, the same node (same time) should be expanded at two different costs
        success, path, max_open, expansions = algo.plan(start_time=0)

        assert success == True
        assert path == [0, 1, 2, 3]
        assert max_open == 2
        assert expansions == 5

        success, path, max_open, expansions = algo.plan_temporal(
            start_time=0, temporal_precision=0
        )

        assert success == True
        assert path == [0, 1, 2, 3]
        assert max_open == 2
        assert expansions == 4

    def test_limited_precision2(self):
        # Test case 2: arriving at the same node at the same time (higher cost)
        # -> skip node, do not add to OL
        env = Environment(obstacles=[])
        env_inst = EnvironmentInstance(
            environment=env,
            query_interval=Interval(0, 200, closed="both"),
            scenario_range_x=(0, 100),
            scenario_range_y=(0, 100),
        )
        graph = Graph(env=env_inst, num_samples=0, quiet=True)

        start = ShapelyPoint(10, 10)
        pt0 = ShapelyPoint(15, 5)
        pt1 = ShapelyPoint(20, 10)
        goal = ShapelyPoint(30, 10)

        graph.vertices = {0: start, 1: pt0, 2: pt1, 3: goal}
        graph.start = 0
        graph.goal = 3

        # edge from start to pt0
        tmp_ln = ShapelyLine([(start.x, start.y), (pt0.x, pt0.y)])
        edge_start_pt0 = TimedEdge(
            geometry=tmp_ln, availability=[], always_available=True, cost=10
        )
        edge_start_pt0.length = 5

        # edge from start to pt1
        tmp_ln = ShapelyLine([(start.x, start.y), (pt1.x, pt1.y)])
        edge_start_pt1 = TimedEdge(
            geometry=tmp_ln, availability=[], always_available=True, cost=15
        )
        edge_start_pt1.length = 10

        # edge from pt0 to pt1
        tmp_ln = ShapelyLine([(pt0.x, pt0.y), (pt1.x, pt1.y)])
        edge_pt0_pt1 = TimedEdge(
            geometry=tmp_ln, availability=[], always_available=True, cost=10
        )
        edge_pt0_pt1.length = 5

        # edge from pt1 to goal
        tmp_ln = ShapelyLine([(pt1.x, pt1.y), (goal.x, goal.y)])
        edge_pt1_goal = TimedEdge(
            geometry=tmp_ln, availability=[], always_available=True, cost=20
        )
        edge_pt1_goal.length = 5

        # add the edges to the graph and index connections
        graph.edges = {
            0: edge_start_pt0,
            1: edge_start_pt1,
            2: edge_pt0_pt1,
            3: edge_pt1_goal,
        }
        graph.connections = {
            0: [(1, 0), (2, 1)],
            1: [(2, 2)],
            2: [(3, 3)],
            3: [],
        }
        graph.heuristic = {0: 12, 1: 11, 2: 10, 3: 0}
        algo = TAPRM(graph=graph)

        # applying the standard algorithm, the same node (same time) should be expanded at two different costs
        success, path, max_open, expansions = algo.plan(start_time=0)

        assert success == True
        assert path == [0, 2, 3]
        assert max_open == 2
        assert expansions == 5

        success, path, max_open, expansions = algo.plan_temporal(
            start_time=0, temporal_precision=0
        )

        assert success == True
        assert path == [0, 2, 3]
        assert max_open == 2
        assert expansions == 4

    def test_limited_precision3(self):
        # Test case 3: arriving at the same node within 0.4 seconds with (lower cost)
        # -> node should be updated if precision casts the instance to the same rounded time interval
        env = Environment(obstacles=[])
        env_inst = EnvironmentInstance(
            environment=env,
            query_interval=Interval(0, 200, closed="both"),
            scenario_range_x=(0, 100),
            scenario_range_y=(0, 100),
        )
        graph = Graph(env=env_inst, num_samples=0, quiet=True)

        start = ShapelyPoint(10, 10)
        pt0 = ShapelyPoint(15, 5)
        pt1 = ShapelyPoint(20, 10)
        goal = ShapelyPoint(30, 10)

        graph.vertices = {0: start, 1: pt0, 2: pt1, 3: goal}
        graph.start = 0
        graph.goal = 3

        # edge from start to pt0
        tmp_ln = ShapelyLine([(start.x, start.y), (pt0.x, pt0.y)])
        edge_start_pt0 = TimedEdge(
            geometry=tmp_ln, availability=[], always_available=True, cost=5
        )
        edge_start_pt0.length = 5

        # edge from start to pt1
        tmp_ln = ShapelyLine([(start.x, start.y), (pt1.x, pt1.y)])
        edge_start_pt1 = TimedEdge(
            geometry=tmp_ln, availability=[], always_available=True, cost=15
        )
        edge_start_pt1.length = 10

        # edge from pt0 to pt1
        tmp_ln = ShapelyLine([(pt0.x, pt0.y), (pt1.x, pt1.y)])
        edge_pt0_pt1 = TimedEdge(
            geometry=tmp_ln, availability=[], always_available=True, cost=5
        )
        edge_pt0_pt1.length = 5.4

        # edge from pt1 to goal
        tmp_ln = ShapelyLine([(pt1.x, pt1.y), (goal.x, goal.y)])
        edge_pt1_goal = TimedEdge(
            geometry=tmp_ln, availability=[], always_available=True, cost=20
        )
        edge_pt1_goal.length = 5

        # add the edges to the graph and index connections
        graph.edges = {
            0: edge_start_pt0,
            1: edge_start_pt1,
            2: edge_pt0_pt1,
            3: edge_pt1_goal,
        }
        graph.connections = {
            0: [(1, 0), (2, 1)],
            1: [(2, 2)],
            2: [(3, 3)],
            3: [],
        }
        graph.heuristic = {0: 12, 1: 11, 2: 10, 3: 0}
        algo = TAPRM(graph=graph)

        # applying the standard algorithm, the same node (same time) should be expanded at two different costs
        success, path, max_open, expansions = algo.plan(start_time=0)

        assert success == True
        assert path == [0, 1, 2, 3]
        assert max_open == 2
        assert expansions == 5

        # with precision 0, the node should be updated
        success, path, max_open, expansions = algo.plan_temporal(
            start_time=0, temporal_precision=0
        )

        assert success == True
        assert path == [0, 1, 2, 3]
        assert max_open == 2
        assert expansions == 4

        # with precision >= 1, two different instances will be created
        success, path, max_open, expansions = algo.plan_temporal(
            start_time=0, temporal_precision=1
        )

        assert success == True
        assert path == [0, 1, 2, 3]
        assert max_open == 2
        assert expansions == 5

        # with precision >= 1, two different instances will be created
        success, path, max_open, expansions = algo.plan_temporal(
            start_time=0, temporal_precision=2
        )

        assert success == True
        assert path == [0, 1, 2, 3]
        assert max_open == 2
        assert expansions == 5

    def test_limited_precision4(self):
        # Test case 4: arriving at the same node within 0.4 seconds with (higher cost)
        # -> node should be skipped if precision casts the instance to the same rounded time interval
        env = Environment(obstacles=[])
        env_inst = EnvironmentInstance(
            environment=env,
            query_interval=Interval(0, 200, closed="both"),
            scenario_range_x=(0, 100),
            scenario_range_y=(0, 100),
        )
        graph = Graph(env=env_inst, num_samples=0, quiet=True)

        start = ShapelyPoint(10, 10)
        pt0 = ShapelyPoint(15, 5)
        pt1 = ShapelyPoint(20, 10)
        goal = ShapelyPoint(30, 10)

        graph.vertices = {0: start, 1: pt0, 2: pt1, 3: goal}
        graph.start = 0
        graph.goal = 3

        # edge from start to pt0
        tmp_ln = ShapelyLine([(start.x, start.y), (pt0.x, pt0.y)])
        edge_start_pt0 = TimedEdge(
            geometry=tmp_ln, availability=[], always_available=True, cost=10
        )
        edge_start_pt0.length = 5

        # edge from start to pt1
        tmp_ln = ShapelyLine([(start.x, start.y), (pt1.x, pt1.y)])
        edge_start_pt1 = TimedEdge(
            geometry=tmp_ln, availability=[], always_available=True, cost=15
        )
        edge_start_pt1.length = 10

        # edge from pt0 to pt1
        tmp_ln = ShapelyLine([(pt0.x, pt0.y), (pt1.x, pt1.y)])
        edge_pt0_pt1 = TimedEdge(
            geometry=tmp_ln, availability=[], always_available=True, cost=10
        )
        edge_pt0_pt1.length = 5.4

        # edge from pt1 to goal
        tmp_ln = ShapelyLine([(pt1.x, pt1.y), (goal.x, goal.y)])
        edge_pt1_goal = TimedEdge(
            geometry=tmp_ln, availability=[], always_available=True, cost=20
        )
        edge_pt1_goal.length = 5

        # add the edges to the graph and index connections
        graph.edges = {
            0: edge_start_pt0,
            1: edge_start_pt1,
            2: edge_pt0_pt1,
            3: edge_pt1_goal,
        }
        graph.connections = {
            0: [(1, 0), (2, 1)],
            1: [(2, 2)],
            2: [(3, 3)],
            3: [],
        }
        graph.heuristic = {0: 12, 1: 11, 2: 10, 3: 0}
        algo = TAPRM(graph=graph)

        # applying the standard algorithm, the same node (same time) should be expanded at two different costs
        success, path, max_open, expansions = algo.plan(start_time=0)

        assert success == True
        assert path == [0, 2, 3]
        assert max_open == 2
        assert expansions == 5

        # with precision 0, the node should be updated
        success, path, max_open, expansions = algo.plan_temporal(
            start_time=0, temporal_precision=0
        )

        assert success == True
        assert path == [0, 2, 3]
        assert max_open == 2
        assert expansions == 4

        # with precision >= 1, two different instances will be created
        success, path, max_open, expansions = algo.plan_temporal(
            start_time=0, temporal_precision=1
        )

        assert success == True
        assert path == [0, 2, 3]
        assert max_open == 2
        assert expansions == 5

        # with precision >= 1, two different instances will be created
        success, path, max_open, expansions = algo.plan_temporal(
            start_time=0, temporal_precision=2
        )

        assert success == True
        assert path == [0, 2, 3]
        assert max_open == 2
        assert expansions == 5

    def test_limited_precision5(self):
        # TODO - test case 5
        pass

    def test_limited_precision6(self):
        # TODO - test case 6
        pass

    def test_negative_precision(self):
        # TODO - test case like 3 and/or 4 but with negative precision required - e.g. summarize 5 and 7 to one
        pass
