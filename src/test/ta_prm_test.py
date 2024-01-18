import pytest
from pandas import Interval

from src.algorithm.graph import Graph
from src.envs.environment import Environment
from src.envs.environment_instance import EnvironmentInstance
from src.algorithm.ta_prm import TAPRM


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
        # TODO - create demo graph with known outcome to test algorithm
        pass

    # def test_plan_real(self):
    #     # create random graph with start and goal node specified
    #     graph = self.__create_random_graph()
    #     graph = self.__add_start_goal(graph)

    #     # initialize algorithm and plan path from start to goal
    #     ta_prm = TAPRM(graph=graph)
    #     success, path = ta_prm.plan(start_time=0)

    #     # TODO - add asserts, which make sense
    #     assert success == True

    #     # TODO - add test cases of known scenarios / graphs with known outcomes
