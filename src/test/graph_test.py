from pandas import Interval

from src.algorithm.graph import Graph
from src.envs.environment import Environment
from src.envs.environment_instance import EnvironmentInstance


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
        default_max_neighbours = 10
        default_max_distance = 10.0

        # create graph
        graph = Graph(
            num_samples=default_samples,
            max_neighbours=default_max_neighbours,
            neighbour_distance=default_max_distance,
            env=env_inst,
        )

        # check if graph vertices are within specified range and collision-free
        assert len(graph.vertices) == default_samples
        for vertex in graph.vertices.values():
            assert vertex.x >= x_range[0] and vertex.x <= x_range[1]
            assert vertex.y >= y_range[0] and vertex.y <= y_range[1]
            assert env_inst.static_collision_free(vertex)
