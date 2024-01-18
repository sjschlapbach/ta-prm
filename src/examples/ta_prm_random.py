import time
from shapely import Point as ShapelyPoint
from pandas import Interval
import matplotlib.pyplot as plt

from src.envs.environment import Environment
from src.envs.environment_instance import EnvironmentInstance
from src.algorithm.graph import Graph
from src.obstacles.point import Point
from src.algorithm.ta_prm import TAPRM


def ta_prm_random(
    interval_end: int = 500, scenario_end: int = 500, plotting: bool = False
):
    # optional parameters
    seed = 0
    max_size = 30

    # time interval
    interval_start = 0
    interval_end = interval_end
    x_range = (0, 300)
    y_range = (0, 300)

    # create environment with random obstacles
    env = Environment()
    env.add_random_obstacles(
        num_points=10,
        num_lines=10,
        num_polygons=10,
        min_x=x_range[0],
        max_x=x_range[1],
        min_y=y_range[0],
        max_y=y_range[1],
        min_interval=interval_start,
        max_interval=interval_end,
        only_static=True,
        seed=seed,
        max_size=max_size,
    )
    env.add_random_obstacles(
        num_points=20,
        num_lines=20,
        num_polygons=20,
        min_x=x_range[0],
        max_x=x_range[1],
        min_y=y_range[0],
        max_y=y_range[1],
        min_interval=interval_start,
        max_interval=interval_end,
        seed=seed,
        max_size=max_size,
        only_dynamic=True,
        random_recurrence=True,
        max_radius=20,
    )

    # create environment instance
    env_inst = EnvironmentInstance(
        environment=env,
        query_interval=Interval(interval_start, scenario_end, closed="both"),
        scenario_range_x=x_range,
        scenario_range_y=y_range,
    )

    # default parameters
    default_samples = 100
    default_max_distance = 100.0

    # create graph
    graph = Graph(
        num_samples=default_samples,
        neighbour_distance=default_max_distance,
        max_connections=10,
        env=env_inst,
    )

    # connect start and goal node
    start_coords = (x_range[0] + 2, y_range[0] + 2)
    goal_coords = (x_range[1] - 2, y_range[1] - 2)
    graph.connect_start(coords=start_coords)
    graph.connect_goal(coords=goal_coords)

    # initialize algorithm and plan path from start to goal
    ta_prm = TAPRM(graph=graph)

    start = time.time()
    success, path, max_length_open = ta_prm.plan(start_time=0)
    runtime = time.time() - start

    assert success == True
    assert len(path) > 0

    # plot the path
    if plotting:
        graph.plot(sol_path=path)
        plt.show()

    return runtime, max_length_open


if __name__ == "__main__":
    ta_prm_random(plotting=True)
