from shapely import Point as ShapelyPoint, LineString as ShapelyLine
from pandas import Interval
import matplotlib.pyplot as plt
import numpy as np
import time

from src.envs.environment import Environment
from src.envs.environment_instance import EnvironmentInstance
from src.algorithms.graph import Graph
from src.obstacles.line import Line
from src.algorithms.ta_prm import TAPRM


def ta_prm_worst_case(
    samples: int = 50,
    max_connections: int = 10,
    plotting: bool = False,
    quiet: bool = False,
    temporal_precision: int = None,
):
    seed = 0
    np.random.seed(seed)

    x_range = (0, 100)
    y_range = (0, 100)
    scenario_interval = Interval(0, 250, closed="both")

    # create worst case obstacle, which is dynamic, but optimal way is around it
    sh_line = ShapelyLine([(10, 90), (90, 10)])
    obstacle = Line(
        geometry=sh_line,
        time_interval=Interval(0, 240, closed="both"),
    )

    # create environment with obstacles
    env = Environment(obstacles=[obstacle])

    # create environment instance
    env_inst = EnvironmentInstance(
        environment=env,
        query_interval=scenario_interval,
        scenario_range_x=x_range,
        scenario_range_y=y_range,
        quiet=quiet,
    )

    ## create path, override the sampling and place samples manually, connect nodes manually
    graph = Graph(
        env=env_inst,
        num_samples=samples,
        neighbour_distance=40.0,
        max_connections=max_connections,
        seed=seed,
        quiet=quiet,
    )

    # connect start and goal nodes
    start_coords = (2, 2)
    goal_coords = (98, 98)
    graph.connect_start(start_coords)
    graph.connect_goal(goal_coords, quiet=quiet)

    # run TA-PRM and check debugging output
    algo = TAPRM(graph=graph)
    start = time.time()
    if temporal_precision is None:
        success, path, max_open, expansions = algo.plan(start_time=0, quiet=quiet)
    else:
        success, path, max_open, expansions = algo.plan_temporal(
            start_time=0, quiet=quiet, temporal_precision=temporal_precision
        )

    runtime = time.time() - start

    assert success == True
    assert len(path) > 0
    assert max_open > 0

    # print("Found optimal solution with fastest path from start to goal being:")
    # print(path)

    if plotting:
        graph.plot(sol_path=path)
        plt.show()

    # compute the cost of the solution path
    path_cost = graph.path_cost(path)

    return runtime, max_open, path_cost


if __name__ == "__main__":
    ta_prm_worst_case(plotting=True)
