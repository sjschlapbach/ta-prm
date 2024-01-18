from shapely import Point as ShapelyPoint
from pandas import Interval
import matplotlib.pyplot as plt

from src.envs.environment import Environment
from src.envs.environment_instance import EnvironmentInstance
from src.algorithm.graph import Graph
from src.obstacles.point import Point
from src.algorithm.ta_prm import TAPRM


def ta_prm_demo():
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
        env=env_inst, num_samples=2, max_connections=10, neighbour_distance=200
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
    success, path = algo.plan(start_time=0, logging=True)

    assert success == True

    print("Found optimal solution with fastest path from start to goal being:")
    print(path)

    graph.plot(sol_path=path)
    plt.show()


if __name__ == "__main__":
    ta_prm_demo()
