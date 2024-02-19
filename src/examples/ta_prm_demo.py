from shapely import Point as ShapelyPoint
from pandas import Interval
import matplotlib.pyplot as plt

from src.envs.environment import Environment
from src.envs.environment_instance import EnvironmentInstance
from src.algorithms.graph import Graph
from src.obstacles.point import Point
from src.algorithms.ta_prm import TAPRM


def ta_prm_demo(plotting: bool = False):
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
    graph = Graph(env=env_inst, num_samples=2)

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
    graph.heuristic[0] = 15
    graph.heuristic[1] = 10
    graph.heuristic[2] = 20
    assert graph.heuristic == {0: 15, 1: 10, 2: 20, 3: 0}

    # override edge costs and flight time
    for edge_id in graph.edges:
        graph.edges[edge_id].length = 2
        graph.edges[edge_id].cost = 20

    # only edge between nodes 0 (A) and 3 (goal) should have cost 15
    for connection in graph.connections[0]:
        if connection[0] == 3:
            graph.edges[connection[1]].cost = 15

    # run TA-PRM and check debugging output
    algo = TAPRM(graph=graph)
    success, path, max_open, expansions = algo.plan(start_time=0, logging=True)
    # success, path, max_open, expansions = algo.plan_temporal(
    #     start_time=0, logging=True, temporal_precision=0
    # )

    assert success == True
    assert len(path) > 0
    assert max_open > 0

    print("Found optimal solution with fastest path from start to goal being:")
    print(path)

    if plotting:
        # Option 1) plot the static result
        graph.plot(sol_path=path)
        plt.show()

        # Option 2) create simulation video
        # graph.simulate(
        #     start_time=0,
        #     sol_path=path,
        #     step=0.05,
        #     fps=15,
        #     plotting=False,
        #     save_simulation=True,
        #     filename="ta_prm_demo",
        #     show_inactive=True,
        # )


if __name__ == "__main__":
    ta_prm_demo(plotting=True)
