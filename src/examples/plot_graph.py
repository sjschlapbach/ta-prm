from src.envs.environment import Environment
from src.envs.environment_instance import EnvironmentInstance
from src.examples.plot_environment import create_environment
from src.algorithms.graph import Graph

from pandas import Interval
import matplotlib.pyplot as plt


def plot_graph(plotting: bool = True):
    seed = 0

    # create an environment with all types of obstacles (static and continuous)
    env = Environment()

    # add random obstacles to environment
    x_range = (0, 100)
    y_range = (0, 100)
    env.add_random_obstacles(
        num_points=20,
        num_lines=20,
        num_polygons=20,
        min_x=x_range[0],
        max_x=x_range[1],
        min_y=y_range[0],
        max_y=y_range[1],
        min_interval=0,
        max_interval=100,
        min_radius=0,
        max_radius=5,
        random_recurrence=True,
        seed=seed,
    )

    # create an environment from it
    env_instance = EnvironmentInstance(
        environment=env,
        query_interval=Interval(20, 100, closed="both"),
        scenario_range_x=x_range,
        scenario_range_y=y_range,
    )

    # create graph
    graph = Graph(
        num_samples=200,
        neighbour_distance=20.0,
        max_connections=12,
        seed=seed,
        env=env_instance,
    )

    # connect start node to the graph
    start_coords = (2, 2)
    graph.connect_start(coords=start_coords)

    # connect goal node to the graph
    goal_coords = (98, 98)
    graph.connect_goal(coords=goal_coords)

    # initialize a figure
    fig = plt.figure(figsize=(8, 8))

    # run the query time from 20 to 100 in increments of 1 and plot the corresponding result and write the current time to the figure
    for query_time in range(0, 150 if plotting else 10):
        graph.plot(query_time=query_time, fig=fig)
        plt.title(f"Query Time: {query_time}")

        if plotting:
            plt.draw()
            plt.pause(0.2)
            plt.clf()


if __name__ == "__main__":
    plot_graph()
