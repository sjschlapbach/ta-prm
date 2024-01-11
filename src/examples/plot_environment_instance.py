from src.envs.environment_instance import EnvironmentInstance
from src.examples.plot_environment import create_environment

from pandas import Interval
import matplotlib.pyplot as plt


def plot_environment_instance(plotting: bool = True):
    # create an environment with all types of obstacles (static and continuous)
    env = create_environment()

    # create an environment from it
    env_instance = EnvironmentInstance(
        environment=env,
        query_interval=Interval(20, 100),
        scenario_range_x=(-1, 15),
        scenario_range_y=(-1, 15),
    )

    # initialize a figure
    fig = plt.figure(figsize=(8, 8))

    # run the query time from 1 to 30 in increments of 1 and plot the corresponding result and write the current time to the figure
    for query_time in range(20, 100):
        env_instance.plot(query_time=query_time, fig=fig)
        plt.title(f"Query Time: {query_time}")

        if plotting:
            plt.draw()
            plt.pause(0.2)
            plt.clf()


if __name__ == "__main__":
    plot_environment_instance()
