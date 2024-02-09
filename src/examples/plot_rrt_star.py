from src.envs.environment import Environment
from src.envs.environment_instance import EnvironmentInstance
from src.algorithms.rrt import RRT

from pandas import Interval
import matplotlib.pyplot as plt


def plot_rrt(plotting: bool = True):
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

    # create tree
    obs_free = 0.95  # free-space percentage should be over-approximation for asymptotic optimality
    rrt = RRT(
        start=(2, 2),
        goal=(98, 98),
        env=env_instance,
        num_samples=1000,
        rewiring=True,
        obs_free=0.5,
        seed=seed,
    )

    # compute solution path
    sol_path = rrt.rrt_find_path()

    # initialize a figure
    fig = plt.figure(figsize=(8, 8))

    # plot the tree
    rrt.plot(fig=fig, sol_path=sol_path)

    if plotting:
        plt.show()


if __name__ == "__main__":
    plot_rrt()
