from src.envs.environment import Environment
import matplotlib.pyplot as plt


def plot_random_environment(plotting: bool = True):
    # initialize an empty environment
    env = Environment()

    # specity min and max x- and y-coordinates
    min_x = 0
    max_x = 300
    min_y = 0
    max_y = 300

    # add random elements
    env.add_random_obstacles(
        num_points=10,
        num_lines=10,
        num_polygons=10,
        min_x=min_x,
        max_x=max_x,
        min_y=min_y,
        max_y=max_y,
        min_radius=0,
        max_radius=4,
        min_interval=0,
        max_interval=100,
        max_size=20,
        min_poly_points=3,
        max_poly_points=10,
        only_static=False,
        only_dynamic=False,
        random_recurrence=True,
    )

    # initialize a figure
    fig = plt.figure(figsize=(8, 8))

    # plot the environment over time
    for query_time in range(0, 160):
        env.plot(query_time=query_time, fig=fig)
        plt.title(f"Query Time: {query_time}")

        # set figure plotting limits
        plt.xlim([min_x, max_x])
        plt.ylim([min_y, max_y])

        if plotting:
            plt.draw()
            plt.pause(0.2)
            plt.clf()


if __name__ == "__main__":
    plot_random_environment()
