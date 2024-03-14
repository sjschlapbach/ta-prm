import time
import matplotlib.pyplot as plt
from pandas import Interval
from typing import List, Tuple
from shapely.geometry import Polygon as ShapelyPolygon, Point as ShapelyPoint

from src.envs.environment import Environment
from src.envs.environment_instance import EnvironmentInstance
from src.obstacles.polygon import Polygon
from src.algorithms.graph import Graph
from src.algorithms.ta_prm import TAPRM
from src.algorithms.replanning_rrt import ReplanningRRT


def get_timed_path(
    graph: Graph, sol_path: List[int], start_time: float
) -> List[Tuple[ShapelyPoint, float]]:

    start_node = graph.vertices[sol_path[0]]
    timed_path: List[Tuple[ShapelyPoint, float]] = [(start_node, start_time)]

    for idx in range(len(sol_path) - 1):
        curr_vertex = sol_path[idx]
        curr_time = timed_path[-1][1]
        next_vertex = sol_path[idx + 1]
        connections = graph.connections[curr_vertex]
        for connection in connections:
            if connection[0] == sol_path[idx + 1]:
                edge_idx = connection[1]
                break

        edge = graph.edges[edge_idx]
        edge_time = edge.length
        next_node = graph.vertices[next_vertex]
        timed_path.append((next_node, curr_time + edge_time))

    goal_time = timed_path[-1][1]

    return timed_path


def get_timed_path_rrt(
    sol_path: List[ShapelyPoint], start_time: float
) -> List[Tuple[ShapelyPoint, float]]:
    timed_path: List[Tuple(ShapelyPoint, float)] = [(sol_path[0], start_time)]

    for idx in range(len(sol_path) - 1):
        curr_vertex = sol_path[idx]
        curr_time = timed_path[-1][1]
        next_vertex = sol_path[idx + 1]
        distance = curr_vertex.distance(next_vertex)

        timed_path.append((next_vertex, curr_time + distance))

    return timed_path


def get_current_pos_timed_path(
    time: float, timed_path: List[Tuple[ShapelyPoint, float]], graph: Graph
) -> Tuple[float, float]:

    # find the index and time at previous and next vertex along path
    prev_vertex = None
    next_vertex = None
    prev_time = None
    next_time = None

    for idx, (vertex, vertex_time) in enumerate(timed_path):
        if time >= vertex_time:
            prev_vertex = vertex
            prev_time = vertex_time
            next_vertex = timed_path[idx + 1][0]
            next_time = timed_path[idx + 1][1]
        else:
            break

    if next_vertex is None:
        print("Simulation failed")
        return None

    # linearly interpolate between vertices to find current position
    alpha = (time - prev_time) / (next_time - prev_time)
    curr_pos_x = prev_vertex.x + alpha * (next_vertex.x - prev_vertex.x)
    curr_pos_y = prev_vertex.y + alpha * (next_vertex.y - prev_vertex.y)

    return (curr_pos_x, curr_pos_y)


def plot_taprm_path(sol_path: List[int], graph: Graph, color: str, label: str = None):
    for idx in range(len(sol_path) - 1):
        connections = graph.connections[sol_path[idx]]
        for connection in connections:
            if connection[0] == sol_path[idx + 1]:
                edge_idx = connection[1]
                break

        if idx == 0:
            plt.plot(
                *graph.edges[edge_idx].geometry.xy,
                color=color,
                linewidth=2,
                label=label,
            )
        else:
            plt.plot(
                *graph.edges[edge_idx].geometry.xy,
                color=color,
                linewidth=2,
            )


def plot_rrt_path(sol_path: List[ShapelyPoint], color: str, label: str = None):
    plt.plot(
        [point.x for point in sol_path],
        [point.y for point in sol_path],
        color=color,
        linewidth=2,
        label=label,
    )


if __name__ == "__main__":
    print("Starting scenario illustration...")

    # ! Basic seed for reproducibility
    seed = 40

    # ! Plotting settings
    plotting_start = 0
    plotting_end = 122
    plotting_step = 1

    # ? Setup specifications
    x_range = (0, 100)
    y_range = (0, 50)
    scenario_start = 0
    scenario_end = 200
    start_coords = (2, 2)
    start_time = 0
    goal_coords = (98, 48)
    min_radius = 2
    max_radius = 8
    stepsize = 0.1

    samples = 100
    pruning = 0
    obstacles = []

    # create static obstacles
    poly1 = ShapelyPolygon([(50, 10), (60, 10), (60, 40), (50, 40)])
    obs1 = Polygon(geometry=poly1)
    obstacles = obstacles + [obs1]

    # create dynamic obstacles
    poly2 = ShapelyPolygon([(15, 0), (30, 0), (30, 25), (15, 25)])
    obs2 = Polygon(geometry=poly2, time_interval=Interval(2, 10, closed="both"))
    obstacles = obstacles + [obs2]

    poly3 = ShapelyPolygon([(75, 25), (95, 25), (95, 50), (75, 50)])
    obs3 = Polygon(geometry=poly3, time_interval=Interval(40, 200, closed="both"))
    obstacles = obstacles + [obs3]

    # create random environment
    env_obj = Environment()
    env_obj.add_obstacles(obstacles)

    # create environment instance
    env = EnvironmentInstance(
        environment=env_obj,
        query_interval=Interval(
            scenario_start,
            scenario_end,
            closed="both",
        ),
        scenario_range_x=x_range,
        scenario_range_y=y_range,
        quiet=True,
    )

    ## Run TA-PRM algorithms

    # prepare the TA-PRM graph
    graph = Graph(
        num_samples=samples,
        env=env,
        seed=seed,
        quiet=True,
    )

    # connect start and goal node
    graph.connect_start(coords=start_coords)
    graph.connect_goal(coords=goal_coords, quiet=True)
    ta_prm = TAPRM(graph=graph)

    # plan a path with TA-PRM*
    start = time.time()
    success_taprm_star, path_taprm_star, _, _ = ta_prm.plan(
        start_time=start_time, quiet=True
    )
    runtime_taprm_star = time.time() - start
    print("Runtime TA-PRM*: ", runtime_taprm_star)
    print("Path:", path_taprm_star)
    print("Cost:", graph.path_cost(path_taprm_star))

    # plan a path with TA-PRM (including temporal pruning)
    start = time.time()
    success_taprm, path_taprm, _, _ = ta_prm.plan_temporal(
        start_time=start_time, temporal_precision=pruning, quiet=True
    )
    runtime_taprm = time.time() - start
    print("Runtime TA-PRM (pruning): ", runtime_taprm)
    print("Path:", path_taprm)
    print("Cost:", graph.path_cost(path_taprm))

    ## run RRT and RRT* with dynamic obstacle replanning
    replanner = ReplanningRRT(env=env, seed=seed)

    start = time.time()
    path_rrt, runs_rrt, prev_paths_rrt = replanner.run(
        samples=samples,
        stepsize=stepsize,
        start=start_coords,
        goal=goal_coords,
        query_time=start_time,
        rewiring=False,
        prev_path=[ShapelyPoint(*start_coords)],
        dynamic_obstacles=True,
        quiet=True,
    )
    runtime_rrt = time.time() - start
    print("Runtime RRT: ", runtime_rrt)
    print("Path:", path_rrt)
    print("Cost:", replanner.get_path_cost(sol_path=path_rrt))

    start = time.time()
    path_rrt_star, runs_rrt_star, prev_paths_rrt_star = replanner.run(
        samples=samples,
        stepsize=stepsize,
        start=start_coords,
        goal=goal_coords,
        query_time=start_time,
        rewiring=True,
        prev_path=[ShapelyPoint(*start_coords)],
        dynamic_obstacles=True,
        quiet=True,
    )
    runtime_rrt_star = time.time() - start
    print("Runtime RRT*: ", runtime_rrt_star)
    print("Path:", path_rrt_star)
    print("Cost:", replanner.get_path_cost(sol_path=path_rrt_star))

    ## Plot all algorithm results in the same figure
    # pre-compute the timed paths for TA-PRM* and TA-PRM
    timed_path_taprm_star = get_timed_path(
        graph=graph, sol_path=path_taprm_star, start_time=start_time
    )
    timed_path_taprm = get_timed_path(
        graph=graph, sol_path=path_taprm, start_time=start_time
    )
    timed_path_rrt = get_timed_path_rrt(sol_path=path_rrt, start_time=start_time)
    timed_path_rrt_star = get_timed_path_rrt(
        sol_path=path_rrt_star, start_time=start_time
    )

    # ensure that both RRT and RRT* used one replanning for plotting to work as expected
    assert len(prev_paths_rrt) == 1
    assert len(prev_paths_rrt_star) == 1

    # iterate from plotting_start to plotting_end with plotting_step
    for plotting_time in range(plotting_start, plotting_end, plotting_step):
        fig = plt.figure(figsize=(6, 3))
        env.plot(fig=fig, query_time=plotting_time, show_inactive=True)

        # plot TA-PRM* path and current position
        plot_taprm_path(
            sol_path=path_taprm_star,
            graph=graph,
            color="orange",
            label="TA-PRM*",
        )
        curr_pos_taprm_star = get_current_pos_timed_path(
            time=plotting_time,
            timed_path=timed_path_taprm_star,
            graph=graph,
        )

        # plot TA-PRM path and current position
        plot_taprm_path(sol_path=path_taprm, graph=graph, color="blue", label="TA-PRM")
        curr_pos_taprm = get_current_pos_timed_path(
            time=plotting_time, timed_path=timed_path_taprm, graph=graph
        )

        # plot RRT path
        if plotting_time < prev_paths_rrt[0][1]:
            plot_rrt_path(sol_path=prev_paths_rrt[0][0], color="green", label="RRT")
        else:
            plot_rrt_path(sol_path=path_rrt, color="green", label="RRT")

        curr_pos_rrt = get_current_pos_timed_path(
            time=plotting_time, timed_path=timed_path_rrt, graph=graph
        )

        # plot RRT* path
        if plotting_time < prev_paths_rrt_star[0][1]:
            plot_rrt_path(
                sol_path=prev_paths_rrt_star[0][0], color="black", label="RRT*"
            )
        else:
            plot_rrt_path(
                sol_path=path_rrt_star,
                color="black",
                label="RRT*",
            )

        curr_pos_rrt_star = get_current_pos_timed_path(
            time=plotting_time,
            timed_path=timed_path_rrt_star,
            graph=graph,
        )

        # plot current positions on all algorithm paths
        plt.plot(
            curr_pos_taprm_star[0],
            curr_pos_taprm_star[1],
            color="red",
            marker="o",
            markersize=6,
        )
        plt.plot(
            curr_pos_taprm[0], curr_pos_taprm[1], color="red", marker="o", markersize=6
        )
        plt.plot(
            curr_pos_rrt[0], curr_pos_rrt[1], color="red", marker="o", markersize=6
        )
        plt.plot(
            curr_pos_rrt_star[0],
            curr_pos_rrt_star[1],
            color="red",
            marker="o",
            markersize=6,
        )

        # add time to figure
        plt.text(2, 45, "$t = {}$".format(plotting_time), fontsize=12)

        # save figure without legend
        fig.tight_layout()
        plt.savefig(
            f"results/illustrations/demo_illustration_{plotting_time}_no_legend.svg",
            format="svg",
        )

        # add legend to the plot
        plt.legend(
            bbox_to_anchor=(0, 1.02, 1, 0.2),
            loc="lower left",
            mode="expand",
            borderaxespad=0,
            ncol=4,
            fontsize=12,
        )
        fig.tight_layout()
        plt.savefig(
            f"results/illustrations/demo_illustration_{plotting_time}_legend.svg",
            format="svg",
        )

        # add version with legend but no x axis labels
        plt.xticks([])
        fig.tight_layout()
        plt.savefig(
            f"results/illustrations/demo_illustration_{plotting_time}_legend_no_xlabels.svg",
            format="svg",
        )

        # hide the legend and scale on the x axis
        plt.legend().remove()
        fig.tight_layout()
        plt.savefig(
            f"results/illustrations/demo_illustration_{plotting_time}_no_legend_no_xlabels.svg",
            format="svg",
        )

        # close the plot
        plt.close()
