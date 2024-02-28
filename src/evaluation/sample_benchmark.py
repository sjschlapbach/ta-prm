import time
import random
import statistics
from pandas import Interval
from shapely.geometry import Point as ShapelyPoint

from src.algorithms.graph import Graph
from src.algorithms.ta_prm import TAPRM
from src.algorithms.rrt import RRT
from src.algorithms.replanning_rrt import ReplanningRRT
from src.envs.environment import Environment
from src.envs.environment_instance import EnvironmentInstance


def create_environment(specifications, seed, obstacles):
    # This function creates an environment instance with random obstacles
    # according to the specifications (both static and dynamic).

    # create environment with random obstacles (25% static obstacles, 75% dynamic obstacles)
    env = Environment()
    static_obstacles = obstacles // 4
    points = obstacles // 4
    lines = obstacles // 4
    polygons = obstacles - points - lines - static_obstacles

    env.add_random_obstacles(
        num_points=0,
        num_lines=0,
        num_polygons=static_obstacles,
        min_x=specifications["x_range"][0],
        max_x=specifications["x_range"][1],
        min_y=specifications["y_range"][0],
        max_y=specifications["y_range"][1],
        min_interval=specifications["scenario_start"],
        max_interval=specifications["scenario_end"],
        max_size=specifications["obstacle_maximum"],
        min_radius=specifications["min_radius"],
        max_radius=specifications["max_radius"],
        seed=seed,
        only_static=True,
    )

    env.add_random_obstacles(
        num_points=points,
        num_lines=lines,
        num_polygons=polygons,
        min_x=specifications["x_range"][0],
        max_x=specifications["x_range"][1],
        min_y=specifications["y_range"][0],
        max_y=specifications["y_range"][1],
        min_interval=specifications["scenario_start"],
        max_interval=specifications["scenario_end"],
        max_size=specifications["obstacle_maximum"],
        min_radius=specifications["min_radius"],
        max_radius=specifications["max_radius"],
        seed=seed + static_obstacles,
        only_dynamic=True,
    )

    # create environment instance
    env_inst = EnvironmentInstance(
        environment=env,
        query_interval=Interval(
            specifications["scenario_start"],
            specifications["scenario_end"],
            closed="both",
        ),
        scenario_range_x=specifications["x_range"],
        scenario_range_y=specifications["y_range"],
        quiet=True,
    )

    return env_inst


def sample_benchmark(specifications, samples, reruns, seed):
    random.seed(seed)
    seeds = random.sample(range(0, 100000), reruns)
    results = {}

    for sample in samples:
        collector_taprm = []
        collector_taprm_pruned = []
        collector_rrt = []
        collector_rrt_star = []

        for rerun in range(reruns):
            # TODO - Note: setting different seeds for each rerun will result in different path costs
            seed = seeds[rerun]

            # initialize random environment with static and dynamic obstacles
            env = create_environment(specifications, seed, obstacles=100)

            ####################################################################
            # run TA-PRM with and without temporal pruning (rounded to integers)
            temporal_precision = 0
            start = time.time()

            # Prepare the TA-PRM graph
            graph = Graph(
                num_samples=sample,
                env=env,
                seed=seed,
                quiet=True,
            )
            graph.connect_start(coords=specifications["start_coords"])
            graph.connect_goal(coords=specifications["goal_coords"], quiet=True)
            ta_prm = TAPRM(graph=graph)
            preptime = time.time() - start

            # run the vanilla TA-PRM algorithm
            start = time.time()
            success, path, max_length_open, expansions = ta_prm.plan(
                start_time=specifications["start_time"], quiet=True
            )
            runtime_taprm = time.time() - start
            pathcost = graph.path_cost(path)
            collector_taprm = collector_taprm + [(preptime, runtime_taprm, pathcost)]

            # run the TA-PRM algorithm with temporal pruning
            start = time.time()
            success, path, max_length_open, expansions = ta_prm.plan_temporal(
                start_time=specifications["start_time"],
                quiet=True,
                temporal_precision=temporal_precision,
            )
            runtime_taprm_pruned = time.time() - start
            pathcost = graph.path_cost(path)
            collector_taprm_pruned = collector_taprm_pruned + [
                (preptime, runtime_taprm_pruned, pathcost)
            ]

            ####################################################################
            # run RRT and RRT* algorithms
            # initialize replanning framework
            replanner = ReplanningRRT(env=env, seed=seed)

            # run RRT algorithm (without rewiring)
            start = time.time()
            sol_path, rrt_runs = replanner.run(
                samples=sample,
                stepsize=specifications["stepsize"],
                start=specifications["start_coords"],
                goal=specifications["goal_coords"],
                query_time=specifications["start_time"],
                rewiring=False,
                prev_path=[ShapelyPoint(*specifications["start_coords"])],
                dynamic_obstacles=True,
                quiet=True,
            )
            runtime_rrt = time.time() - start
            pathcost = replanner.get_path_cost(sol_path)
            collector_rrt = collector_rrt + [(rrt_runs, runtime_rrt, pathcost)]

            # run RRT* algorithm (with rewiring)
            start = time.time()
            sol_path, rrt_star_runs = replanner.run(
                samples=sample,
                stepsize=specifications["stepsize"],
                start=specifications["start_coords"],
                goal=specifications["goal_coords"],
                query_time=specifications["start_time"],
                rewiring=True,
                prev_path=[ShapelyPoint(*specifications["start_coords"])],
                dynamic_obstacles=True,
                quiet=True,
            )
            runtime_rrt_star = time.time() - start
            pathcost = replanner.get_path_cost(sol_path)
            collector_rrt_star = collector_rrt_star + [
                (rrt_star_runs, runtime_rrt_star, pathcost)
            ]

        results[(1, sample)] = collector_taprm
        results[(2, sample)] = collector_taprm_pruned
        results[(3, sample)] = collector_rrt
        results[(4, sample)] = collector_rrt_star

    return results


def sample_benchmark_results(results, samples):
    print()
    print("Vanilla TA-PRM results:")
    for sample in samples:
        result = results[(1, sample)]
        aggregate_taprm_results(sample, result)
    print()

    print("TA-PRM with pruning results:")
    for sample in samples:
        result = results[(2, sample)]
        aggregate_taprm_results(sample, result)
    print()

    print("RRT results:")
    for sample in samples:
        result = results[(3, sample)]
        aggregate_rrt_results(sample, result)
    print()

    print("RRT* results:")
    for sample in samples:
        result = results[(4, sample)]
        aggregate_rrt_results(sample, result)
    print()

    print()


def aggregate_taprm_results(sample, results):
    # This function simplified and aggregates the TA-PRM results (preparation time, runtime, path cost)

    preptimes = [x[0] for x in results]
    runtimes = [x[1] for x in results]
    pathcosts = [x[2] for x in results]
    avg_prep = round(statistics.mean(preptimes), 5)
    std_prep = round(statistics.stdev(preptimes), 5)
    avg_runtime = round(statistics.mean(runtimes), 5)
    std_runtime = round(statistics.stdev(runtimes), 5)
    avg_cost = round(statistics.mean(pathcosts), 5)
    std_cost = round(statistics.stdev(pathcosts), 5)
    print(
        f"Samples: {sample},",
        avg_prep,
        "\u00B1",
        std_prep,
        "preparation time;",
        avg_runtime,
        "\u00B1",
        std_runtime,
        "runtime;",
        avg_cost,
        "\u00B1",
        std_cost,
        "path cost",
    )


def aggregate_rrt_results(sample, results):
    # This function simplified and aggregates the RRT results (runtime, path cost)

    rrt_runs = [x[0] for x in results]
    runtimes = [x[1] for x in results]
    pathcosts = [x[2] for x in results]
    avg_runs = round(statistics.mean(rrt_runs), 5)
    std_runs = round(statistics.stdev(rrt_runs), 5)
    avg_runtime = round(statistics.mean(runtimes), 5)
    std_runtime = round(statistics.stdev(runtimes), 5)
    avg_cost = round(statistics.mean(pathcosts), 5)
    std_cost = round(statistics.stdev(pathcosts), 5)
    print(
        f"Samples: {sample},",
        avg_runs,
        "\u00B1",
        std_runs,
        "runs;",
        avg_runtime,
        "\u00B1",
        std_runtime,
        "runtime;",
        avg_cost,
        "\u00B1",
        std_cost,
        "path cost",
    )
