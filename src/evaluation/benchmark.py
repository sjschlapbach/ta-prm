from pandas import Interval
import numpy as np
import random
import time

from src.algorithms.graph import Graph
from src.algorithms.ta_prm import TAPRM
from src.algorithms.rrt import RRT
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
            # run RRT algorithm (without rewiring)
            start = time.time()
            rrt = RRT(
                start=specifications["start_coords"],
                goal=specifications["goal_coords"],
                env=env,
                num_samples=sample,
                rewiring=False,
                seed=seed,
            )
            sol_path = rrt.rrt_find_path()
            runtime_rrt = time.time() - start
            pathcost = rrt.get_path_cost(sol_path)
            collector_rrt = collector_rrt + [(None, runtime_rrt, pathcost)]

            ####################################################################
            # run RRT* algorithm (with rewiring)
            start = time.time()
            rrt = RRT(
                start=specifications["start_coords"],
                goal=specifications["goal_coords"],
                env=env,
                num_samples=sample,
                rewiring=True,
                seed=seed,
            )
            sol_path = rrt.rrt_find_path()
            runtime_rrt_star = time.time() - start
            pathcost = rrt.get_path_cost(sol_path)
            collector_rrt_star = collector_rrt_star + [(None, runtime_rrt, pathcost)]

        results[(1, sample)] = collector_taprm
        results[(2, sample)] = collector_taprm_pruned
        results[(3, sample)] = collector_rrt
        results[(4, sample)] = collector_rrt_star

    return results


if __name__ == "__main__":
    print("Starting benchmarking...")

    # ? Description of algorithms
    # 1) Vanilla TA-PRM without temporal pruning and all guarantees
    # 2) TA-PRM with temporal pruning and limited temporal resolution to integers
    # 3) RRT with dynamic obstacle replanning
    # 4) RRT* with dynamic obstacle replanning and rewiring

    # ! Benchmark selection
    sampling = True
    obstacles = True
    pruning = True

    # ! Basic seed for reproducibility
    # TODO: for certain seeds, RRT cannot see that the goal node is only blocked dynamically - not used for evaluation -> but note in paper!!
    seed = 0

    # ! Specify workspace and task
    specifications: dict = {
        "x_range": (0, 1000),
        "y_range": (0, 1000),
        "scenario_start": 0,
        "scenario_end": 2000,
        "start_coords": (2, 2),
        "start_time": 20,
        "goal_coords": (998, 998),
        "obstacle_maximum": 3,
        "min_radius": 1.1,
        "max_radius": 4,
        "stepsize": 1,
    }

    # ? How many reruns per scenario should be performed to compute average values?
    reruns = 2
    # TODO: reruns = 20

    # ? Specifications
    ###########################################################
    # Sample benchmarking - track runtime and path cost with increasing number of samples
    # (fixed number of static obstacles)
    if sampling:
        print("Running sample benchmark...")
        # TODO: samples = [50, 100, 500, 1000]
        samples = [50, 100]

        # Results: (algorithm, sample): (preptime, runtime, path_cost)[]
        sample_benchmarks = sample_benchmark(specifications, samples, reruns, seed)
        print(sample_benchmarks)  # TODO: remove
        print("Sample benchmarking completed.")

    ###########################################################
    # OBSTACLE BENCHMARKING - track runtime and path cost with increasing number of dynamic obstacles
    if obstacles:
        print("Running obstacle benchmark...")
        obstacles = [10, 20, 50, 100, 500]

        # Results: (algorithm, obstacles): (preptime, runtime, path_cost)[]
        obstacle_benchmarks = {}
        print("Obstacle benchmarking completed.")

    ###########################################################
    # Pruning benchmarking - track the performance for increased pruning levels (compared against vanilla TA-PRM)
    if pruning:
        print("Running pruning benchmark...")
        pruning = [np.inf, 0, -1, -2]
        obstacles = [10, 100, 200]

        # Results: (pruning, obstacles): (preptime, runtime, path_cost)[]
        pruning_benchmarks = {}
        print("Pruning benchmarking completed.")
