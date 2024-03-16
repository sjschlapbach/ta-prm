import os
import json
import numpy as np

from src.evaluation.sample_benchmark import sample_benchmark
from src.evaluation.obstacle_benchmark import obstacle_benchmark
from src.evaluation.pruning_benchmark import (
    pruning_benchmark,
    print_pruning_analytics,
    aggregate_pruning_benchmark_results,
)
from src.evaluation.helpers import aggregate_benchmark_results, print_analytics


def remap_keys(mapping):
    return [{"key": k, "value": v} for k, v in mapping.items()]


if __name__ == "__main__":
    print("Starting benchmarking...")

    # ? Description of algorithms
    # 1) Vanilla TA-PRM without temporal pruning and all guarantees
    # 2) TA-PRM with temporal pruning and limited temporal resolution to integers
    # 3) RRT with dynamic obstacle replanning
    # 4) RRT* with dynamic obstacle replanning and rewiring

    # ! Benchmark selection
    sampling_benchmark = True
    obstacles_benchmark = True
    pruning_benchmarking = True

    # ! Basic seed for reproducibility
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
        "min_radius": 2,
        "max_radius": 80,
        "stepsize": 1,
    }

    # ? How many reruns per scenario should be performed to compute average values?
    reruns = 1000

    # ? Specifications
    ###########################################################
    # Sample benchmarking - track runtime and path cost with increasing number of samples
    # (fixed number of static obstacles)
    if sampling_benchmark:
        print("Running sample benchmark...")
        # Note: 25% of the obstacles will be static, 75% dynamic
        obstacles = 50
        samples = [50, 100, 200]

        # set the timeouts to fixed values for each sample number (in seconds)
        timeouts = [60, 120, 120]

        # Results: (algorithm, sample): (preptime, runtime, path_cost)[]
        sample_benchmarks, sample_analytics = sample_benchmark(
            specifications=specifications,
            samples=samples,
            obstacles=obstacles,
            reruns=reruns,
            timeouts=timeouts,
            seed=seed,
            dynamic_obs_only=False,
        )
        print("Sample benchmarking completed:")
        print_analytics(sample_analytics)
        aggregate_benchmark_results(sample_benchmarks, samples, None)

        # save the results in a JSON file
        if not os.path.exists("results"):
            os.makedirs("results")

        with open(
            "results/sample_benchmarks_" + str(reruns) + "_reruns.json", "w"
        ) as file:
            json.dump(remap_keys(sample_benchmarks), file)

        with open(
            "results/sample_analytics_" + str(reruns) + "_reruns.json", "w"
        ) as file:
            sample_analytics["taprm_timeouts"] = remap_keys(
                sample_analytics["taprm_timeouts"]
            )
            json.dump(remap_keys(sample_analytics), file)

    ###########################################################
    # OBSTACLE BENCHMARKING - track runtime and path cost with increasing number of dynamic obstacles
    if obstacles_benchmark:
        print("Running obstacle benchmark...")
        specifications["max_radius"] = 40
        samples = 100
        obstacles = [10, 20, 50, 100, 200]

        # set the timeouts to fixed values for each obstacle number (in seconds)
        timeouts = [60, 120, 120, 120, 120]

        # Results: (algorithm, obstacles): (preptime, runtime, path_cost)[]
        obstacle_benchmarks, obstacle_analytics = obstacle_benchmark(
            specifications=specifications,
            samples=samples,
            obstacles=obstacles,
            reruns=reruns,
            timeouts=timeouts,
            seed=seed,
        )
        print("Obstacle benchmarking completed:")
        print_analytics(obstacle_analytics)
        aggregate_benchmark_results(obstacle_benchmarks, None, obstacles)

        # save the results in a JSON file
        if not os.path.exists("results"):
            os.makedirs("results")

        with open(
            "results/obstacle_benchmarks_" + str(reruns) + "_reruns.json", "w"
        ) as file:
            json.dump(remap_keys(obstacle_benchmarks), file)

        with open(
            "results/obstacle_analytics_" + str(reruns) + "_reruns.json", "w"
        ) as file:
            obstacle_analytics["taprm_timeouts"] = remap_keys(
                obstacle_analytics["taprm_timeouts"]
            )
            json.dump(remap_keys(obstacle_analytics), file)

    ###########################################################
    # Pruning benchmarking - track the performance for increased pruning levels (compared against vanilla TA-PRM)
    if pruning_benchmarking:
        print("Running pruning benchmark...")
        samples = 100
        obstacles = 100
        prunings = [np.inf, 1, 0, -1, -2]  # np.inf = vanilla TA-PRM
        timeout = 120

        # Results: {pruning: (preptime, runtime, path_cost)}[]
        results, analytics = pruning_benchmark(
            specifications=specifications,
            prunings=prunings,
            samples=samples,
            obstacles=obstacles,
            reruns=reruns,
            taprm_timeout=timeout,
            seed=seed,
            dynamic_obs_only=True,
        )
        print("Pruning benchmarking completed.")
        print_pruning_analytics(analytics)
        aggregate_pruning_benchmark_results(results, samples, prunings)

        # save the results in a JSON file
        if not os.path.exists("results"):
            os.makedirs("results")

        with open(
            "results/pruning_benchmarks_" + str(reruns) + "_reruns.json", "w"
        ) as file:
            json.dump(results, file)

        with open(
            "results/pruning_analytics_" + str(reruns) + "_reruns.json", "w"
        ) as file:
            json.dump(analytics, file)
