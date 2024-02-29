import os
import json
import numpy as np

from src.evaluation.sample_benchmark import sample_benchmark
from src.evaluation.sample_benchmark import sample_benchmark_results


if __name__ == "__main__":
    print("Starting benchmarking...")

    # ? Description of algorithms
    # 1) Vanilla TA-PRM without temporal pruning and all guarantees
    # 2) TA-PRM with temporal pruning and limited temporal resolution to integers
    # 3) RRT with dynamic obstacle replanning
    # 4) RRT* with dynamic obstacle replanning and rewiring

    # ! Benchmark selection
    sampling = True
    obstacles = False
    pruning = False

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
        "min_radius": 2,
        "max_radius": 80,
        "stepsize": 0.1,
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
        # Note: 25% of the obstacles will be static, 75% dynamic
        obstacles = 50
        samples = [50, 100, 200, 300]

        # Results: (algorithm, sample): (preptime, runtime, path_cost)[]
        sample_benchmarks = sample_benchmark(
            specifications, samples, obstacles, reruns, seed
        )
        print("Sample benchmarking completed:")
        sample_benchmark_results(sample_benchmarks, samples)

        # TODO: save the results in a JSON file
        # if not os.path.exists("results"):
        #     os.makedirs("results")

        # with open("results/sample_benchmarks.json", "w") as file:
        #     json.dump(sample_benchmarks, file)

    ###########################################################
    # OBSTACLE BENCHMARKING - track runtime and path cost with increasing number of dynamic obstacles
    if obstacles:
        print("Running obstacle benchmark...")
        obstacles = [10, 20, 50, 100, 200]

        # Results: (algorithm, obstacles): (preptime, runtime, path_cost)[]
        obstacle_benchmarks = {}
        print("Obstacle benchmarking completed.")

    ###########################################################
    # Pruning benchmarking - track the performance for increased pruning levels (compared against vanilla TA-PRM)
    if pruning:
        print("Running pruning benchmark...")
        samples = 100
        pruning = [np.inf, 0, -1, -2]  # np.inf = vanilla TA-PRM
        obstacles = [10, 50, 100, 200]

        # Results: (pruning, obstacles): (preptime, runtime, path_cost)[]
        pruning_benchmarks = {}
        print("Pruning benchmarking completed.")

    # TODO: think about adding a benchmark with a worst-case szenario for TA-PRM to compare vanilla and pruning versions
