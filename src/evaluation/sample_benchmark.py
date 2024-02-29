import random
from src.evaluation.helpers import (
    create_environment,
    run_algorithms,
    aggregate_benchmark_results,
)


def sample_benchmark(
    specifications, samples, obstacles, reruns, seed, dynamic_obs_only: bool = False
):
    random.seed(seed)
    seeds = random.sample(range(0, 100000), 10 * reruns)
    results = {}

    # track the number of total runs and discarded runs
    total_runs = 0
    # track number of runs discarded due to potential unavailability of start and goal
    discarded_start_goal_runs = 0
    # track number of runs discarded due to dynamic collision issue on replanning with RRT
    failed_replanning_runs = 0
    # track the number of times the goal node could not be connected to the RRT tree
    rrt_goal_connection_failures = 0

    for sample in samples:
        (
            total_runs,
            discarded_start_goal_runs,
            failed_replanning_runs,
            rrt_goal_connection_failures,
            collector_taprm,
            collector_taprm_pruned,
            collector_rrt,
            collector_rrt_star,
        ) = run_algorithms(
            specifications,
            total_runs,
            discarded_start_goal_runs,
            failed_replanning_runs,
            rrt_goal_connection_failures,
            sample,
            obstacles,
            reruns,
            seeds,
            dynamic_obs_only,
        )

        results[(1, sample)] = collector_taprm
        results[(2, sample)] = collector_taprm_pruned
        results[(3, sample)] = collector_rrt
        results[(4, sample)] = collector_rrt_star

    print()
    print("Total runs:", total_runs)
    print("Discarded start/goal runs:", discarded_start_goal_runs)
    print("Failed replanning runs:", failed_replanning_runs)
    print("Goal connection failures:", rrt_goal_connection_failures)
    print()

    return results
