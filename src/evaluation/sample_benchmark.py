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
    # track the number of times the start or goal node could not be connected to the RRT tree / TA-PRM roadmap
    # (caused by probabilistic completeness and fixed sample sizes for comparability)
    no_start_goal_connection = 0
    # track the number of times the maximum number of connection trials was exceeded
    rrt_exceeded_max_connection_trials = 0

    for sample in samples:
        (
            total_runs,
            discarded_start_goal_runs,
            failed_replanning_runs,
            no_start_goal_connection,
            rrt_exceeded_max_connection_trials,
            collector_taprm,
            collector_taprm_pruned,
            collector_rrt,
            collector_rrt_star,
        ) = run_algorithms(
            specifications=specifications,
            total_runs=total_runs,
            discarded_start_goal_runs=discarded_start_goal_runs,
            failed_replanning_runs=failed_replanning_runs,
            no_start_goal_connection=no_start_goal_connection,
            rrt_exceeded_max_connection_trials=rrt_exceeded_max_connection_trials,
            samples=sample,
            obstacles=obstacles,
            reruns=reruns,
            seeds=seeds,
            dynamic_obs_only=dynamic_obs_only,
            quantitiy_print="Samples: " + str(sample),
        )

        results[(1, sample)] = collector_taprm
        results[(2, sample)] = collector_taprm_pruned
        results[(3, sample)] = collector_rrt
        results[(4, sample)] = collector_rrt_star

    print()
    print("Total runs:", total_runs)
    print("Discarded start/goal runs:", discarded_start_goal_runs)
    print("Failed replanning runs:", failed_replanning_runs)
    print(
        "Start/Goal not connected - probabilistic completeness limitation:",
        no_start_goal_connection,
    )
    print("Exceeded max connection trials (RRT):", rrt_exceeded_max_connection_trials)
    print()

    return results
