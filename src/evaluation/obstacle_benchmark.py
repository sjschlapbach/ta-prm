import random
from src.evaluation.helpers import run_algorithms


def obstacle_benchmark(specifications, samples, obstacles, reruns, seed):
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
    prob_completness_failures = 0
    # track the number of times the maximum number of connection trials was exceeded
    rrt_exceeded_max_connection_trials = 0

    for num_obstacles in obstacles:
        (
            total_runs,
            discarded_start_goal_runs,
            failed_replanning_runs,
            prob_completness_failures,
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
            prob_completness_failures=prob_completness_failures,
            rrt_exceeded_max_connection_trials=rrt_exceeded_max_connection_trials,
            samples=samples,
            obstacles=num_obstacles,
            reruns=reruns,
            seeds=seeds,
            dynamic_obs_only=True,
            quantitiy_print="Obstacles: " + str(num_obstacles),
        )

        results[(1, num_obstacles)] = collector_taprm
        results[(2, num_obstacles)] = collector_taprm_pruned
        results[(3, num_obstacles)] = collector_rrt
        results[(4, num_obstacles)] = collector_rrt_star

    print()
    print("Total runs:", total_runs)
    print("Discarded start/goal runs:", discarded_start_goal_runs)
    print("Failed replanning runs:", failed_replanning_runs)
    print(
        "Start/Goal not connected or not valid path found on roadmap - probabilistic completeness limitation:",
        prob_completness_failures,
    )
    print("Exceeded max connection trials (RRT):", rrt_exceeded_max_connection_trials)
    print()

    return results
