import time
import random
import numpy as np

from shapely.geometry import Point as ShapelyPoint
from src.evaluation.helpers import (
    create_environment,
    run_algorithms,
    aggregate_benchmark_results,
)
from src.algorithms.graph import Graph
from src.algorithms.ta_prm import TAPRM
from src.evaluation.helpers import taprm_statistics


def pruning_benchmark(
    specifications,
    prunings,
    samples,
    obstacles,
    reruns,
    taprm_timeout,
    seed,
    dynamic_obs_only: bool = False,
):
    random.seed(seed)
    seeds = random.sample(range(0, 100000), 10 * reruns)
    results = {}

    # track the number of total runs and discarded runs
    total_runs = 0
    # track number of runs discarded due to potential unavailability of start and goal
    discarded_start_goal_runs = 0
    # track the number of times the start or goal node could not be connected to the RRT tree / TA-PRM roadmap
    # (caused by probabilistic completeness and fixed sample sizes for comparability)
    prob_completness_failures = 0
    # number of timeouts for TA-PRM (with and without temporal pruning)
    # collected as {pruning_param: number_of_timeouts, ...}
    timeouts = {}

    # keep track of valid reruns
    seed_idx = 0
    rerun = 0

    # initialize collectors for all prunign parameters
    collectors = {}
    for pruning_param in prunings:
        collectors[str(pruning_param)] = []

    while rerun < reruns:
        # Note: different seeds will result in different obstacle distributions
        # and different algorithm results
        seed = seeds[seed_idx]
        total_runs += 1

        ####################################################################
        # initialize random environment with static and dynamic obstacles
        env = create_environment(
            specifications=specifications,
            seed=seed,
            obstacles=obstacles,
            dynamic_obs_only=dynamic_obs_only,
        )

        # if start and/or goal node are in static collision, skip this seed
        # RRT / RRT* might not be able to find solutions in these scenarios,
        # not allow for comparability
        # (not counting towards algorithm preparation time)
        start_coords = specifications["start_coords"]
        goal_coords = specifications["goal_coords"]
        start_pt = ShapelyPoint(start_coords[0], start_coords[1])
        goal_pt = ShapelyPoint(goal_coords[0], goal_coords[1])
        if not env.static_collision_free(
            point=start_pt, check_all_dynamic=True
        ) or not env.static_collision_free(point=goal_pt, check_all_dynamic=True):
            print("Start or goal node in collision - skipping seed")
            seed_idx += 1
            discarded_start_goal_runs += 1
            continue

        ####################################################################
        # Prepare graph for TA-PRM algorithm
        temporal_precision = 0
        start = time.time()

        # Prepare the TA-PRM graph
        graph = Graph(
            num_samples=samples,
            env=env,
            seed=seed,
            quiet=True,
        )
        preptime_p1 = time.time() - start

        # connect start and goal node to the roadmap
        start = time.time()
        try:
            graph.connect_start(coords=start_coords)
        except RuntimeError as e:
            print("TA-PRM: Start node not connected to roadmap - skipping seed")
            print()
            prob_completness_failures += 1
            seed_idx += 1
            continue

        try:
            graph.connect_goal(coords=goal_coords, quiet=True)
        except RuntimeError as e:
            print("TA-PRM: Goal node not connected to roadmap - skipping seed")
            print()
            prob_completness_failures += 1
            seed_idx += 1
            continue

        ta_prm = TAPRM(graph=graph)
        preptime_p2 = time.time() - start
        preptime = preptime_p1 + preptime_p2
        current_collector = {}

        try:
            for pruning_param in prunings:
                # prepare logging
                quantitiy_print = "Pruning: " + str(pruning_param)

                ####################################################################
                # Run the TA-PRM algorithm with / without temporal pruning
                if np.isinf(pruning_param):
                    # run vanilla TA-PRM algorithm
                    start = time.time()
                    success, path, max_length_open, expansions = ta_prm.plan(
                        start_time=specifications["start_time"],
                        timeout=taprm_timeout,
                        quiet=True,
                    )
                    runtime = time.time() - start

                else:
                    # run TA-PRM with temporal pruning
                    start = time.time()
                    success, path, max_length_open, expansions = ta_prm.plan_temporal(
                        start_time=specifications["start_time"],
                        timeout=taprm_timeout,
                        temporal_precision=pruning_param,
                        quiet=True,
                    )
                    runtime = time.time() - start

                pathcost = graph.path_cost(path)
                print(
                    "TA-PRM -",
                    quantitiy_print,
                    "Rerun:",
                    rerun,
                    "Path Cost:",
                    pathcost,
                )

                # collect results
                current_collector[str(pruning_param)] = [(preptime, runtime, pathcost)]

        except RuntimeError as e:
            if (
                str(e)
                == "No valid path found from start to goal within the specified scenario horizon."
            ):
                print(
                    "TA-PRM -",
                    quantitiy_print,
                    "(no valid path found / probabilistic completeness limitation)",
                )
                print("Skipping seed...")
                print()
                prob_completness_failures += 1
                seed_idx += 1
                continue

            else:
                raise e

        except TimeoutError:
            print(
                "TA-PRM -",
                quantitiy_print,
                "Rerun:",
                rerun,
                "Timeout reached",
            )
            timeouts[str(pruning_param)] = timeouts.get(str(pruning_param), 0) + 1
            seed_idx += 1
            continue

        print("Successfully collected results for rerun", rerun)
        print()

        # if all runs were successful, store the results in the global collector
        for pruning_param in prunings:
            collectors[str(pruning_param)] = (
                collectors[str(pruning_param)] + current_collector[str(pruning_param)]
            )

        # increment rerun counter
        rerun += 1
        seed_idx += 1

    # collect analytics results to be save alongside results
    analytics = {
        "total_runs": total_runs,
        "discarded_start_goal_runs": discarded_start_goal_runs,
        "prob_completness_failures": prob_completness_failures,
        "timeouts": timeouts,
    }

    return collectors, analytics


def print_pruning_analytics(analytics):
    # Printing function for the analytics of the pruning benchmark
    print("Total runs:", analytics["total_runs"])
    print("Discarded start/goal runs:", analytics["discarded_start_goal_runs"])
    print(
        "Start/Goal not connected or no valid path found on roadmap - probabilistic completeness limitation:",
        analytics["prob_completness_failures"],
    )
    print()

    for key, value in analytics["timeouts"].items():
        print("Timeouts for TA-PRM with pruning parameter", key, ":", value)
    print()


def aggregate_pruning_benchmark_results(results, samples, prunings):
    # Print the aggregated results of the pruning benchmark

    for pruning in prunings:
        print("TA-PRM with Pruning Parameter:", pruning)
        taprm_statistics(samples, None, results=results[str(pruning)])
        print()
