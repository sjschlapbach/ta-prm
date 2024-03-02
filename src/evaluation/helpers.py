import time
import random
import statistics
import numpy as np
from pandas import Interval
from shapely.geometry import Point as ShapelyPoint

from src.algorithms.graph import Graph
from src.algorithms.ta_prm import TAPRM
from src.algorithms.rrt import RRT
from src.algorithms.replanning_rrt import ReplanningRRT
from src.envs.environment import Environment
from src.envs.environment_instance import EnvironmentInstance


def create_environment(specifications, seed, obstacles, dynamic_obs_only: bool = False):
    # This function creates an environment instance with random obstacles
    # according to the specifications (both static and dynamic).

    # create environment with random obstacles (25% static obstacles, 75% dynamic obstacles)
    env = Environment()
    if dynamic_obs_only:
        static_obstacles = 0
        points = obstacles // 3
        lines = obstacles // 3
        polygons = obstacles - points - lines
    else:
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


def run_algorithms(
    specifications,
    total_runs,
    discarded_start_goal_runs,
    failed_replanning_runs,
    prob_completness_failures,
    rrt_exceeded_max_connection_trials,
    taprm_timeouts,
    samples,
    obstacles,
    reruns,
    timeout,
    seeds,
    dynamic_obs_only,
    quantitiy_print,
):
    # keep track of valid reruns
    seed_idx = 0
    rerun = 0

    # track the results of the different algorithm runs
    collector_taprm = []
    collector_taprm_pruned = []
    collector_rrt = []
    collector_rrt_star = []

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
        # run RRT and RRT* algorithms
        # initialize replanning framework
        replanner = ReplanningRRT(env=env, seed=seed)

        # run RRT algorithm (without rewiring)
        start = time.time()
        try:
            sol_path, rrt_runs = replanner.run(
                samples=samples,
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

        except RuntimeError as e:
            if (
                str(e)
                == "Goal node is not reachable from the tree or not collision free."
            ):
                print("RRT -", quantitiy_print, "(goal node not connected)")
                print("Skipping seed...")
                print()
                prob_completness_failures += 1
                seed_idx += 1
                continue

            elif (
                str(e) == "Edge from new starting point is in collision on replanning."
            ):
                print("RRT -", quantitiy_print, "(replanning issue)")
                print("Skipping seed...")
                print()
                failed_replanning_runs += 1
                seed_idx += 1
                continue

            elif (
                str(e) == "Exceeded maximum number of connection trials for new sample."
            ):
                print("RRT -", quantitiy_print, "Rerun:", rerun)
                print("Skipping seed...")
                print()
                rrt_exceeded_max_connection_trials += 1
                seed_idx += 1
                continue

            else:
                raise e

        pathcost_rrt = replanner.get_path_cost(sol_path)
        print("RRT -", quantitiy_print, "Rerun:", rerun, "Path Cost:", pathcost_rrt)

        # run RRT* algorithm (with rewiring)
        try:
            start = time.time()
            sol_path, rrt_star_runs = replanner.run(
                samples=samples,
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
        except RuntimeError as e:
            if (
                str(e)
                == "Goal node is not reachable from the tree or not collision free."
            ):
                print("RRT* -", quantitiy_print, "(goal node not connected)")
                print("Skipping seed...")
                print()
                prob_completness_failures += 1
                seed_idx += 1
                continue

            elif (
                str(e) == "Edge from new starting point is in collision on replanning."
            ):
                print("RRT* -", quantitiy_print, "(replanning issue)")
                print("Skipping seed...")
                print()
                failed_replanning_runs += 1
                seed_idx += 1
                continue

            elif (
                str(e) == "Exceeded maximum number of connection trials for new sample."
            ):
                print("RRT* -", quantitiy_print, "Rerun:", rerun)
                print("Skipping seed...")
                print()
                rrt_exceeded_max_connection_trials += 1
                seed_idx += 1
                continue

            else:
                raise e

        pathcost_rrt_star = replanner.get_path_cost(sol_path)
        print(
            "RRT* -",
            quantitiy_print,
            "Rerun:",
            rerun,
            "Path Cost:",
            pathcost_rrt_star,
        )

        ####################################################################
        # run TA-PRM with and without temporal pruning (rounded to integers)
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

        # run the vanilla TA-PRM algorithm
        try:
            start = time.time()
            success, path, max_length_open, expansions = ta_prm.plan(
                start_time=specifications["start_time"], timeout=timeout, quiet=True
            )
            runtime_taprm = time.time() - start

        except RuntimeError as e:
            if (
                str(e)
                == "No valid path found from start to goal within the specified scenario horizon."
            ):
                print(
                    "Vanilla TA-PRM -",
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
                "Vanilla TA-PRM -",
                quantitiy_print,
                "Rerun:",
                rerun,
                "Timeout reached",
            )
            taprm_timeouts[(str(np.inf), samples, obstacles)] = (
                taprm_timeouts.get((str(np.inf), samples, obstacles), 0) + 1
            )
            seed_idx += 1
            continue

        pathcost_taprm = graph.path_cost(path)
        print(
            "Vanilla TA-PRM -",
            quantitiy_print,
            "Rerun:",
            rerun,
            "Path Cost:",
            pathcost_taprm,
        )

        # run the TA-PRM algorithm with temporal pruning
        try:
            start = time.time()
            success, path, max_length_open, expansions = ta_prm.plan_temporal(
                start_time=specifications["start_time"],
                timeout=timeout,
                temporal_precision=temporal_precision,
                quiet=True,
            )
            runtime_taprm_pruned = time.time() - start

        except RuntimeError as e:
            if (
                str(e)
                == "No valid path found from start to goal within the specified scenario horizon."
            ):
                print(
                    "TA-PRM with pruning -",
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
                "TA-PRM with pruning -",
                quantitiy_print,
                "Rerun:",
                rerun,
                "Timeout reached",
            )
            taprm_timeouts[(str(temporal_precision), samples, obstacles)] = (
                taprm_timeouts.get((str(temporal_precision), samples, obstacles), 0) + 1
            )
            seed_idx += 1
            continue

        pathcost_taprm_pruning = graph.path_cost(path)
        print(
            "TA-PRM with pruning -",
            quantitiy_print,
            "Rerun:",
            rerun,
            "Path Cost:",
            pathcost_taprm_pruning,
        )

        # collect all results
        collector_taprm = collector_taprm + [(preptime, runtime_taprm, pathcost_taprm)]
        collector_taprm_pruned = collector_taprm_pruned + [
            (preptime, runtime_taprm_pruned, pathcost_taprm_pruning)
        ]
        collector_rrt = collector_rrt + [(rrt_runs, runtime_rrt, pathcost_rrt)]
        collector_rrt_star = collector_rrt_star + [
            (rrt_star_runs, runtime_rrt_star, pathcost_rrt_star)
        ]
        print("Successfully collected results for rerun", rerun)
        print()

        # increment rerun counter
        rerun += 1
        seed_idx += 1

    return (
        total_runs,
        discarded_start_goal_runs,
        failed_replanning_runs,
        prob_completness_failures,
        rrt_exceeded_max_connection_trials,
        taprm_timeouts,
        collector_taprm,
        collector_taprm_pruned,
        collector_rrt,
        collector_rrt_star,
    )


def aggregate_benchmark_results(results, samples, obstacles):
    if samples is not None:
        print("Vanilla TA-PRM results:")
        for sample in samples:
            result = results[(1, sample)]
            taprm_statistics(sample, None, result)
        print()

        print("TA-PRM with pruning results:")
        for sample in samples:
            result = results[(2, sample)]
            taprm_statistics(sample, None, result)
        print()

        print("RRT results:")
        for sample in samples:
            result = results[(3, sample)]
            rrt_statistics(sample, None, result)
        print()

        print("RRT* results:")
        for sample in samples:
            result = results[(4, sample)]
            rrt_statistics(sample, None, result)
        print()

    elif obstacles is not None:
        print("Vanilla TA-PRM results:")
        for obstacle in obstacles:
            result = results[(1, obstacle)]
            taprm_statistics(None, obstacle, result)
        print()

        print("TA-PRM with pruning results:")
        for obstacle in obstacles:
            result = results[(2, obstacle)]
            taprm_statistics(None, obstacle, result)
        print()

        print("RRT results:")
        for obstacle in obstacles:
            result = results[(3, obstacle)]
            rrt_statistics(None, obstacle, result)
        print()

        print("RRT* results:")
        for obstacle in obstacles:
            result = results[(4, obstacle)]
            rrt_statistics(None, obstacle, result)
        print()


def taprm_statistics(sample, obstacle, results):
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

    if sample is not None:
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

    elif obstacle is not None:
        print(
            f"Obstacles: {obstacle},",
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


def rrt_statistics(sample, obstacle, results):
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

    if sample is not None:
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

    elif obstacle is not None:
        print(
            f"Obstacles: {obstacle},",
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


# TODO: add loader functions to print the results from the JSON file
