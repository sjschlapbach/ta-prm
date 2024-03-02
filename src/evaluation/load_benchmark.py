import os
import json

from src.evaluation.helpers import results_from_file, analytics_from_file

if __name__ == "__main__":
    # ? Benchmark selection
    sampling_benchmark = True
    obstacles_benchmark = True
    pruning_benchmark = False

    # ? Number of reruns, which were performed in the benchmark
    reruns = 100

    print("Loading Benchmarks...")
    if sampling_benchmark:
        benchmark_file = "sample_benchmarks_" + str(reruns) + "_reruns.json"
        analytics_file = "sample_analytics_" + str(reruns) + "_reruns.json"

        # check if both files exist in the results folder
        if not os.path.exists("results/" + benchmark_file) or not os.path.exists(
            "results/" + analytics_file
        ):
            print("Sample benchmark results not found.")
            print("Please run the sample benchmark first.")
            exit()

        # load the benchmark and analytics results
        with open("results/" + benchmark_file, "r") as file:
            sample_benchmarks = json.load(file)

        with open("results/" + analytics_file, "r") as file:
            sample_analytics = json.load(file)

        print()
        print("################################################################")
        print("SAMPLE BENCHMARK RESULTS:")
        results_from_file(sample_benchmarks, samples=True)

        print()
        print("SAMPLE BENCHMARK ANALYTICS:")
        analytics_from_file(sample_analytics)

    if obstacles_benchmark:
        benchmark_file = "obstacle_benchmarks_" + str(reruns) + "_reruns.json"
        analytics_file = "obstacle_analytics_" + str(reruns) + "_reruns.json"

        # check if both files exist in the results folder
        if not os.path.exists("results/" + benchmark_file) or not os.path.exists(
            "results/" + analytics_file
        ):
            print("Obstacle benchmark results not found.")
            print("Please run the obstacle benchmark first.")
            exit()

        # load the benchmark and analytics results
        with open("results/" + benchmark_file, "r") as file:
            obstacle_benchmarks = json.load(file)

        with open("results/" + analytics_file, "r") as file:
            obstacle_analytics = json.load(file)

        print()
        print("################################################################")
        print("OBSTACLE BENCHMARK RESULTS:")
        results_from_file(obstacle_benchmarks, obstacles=True)

        print()
        print("OBSTACLE BENCHMARK ANALYTICS:")
        analytics_from_file(obstacle_analytics)

    if pruning_benchmark:
        # TODO: Implement once pruning benchmark is available
        pass

    print("Benchmark loading completed.")
