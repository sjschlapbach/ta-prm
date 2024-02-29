from src.evaluation.sample_benchmark import create_environment


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
    # track the number of times the goal node could not be connected to the RRT tree
    rrt_goal_connection_failures = 0

    for obstacle in obstacles:
        collector_taprm = []
        collector_taprm_pruned = []
        collector_rrt = []
        collector_rrt_star = []

        # keep track of valid reruns
        seed_idx = 0
        rerun = 0

        while rerun < reruns:
            # TODO: possibly re-use the same function as in sample_benchmark
            pass
