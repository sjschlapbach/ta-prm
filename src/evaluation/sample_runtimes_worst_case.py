from tqdm import tqdm
import matplotlib.pyplot as plt

from src.examples.ta_prm_worst_case import ta_prm_worst_case


if __name__ == "__main__":
    # values used as scenario end
    samples = []

    # track path costs
    costs = []

    # runtimes of the random example scenario
    runtimes = []

    # use different numbers of samples and track runtime
    min_samples = 50
    max_samples = 1000
    step = 50

    for k in tqdm(range(min_samples, max_samples, step)):
        runtime, path_cost = ta_prm_worst_case(plotting=False, samples=k)
        samples.append(k)
        costs.append(path_cost)
        runtimes.append(runtime * 1000)

    # plot the result in two subplots of the same figure
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    fig.suptitle("Algorithm metrics over the number of samples - Worst Case Scenario")

    ax1.plot(samples, runtimes)
    ax1.set_ylabel("Runtime (ms)")

    ax2.plot(samples, costs)
    ax2.set_ylabel("Path Cost")
    ax2.set_xlabel("Samples")

    # increase connectivity at the same time as the samples and observe complexity
    # values used as scenario end
    connections_per_node = []

    # track path costs
    costs = []

    # runtimes of the random example scenario
    runtimes = []

    # use different numbers of samples and track runtime
    min_connections = 10
    max_connections = 20
    step = 2

    for k in tqdm(range(min_connections, max_connections, step)):
        runtime, path_cost = ta_prm_worst_case(
            plotting=False, samples=500, max_connections=k
        )
        connections_per_node.append(k)
        costs.append(path_cost)
        runtimes.append(runtime * 1000)

    # plot the result in two subplots of the same figure
    fig2, (ax1_2, ax2_2) = plt.subplots(2, sharex=True)
    fig2.suptitle(
        "Algorithm metrics over the number of connections - Worst Case Scenario"
    )

    ax1_2.plot(connections_per_node, runtimes)
    ax1_2.set_ylabel("Runtime (ms)")

    ax2_2.plot(connections_per_node, costs)
    ax2_2.set_ylabel("Path Cost")
    ax2_2.set_xlabel("Maximum connections to other nodes (per node)")
    plt.show()
