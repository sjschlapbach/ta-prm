from tqdm import tqdm
import matplotlib.pyplot as plt

from src.examples.ta_prm_random import ta_prm_random


if __name__ == "__main__":
    print("Running experiment with random scenario and increasing sample number...")
    # values used as scenario end
    samples = []

    # track path costs
    costs = []

    # runtimes of the random example scenario
    runtimes = []

    # maximum length of the open list during expansion
    max_open_list = []

    # use different numbers of samples and track runtime
    min_samples = 70
    max_samples = 200
    step = 10

    for k in tqdm(range(min_samples, max_samples, step)):
        runtime, max_open, path_cost = ta_prm_random(
            plotting=False, samples=k, quiet=True
        )
        samples.append(k)
        costs.append(path_cost)
        runtimes.append(runtime * 1000)
        max_open_list.append(max_open)

    # plot the result in two subplots of the same figure
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
    fig.suptitle("Algorithm metrics over the number of samples - Random Scenario")

    ax1.plot(samples, runtimes)
    ax1.set_ylabel("Runtime (ms)")

    ax2.plot(samples, costs)
    ax2.set_ylabel("Path Cost")

    ax3.plot(samples, max_open_list)
    ax3.set_ylabel("Max Open List")
    ax3.set_xlabel("Samples (#)")

    print("Running experiment with random scenario and increasing connectivity...")
    # increase connectivity at the same time as the samples and observe complexity
    # values used as scenario end
    connections_per_node = []

    # track path costs
    costs = []

    # runtimes of the random example scenario
    runtimes = []

    # use different numbers of samples and track runtime
    min_connections = 8
    max_connections = 12
    step = 1

    for k in tqdm(range(min_connections, max_connections, step)):
        runtime, _, path_cost = ta_prm_random(
            plotting=False, samples=200, max_connections=k, quiet=True
        )
        connections_per_node.append(k)
        costs.append(path_cost)
        runtimes.append(runtime * 1000)

    # plot the result in two subplots of the same figure
    fig2, (ax1_2, ax2_2) = plt.subplots(2, sharex=True)
    fig2.suptitle("Algorithm metrics over the number of connections - Random Scenario")

    ax1_2.plot(connections_per_node, runtimes)
    ax1_2.set_ylabel("Runtime (ms)")

    ax2_2.plot(connections_per_node, costs)
    ax2_2.set_ylabel("Path Cost")
    ax2_2.set_xlabel("Maximum connections to other nodes (per node)")
    plt.show()
