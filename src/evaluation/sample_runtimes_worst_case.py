from tqdm import tqdm
import matplotlib.pyplot as plt

from src.examples.ta_prm_worst_case import ta_prm_worst_case


if __name__ == "__main__":
    # values used as scenario end
    samples = []

    # track path costs
    costs_standard = []
    costs_temporal = []

    # runtimes of the random example scenario
    runtimes_standard = []
    runtimes_temporal = []

    # track the maximum open list size
    max_open_list_standard = []
    max_open_list_temporal = []

    # use different numbers of samples and track runtime
    min_samples = 50
    max_samples = 120
    step = 5

    # run the scenario with standard TA-PRM algorithm and increasing number of samples
    for k in tqdm(range(min_samples, max_samples, step)):
        runtime, max_open, path_cost = ta_prm_worst_case(
            plotting=False, samples=k, quiet=True
        )
        samples.append(k)
        costs_standard.append(path_cost)
        runtimes_standard.append(runtime * 1000)
        max_open_list_standard.append(max_open)

    # re-run the algorithm with a version of the algorithm using temporal pruning
    for k in tqdm(range(min_samples, max_samples, step)):
        runtime, max_open, path_cost = ta_prm_worst_case(
            plotting=False, samples=k, quiet=True, temporal_precision=0
        )
        costs_temporal.append(path_cost)
        runtimes_temporal.append(runtime * 1000)
        max_open_list_temporal.append(max_open)

    # plot the result in two subplots of the same figure
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
    fig.suptitle("Algorithm metrics over the number of samples - Worst Case Scenario")

    ax1.plot(samples, runtimes_standard, label="Vanilla TA-PRM")
    ax1.plot(samples, runtimes_temporal, label="TA-PRM with precision 0")
    ax1.set_ylabel("Runtime (ms)")
    ax1.legend()

    ax2.plot(samples, costs_standard, label="Vanilla TA-PRM")
    ax2.plot(samples, costs_temporal, label="TA-PRM with precision 0")
    ax2.set_ylabel("Path Cost")
    ax2.legend()

    ax3.plot(samples, max_open_list_standard, label="Vanilla TA-PRM")
    ax3.plot(samples, max_open_list_temporal, label="TA-PRM with precision 0")
    ax3.set_ylabel("Maximum Open List Size")
    ax3.set_xlabel("Samples")
    ax3.legend()
    plt.show()
