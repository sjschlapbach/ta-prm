from tqdm import tqdm
import matplotlib.pyplot as plt

from src.examples.ta_prm_random import ta_prm_random


if __name__ == "__main__":
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
    max_samples = 300
    step = 10

    for k in tqdm(range(min_samples, max_samples, step)):
        runtime, max_open, path_cost = ta_prm_random(plotting=False, samples=k)
        samples.append(k)
        costs.append(path_cost)
        runtimes.append(runtime * 1000)
        max_open_list.append(max_open)

    # plof the result in two subplots of the same figure
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
    fig.suptitle("Algorithm metrics over the number of samples")

    ax1.plot(samples, runtimes)
    ax1.set_ylabel("Runtime (ms)")

    ax2.plot(samples, costs)
    ax2.set_ylabel("Path Cost")

    ax3.plot(samples, max_open_list)
    ax3.set_ylabel("Max Open List")
    ax3.set_xlabel("Samples (#)")
    plt.show()
