from tqdm import tqdm
import matplotlib.pyplot as plt

from src.examples.ta_prm_random import ta_prm_random


if __name__ == "__main__":
    # values used as scenario end
    scenario_ends = []

    # runtimes of the random example scenario
    runtimes = []

    # maximum length of the open list during expansion
    max_open_list = []

    for k in tqdm(range(460, 620)):
        print(k)
        runtime, max_open = ta_prm_random(plotting=False, scenario_end=k)
        scenario_ends.append(k)
        runtimes.append(runtime)
        max_open_list.append(max_open)

    # plof the result in two subplots of the same figure
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    fig.suptitle("Runtime and maximum length of open list over scenario end")

    ax1.plot(scenario_ends, runtimes)
    ax1.set_ylabel("Runtime (s)")

    ax2.plot(scenario_ends, max_open_list)
    ax2.set_ylabel("Maximum length of open list")
    ax2.set_xlabel("Scenario End Time (s)")
    plt.show()
