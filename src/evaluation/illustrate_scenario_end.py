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

    # set values for interval_end and minimum length to find a solution
    min_for_solution = 460
    max_scenario_end = 1500

    for k in tqdm(range(min_for_solution, max_scenario_end, 10)):
        runtime, max_open = ta_prm_random(plotting=False, scenario_end=k)
        scenario_ends.append(k)
        runtimes.append(runtime * 1000)
        max_open_list.append(max_open)

    # plof the result in two subplots of the same figure
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    fig.suptitle("Runtime and maximum length of open list over scenario end")

    ax1.plot(scenario_ends, runtimes)
    ax1.set_ylabel("Runtime (ms)")

    ax2.plot(scenario_ends, max_open_list)
    ax2.set_ylabel("Maximum length of open list")
    ax2.set_xlabel("Scenario End Time (s)")
    plt.show()
