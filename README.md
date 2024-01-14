# Time-Aware PRM

[![Python Testing](https://github.com/sjschlapbach/ta-prm/actions/workflows/python_testing.yml/badge.svg)](https://github.com/sjschlapbach/ta-prm/actions/workflows/python_testing.yml)
[![Example Scripts](https://github.com/sjschlapbach/ta-prm/actions/workflows/python_scripts.yml/badge.svg)](https://github.com/sjschlapbach/ta-prm/actions/workflows/python_scripts.yml)

This repository contains a Python implementation of the Time-Aware PRM (TA-PRM) algorithm. TA-PRM is a sampling-based motion planning algorithm that is able to find a solution to a motion planning problem in a time-varying environment.

## Setup

Consider setting up a virtual environment for this project. The following commands will create a virtual environment and install the required dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

In some cases, it might be necessary to set the `PYTHONPATH` environment variable to the root of this repository. This can be done by running the following command:

```bash
export PYTHONPATH=.
```

## Example Scripts

The `src/examples/` directory contains a number of example scripts. They showcase the core functionalities of this repository and offer an entry point into the codebase. For more detailed insights and edge cases, check out the test suite (below). The following scripts are currently available:

- `src/examples/plot_environment.py`: Create an environment with obstacles and plot it using `matplotlib`. Besides the geometric illustration, the scripts also showcases the usage of temporal parameters and querying.
- `src/examples/plot_random_environment.py`: Plots an automatically generated environment with random color-coded static and dynamic obstacles.
- `src/examples/plot_environment_instance`: Script to illustrate the obstacles (static and dynamic) in an environment instance.
- `src/examples/plot_geometries.py`: Simple script, creating geometric obstacles and using their member functions to illustrate them.
- `src/examples/plot_graph.py`: Generates a graph from random samples in an environment with randomly generated obstacles (both static and dynamic).

A separate GitHub action is run on every push or pull_request. The status of the latest run can be seen in the badge at the top of this README.

## Test Suite

After installing the pytest package, the test suite can be run with the following command:

```bash
pytest src/test/
```

A corresponding GitHub action is also available. The status of the latest run can be seen in the badge at the top of this README.
