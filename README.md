# Time-Aware Probabilistic Roadmaps

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

The `src/examples/` directory contains a number of example scripts. They showcase the core functionalities of this repository and offer an entry point into the codebase. For more detailed insights and edge cases, check out the test suite (below). Demonstration scripts for the following topics are currently available:

- Creation and plotting of defined or random `Environment` and `EnvironmentInstance` objects with static and dynamic obstacles, using `matplotlib`.
- Creation and plotting of `Geometry` objects, which can be of types `Point`, `Line` or `Polygon`.
- Creation and plotting of `Graph` objects, which can be generated from random samples in an environment.
- Example scenarios for the usage of the `TimeAwarePRM` class, which implements the TA-PRM algorithm.

A separate GitHub action is run on every push or pull_request. The status of the latest run can be seen in the badge at the top of this README.

## Simulation Videos

The `simulate`-function of the graph class, allows to simulate a solution path in a time-varying environment and either illustrate the result using `matplotlib` or save it as a video. While the video is initially save in the `avi`-format, this results in a very large file size. A subsequent conversion to the more compressed `mp4`-format is implemented through the `ffmpeg`-tool. To install it on MacOS, run the following command:

```bash
brew install ffmpeg
```

For the installation of `ffmpeg` on other operating systems, please refer to the [official documentation](https://ffmpeg.org/).

## Test Suite

After installing the pytest package, the test suite can be run with the following command:

```bash
pytest src/test/
```

A corresponding GitHub action is also available. The status of the latest run can be seen in the badge at the top of this README.
