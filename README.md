# Time-Aware PRM

[![Python Testing](https://github.com/sjschlapbach/ta-prm/actions/workflows/python_testing.yml/badge.svg)](https://github.com/sjschlapbach/ta-prm/actions/workflows/python_testing.yml)

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
