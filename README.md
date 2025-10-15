# scikit-agent

[![Actions Status][actions-badge]][actions-link]
[![Documentation Status][rtd-badge]][rtd-link]

[![PyPI version][pypi-version]][pypi-link]
[![Conda-Forge][conda-badge]][conda-link]
[![PyPI platforms][pypi-platforms]][pypi-link]

[![GitHub Discussion][github-discussions-badge]][github-discussions-link]

<!-- SPHINX-START -->

**scikit-agent** is a scientific Python toolkit for agent-based economic
modeling and multi-agent systems design. It provides a unified interface for
creating, solving, and simulating economic models using modern computational
methods including deep reinforcement learning as well as more traditional
numerical techniques.

Our goal is for `scikit-agent` to be for computational social scientific
modeling and statistics what `scikit-learn` is for machine learning.

## Key Features

- **Built on Scientific Python and Torch** for easy integration with the Python
  ecosystem
- **Modular modeling system**. Construct multi-agent environments from modular
  blocks of structural equations.
- **Solution algorithms** including deep reinforcement learning methods.
- **Simulation tools** for generating synthetic data and running policy
  experiments

## Installation

```bash
pip install scikit-agent
```

For development installation:

```bash
git clone https://github.com/scikit-agent/scikit-agent.git
cd scikit-agent
pip install -e ".[dev,docs]"
```

<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/scikit-agent/scikit-agent/workflows/CI/badge.svg
[actions-link]:             https://github.com/scikit-agent/scikit-agent/actions
[conda-badge]:              https://img.shields.io/conda/vn/conda-forge/scikit-agent
[conda-link]:               https://github.com/conda-forge/scikit-agent-feedstock
[github-discussions-badge]: https://img.shields.io/static/v1?label=Discussions&message=Ask&color=blue&logo=github
[github-discussions-link]:  https://github.com/scikit-agent/scikit-agent/discussions
[pypi-link]:                https://pypi.org/project/scikit-agent/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/scikit-agent
[pypi-version]:             https://img.shields.io/pypi/v/scikit-agent
[rtd-badge]:                https://readthedocs.org/projects/scikit-agent/badge/?version=latest
[rtd-link]:                 https://scikit-agent.readthedocs.io/en/latest/?badge=latest

<!-- prettier-ignore-end -->
