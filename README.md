# scikit-agent

[![Actions Status][actions-badge]][actions-link]
[![Documentation Status][rtd-badge]][rtd-link]

[![PyPI version][pypi-version]][pypi-link]
[![Conda-Forge][conda-badge]][conda-link]
[![PyPI platforms][pypi-platforms]][pypi-link]

[![GitHub Discussion][github-discussions-badge]][github-discussions-link]

<!-- SPHINX-START -->

**scikit-agent** is a scikit-learn compatible toolkit for agent-based economic
modeling. It provides a unified interface for creating, solving, and simulating
economic models using modern computational methods including reinforcement
learning, neural networks, and traditional numerical techniques.

## Key Features

- **Scikit-learn compatible API** for easy integration with the Python
  scientific ecosystem
- **Economic model classes** for consumption-savings, portfolio choice, and
  other standard models
- **Multiple solution algorithms** including value function iteration, policy
  iteration, and neural network approaches
- **Simulation tools** for generating synthetic data and running policy
  experiments
- **Comprehensive documentation** with examples and tutorials

## Quick Start

```python
import skagent

# Create a basic consumption-savings model
model = skagent.models.ConsumptionSavingsModel(
    periods=50, discount_factor=0.95, risk_aversion=2.0
)

# Solve the model
model.fit()

# Run simulations
results = model.simulate(n_agents=1000, n_periods=50)
```

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
