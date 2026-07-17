# scikit-agent

[![Actions Status][actions-badge]][actions-link]
[![Documentation][docs-badge]][docs-link]
[![PyPI version][pypi-version]][pypi-link]
[![PyPI platforms][pypi-platforms]][pypi-link]
[![GitHub Discussion][github-discussions-badge]][github-discussions-link]

A scientific Python toolkit for agent-based economic modeling: build models from
modular blocks, solve them with deep learning or classical numerical methods,
and simulate.

📖 **[Documentation][docs-link]** &nbsp;·&nbsp;
[Quickstart](https://scikit-agent.github.io/scikit-agent/user_guide/quickstart.html)
&nbsp;·&nbsp;
[Examples](https://scikit-agent.github.io/scikit-agent/auto_examples/index.html)
&nbsp;·&nbsp;
[API Reference](https://scikit-agent.github.io/scikit-agent/api/index.html)

## Quick example

```python
import skagent as ska
from skagent.models.consumer import cons_problem, calibration

# `cons_problem` is a prebuilt consumption-saving model. Supply a decision
# rule for the control `c` and simulate a population of agents forward.
simulator = ska.MonteCarloSimulator(
    calibration=calibration,
    block=cons_problem,
    dr={"c": lambda m: 0.9 * m},
    initial={"k": 1.0},
    agent_count=1000,
    T_sim=50,
    seed=42,
)
simulator.initialize_sim()
history = simulator.simulate()
```

The
[Quickstart](https://scikit-agent.github.io/scikit-agent/user_guide/quickstart.html)
goes further, _solving_ the model for an optimal policy instead of hand-coding a
rule.

<!-- SPHINX-START -->

**scikit-agent** is a scientific Python toolkit for agent-based economic
modeling and multi-agent systems design. It provides a unified interface for
creating, solving, and simulating economic models using modern computational
methods — including deep learning — alongside more traditional numerical
techniques.

Our goal is for `scikit-agent` to be for computational social science what
`scikit-learn` is for machine learning.

## Key Features

- 🧱 **Modular modeling system.** Construct multi-agent environments from
  composable blocks of structural equations.
- ⚡ **Solution algorithms.** Solve models with deep-learning methods (following
  Maliar, Maliar, and Winant, 2021), value backwards induction, and
  reinforcement learning via
  [Stable-Baselines3](https://stable-baselines3.readthedocs.io/).
- 📊 **Simulation tools.** Generate synthetic data and run policy experiments
  with a Monte Carlo engine.
- 🐍 **Built on Scientific Python and PyTorch** for easy integration with the
  wider Python ecosystem.

## Installation

```bash
pip install scikit-agent
```

For a development installation:

```bash
git clone https://github.com/scikit-agent/scikit-agent.git
cd scikit-agent
pip install -e ".[dev,docs]"
```

See the [documentation][docs-link] for the user guide, a gallery of runnable
examples, and the full API reference.

<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/scikit-agent/scikit-agent/workflows/CI/badge.svg
[actions-link]:             https://github.com/scikit-agent/scikit-agent/actions
[docs-badge]:               https://img.shields.io/badge/docs-latest-blue
[docs-link]:                https://scikit-agent.github.io/scikit-agent/
[github-discussions-badge]: https://img.shields.io/static/v1?label=Discussions&message=Ask&color=blue&logo=github
[github-discussions-link]:  https://github.com/scikit-agent/scikit-agent/discussions
[pypi-link]:                https://pypi.org/project/scikit-agent/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/scikit-agent
[pypi-version]:             https://img.shields.io/pypi/v/scikit-agent

<!-- prettier-ignore-end -->
