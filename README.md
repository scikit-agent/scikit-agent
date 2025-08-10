# scikit-agent

[![Actions Status][actions-badge]][actions-link]
[![Documentation Status][rtd-badge]][rtd-link]

[![PyPI version][pypi-version]][pypi-link]
[![Conda-Forge][conda-badge]][conda-link]
[![PyPI platforms][pypi-platforms]][pypi-link]

[![GitHub Discussion][github-discussions-badge]][github-discussions-link]

<!-- SPHINX-START -->

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

## Parameter sweep for infinite-horizon models

Use `skagent.simulation.monte_carlo.sweep` to generate a table of ergodic
moments across a parameter grid Θ for an infinite-horizon model `M(θ)`.

Example (consumption–savings benchmark D-3):

```python
from skagent.models.benchmarks import d3_block, d3_calibration
from skagent.simulation.monte_carlo import sweep

H = sweep(
    block=d3_block,
    base_calibration=d3_calibration,
    dr={"c": lambda m: 0.3 * m},
    initial={"a": 0.5},
    param_grid={"DiscFac": [0.94, 0.96], "CRRA": [1.5, 2.0]},
    agent_count=1000,
    T_sim=5000,
    burn_in=0.5,
    variables=["a", "c", "m", "u"],
)

print(H.head())  # Columns: parameters + <var>_<stat> moments
```

- `param_grid`: dict-of-lists or `skagent.grid.Grid`
- Moments are computed over post–burn-in samples from Monte Carlo histories.
- Output is a pandas DataFrame with one row per θ.
