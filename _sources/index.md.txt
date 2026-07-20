# scikit-agent

```{toctree}
:maxdepth: 2
:hidden:

user_guide/index
auto_examples/index
api/index
community/index
ecosystem/index
```

```{include} ../README.md
:start-after: <!-- SPHINX-START -->
```

## Quick Example

```python
import skagent as ska
from skagent.models.consumer import cons_problem, calibration

# A consumption-saving model: a consumption block chained with a
# tick block that carries end-of-period assets into next period's
# capital. The simulator constructs the shock distributions from
# the calibration internally.
model = cons_problem

# Define simple decision rule
decision_rules = {"c": lambda m: 0.9 * m}

# Run simulation
simulator = ska.MonteCarloSimulator(
    calibration=calibration,
    block=model,
    dr=decision_rules,
    initial={"k": 1.0},
    agent_count=1000,
    T_sim=50,
)

simulator.initialize_sim()
results = simulator.simulate()
```

## Quick Links

::::{grid} 1 1 2 3
:gutter: 3

:::{grid-item-card} 🚀 Quickstart Guide
:link: user_guide/quickstart
:link-type: doc

Get up and running in minutes with your first economic model
:::

:::{grid-item-card} 🧠 Blocks Guide
:link: user_guide/blocks
:link-type: doc

Learn to build custom models using DBlocks and economic building blocks
:::

:::{grid-item-card} ⚡ Algorithms Guide
:link: user_guide/algorithms
:link-type: doc

Explore solution methods from value function iteration to neural networks
:::

:::{grid-item-card} 📈 Simulation Guide
:link: user_guide/simulation
:link-type: doc

Master Monte Carlo simulation and result analysis
:::

:::{grid-item-card} 🔬 Examples Gallery
:link: auto_examples/index
:link-type: doc

Browse complete working examples and use cases
:::

:::{grid-item-card} 📚 API Reference
:link: api/index
:link-type: doc

Detailed documentation of all classes and functions
:::

::::

## Next Steps

New to scikit-agent? Start with the {doc}`user_guide/quickstart` guide.

Want to dive deeper? Check out:

- {doc}`user_guide/blocks` for model building concepts
- {doc}`auto_examples/index` for complete working examples
- {doc}`api/index` for detailed API documentation

## Community & Support

- **GitHub Repository**: [github.com/scikit-agent/scikit-agent](https://github.com/scikit-agent/scikit-agent)
