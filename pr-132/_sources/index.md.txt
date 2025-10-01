# scikit-agent

**A scikit-learn compatible toolkit for agent-based economic modeling.**

scikit-agent provides a unified, pythonic interface for creating, solving, and simulating
economic models using modern computational methods including neural networks, value function
iteration, and Monte Carlo simulation.

```{toctree}
:maxdepth: 2
:hidden:

user_guide/index
auto_examples/index
api/index
```

```{include} ../README.md
:start-after: <!-- SPHINX-START -->
```

## Why scikit-agent?

### 🧱 **Modular Design**

Build complex economic models using intuitive building blocks. Compose **DBlocks** and **RBlocks**
to represent different stages of economic behavior with clear separation of shocks, dynamics,
controls, and rewards.

### 🔬 **Multiple Solution Methods**

Choose from traditional methods like value function iteration or modern neural network approaches.
All algorithms follow consistent APIs and integrate seamlessly with the model framework.

### 📊 **Rich Simulation Engine**

Generate synthetic data with sophisticated Monte Carlo simulators that handle agent heterogeneity,
life-cycle dynamics, and complex stochastic processes.

### 🐍 **Scikit-Learn Compatible**

Familiar fit-predict patterns, consistent APIs, and integration with the broader Python scientific ecosystem.

## Quick Example

```python
import skagent as ska
from skagent.models.consumer import consumption_block, calibration

# Create a consumption-saving model
model = consumption_block
model.construct_shocks(calibration)

# Define simple decision rule
decision_rules = {"c": lambda m: 0.9 * m}

# Run simulation
simulator = ska.MonteCarloSimulator(
    calibration=calibration,
    block=model,
    dr=decision_rules,
    initial={"k": 1.0, "p": 1.0},
    agent_count=1000,
    T_sim=50,
)

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

:::{grid-item-card} 🧠 Models Guide
:link: user_guide/models
:link-type: doc

Learn to build custom models using DBlocks and economic building blocks
:::

:::{grid-item-card} ⚡ Algorithms Guide
:link: user_guide/algorithms
:link-type: doc

Explore solution methods from VFI to neural networks
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

## What Can You Build?

### **Consumption-Saving Models**

- Life-cycle consumption with income uncertainty
- Precautionary saving with borrowing constraints
- Habit formation and durable goods

### **Portfolio Choice Models**

- Mean-variance optimization with labor income
- Multi-asset allocation problems
- Portfolio constraints and transaction costs

### **General Equilibrium Models**

- Overlapping generations models
- Heterogeneous agent models with aggregate shocks
- Market clearing and equilibrium computation

### **Policy Analysis**

- Tax and transfer policy evaluation
- Social Security and pension reform
- Regulatory impact assessment

## Key Features

✅ **Block-based model architecture** for composable, reusable components
✅ **String-based equation parsing** with automatic differentiation support
✅ **Multiple solution algorithms** (VFI, neural networks, policy iteration)
✅ **Sophisticated simulation engine** with aging, mortality, and heterogeneity
✅ **Rich visualization tools** and analysis functions
✅ **Extensive documentation** with examples and tutorials
✅ **Active development** with growing community

## Installation

Install the latest stable version:

```bash
pip install scikit-agent
```

For development installation:

```bash
git clone https://github.com/scikit-agent/scikit-agent.git
cd scikit-agent
pip install -e ".[dev,docs]"
```

## Next Steps

New to scikit-agent? Start with the {doc}`user_guide/quickstart` guide.

Want to dive deeper? Check out:

- {doc}`user_guide/models` for model building concepts
- {doc}`auto_examples/index` for complete working examples
- {doc}`api/index` for detailed API documentation

## Community & Support

- **GitHub Repository**: [github.com/scikit-agent/scikit-agent](https://github.com/scikit-agent/scikit-agent)
- **Issue Tracker**: Report bugs and request features
- **Discussions**: Ask questions and share examples
- **Contributing**: Help improve scikit-agent for everyone

---

## Indices and tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
