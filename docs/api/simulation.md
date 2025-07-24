# Simulation

This section contains the API documentation for simulation tools and analysis
functions.

## Monte Carlo Simulation

The simulation module provides Monte Carlo simulation engines for economic
models.

### AgentTypeMonteCarloSimulator

A Monte Carlo simulation engine based on the HARK.core.AgentType framework. This
simulator includes aging and mortality assumptions.

```{eval-rst}
.. autoclass:: skagent.simulation.monte_carlo.AgentTypeMonteCarloSimulator
   :members:
   :undoc-members:
   :show-inheritance:
```

### MonteCarloSimulator

A simplified Monte Carlo simulation engine that doesn't make assumptions about
aging or mortality.

```{eval-rst}
.. autoclass:: skagent.simulation.monte_carlo.MonteCarloSimulator
   :members:
   :undoc-members:
   :show-inheritance:
```

### Base Simulator

```{eval-rst}
.. autoclass:: skagent.simulation.monte_carlo.Simulator
   :members:
   :undoc-members:
   :show-inheritance:
```

## Simulation Utility Functions

### Drawing Shocks

```{eval-rst}
.. autofunction:: skagent.simulation.monte_carlo.draw_shocks
```

### Age-Varying Calibration

```{eval-rst}
.. autofunction:: skagent.simulation.monte_carlo.calibration_by_age
```

## Example Usage

### Basic Simulation

```python
import skagent as ska
from skagent.models.consumer import consumption_block, calibration

# Set up decision rules (simplified example)
dr = {"c": lambda m: 0.9 * m}

# Initialize simulation
simulator = ska.MonteCarloSimulator(
    calibration=calibration,
    block=consumption_block,
    dr=dr,
    initial={"k": 1.0, "p": 1.0},
    agent_count=1000,
    T_sim=50,
)

# Run simulation
results = simulator.simulate()
```

### Agent-Type Simulation with Aging

```python
# For models with aging and mortality
simulator = ska.AgentTypeMonteCarloSimulator(
    calibration=calibration,
    block=consumption_block,
    dr=dr,
    initial={"k": 1.0, "p": 1.0},
    agent_count=1000,
    T_sim=50,
)

results = simulator.simulate()
```

---

_This page is under construction. Content will be added as the API develops._
