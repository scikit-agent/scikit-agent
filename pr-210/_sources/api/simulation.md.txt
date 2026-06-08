# Simulation

This section contains the API documentation for simulation tools and analysis
functions.

## Monte Carlo Simulation

The simulation module provides Monte Carlo simulation engines for economic
models.

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
simulator.initialize_sim()
results = simulator.simulate()
```

### Simulation with Mortality

Mortality is expressed declaratively as a block rather than baked into the
simulator. The `mortality_block` in `skagent.models.consumer` resets an agent's
state to a newborn draw whenever the `live` shock is zero, so the base
`MonteCarloSimulator` is sufficient:

```python
import skagent as ska
from skagent.distributions import Lognormal
from skagent.models.consumer import calibration, mortal_cons_problem

simulator = ska.MonteCarloSimulator(
    calibration=calibration,
    block=mortal_cons_problem,
    dr={"c": lambda m: m / 3},
    initial={"k": Lognormal(-6, 0), "p": 1.0, "age": 0},
    agent_count=1000,
    T_sim=50,
)

simulator.initialize_sim()
results = simulator.simulate()
```

---

_This page is under construction. Content will be added as the API develops._
