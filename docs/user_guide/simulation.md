# Simulation Guide

This guide covers how to run simulations and analyze results with scikit-agent.

While it may support more simulation engines in the future, scikit-agent
currently supports generic Monte Carlo simulation of its models.

The simulator accepts as arguments a calibration dictionary, a model, and
decision rules for all agent decisions. It also takes initial state values, a
number of agents to simulate, and a number of time steps to simulate through.

The simulator then runs through the model equations, sampling random variables
and applying transition and decision rules. It generates a complete history of
all variables in the simulation, that can then be inspected.

## Simulation Configuration

The simulator takes the following elements in configuration:

- **Calibration dictionary**. A dictionary specifying values for any free
  parameters of the model.
- **Model**. A block model (`DBlock` or `RBlock`), defining states, shocks,
  control, and reward variables of an agent (or population of agents).
- **Decision rules**. A dictionary of decision rules governing the informed
  choices of agents at their decision variables.
- **Initial values**. A dictionary of starting values, or starting
  distributions, for arrival state variables.

This provides all the data needed for the simulation to run forward.

```python
from skagent.distributions import Lognormal
import skagent.models.consumer as cons
from skagent.simulation.monte_carlo import MonteCarloSimulator

simulator = MonteCarloSimulator(
    calibration=cons.calibration,
    block=cons.cons_problem,
    dr={  # decision rules passed in as dictionary
        "c": lambda m: 0.5 * m,
    },
    initial={  # distributions of starting values, in levels
        "k": Lognormal(1.0, 0.5),
    },
    agent_count=5,
    T_sim=10,
)
```

## Running Simulations

Running the simulation is a simple two-step process:

```python
simulator.initialize_sim()
simulator.simulate()
```

## Analyzing and Visualizing Data

The data for all variables of the simulation is made available as a dictionary
of arrays.

The following will show the mean across all agents, at each time step, for the
`k` variable (the history array is shaped `(T_sim, agent_count)`, so `axis=1`
averages over agents).

```python
import matplotlib.pyplot as plt

plt.plot(simulator.history["k"].mean(axis=1))
plt.show()
```
