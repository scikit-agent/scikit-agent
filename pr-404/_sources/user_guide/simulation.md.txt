# Simulation Guide

This guide covers how to run simulations and analyze results with scikit-agent.

While it may support more simulation engines in the future, scikit-agent
currently supports generic Monte Carlo simulation of its models.

The simulator accepts as arguments a calibration dictionary, a model, and
decision rules for all agent decisions. It also takes initial state values, a
number of independent trajectories to simulate, and a number of time steps to
simulate through.

The simulator then runs through the model equations, sampling random variables
and applying transition and decision rules. It generates a complete history of
all variables in the simulation, that can then be inspected.

## Simulation Configuration

The simulator takes the following elements in configuration:

- **Calibration dictionary**. A dictionary specifying values for any free
  parameters of the model. If the model declares entity classes, this also gives
  how many instances each class has, under a key equal to the entity's name.
- **Model**. A block model (`DBlock` or `RBlock`), defining states, shocks,
  control, and reward variables of an agent (or population of agents). See
  {doc}`blocks` on entities for how a model declares a population rather than a
  single agent.
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
    sample_count=5,
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

A variable's history carries the period axis first, then the replication axis,
then one axis per entity class it is an attribute of:

| the variable is                               | its history is shaped         |
| --------------------------------------------- | ----------------------------- |
| axis-free, or in a model with no entities     | `(T_sim, sample_count)`       |
| an attribute of a class with `size` instances | `(T_sim, sample_count, size)` |

So a model that declares no entity keeps the two-dimensional histories it has
always had, and `axis=1` is the replication axis in every case.

The following will show the mean across all trajectories, at each time step, for
the `k` variable.

```python
import matplotlib.pyplot as plt

plt.plot(simulator.history["k"].mean(axis=1))
plt.show()
```

For a per-instance variable, averaging over `axis=1` leaves one series per
instance, while averaging over both the replication and instance axes gives the
population mean at each time step:

```python
# one series per firm, averaged over trajectories
simulator.history["q"].mean(axis=1)

# the average across firms and trajectories, at each time step
simulator.history["q"].mean(axis=(1, 2))
```

Two checks run every period, and both raise rather than letting a bad value
travel. An equation declared outside every entity class that returns an array
over one has failed to aggregate, and is reported where it happens. And a
non-finite value entering a reduction is refused, because one instance's bad
draw would otherwise become every instance's aggregate for the rest of the run.
