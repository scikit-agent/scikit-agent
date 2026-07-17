# Quickstart Guide

Get up and running with scikit-agent in minutes. This guide takes you through
the full loop: build a model, simulate it, and then **solve** it for an optimal
policy.

## Installation

```bash
pip install scikit-agent
```

See the {doc}`installation` guide for development installs and system
dependencies.

## Core Concepts

scikit-agent is built around a few key objects:

- **`DBlock` (Dynamic Block)** — one period of model behavior, defined by its
  **shocks** (random variables), **dynamics** (transition equations),
  **controls** (decision variables agents choose), and **rewards** (what agents
  maximize).
- **`RBlock` (Recursive Block)** — several blocks chained together, e.g. to
  carry state from one period into the next.
- **`BellmanPeriod`** — wraps a block with its discount variable and
  calibration, turning it into one period of a dynamic optimization problem.
  This is the object the solvers consume.
- **`Grid`** — a discretized set of states to solve or evaluate a model over.
- **`MonteCarloSimulator`** — generates synthetic panel data from a model and a
  set of decision rules.

## Simulate a Prebuilt Model

Let's start with the consumption-saving model that ships with scikit-agent.

### Step 1: Import scikit-agent

```python
import numpy as np
import skagent as ska
from skagent.models.consumer import (
    consumption_block_normalized,
    cons_problem,
    calibration,
)
```

### Step 2: Examine the Model

`consumption_block_normalized` describes a single period of a normalized
consumption-saving problem — market resources `m` arrive, the agent chooses
consumption `c`, and end-of-period assets `a` remain:

```python
print("Dynamics:")
for var, eq in consumption_block_normalized.dynamics.items():
    print(f"  {var}: {eq}")

print("Shocks:", list(consumption_block_normalized.shocks.keys()))
```

A single `DBlock` is only one period. To simulate over time, end-of-period
assets `a` must become next period's capital `k`. The prebuilt `cons_problem` is
an `RBlock` that chains the consumption block with a small "tick" block doing
exactly that:

```python
print("Recursive problem blocks:")
for blk in cons_problem.blocks:
    print(f"  {blk.name}: {list(blk.get_dynamics().keys())}")
```

There is no need to construct the shock distributions by hand; the simulator
builds them from the calibration internally.

### Step 3: Set Up Model Parameters

```python
# Start from the default calibration and adjust a few parameters.
my_calibration = calibration.copy()
my_calibration.update(
    {
        "DiscFac": 0.95,  # Discount factor
        "CRRA": 2.5,  # Risk aversion
    }
)
```

### Step 4: Simulate

A **decision rule** maps an agent's information to its choice. For now we supply
a simple hand-picked rule — consume 90% of market resources — just to see the
model run. (The next section replaces it with a rule a solver _finds_.)

```python
simulator = ska.MonteCarloSimulator(
    calibration=my_calibration,
    block=cons_problem,
    dr={"c": lambda m: 0.9 * m},  # decision rule for the control `c`
    initial={"k": 1.0},  # starting capital for every agent
    agent_count=1000,
    T_sim=50,
    seed=42,
)

simulator.initialize_sim()
history = simulator.simulate()

print("History keys:", list(history.keys()))
```

### Step 5: Analyze Results

Each entry of `simulator.history` is a NumPy array of shape
`(T_sim, agent_count)`, so averaging over `axis=1` gives the cross-agent mean at
each period:

```python
import matplotlib.pyplot as plt

mean_consumption = simulator.history["c"].mean(axis=1)

plt.plot(mean_consumption)
plt.title("Average Consumption Over Time")
plt.xlabel("Period")
plt.ylabel("Consumption")
plt.grid(True)
plt.show()
```

## Solve for an Optimal Policy

The rule above was picked by hand. The point of scikit-agent is to _solve_ for
the optimal one. scikit-agent offers several solution methods (see the
{doc}`algorithms` guide); here we use the most direct: train a policy network to
maximize the reward earned within a block.

To see the machinery clearly, we use a deliberately tiny block whose optimal
policy we already know. The agent chooses a control `c` to match a random shock
`theta`, with reward `-(theta - c)**2`, so the optimum is simply `c = theta`.
Recovering that confirms the training loop works before you point it at a model
whose answer you _don't_ know.

### Step 1: Define the block and wrap it

```python
import torch
import skagent.ann as ann
import skagent.bellman as bellman
import skagent.block as block
import skagent.grid as grid
import skagent.loss as loss
from skagent.distributions import Normal

track_calibration = {"beta": 0.9}

track_block = block.DBlock(
    name="track the shock",
    shocks={"theta": (Normal, {"mu": 0, "sigma": 1})},
    dynamics={
        "c": block.Control(["a", "theta"]),  # may condition its choice on theta
        "a": lambda a, c, theta: a - c + theta,
        "u": lambda theta, c: -((theta - c) ** 2),  # maximized at c = theta
    },
    reward={"u": "consumer"},
)

# Wrap the block as one period of an optimization problem, naming its
# discount variable.
bp = bellman.BellmanPeriod(track_block, "beta", track_calibration)
```

### Step 2: Build a grid of starting points

The network is trained over a `Grid` of states (and shock realizations). Each
grid point is one scenario the policy must do well on:

```python
states = grid.Grid.from_config(
    {
        "a": {"min": 0, "max": 1, "count": 7},
        "theta": {"min": -1, "max": 1, "count": 7},
    }
)
```

### Step 3: Train a policy network

`StaticRewardLoss` is the negative of this period's reward, so minimizing it
maximizes reward. `train_block_nn` runs the optimizer:

```python
policy = ska.BlockPolicyNet(bp, width=16)
loss_fn = loss.StaticRewardLoss(bp, track_calibration)
ann.train_block_nn(policy, states, loss_fn, epochs=500)
```

### Step 4: Read off the learned policy

`decision_function` returns the control values the trained network chooses at
each grid point. We compare them against the known optimum `c = theta`:

```python
learned_c = (
    policy.decision_function({"a": states["a"]}, {"theta": states["theta"]}, {})["c"]
    .detach()
    .flatten()
    .cpu()
    .numpy()
)
theta = states["theta"].flatten().cpu().numpy()

print("Max deviation from optimum:", np.max(np.abs(learned_c - theta)))
# a small number: the network has recovered c = theta
```

For a reusable rule you can hand to the simulator, call
`policy.get_decision_rule()`.

## Next Steps

- **Solve real economic models**: the {doc}`algorithms` guide covers value
  backwards induction, the Maliar deep-learning method, and reinforcement
  learning, with runnable versions in the
  {doc}`../auto_examples/algorithms/index`.
- **Build custom models**: the {doc}`blocks` guide explains blocks, controls,
  and how to compose them.
- **Constrained problems**: the {doc}`constraints` guide covers borrowing limits
  and other bounds.
- **Simulation in depth**: the {doc}`simulation` guide covers analysis and
  visualization.

## Key Takeaways

- **Blocks** (`DBlock`, `RBlock`) are the building units of economic models.
- **Decision rules** map an agent's information to its choices; a **solver**
  finds the rule that is optimal.
- **`BellmanPeriod`** turns a block into a period a solver can optimize.
- **`Grid`s** discretize the states to solve or evaluate over.
- **Simulators** generate synthetic data from a model and its decision rules.

You're now ready to build and solve more sophisticated models with scikit-agent!
