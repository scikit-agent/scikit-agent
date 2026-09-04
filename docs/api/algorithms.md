# Algorithms

This section contains the API documentation for solution algorithms, neural
network components, and grid tools used to solve dynamic stochastic optimization
problems.

## Value Function Iteration (VFI)

The value function iteration (VFI) algorithm derives arrival value functions
from a continuation value function and the stage dynamics of model blocks.

```{eval-rst}
.. automodule:: skagent.algos.vfi
   :members:
```

### Core VFI Functions

```{eval-rst}
.. autofunction:: skagent.algos.vfi.solve_step
   :no-index:
```

```{eval-rst}
.. autofunction:: skagent.algos.vfi.solve_bellman
   :no-index:
```

```{eval-rst}
.. autofunction:: skagent.algos.vfi.get_action_rule
   :no-index:
```

```{eval-rst}
.. autofunction:: skagent.algos.vfi.ar_from_data
   :no-index:
```

```{eval-rst}
.. autofunction:: skagent.algos.vfi.grid_to_data_array
   :no-index:
```

## Tabular Best Response

The tabular best-response solver solves one decision at a time from a tabulated
payoff table. For every value of what the decision-maker observes, it reports
the action that maximizes the payoff of the agent making that decision,
conditional on the observation and on a supplied rule for every other decision.
The order in which the decisions are solved belongs to a schedule rather than to
the method. {func}`skagent.solver.solve_in_relevance_order` supplies one such
order, which it takes from the block's relevance graph (see {doc}`analysis`),
and it raises an error when that graph has a cyclic component, because the
decisions in a cycle have to be solved together as a simultaneous-move
equilibrium.

```{eval-rst}
.. automodule:: skagent.algos.tabular
   :members:
```

### Core Best-Response Classes

```{eval-rst}
.. autoclass:: skagent.algos.tabular.TabularBestResponseSolver
   :members:
   :no-index:
```

```{eval-rst}
.. autoclass:: skagent.algos.tabular.TabulatedRule
   :members:
   :no-index:
```

## Maliar-Style Algorithms

Neural network-based solution methods following Maliar et al.

```{eval-rst}
.. automodule:: skagent.algos.maliar
   :members:
```

## Reinforcement Learning (Stable-Baselines3)

Proximal Policy Optimization (PPO) for `BellmanPeriod` models, via a
[Stable-Baselines3](https://stable-baselines3.readthedocs.io/) backend. The
agent wraps a model in a gymnasium environment (see {doc}`environments`), trains
PPO, and emits a standard skagent decision rule.

```{eval-rst}
.. automodule:: skagent.algos.sb3
   :members:
```

## Loss Functions

Objective functions passed to {func}`skagent.ann.train_block_nn`. The
reward-based losses ({class}`~skagent.loss.StaticRewardLoss`,
{class}`~skagent.loss.EstimatedDiscountedLifetimeRewardLoss`) solve a block
directly for the non-recurring case; the equation-residual losses
({class}`~skagent.loss.BellmanEquationLoss`,
{class}`~skagent.loss.EulerEquationLoss`) target the recurring, dynamic case.
See {doc}`loss` for the full reference.

## Neural Network Components

### Net

Base neural network class with device management.

```{eval-rst}
.. autoclass:: skagent.ann.Net
   :members:
   :undoc-members:
   :show-inheritance:
```

### BlockPolicyNet

Specialized neural network for policy functions in economic models.

```{eval-rst}
.. autoclass:: skagent.ann.BlockPolicyNet
   :members:
   :undoc-members:
   :show-inheritance:
```

### BlockValueNet

A neural network for value functions in dynamic programming problems.

```{eval-rst}
.. autoclass:: skagent.ann.BlockValueNet
   :members:
   :undoc-members:
   :show-inheritance:
```

### BlockPolicyValueNet

A shared-backbone neural network that jointly represents the policy and value
functions.

```{eval-rst}
.. autoclass:: skagent.ann.BlockPolicyValueNet
   :members:
   :undoc-members:
   :show-inheritance:
```

### Training Functions

```{eval-rst}
.. autofunction:: skagent.ann.train_block_nn
```

```{eval-rst}
.. autofunction:: skagent.ann.aggregate_net_loss
```

## Equilibrium Schedules and Methods

A schedule decides when each decision is solved, and it asks a method to carry
out each solve. A method carries one algorithm together with the configuration
that algorithm needs. The pairing is free: the same schedule accepts either a
policy network or an exact backup, and it returns the same answer in either
case.

```{eval-rst}
.. autofunction:: skagent.solver.solve_in_order
```

```{eval-rst}
.. autofunction:: skagent.solver.solve_in_relevance_order
```

```{eval-rst}
.. autofunction:: skagent.solver.project
```

```{eval-rst}
.. autofunction:: skagent.solver.solve_symmetric_equilibrium
```

```{eval-rst}
.. autoclass:: skagent.solver.NeuralBestResponse
   :members:
```

```{eval-rst}
.. autoclass:: skagent.solver.ExactBestResponse
   :members:
```

## Grid and Computational Tools

### Grid Class

```{eval-rst}
.. autoclass:: skagent.grid.Grid
   :members:
   :undoc-members:
   :show-inheritance:
```

### Grid Utility Functions

```{eval-rst}
.. autofunction:: skagent.grid.make_grid
```

```{eval-rst}
.. autofunction:: skagent.grid.cartesian_product
```
