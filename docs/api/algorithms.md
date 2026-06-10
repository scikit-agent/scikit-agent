# Algorithms

This section contains the API documentation for solution algorithms and
optimization methods used to solve economic models.

## Value Function Iteration

The Value Function Iteration (VBI) algorithm implements backwards induction to
derive value functions from model blocks.

```{eval-rst}
.. automodule:: skagent.algos.vbi
   :members:
```

### Core VBI Functions

```{eval-rst}
.. autofunction:: skagent.algos.vbi.solve
   :no-index:
```

```{eval-rst}
.. autofunction:: skagent.algos.vbi.get_action_rule
   :no-index:
```

```{eval-rst}
.. autofunction:: skagent.algos.vbi.ar_from_data
   :no-index:
```

```{eval-rst}
.. autofunction:: skagent.algos.vbi.grid_to_data_array
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

### Training Functions

```{eval-rst}
.. autofunction:: skagent.ann.aggregate_net_loss
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

---

_This page is under construction. Content will be added as the API develops._
