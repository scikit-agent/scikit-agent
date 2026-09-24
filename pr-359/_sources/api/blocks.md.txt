# Blocks

This section contains the API documentation for model building blocks.

## Block Classes

### Block

Base class for all blocks. Provides shared analysis methods, including
strategic-relevance analysis (`relevance_graph`, `relies_on`).

```{eval-rst}
.. autoclass:: skagent.block.Block
   :members:
   :undoc-members:
   :show-inheritance:
```

### DBlock

```{eval-rst}
.. autoclass:: skagent.block.DBlock
   :members:
   :undoc-members:
   :show-inheritance:
```

### RBlock

```{eval-rst}
.. autoclass:: skagent.block.RBlock
   :members:
   :undoc-members:
   :show-inheritance:
```

### GroundedBlock

A block together with the calibration and generator it is read against. A block
is calibration-independent, so a block alone does not fix its shocks; this pair
does, and is what solvers, simulators and environments are handed.

```{eval-rst}
.. autoclass:: skagent.ground.GroundedBlock
   :members:
   :undoc-members:
   :show-inheritance:
```

### Measures

What a payoff expectation is reduced over. A profile's worth is asked for with
one of these, since a sampled estimate and one taken over a discretization carry
different error and a caller holding a number to a tolerance has to know which
it has.

```{eval-rst}
.. autoclass:: skagent.ground.Measure
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: skagent.ground.Sampled
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: skagent.ground.Discretized
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: skagent.ground.ExpectedPayoff
   :members:
   :undoc-members:
   :show-inheritance:
```

### Control

```{eval-rst}
.. autoclass:: skagent.block.Control
   :members:
   :undoc-members:
   :show-inheritance:
```

### Aggregate

```{eval-rst}
.. autoclass:: skagent.block.Aggregate
   :members:
   :undoc-members:
   :show-inheritance:
```

### Entity

```{eval-rst}
.. autoclass:: skagent.block.Entity
   :members:
   :undoc-members:
   :show-inheritance:
```

## Model Utilities

### Simulation Dynamics

```{eval-rst}
.. autofunction:: skagent.block.simulate_dynamics
```

### Shock Construction

```{eval-rst}
.. autofunction:: skagent.block.construct_shocks
```

### Discretized Shock Distribution

```{eval-rst}
.. autofunction:: skagent.block.discretized_shock_dstn
```
