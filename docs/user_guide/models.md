# Models

This section describes how to build and work with economic models in scikit-agent.

## Block-based modeling

scikit-agent uses a block-based approach to model construction, where complex economic models are built by combining simpler building blocks.

### Dynamic Blocks (DBlock)

Dynamic blocks define how state variables evolve over time. They specify:

- **Shocks**: Random variables that affect the system
- **Dynamics**: Equations governing state transitions
- **Parameters**: Calibrated values used in computations

Example from the consumer model:

```python
from skagent.models.consumer import consumption_block

print(f"Shocks: {list(consumption_block.shocks.keys())}")
print(f"Dynamics: {list(consumption_block.dynamics.keys())}")
```

### Reward Blocks (RBlock)

Reward blocks define optimization objectives, typically utility functions or profit functions.

### Control Variables

Control variables represent choices made by economic agents, such as consumption, investment, or labor supply decisions.

## Pre-built models

scikit-agent includes several pre-built models:

### Consumer model

A standard consumption-savings model with:
- CRRA utility
- Income uncertainty
- Mortality risk
- Borrowing constraints

### Benchmarks

Various benchmark models for testing and comparison purposes.

## Building custom models

You can create custom models by defining your own blocks:

```python
from skagent.model import DBlock
from skagent.distributions import Normal

# Define a simple growth model
growth_block = DBlock(
    name="growth",
    shocks={
        "productivity": (Normal, {"mu": 0, "sigma": 0.1})
    },
    dynamics={
        "output": "capital ** alpha * productivity",
        "investment": "savings_rate * output",
        "capital_next": "(1 - depreciation) * capital + investment"
    }
)
```