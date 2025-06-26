# Models Guide

This guide explains how to work with economic models in scikit-agent. You'll
learn about the core modeling concepts, how to build custom models, and how to
use predefined models.

## Understanding Model Structure

### The Block Architecture

scikit-agent uses a "block" architecture where economic models are composed of
building blocks:

- **DBlock (Dynamic Block)**: Represents a stage or period of model behavior
- **RBlock (Recursive Block)**: Combines multiple blocks into a multi-period
  model
- **Control**: Represents decision variables that agents optimize
- **Aggregate**: Represents aggregate (economy-wide) shocks

### DBlock Components

A DBlock has four main components:

```python
import skagent as ska
from HARK.distributions import MeanOneLogNormal

# Example: Simple consumption block
consumption_block = ska.DBlock(
    name="consumption_stage",
    # 1. Shocks: Random variables
    shocks={"theta": (MeanOneLogNormal, {"sigma": 0.1})},  # Income shock
    # 2. Dynamics: State transition equations
    dynamics={
        "y": lambda p, theta: p * theta,  # Income = permanent * transitory
        "m": lambda b, y: b + y,  # Market resources = beginning + income
        "c": ska.Control(["m"]),  # Consumption (control variable)
        "a": lambda m, c: m - c,  # Assets = resources - consumption
    },
    # 3. Rewards: What agents maximize
    reward={"u": "consumer"},  # Utility goes to consumer agent
)
```

## Building Custom Models

### Step-by-Step Model Creation

Let's build a portfolio choice model from scratch:

#### Step 1: Define the Economic Problem

We want to model an agent who:

1. Receives labor income with shocks
2. Chooses consumption and savings
3. Allocates savings between safe and risky assets
4. Maximizes expected utility

#### Step 2: Set Up Shocks

```python
from HARK.distributions import Lognormal, MeanOneLogNormal

# Define all random variables
shocks = {
    "theta": (MeanOneLogNormal, {"sigma": "TranShkStd"}),  # Transitory income shock
    "psi": (MeanOneLogNormal, {"sigma": "PermShkStd"}),  # Permanent income shock
    "risky_return": (
        Lognormal,
        {"mean": "EqP", "std": "RiskyStd"},
    ),  # Risky asset return
}
```

#### Step 3: Define State Transitions

```python
# Economic dynamics
dynamics = {
    # Income process
    "y": lambda p, theta: p * theta,
    "p": lambda p_prev, psi, PermGroFac: p_prev * psi * PermGroFac,
    # Portfolio returns
    "R_portfolio": lambda alpha, Rfree, risky_return: (
        Rfree + alpha * (risky_return - Rfree)
    ),
    # Market resources
    "b": lambda a_prev, R_portfolio: a_prev * R_portfolio,
    "m": lambda b, y: b + y,
    # Decisions (controls)
    "c": ska.Control(["m"], upper_bound=lambda m: m),
    "alpha": ska.Control(["a"], lower_bound=0.0, upper_bound=1.0),
    # End-of-period states
    "a": lambda m, c: m - c,
    # Utility
    "u": lambda c, CRRA: c ** (1 - CRRA) / (1 - CRRA),
}

# Create the block
portfolio_block = ska.DBlock(
    name="portfolio_choice", shocks=shocks, dynamics=dynamics, reward={"u": "investor"}
)
```

#### Step 4: Add Parameter Calibration

```python
calibration = {
    "CRRA": 2.0,  # Risk aversion
    "DiscFac": 0.96,  # Discount factor
    "Rfree": 1.03,  # Risk-free rate
    "EqP": 0.06,  # Equity premium
    "RiskyStd": 0.20,  # Stock volatility
    "PermGroFac": 1.01,  # Permanent income growth
    "TranShkStd": 0.1,  # Transitory shock std
    "PermShkStd": 0.05,  # Permanent shock std
}

# Apply calibration to construct actual distributions
portfolio_block.construct_shocks(calibration)
```

### Multi-Period Models with RBlock

For multi-period models, combine blocks with RBlock:

```python
# Retirement transition block
retirement_block = ska.DBlock(
    name="retirement",
    dynamics={
        "p": lambda p: p * 0.8,  # Retirement income drop
        "retired": lambda: 1,  # Retirement indicator
    },
)

# Life-cycle model
lifecycle_model = ska.RBlock(
    name="lifecycle_model", blocks=[portfolio_block, retirement_block]
)
```

## Working with Predefined Models

scikit-agent includes several predefined models in `skagent.models`:

### Consumer Models

```python
from skagent.models.consumer import (
    consumption_block,
    consumption_block_normalized,
    portfolio_block,
    calibration,
)

# Use predefined consumption-saving model
print("Available dynamics:")
for var, eq in consumption_block.dynamics.items():
    print(f"  {var}: {eq}")
```

### Benchmark Models

```python
from skagent.models.benchmarks import U6HabitSolver

# Access benchmark problem specifications
solver = U6HabitSolver()
```

### Perfect Foresight Models

```python
from skagent.models import perfect_foresight

# Simple deterministic model
pf_model = perfect_foresight.get_perfect_foresight_model()
```

## Advanced Model Features

### String-Based Dynamics

You can define dynamics using string expressions that get parsed automatically:

```python
dynamics = {
    "c": ska.Control(["m"]),
    "u": "c**(1-CRRA)/(1-CRRA)",  # String expression
    "mpc": "CRRA * c**(-CRRA)",  # Marginal propensity to consume
    "a": "m - c",  # Simple arithmetic
}
```

### Control Variable Constraints

Add bounds and information sets to controls:

```python
consumption_control = ska.Control(
    iset=["m", "p"],  # Information set
    lower_bound=lambda: 0.001,  # Minimum consumption
    upper_bound=lambda m: 0.99 * m,  # Maximum consumption
    agent="consumer",  # Agent assignment
)
```

### Aggregate Shocks

Model economy-wide shocks that affect all agents identically:

```python
aggregate_shock_block = ska.DBlock(
    shocks={
        "TFP": ska.Aggregate(
            Lognormal(mean=1.0, std=0.02)
        ),  # Total factor productivity
        "theta": MeanOneLogNormal(sigma=0.1),  # Idiosyncratic shock
    },
    dynamics={
        "Y": lambda TFP, L: TFP * L,  # Aggregate production
        "w": lambda TFP: TFP,  # Wage rate
        "y": lambda w, theta: w * theta,  # Individual income
    },
)
```

## Model Validation and Inspection

### Examining Model Structure

```python
# Get all variables in the model
variables = portfolio_block.get_vars()
print("Model variables:", variables)

# Get control variables
controls = portfolio_block.get_controls()
print("Control variables:", controls)

# Get shock variables
shocks = portfolio_block.get_shocks()
print("Shock variables:", list(shocks.keys()))
```

### Testing Model Dynamics

```python
# Test model transition with simple decision rules
pre_state = {
    "m": 2.0,
    "p": 1.0,
    "a_prev": 1.0,
}

decision_rules = {
    "c": lambda m: 0.8 * m,
    "alpha": lambda a: 0.6,  # 60% in risky asset
}

# Simulate one period
post_state = portfolio_block.transition(pre_state, decision_rules)
print("Post-transition state:", post_state)
```

## Best Practices

### Model Design

1. **Start Simple**: Begin with basic models and add complexity gradually
2. **Use Clear Names**: Choose descriptive variable names
3. **Document Dynamics**: Add comments explaining economic meaning
4. **Validate Calibration**: Check parameter values make economic sense

### Code Organization

```python
# Organize related models in modules
# my_models.py

from skagent import DBlock, Control
from HARK.distributions import MeanOneLogNormal

# Standard calibration for consumption models
CONSUMPTION_CALIBRATION = {
    "CRRA": 2.0,
    "DiscFac": 0.96,
    "Rfree": 1.03,
    # ... other parameters
}


def make_consumption_block(calibration=None):
    """Factory function for consumption blocks."""
    if calibration is None:
        calibration = CONSUMPTION_CALIBRATION

    block = DBlock(
        name="consumption",
        shocks={"theta": (MeanOneLogNormal, {"sigma": "TranShkStd"})},
        dynamics={
            "c": Control(["m"]),
            "u": "c**(1-CRRA)/(1-CRRA)",
            # ... other dynamics
        },
    )

    block.construct_shocks(calibration)
    return block
```

## Next Steps

- **Solution Methods**: Learn how to solve models using {doc}`algorithms`
- **Simulation**: Generate synthetic data with {doc}`simulation`
- **Examples**: See complete working examples in {doc}`../auto_examples/index`

## Common Patterns

### Habit Formation Models

```python
habit_block = ska.DBlock(
    dynamics={
        "c": ska.Control(["m", "h"]),
        "x": lambda c, h: c / h,  # Consumption relative to habit
        "u": lambda x, CRRA: x ** (1 - CRRA) / (1 - CRRA),
        "h": lambda h_prev, c_prev, rho: rho * h_prev + (1 - rho) * c_prev,
    }
)
```

### Durable Goods Models

```python
durables_block = ska.DBlock(
    dynamics={
        "c_nd": ska.Control(["m"]),  # Non-durable consumption
        "i_d": ska.Control(["m", "d"]),  # Durable investment
        "d": lambda d_prev, i_d, delta: (1 - delta) * d_prev + i_d,  # Durable stock
        "c_d": lambda d: d,  # Durable services
        "u": lambda c_nd, c_d, alpha: (c_nd**alpha * c_d ** (1 - alpha)) ** (1 - CRRA)
        / (1 - CRRA),
    }
)
```

This guide provides the foundation for building and working with economic models
in scikit-agent. The block-based architecture provides flexibility while
maintaining clear economic interpretation.
