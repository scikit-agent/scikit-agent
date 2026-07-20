# Block Guide

This guide explains how to work with economic models in scikit-agent. You'll
learn about the core modeling concepts, how to build custom models, and how to
use predefined models.

## Understanding Model Structure

### The Block Architecture

scikit-agent uses a "block" architecture where models are composed of building
blocks:

- **DBlock (Dynamic Block)**: Represents a structured environment in which
  agents act
- **RBlock (Recursive Block)**: Combines multiple blocks into a more complex
  block
- **BellmanPeriod**: Wraps a block (it is not itself a `Block`) together with a
  discount variable and calibration, turning it into one period of a dynamic
  program

These blocks all define the ways that _variables_ change. The relationships
between variables are defined in terms of _structural equations_, which in
scikit-agent are represented as functions.

Some variables are reserved as **control** variables, which are assigned to
particular agent roles. Agents choose a decision rule that determines the value
of each of their control variables.

Some variables are reserved as **reward** variables, which provide the agent
_utility_ or incentive.

scikit-agent models normally involve agents who are trying to maximize their
reward through their choices.

### DBlocks

A DBlock has four main components:

- **Shocks**: Variables which are drawn from probabilistic distributions.
- **Dynamics**: Variables determined by _structural equations_, or functions, of
  other variables
- **Controls**: special dynamic variables for which agents decide _decision
  rules_.
- **Rewards**: special dynamic variables that agents try to optimize

Here is an example of a simple DBlock representing a single stage of a
consumption-saving problem:

```python
import math
import skagent as ska
from skagent.distributions import MeanOneLogNormal

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
        "u": lambda c: math.log(c),
    },
    # 3. Rewards: What agents maximize
    reward={"u": "consumer"},  # Utility goes to consumer agent
)
```

This corresponds to the following mathematical model, where the income shock
$\theta$ is a mean-one lognormal with log-space standard deviation
$\sigma = 0.1$:

$$
\begin{aligned}
    \log \theta &\sim \mathcal{N}(-\sigma^2 / 2,\ \sigma^2) \\
    y &= p \theta \\
    m &= b_{-1} + y \\
    c &= c(m) \\
    a &= m - c \\
    u &= \log(c)
\end{aligned}
$$

Here, the agent can choose its level of consumption $c$ given an _information
set_ $m$. It receives $u$ as a reward.

#### Arrival states

Dynamic equations are interpreted in sequence. Each variable is assigned based
on the values of other variables in scope. If a variable is referenced in a
dynamic equation before it is assigned, it is an _arrival state_, or _lag
variable_, which refers to a value that is assigned in some preceding block or
time step.

In the example above, $b_{-1}$ is such a variable.

Arrival states can be provided by a previous block (see _RBlocks_, below), in a
previous time period (see _BellmanPeriod_), or by _initialization data_ before a
simulation.

### Control Variables

A **Control** variable is under the control of some agent. Instead of providing
a dynamic equation, the modeler specifies an information set -- what information
(variables) are available to the agent when they decide this variable's value.

#### Constraints

Control variables can be upper and lower bound to values that are themselves
functions of state variables. Each bound is a callable; a constant bound is a
zero-argument callable.

```python
consumption_control = ska.Control(
    iset=["m", "p"],  # Information set
    lower_bound=lambda: 0.001,  # Minimum consumption
    upper_bound=lambda m: 0.99 * m,  # Maximum consumption
    agent="consumer",  # Agent assignment
)
```

How the solvers enforce these bounds, and how to encode the optimality
conditions that hold where a constraint binds, is the subject of the
{doc}`constraints` guide.

#### Calibration

Shock parameters can be given as strings naming calibration parameters rather
than as literal values. Calling `construct_shocks` with a calibration dictionary
then builds the actual distributions:

```python
income_block = ska.DBlock(
    name="income",
    shocks={"theta": (MeanOneLogNormal, {"sigma": "TranShkStd"})},
    dynamics={"y": lambda p, theta: p * theta},
)

calibration = {
    "CRRA": 2.0,  # Risk aversion
    "DiscFac": 0.96,  # Discount factor
    "Rfree": 1.03,  # Risk-free rate
    "PermGroFac": 1.01,  # Permanent income growth
    "TranShkStd": 0.1,  # Transitory shock std
}

# Apply calibration to construct actual distributions
income_block.construct_shocks(calibration)
```

#### String-Based Dynamics

You can define dynamics using string expressions that get parsed automatically:

```python
dynamics = {
    "c": ska.Control(["m"]),
    "u": "c**(1-CRRA)/(1-CRRA)",  # String expression
    "mpc": "CRRA * c**(-CRRA)",  # Marginal propensity to consume
    "a": "m - c",  # Simple arithmetic
}
```

### RBlocks: Composing Blocks

The RBlock is for composing other blocks together.

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
    name="lifecycle_model", blocks=[consumption_block, retirement_block]
)
```

### Bellman Periods

A `BellmanPeriod` (from `skagent.bellman`) wraps a block together with its
discount variable and calibration, turning the block into one period of a
dynamic stochastic optimization problem. The wrapped period exposes the reward,
transition, and gradient functions that the neural network solution methods and
loss functions consume:

```python
from skagent.bellman import BellmanPeriod

bp = BellmanPeriod(consumption_block, "DiscFac", calibration)
```

See the {doc}`../api/bellman` reference for the period timing notation (arrival
states, shocks, pre-decision states, controls, and rewards) and the full API.

## Model Validation and Inspection

### Examining Model Structure

```python
# Get all variables in the model
variables = consumption_block.get_vars()
print("Model variables:", variables)

# Get control variables
controls = consumption_block.get_controls()
print("Control variables:", list(controls.keys()))

# Get shock variables
shocks = consumption_block.get_shocks()
print("Shock variables:", list(shocks.keys()))
```

### Testing Model Dynamics

The `transition` method advances the block by one period from a dictionary of
arrival states and realized shock values:

```python
# Arrival states and a realized shock value
pre_state = {
    "b": 1.0,
    "p": 1.0,
    "theta": 1.0,
}

decision_rules = {
    "c": lambda m: 0.8 * m,
}

# Simulate one period
post_state = consumption_block.transition(pre_state, decision_rules)
print("Post-transition state:", post_state)
```

## Next Steps

- **Solution Methods**: Learn how to solve models using {doc}`algorithms`
- **Simulation**: Generate synthetic data with {doc}`simulation`
- **Examples**: See complete working examples in {doc}`../auto_examples/index`

## Common Patterns

Recall that a variable referenced before it is assigned within a block is an
arrival state: in the habit block below, the `h` appearing in the information
set of `c` and in `x` is last period's habit stock, while the final equation
assigns this period's value.

### Habit Formation Models

```python
habit_block = ska.DBlock(
    dynamics={
        "c": ska.Control(["m", "h"]),
        "x": lambda c, h: c / h,  # Consumption relative to habit
        "u": lambda x, CRRA: x ** (1 - CRRA) / (1 - CRRA),
        "h": lambda h, c, rho: rho * h + (1 - rho) * c,  # Habit stock update
    }
)
```

### Durable Goods Models

```python
durables_block = ska.DBlock(
    dynamics={
        "c_nd": ska.Control(["m"]),  # Non-durable consumption
        "i_d": ska.Control(["m", "d"]),  # Durable investment
        "d": lambda d, i_d, delta: (1 - delta) * d + i_d,  # Durable stock
        "c_d": lambda d: d,  # Durable services
        "u": lambda c_nd, c_d, alpha, CRRA: (c_nd**alpha * c_d ** (1 - alpha))
        ** (1 - CRRA)
        / (1 - CRRA),
    }
)
```

This guide provides the foundation for building and working with economic models
in scikit-agent. The block-based architecture provides flexibility while
maintaining clear economic interpretation.
