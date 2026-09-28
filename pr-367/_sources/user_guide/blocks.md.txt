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

#### Declaration order and symbol aliasing

Because dynamic equations are interpreted in sequence, **the same symbol can
mean two different things at two points in one block**: its arrival value before
it is reassigned, and its newly computed value afterwards. This matters most for
rewards, because a reward equation reads its inputs _as of its own declaration
point_.

Consider a block whose reward penalizes the distance between an asset `a` and a
target `b` that the control sets for next period:

```python
tracking_block = ska.DBlock(
    name="tracking",
    dynamics={
        "c": ska.Control(["a"]),
        "u": lambda a, b: -((a - b) ** 2),  # reads the ARRIVAL b
        "b": lambda c: c,  # assigns next period's b
    },
    reward={"u": "tracker"},
)
```

`u` is declared _before_ `b`, so the reward sees last period's `b` while the
transition reports `b = c` for next period. Today's choice therefore does not
enter today's reward at all: `c` influences the objective only through the
continuation value. Swapping the last two equations gives a different model.

The practical rule is to **place a reward equation where you want its inputs
read from**. A survival model makes the stakes concrete, with an alive indicator
`liv` and a survival shock `live`:

```python
# Utility accrues to those alive on ARRIVAL, so u is declared BEFORE liv updates.
dynamics = {
    "c": ska.Control(["m"]),
    "u": lambda liv, c: liv * math.log(c),
    "liv": lambda liv, live: liv * live,
}
```

Declaring `u` after the `liv` update instead means utility requires _surviving_
the period. That is a different model, and it differs from the intended one only
by a term of order $1 - \mathbb{E}[\texttt{live}]$ — small enough to read as a
numerical problem rather than as the specification error it is.

### Control Variables

A **Control** variable is under the control of some agent. Instead of providing
a dynamic equation, the modeler specifies an information set -- what information
(variables) are available to the agent when they decide this variable's value.

#### Constraints

Control variables can be upper and lower bound to values that are themselves
functions of state variables. Each bound is a number (a constant bound) or a
callable of variables in the control's information set.

```python
consumption_control = ska.Control(
    iset=["m", "p"],  # Information set
    lower_bound=0.001,  # Minimum consumption (constant)
    upper_bound=lambda m: 0.99 * m,  # Maximum consumption (state-dependent)
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

### Entities: several of the same kind of agent

A block may declare that its variables describe a _class_ of things there are
several of, rather than a single agent. That is an `Entity`, and it is set on
the block with the `entity` field:

```python
firms = ska.RBlock(
    name="firms",
    entity=ska.Entity("firm"),
    blocks=[
        ska.DBlock(
            name="offer",
            shocks={"c": (Uniform, {"low": "cl", "high": "ch"})},
            dynamics={"q": ska.Control(["c"], agent="firm")},
        )
    ],
)
```

A variable defined in a block carrying an entity is an **attribute** of that
entity class, so each instance has its own value. A variable defined in a block
carrying no entity is **axis-free**: there is one of it for the whole model.

An `Entity` has a name and no size. How many instances exist is a fact about the
population being simulated rather than about the model, so it is read from the
calibration under a key equal to the entity's name -- a model declaring
`Entity("firm")` is simulated against a calibration containing `{"firm": 3}`.
One consequence to keep in mind when naming: an entity therefore shares a
namespace with the model's parameters, so a model with an entity `"firm"` cannot
also have an unrelated parameter `"firm"`.

An agent name declares a **role**, and never creates an entity class. The role
attaches to the entity class of the block its control is declared in, and to
nothing at all when that block has none -- so a single analyst observing many
subjects is one agent, not a population of one.

#### Aggregation, and what a crossing is

An axis-free equation may read a per-instance variable, in which case it is
handed the whole array and must reduce it to one value:

```python
market = ska.DBlock(
    name="market",
    dynamics={
        "Q": lambda q: q.mean(),  # reads out of the firm class: a crossing
        "P": lambda A, b, Q: A - b * Q,
    },
)
```

Reading _out of_ an entity class like this is a **crossing**, and the library
reports it. Reading _into_ one -- a per-instance equation using the single
market price -- is an ordinary broadcast and needs nothing:

```python
payoff = ska.DBlock(
    name="payoff",
    dynamics={"u": lambda P, c, q: (P - c) * q},  # broadcast, not a crossing
    reward={"u": "firm"},
)
```

A crossing is what distinguishes a _population_ from a set of independent
samples. Where a model has no crossing, running it over many parallel copies is
Monte Carlo replication: the copies do not interact. Where it has one, the
copies interact through the aggregate and are a population inside one run.

Because a solver treats each state variable as an axis to optimise over point by
point, and a crossing is not that, blocks with crossings are refused by the
value-function solvers. Simulating them forward under supplied decision rules is
fully supported.

The same entity class may be declared more than once in a model, and that is how
timing is expressed. In a Cournot market the firms act, then the market clears,
then the firms are paid, so the firm class appears both before and after the
market block:

```python
cournot = ska.RBlock(
    name="cournot",
    blocks=[
        firms,
        market,
        ska.RBlock(name="payoffs", entity=ska.Entity("firm"), blocks=[payoff]),
    ],
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

## Authoring Blocks for Dynamic Programming

A block that simulates correctly can still be difficult to _solve_. Three
authoring choices matter to the solvers, and each one has produced a
plausible-looking wrong answer rather than an error.

### Declare both bounds when the reward is undefined outside the feasible set

Log and CRRA utility are undefined at $c \le 0$. If such a control declares no
`lower_bound`, its feasible box extends into the region where the objective is
not a number, and a gradient-based optimizer can step there: the line search
fails, the optimizer terminates abnormally, and it **returns its own starting
point unchanged**.

Nothing raises. Because the returned point is the point the solver started from,
value iteration sees no change between iterations and reports a converged solve
with a residual of exactly zero — and the answer is insensitive to grid
refinement, which normally rules out numerical explanations. Declare the bound
even when it feels implied by the economics:

```python
{
    "c": ska.Control(["m"], lower_bound=lambda m: 1e-4, upper_bound=lambda m: m),
}
```

See {doc}`constraints` for how the solvers enforce bounds and how to handle
constraints that bind at the optimum.

### Extend a terminal or absorbing axis one step past the last nonzero reward

A finite-horizon model can carry its own time index as an ordinary arrival
state, with the horizon living in the reward:

```python
dynamics = {
    "c": ska.Control(["W", "t"], lower_bound=lambda W: 1e-4, upper_bound=lambda W: W),
    "u": lambda t, T, c: (t < T) * math.log(c),  # no reward from period T on
    "W": lambda W, c, R: (W - c) * R,
    "t": lambda t: t + 1,
}
```

When solving such a model on a grid, **the `t` axis must extend to `T + 1`**,
one slice beyond the last period with a nonzero reward. Value functions are
interpolated over the grid and _extrapolated linearly_ past its edges, which is
deliberate: a flat clamp at the boundary would zero out the marginal value of
saving. But if the axis stops at `T`, the last two slices are a nonzero value
and zero, so the linear extrapolation invents a nonzero value at `t = T + 1`
with the wrong sign — an afterlife — and the backup happily maximizes it. The
residual diverges instead of converging.

Two identically-zero slices (`t = T` and `t = T + 1`) make the extrapolation
flat at zero, which is the correct terminal condition. The same requirement
applies to any absorbing state whose value is zero.

### Keep dynamics agnostic about array types

The same block runs under two numeric backends: torch tensors for the neural
solvers and their autodiff, and plain numpy floats for grid-based backward
induction. Equations written for only one of them fail on the other —
`numpy_array * torch_tensor` raises, and `torch.clamp` rejects non-tensor input,
so a guard like `m = R * a / torch.clamp(psi, min=1e-8)` makes the block
unsolvable on the numpy path.

Prefer plain arithmetic, which broadcasts across both, and route anything else
through a helper that accepts either:

```python
import numpy as np
import torch


def clamp_min(x, lo):
    """Lower-clamp that works on torch tensors and numpy/Python scalars."""
    if torch.is_tensor(x):
        return torch.clamp(x, min=lo)
    return np.maximum(x, lo)
```

`skagent.models.benchmarks` uses exactly this pattern, alongside `as_tensor` for
comparisons that need to hold on both paths. When a block is intended for both,
exercise it on both.

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

For a model declaring entities, four further queries report its population
structure. All are derived from the model's own declarations; nothing extra is
annotated.

```python
# Which entity classes the model declares
cournot.entities()
# {'firm': Entity(name='firm')}

# Which classes each symbol is an attribute of; empty means axis-free
cournot.signatures()
# {'c': frozenset({'firm'}), 'q': frozenset({'firm'}),
#  'Q': frozenset(), 'P': frozenset(), 'u': frozenset({'firm'})}

# The equations that read out of a class, with the axes to reduce and broadcast
cournot.crossings()
# {'Q': [('q', frozenset({'firm'}), frozenset())]}

# Each agent role, and the class whose instances hold it; None means one agent
cournot.agent_populations()
# {'firm': 'firm'}
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
assigns this period's value. Both patterns below rely on that aliasing, so they
are worth reading against the rule described under _Declaration order and symbol
aliasing_ above.

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

Note the deliberate use of declaration order here: `i_d`'s information set reads
the _arrival_ durable stock `d`, while `c_d` is declared after the stock updates
and so measures services from the _post-investment_ stock. Moving `c_d` above
the `d` equation would make this period's utility depend on last period's stock
instead.

This guide provides the foundation for building and working with economic models
in scikit-agent. The block-based architecture provides flexibility while
maintaining clear economic interpretation.
