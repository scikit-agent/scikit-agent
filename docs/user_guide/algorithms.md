# Algorithms Guide

This guide covers the solution algorithms and optimization methods available in
scikit-agent.

## Solution methods

scikit-agent offers several families of solution method, each producing a
standard `{control: callable}` decision rule that plugs into simulators and the
rest of the toolkit:

- **Maliar-style deep learning methods**: Neural network solvers following
  Maliar, Maliar, and Winant (2021), which train on an all-in-one (AiO)
  objective function
- **Value backwards induction (VBI)**: Classical dynamic programming via
  backwards induction on a grid
- **Reinforcement Learning**: Learn a policy by trial-and-error interaction with
  the model, using established RL libraries (see below)

The rest of this guide covers these in turn.

## Reinforcement Learning

Instead of solving a model with a dedicated dynamic-programming method, you can
let a reinforcement-learning (RL) agent _learn_ a good decision rule by
repeatedly interacting with the model and observing the rewards it earns. This
is handy when a model is hard to solve analytically, or when you simply want a
quick baseline to compare against.

One way to do this is to integrate with an established RL library. scikit-agent
adapts your model into a standard environment (see {doc}`environments`) and
hands it to [Stable-Baselines3](https://stable-baselines3.readthedocs.io/),
which provides **PPO** (Proximal Policy Optimization), a robust, general-purpose
algorithm.

The entry point is {class}`~skagent.algos.sb3.PPOAgent`. You give it a model
(`BellmanPeriod`) and a distribution of starting states, train for a number of
timesteps, and ask for a decision rule:

```python
from skagent.algos.sb3 import PPOAgent
from skagent.bellman import BellmanPeriod
from skagent.distributions import Uniform
from skagent.models.benchmarks import d2_block, d2_calibration

# Wrap a model block together with its discount variable and calibration.
bp = BellmanPeriod(d2_block, "DiscFac", d2_calibration)

# Train PPO, sampling fresh initial states from this distribution each episode.
agent = PPOAgent(bp, {"a": Uniform(low=0.01, high=5.0)}, seed=0)
agent.learn(total_timesteps=100_000)

# Get a standard skagent decision rule and use it like any other.
dr = agent.decision_rule()
```

The returned `dr` is an ordinary `{control: callable}` decision rule — the same
shape produced by other solvers — so it plugs straight into simulators and the
rest of the toolkit.

For a complete, runnable walkthrough — training PPO on a benchmark with a known
closed-form solution and comparing the learned policy against it — see the
{doc}`../auto_examples/algorithms/plot_sb3_ppo` example.

```{note}
PPO uses a single, constant discount factor (`gamma`), taken from your model's
discount variable. Models whose discount factor varies with the state are not
yet supported through this path.
```

## Solving a block directly

A {doc}`block <blocks>` describes a single decision period: arrival states and
shocks come in, the agent chooses its controls, and a reward is produced. The
most direct use of the deep-learning solver is to train a
{py:class}`~skagent.ann.BlockPolicyNet` so that its decision rule maximizes the
reward earned within the block, over a grid of starting points.

What you optimize — this period's immediate reward, a fixed finite horizon, or a
recurring problem with a continuation value — is determined by the _loss
function_ you choose in step 3, not by the model objects themselves. This
section uses reward-based losses; the **Value Function Iteration** section below
covers the recurring case.

### 1. Load a model and wrap it

We use the **D-2** benchmark — an infinite-horizon, perfect-foresight
consumption–savings model — from {py:mod}`skagent.models.benchmarks`. Each model
in that registry is a {py:class}`~skagent.block.DBlock` paired with a
calibration. Wrap the block in a {py:class}`~skagent.bellman.BellmanPeriod`,
naming the discount factor variable (`"DiscFac"`); the wrapper attaches the
calibration and exposes the methods the solver needs (transitions, decisions,
rewards).

```python
import skagent.bellman as bellman
import skagent.grid as grid
from skagent.models.benchmarks import get_benchmark_model, get_benchmark_calibration

block_d2 = get_benchmark_model("D-2")
calibration = get_benchmark_calibration("D-2")

# Build the shock distributions (a no-op for the deterministic D-2, but good
# practice — required once a model has live shocks, as U-2 does below).
block_d2.construct_shocks(calibration)

bp = bellman.BellmanPeriod(block_d2, "DiscFac", calibration)
```

In D-2 the agent arrives with assets `a`, cash-on-hand `m = R*a + y` is formed,
and the single control `c` (consumption) is chosen subject to the natural
borrowing limit `c <= m + H`, where `H = y/(R-1)` is human wealth (the agent may
borrow against future income).

### 2. Build the grid of starting points

The network is trained over a {py:class}`~skagent.grid.Grid` of arrival states
(and shock realizations, if the block has shocks). Each grid point is one
scenario the policy must do well on. D-2's only arrival state is `a`:

```python
states = grid.Grid.from_config({"a": {"min": 0.5, "max": 4.0, "count": 15}})
```

### 3. Choose a reward-based loss

To solve the block directly, maximize reward. Two loss functions cover the
common cases:

- {py:class}`~skagent.loss.StaticRewardLoss` — the negative of this period's
  immediate reward. Use it when only the current period's reward matters.
- {py:class}`~skagent.loss.EstimatedDiscountedLifetimeRewardLoss` — the negative
  discounted reward accumulated over a fixed horizon of `big_t` simulated
  periods. With `big_t=1` this reduces to the immediate-reward case; larger
  values roll the block forward a known, finite number of steps. Because the
  horizon is fixed, there is still no continuation value to solve for.

Here we give the agent a 10-period planning horizon so it trades off consumption
over time rather than acting myopically:

```python
import skagent.loss as loss

loss_fn = loss.EstimatedDiscountedLifetimeRewardLoss(
    bp, big_t=10, parameters=calibration
)
```

### 4. Train and read off the decision rule

{py:func}`~skagent.ann.train_block_nn` runs the optimizer. Afterwards, query the
network's {py:meth}`~skagent.ann.BlockPolicyNet.decision_function` to get
control values, or {py:meth}`~skagent.ann.BlockPolicyNet.get_decision_rule` to
get a reusable decision rule. Pass the calibration to `decision_function`: the
control's information variable `m` is derived from the arrival state and the
parameters `R` and `y`, so the solver needs them to reconstruct it.

```python
import skagent.ann as ann

policy = ann.BlockPolicyNet(bp, width=16)
ann.train_block_nn(policy, states, loss_fn, epochs=250)

c = policy.decision_function({"a": states["a"]}, {}, calibration)["c"]
```

Over a 10-period horizon the agent saves: the learned `c` sits well below
cash-on-hand `m`, in the neighborhood of D-2's infinite-horizon optimum.
Shortening the horizon toward `big_t=1` makes it myopic (consume everything, `c`
→ `m`); lengthening it pushes the stationary rule toward the infinite-horizon
policy. That true optimum — D-2 has a known closed form, available via
{py:func}`skagent.models.benchmarks.get_analytical_policy` — is solved directly
with the recurring methods in the **Value Function Iteration** section below.

### Working with shocks

The **U-2** benchmark adds a permanent-income shock `psi` to a normalized
log-utility model. To keep the focus on shock handling rather than the planning
horizon, this example uses {py:class}`~skagent.loss.StaticRewardLoss` — a
single-period (myopic) objective. With this loss, each shock gets a column in
the grid under its own name (`"psi"`), and the realized shocks are passed to
`decision_function`.

```{note}
The two reward losses use different grid conventions for shocks.
{py:class}`~skagent.loss.StaticRewardLoss` reads each shock by its base name
(``"psi"``). {py:class}`~skagent.loss.EstimatedDiscountedLifetimeRewardLoss`
simulates ``big_t`` periods and reads one draw per period under the
``<shock>_<t>`` convention (``"psi_0"``, ``"psi_1"``, …).
```

The registry ships U-2 with `sigma_psi=0` (the shock switched off, so the
permanent-income-hypothesis solution is exact); here we switch it on to
illustrate shock handling.

```python
import numpy as np

block_u2 = get_benchmark_model("U-2")
calibration = get_benchmark_calibration("U-2")
calibration["sigma_psi"] = 0.1  # switch the permanent-income shock on

# Construct shocks with an explicit RNG for reproducible draws.
block_u2.construct_shocks(calibration, rng=np.random.default_rng(0))

bp = bellman.BellmanPeriod(block_u2, "DiscFac", calibration)

states = grid.Grid.from_config(
    {
        "a": {"min": 0.5, "max": 5.0, "count": 8},
        "psi": {"min": 0.8, "max": 1.2, "count": 5},
    }
)

policy = ann.BlockPolicyNet(bp, width=16)
loss_fn = loss.StaticRewardLoss(bp, calibration)
ann.train_block_nn(policy, states, loss_fn, epochs=500)

c = policy.decision_function({"a": states["a"]}, {"psi": states["psi"]}, calibration)[
    "c"
]
```

U-2's control is bounded (`0.01 <= c <= 0.1*m + 2`), set via the `lower_bound` /
`upper_bound` arguments to {py:class}`~skagent.block.Control` in its definition;
the policy network enforces such bounds automatically.

## Blocks with multiple controls

When a block has more than one control, train one policy network per control and
let each network treat the others' current policies as fixed — a best-response
sweep. {py:func}`skagent.solver.solve_multiple_controls` automates this. The
benchmark registry has no multi-control model, so we use a small illustrative
block whose reward is maximized at `c = a` and `d = k`:

```python
import skagent.block as block
from skagent.solver import solve_multiple_controls

calibration = {"k": 3, "beta": 0.9}

b = block.DBlock(
    name="two controls",
    dynamics={
        "c": block.Control(["a"], agent="agent"),
        "d": block.Control([], agent="agent"),  # empty information set
        "u": lambda a, c, d, k: -((a - c) ** 2) - (k - d) ** 2,
    },
    reward={"u": "agent"},
)
bp = bellman.BellmanPeriod(b, "beta", calibration)

states = grid.Grid.from_config({"a": {"min": -2, "max": 2, "count": 11}})

# Repeat a symbol to schedule an extra refinement pass after its neighbour
# has been updated.
decision_rules = solve_multiple_controls(
    ["c", "d", "c"], bp, states, calibration, epochs=200
)
# optimal: c = a and d = 3, so the reward u is approximately 0
```

The return value is a dictionary mapping each control symbol to its trained
decision rule, suitable for passing to `reward_function` or to simulation.

## Value Function Iteration

Classical dynamic programming for recurring problems, implemented as backwards
induction over a block's value function. See {py:func}`skagent.algos.vbi.solve`.
The neural Bellman- and Euler-equation losses
({py:class}`~skagent.loss.BellmanEquationLoss`,
{py:class}`~skagent.loss.EulerEquationLoss`) provide deep-learning alternatives
for the recurring case.

---

_For a runnable, end-to-end version of these workflows, see the
{doc}`Algorithms examples gallery </auto_examples/algorithms/index>`._
