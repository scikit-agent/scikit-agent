# Environments Guide

This guide introduces _environments_ — the adapters that let an external
algorithm interact with a scikit-agent model one step at a time. They are the
bridge between your model and the reinforcement-learning tools described in the
{doc}`algorithms` guide.

## Why environments?

A {class}`~skagent.bellman.BellmanPeriod` describes a model: its shocks,
dynamics, controls, and rewards. Many algorithms, though, expect to _drive_ a
model interactively — propose an action, see what reward and next state result,
and repeat. An environment wraps your model in exactly that interactive loop, so
you don't have to wire up the stepping logic yourself.

scikit-agent provides two, depending on who is doing the driving.

## `Environment`: stepping a model in plain Python

{class}`~skagent.env.Environment` advances a model one period at a time and
hands back the full transition — the state, the action taken, the reward, the
next state, the period's discount factor, and the observation the policy saw.
You supply a decision rule (a `{control: callable}` dict) and call `step`:

```python
import numpy as np
from skagent.env import Environment
from skagent.distributions import Uniform
from skagent.models.benchmarks import d2_block, d2_calibration
from skagent.bellman import BellmanPeriod

bp = BellmanPeriod(d2_block, "DiscFac", d2_calibration)
env = Environment(bp, {"a": Uniform(low=0.5, high=2.0)}, rng=np.random.default_rng(0))

env.reset()
state, action, reward, next_state, discount, obs = env.step({"c": lambda m: m / 2})
```

This is useful whenever you want direct control over the simulation loop — for
custom analysis, or for algorithms that consume full transitions.

### Scoring a policy with rollouts

A common thing to do with an `Environment` is to _score_ a decision rule by the
total discounted reward it earns over a rollout. The helper
{func}`~skagent.env.discounted_rollout_reward` does this for you:

```python
from skagent.env import discounted_rollout_reward

total = discounted_rollout_reward(
    bp,
    {"c": lambda m: m / 2},  # the decision rule to score
    {"a": Uniform(low=0.5, high=2.0)},  # initial state distribution
    steps=200,
    rng=np.random.default_rng(0),
)
```

Running this for several policies (and averaging over many rollouts) is a simple
way to compare how well different decision rules actually perform.

## `GymEnv`: a gymnasium environment for RL libraries

{class}`~skagent.env.GymEnv` presents your model through the standard
[gymnasium](https://gymnasium.farama.org/) interface, so reinforcement-learning
libraries can train on it directly. It is what
{class}`skagent.algos.sb3.PPOAgent` uses under the hood; most users never need
to construct it by hand.

A couple of details worth knowing:

- **Actions are normalised.** The agent works with actions in `[-1, 1]`, and
  `GymEnv` automatically rescales them to each control's real bounds (for
  example, the borrowing constraint `c ≤ m`) before applying them. Your model's
  bounds are respected without any extra effort.
- **Single control, single agent.** For now, `GymEnv` handles models with one
  control variable and one agent.

If you are using PPO through `PPOAgent`, this all happens for you — see the
{doc}`algorithms` guide to get started.
