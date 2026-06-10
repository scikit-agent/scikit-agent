# Algorithms Guide

This guide covers the solution algorithms and optimization methods available in
scikit-agent.

## Solution Methods

Learn about different approaches to solving economic models:

- **All-in-One Deep Learning Methods** : Deep learning solvers that use an
  All-in-One (AiO) objective function
- **Value Function Iteration**: Classical dynamic programming approaches
- **Reinforcement Learning**: Learn a policy by trial-and-error interaction with
  the model, using established RL libraries (see below)

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

---

_This page is under construction. Content will be added as algorithms are
implemented._
