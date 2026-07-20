# Environments

This section documents the environment adapters that step a
{class}`~skagent.bellman.BellmanPeriod` one transition at a time, for use by
reinforcement-learning algorithms. The {class}`~skagent.env.GymEnv` adapter
below is the [gymnasium](https://gymnasium.farama.org/) backend behind
{class}`skagent.algos.sb3.PPOAgent` (see {doc}`algorithms`). The module
docstring describes the two available interfaces.

```{eval-rst}
.. automodule:: skagent.env
   :members:
```
