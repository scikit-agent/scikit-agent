# Environments

This section documents the environment adapters that step a
{class}`~skagent.bellman.BellmanPeriod` one transition at a time, for use by
reinforcement-learning algorithms.

Two interfaces are provided:

- {class}`~skagent.env.Environment` — a plain Python environment that returns
  the full transition `(state, action, reward, next_state, discount, obs)` as
  dicts keyed by symbol, suitable for off-policy RL and for rollouts.
- {class}`~skagent.env.GymEnv` — a [gymnasium](https://gymnasium.farama.org/)
  adapter so Stable-Baselines3 algorithms (PPO, SAC, TD3, …) can drive a
  `BellmanPeriod` directly. It is the backend used by
  {class}`skagent.algos.sb3.PPOAgent` (see {doc}`algorithms`).

```{eval-rst}
.. automodule:: skagent.env
   :members:
```
