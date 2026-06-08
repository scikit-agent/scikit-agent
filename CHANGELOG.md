# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Refactored `BellmanPeriod` with type hints, docstrings, and improved parameter
  handling
- Introduced `_resolve_parameters`, `_resolve_decision_rules`, and
  `_resolve_shocks` helper methods for consistent fallback logic
- Added gradient computation methods (`grad_reward_function`,
  `grad_transition_function`, `grad_pre_state_function`) to `BellmanPeriod`
- Added shock resolution support in `BellmanPeriod` methods
- Moved `compute_gradients_for_tensors` tests from `test_bellman.py` to
  `test_utils.py`
- Introduced `mortality_block` (and `mortal_cons_problem`) to demonstrate how to
  encode stochastic mortality and agent rebirth as a composable `DBlock`.

### Added

- PPO solution algorithm via Stable-Baselines3: `skagent.algos.sb3.PPOAgent`
  wraps a `BellmanPeriod` in a gymnasium environment, trains SB3's PPO, and
  emits a standard skagent decision rule (`#205`)
- `PPOAgent.snapshot()` and the `PolicySnapshot` class, capturing a frozen copy
  of the trained policy (unaffected by later `learn` calls) for comparing
  checkpoints during training
- `skagent.env` module with `Environment` (single-transition stepping of a
  `BellmanPeriod`) and `GymEnv` (gymnasium adapter for Stable-Baselines3)
- `skagent.env.discounted_rollout_reward` for scoring a decision rule by its
  realized discounted return over a rollout
- `skagent.models.benchmarks.d2_constrained_optimal_c`, the D-2 closed-form
  consumption function keyed on cash-on-hand with the borrowing constraint
  applied
- Gallery example `examples/algorithms/plot_sb3_ppo.py` demonstrating PPO on the
  D-2 benchmark
- NumFOCUS Code of Conduct adopted
- Created a working `Consumption-Saving Model` example in the documentation
  gallery
- Added a **Benchmark Models** user-guide page (a model-agnostic onramp: the
  registry roster and how to fetch and validate models) alongside a runnable
  `plot_benchmark_models.py` gallery tour that introduces each model with its
  equations and plots the lesson it teaches
- Added the public `has_analytical_policy` registry helper to
  `skagent.models.benchmarks`, replacing duplicated closed-form checks in the
  tests and the gallery

### Removed

- Removed `AgentTypeMonteCarloSimulator`; mortality is now expressed
  declaratively via `mortality_block` (see Changed). The now-unused
  `calibration_by_age` helper and its API documentation entry were removed with
  it.

### Fixed

- Fixed the `CRRA` calibration in `perfect_foresight_normalized`: it was a
  1-tuple `(2.0,)`, which broke the CRRA utility power; it is now the scalar
  `2.0`.

...

[Unreleased]: https://github.com/scikit-agent/scikit-agent/commits/main
