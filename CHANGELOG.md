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
- Added an **Algorithms** user-guide page documenting the direct (non-recurring)
  solve workflow — training a `BlockPolicyNet` against reward-based losses
  (`StaticRewardLoss`, `EstimatedDiscountedLifetimeRewardLoss`) on benchmark
  models (D-2, U-2), including multiple-control solves — with a runnable
  `plot_direct_block_solve.py` gallery example
- Expanded the Algorithms API reference with the `skagent.solver` and
  `skagent.loss` modules and `skagent.ann.train_block_nn`

### Removed

- Removed `AgentTypeMonteCarloSimulator`; mortality is now expressed
  declaratively via `mortality_block` (see Changed). The now-unused
  `calibration_by_age` helper and its API documentation entry were removed with
  it.

### Fixed

- Fixed the `CRRA` calibration in `perfect_foresight_normalized`: it was a
  1-tuple `(2.0,)`, which broke the CRRA utility power; it is now the scalar
  `2.0`.
- Fixed `skagent.solver.solve_multiple_controls`, which previously crashed on
  its default loss and passed incorrect arguments to `StaticRewardLoss`; it now
  trains a policy network per control via a best-response sweep and returns the
  trained decision rules.

...

[Unreleased]: https://github.com/scikit-agent/scikit-agent/commits/main
