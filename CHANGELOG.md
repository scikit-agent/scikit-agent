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
- `EulerEquationLoss` no longer takes a `discount_factor` parameter; the
  discount factor is now resolved from `bellman_period.discount_variable`
- `EulerEquationLoss` constrained mode uses the Fischer-Burmeister function
  (equation 25) when controls have an `upper_bound` defined
- `estimate_euler_residual` resolves the discount factor dynamically from the
  model and supports multi-control models (returns a dict for >1 controls)
- Control bounds (`lower_bound`, `upper_bound`) must now be callables; numeric
  values raise a clear `TypeError` instead of being silently ignored.
- Introduced `mortality_block` (and `mortal_cons_problem`) to demonstrate how to
  encode stochastic mortality and agent rebirth as a composable `DBlock`.
- `train_block_nn` now always returns a 3-tuple
  `(network, final_loss, optimizer)`; previously it returned a 2-tuple unless an
  optimizer was passed in. Callers should unpack three values.
- `maliar_training_loop` accepts an `lr` argument controlling the learning rate
  of its internal Adam optimizer.
- Consolidated the open-bounds scaling and decision-function plumbing shared by
  `BlockPolicyNet` and `BlockPolicyValueNet` into `BellmanPeriodMixin`.

### Added

- `fischer_burmeister(a, h)` utility for smooth complementarity conditions
- `examples/algorithms/plot_maliar_training.py` gallery example: trains a
  shared-backbone policy/value network and compares the trained policy against
  the U-2 analytical permanent-income solution.
- `estimate_bellman_foc_residual` for the first-order condition from the Bellman
  equation, using autograd to differentiate the value network
- `BellmanEquationLoss` gains a `foc_weight` parameter for adding a weighted FOC
  term to the Bellman loss (Maliar et al. 2021, equation 14)
- `BlockPolicyValueNet` (shared-backbone single network with policy and value
  heads) for use with `BellmanEquationLoss` under a single optimizer
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

- Standalone `BlockValueNet` class and `train_block_value_and_policy_nn`
  trainer; the alternating dual-optimizer pattern is replaced by a single
  shared-backbone network (`BlockPolicyValueNet`) trained with one optimizer
- `value_network` and `value_loss_function` parameters from
  `maliar_training_loop`
- Removed `AgentTypeMonteCarloSimulator`; mortality is now expressed
  declaratively via `mortality_block` (see Changed). The now-unused
  `calibration_by_age` helper and its API documentation entry were removed with
  it.

### Fixed

- Fixed the `CRRA` calibration in `perfect_foresight_normalized`: it was a
  1-tuple `(2.0,)`, which broke the CRRA utility power; it is now the scalar
  `2.0`.
- `train_block_nn` now halts early with a warning on a non-finite (NaN/Inf) loss
  instead of continuing to train on poisoned weights.

...

[Unreleased]: https://github.com/scikit-agent/scikit-agent/commits/main
