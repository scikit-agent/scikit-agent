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

### Added

- `fischer_burmeister(a, h)` utility for smooth complementarity conditions
- `estimate_bellman_foc_residual` for the first-order condition from the Bellman
  equation, using autograd to differentiate the value network
- `BellmanEquationLoss` gains a `foc_weight` parameter for adding a weighted FOC
  term to the Bellman loss (Maliar et al. 2021, equation 14)
- `maliar_training_loop` gains optional `value_network` and
  `value_loss_function` parameters for joint policy-value training
- Initial release features

...

[Unreleased]: https://github.com/user/repo/commits/main
