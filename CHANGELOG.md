# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- `compute_gradients_for_tensors` returns a zero tensor (instead of `None`) for
  a variable with no computational path to the target, and raises `ValueError`
  when a `wrt` tensor does not require gradients; the `BellmanPeriod` gradient
  methods (`grad_reward_function`, `grad_transition_function`,
  `grad_pre_state_function`) inherit the tensor-only contract (#129)
- Declared `torch >=2.0` as the minimum supported PyTorch version
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
- `EulerEquationLoss` now estimates the squared expected Euler residual with the
  all-in-one operator: the _product_ of two residuals at independent next-period
  shock draws (Maliar, Maliar, and Winant 2021, JME), rather than the square of
  a single draw. The product is an unbiased estimate of `(E[f])**2`, whereas
  squaring one draw adds `Var(f) >= 0` and biases the solution of any stochastic
  model. For deterministic models the two draws coincide and the loss is
  unchanged.
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

- **Constraints** user-guide page documenting the ways to constrain an
  optimization problem: bound declaration on `Control`, the open-bounds
  policy-network transforms, the Fischer-Burmeister complementarity loss
  (including its current upper-bound-only scope), how the mechanisms compose,
  and VBI's box-constraint handling (#191). The `blocks.md` portfolio example
  now passes callable bounds, matching the enforced API.
- `fischer_burmeister(a, h)` utility for smooth complementarity conditions
- `examples/algorithms/plot_train_against_known_solution.py` gallery example
  (renamed from `plot_maliar_training.py`): trains a shared-backbone
  policy/value network with `train_block_nn` and compares the trained policy
  against the U-2 analytical permanent-income solution. Its docstring now states
  that it uses direct SGD on the MMW'21 objective rather than the iterative
  `maliar_training_loop`.
- `examples/algorithms/plot_maliar_training_loop.py` gallery example: runs the
  full `maliar_training_loop` (all-in-one operator + forward-simulation
  resampling + inner SGD) on the U-3 buffer-stock model, which has no
  closed-form solution, and validates the trained policy against the
  buffer-stock properties `0 < c <= m` and an average propensity to consume that
  declines monotonically with wealth.
- `D-4` benchmark: a deterministic CRRA consumption-savings model with a binding
  borrowing constraint (`c <= m`) and impatience (`betaR = 0.9568 < 1`). The
  binding constraint precludes a closed form, so instead of an analytical policy
  it ships `d4_vfi_reference_policy` (a value-function-iteration oracle, reached
  via the new `get_reference_policy` accessor). The model dynamics and the
  constraint live entirely in the `DBlock`. The accompanying
  `TestD4ConstrainedEulerVFI` trains a policy-only network on the
  Fischer-Burmeister Euler/KKT residual (`EulerEquationLoss(constrained=True)`)
  and matches the oracle to a mean gap of 0.30% (max 0.83%): the in-package
  demonstration that the MMW'21 Euler method reaches benchmark accuracy on a
  constrained problem once the constraint anchors the consumption level.
- `get_reference_policy(model_id)` accessor for benchmarks that have a numerical
  oracle but no closed-form policy.
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
- Added an **Algorithms** user-guide page documenting the direct (non-recurring)
  solve workflow — training a `BlockPolicyNet` against reward-based losses
  (`StaticRewardLoss`, `EstimatedDiscountedLifetimeRewardLoss`) on benchmark
  models (D-2, U-2), including multiple-control solves — with a runnable
  `plot_direct_block_solve.py` gallery example
- Expanded the Algorithms API reference with the `skagent.solver` and
  `skagent.loss` modules and `skagent.ann.train_block_nn`

### Removed

- `train_block_value_and_policy_nn` trainer and its alternating dual-optimizer
  pattern; value-aware training now uses the single shared-backbone
  `BlockPolicyValueNet` trained with one optimizer. The standalone
  `BlockValueNet` is retained for value-function approximation in future
  algorithms but is no longer used in the Maliar training path.
- `value_network` and `value_loss_function` parameters from
  `maliar_training_loop`
- Removed `AgentTypeMonteCarloSimulator`; mortality is now expressed
  declaratively via `mortality_block` (see Changed). The now-unused
  `calibration_by_age` helper and its API documentation entry were removed with
  it.

### Fixed

- The U-1 (Hall random walk) benchmark passed `mean`/`std` to `Normal`, whose
  constructor takes `mu`/`sigma`, so `construct_shocks("U-1")` raised
  `TypeError` and the model was unusable. The income shock is now
  `Normal(mu=0.0, sigma=income_std)`.
- `get_benchmark_model` now returns an independent deep copy of the registered
  block. Previously it returned the shared module-level singleton, so
  `construct_shocks` (which rewrites a block's shock specs in place) leaked
  across callers: a non-default calibration was silently ignored once any other
  caller had constructed the same model, making results depend on execution
  order.
- Fixed the `CRRA` calibration in `perfect_foresight_normalized`: it was a
  1-tuple `(2.0,)`, which broke the CRRA utility power; it is now the scalar
  `2.0`.
- Fixed `skagent.solver.solve_multiple_controls`, which previously crashed on
  its default loss and passed incorrect arguments to `StaticRewardLoss`; it now
  trains a policy network per control via a best-response sweep and returns the
  trained decision rules.
- `train_block_nn` now halts early with a warning on a non-finite (NaN/Inf) loss
  instead of continuing to train on poisoned weights.
- Documentation correctness pass across `docs/` and the examples gallery: the
  `index.md` and quickstart simulation snippets ran `simulate()` without
  `initialize_sim()` and paired `consumption_block` with an initial state it
  does not carry (both crashed verbatim; they now simulate `cons_problem` and
  were verified by execution); the quickstart built `BlockPolicyNet` from a raw
  `DBlock` and read a nonexistent `hidden1` attribute; the parser API example
  used `gamma`, which SymPy parses as the gamma function; `blocks.md` live
  sections referenced a `portfolio_block` defined only inside a comment; the
  benchmark docs said six registry entries instead of seven (D-4 was
  undocumented); gallery narratives contradicted the actual calibrations
  (resource-extraction parameters, the U-2 policy's human-wealth intercept, the
  D-3 median lifetime, and the U-1 smoothing factor).
- New API reference pages for `skagent.bellman`, `skagent.loss`,
  `skagent.distributions`, `skagent.model_analyzer`/`model_visualizer`, and
  `skagent.rule`; added missing entries for `BlockValueNet`,
  `BlockPolicyValueNet`, `train_block_nn`, and `get_reference_policy`; the
  simulation API page no longer documents `MonteCarloSimulator` as a class
  distinct from `Simulator` (it is an alias).
- `solve_multiple_controls` default-loss path crashed (`AttributeError` on
  `None`) and called the loss constructor with a stray positional argument; it
  now resolves `skagent.loss.StaticRewardLoss` and matches its signature. The
  function still has no tests or callers.
- Importing `skagent.simulation.monte_carlo` or `skagent.models.benchmarks` on
  Python 3.9 raised `TypeError` from PEP 604 unions evaluated without
  `from __future__ import annotations`; the imports were added. The `Simulator`
  block parameter is annotated `Union[DBlock, RBlock]` to match documented
  usage.
- Docs build hygiene: `plot_gallery` is a bool; the unused
  `autosummary_generate` flag is gone; the copyright year derives from the build
  date; `sphinx-autobuild` joins the `docs` extra so `make livehtml` works; Read
  the Docs builds with `-W --keep-going` like CI; the empty
  `examples/simulation/` gallery section was removed.
- Project metadata: the PyPI description placeholder ("A great package.") was
  replaced and the Development Status classifier bumped from Planning to
  Pre-Alpha, matching the roadmap's v0.1 proof-of-concept status.

...

[Unreleased]: https://github.com/scikit-agent/scikit-agent/commits/main
