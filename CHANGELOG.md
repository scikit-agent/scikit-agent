# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- Benchmark blocks `d2_block` and `d3_block` imposed a no-borrowing constraint
  (`c <= m`, i.e. end-of-period assets `a' >= 0`) that contradicts their
  unconstrained perfect-foresight closed-form policies, which borrow against
  human wealth. The blocks now use the natural borrowing limit `c <= m + H`,
  with `H = y / r` derived from each calibration as a module-level constant, so
  each coded model matches the analytical policy it is validated against. The
  mismatch was latent until a solver exercised the control bounds; single-step
  and analytical-policy tests missed it because their test states (`a >= 0.5`)
  never reach the borrowing region.

### Added

- `tests/test_benchmark_bound_consistency.py`: regression test asserting each
  unconstrained closed-form benchmark's analytical policy is feasible under the
  block's own control bounds on states that reach the borrowing region
  (`a' < 0`).

### Changed

- `GymEnv._bounds_at` treats a single-point feasible set (`lo == hi`, which the
  natural borrowing limit produces at `m = -H`) as valid, returning that point;
  it now raises only on a genuinely inverted bound (`hi < lo`).

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
  (equation 25) for both the lower-bound and upper-bound sides of the
  complementarity condition. A control with an `upper_bound` uses
  `FB(f, ub - x)`, a `lower_bound` uses `FB(-f, x - lb)`, and a control with
  both uses a two-sided form that reduces to either one-sided residual when the
  opposite bound is slack (#191)
- `EulerEquationLoss` now estimates the squared expected Euler residual with the
  all-in-one operator: the _product_ of two residuals at independent next-period
  shock draws (Maliar, Maliar, and Winant 2021, JME), rather than the square of
  a single draw. The product is an unbiased estimate of `(E[f])**2`, whereas
  squaring one draw adds `Var(f) >= 0` and biases the solution of any stochastic
  model. For deterministic models the two draws coincide and the loss is
  unchanged.
- `estimate_euler_residual` resolves the discount factor dynamically from the
  model and supports multi-control models (returns a dict for >1 controls)
- Control bounds (`lower_bound`, `upper_bound`) accept either a number (a
  constant bound) or a callable of the control's information-set variables.
  Numbers are normalized to zero-argument callables at the `Control` boundary,
  so every downstream consumer sees a uniform callable interface (#191).
- Introduced `mortality_block` (and `mortal_cons_problem`) to demonstrate how to
  encode stochastic mortality and agent rebirth as a composable `DBlock`.
- `train_block_nn` now always returns a 3-tuple
  `(network, final_loss, optimizer)`; previously it returned a 2-tuple unless an
  optimizer was passed in. Callers should unpack three values.
- `maliar_training_loop` accepts an `lr` argument controlling the learning rate
  of its internal Adam optimizer.
- Consolidated the open-bounds scaling and decision-function plumbing shared by
  `BlockPolicyNet` and `BlockPolicyValueNet` into `BellmanPeriodMixin`.
- `skagent.algos.vbi.ar_from_data` now produces decision rules that follow the
  library's calling convention — positional arguments in `control.iset` order
  (`dr(*iset_values)`) instead of the previous keyword form (`dr(m=…)`) — so a
  VBI-fitted rule is a drop-in for `BellmanPeriod`, `loss`, and `solver`.
  `vbi.solve` transposes each fitted policy to `control.iset` order to guarantee
  the positional argument order regardless of how the caller ordered the grid.
- Renamed `vbi.solve`'s `calibration` argument to `scope`. VBI uses it as the
  general evaluation scope (merged with each grid point to form `pre_states`),
  which legacy usage populates with fixed parameters _and_ fixed exogenous
  values such as a shock realization — broader than the parameters-only
  `calibration` used elsewhere in the library.
- Rewrote `skagent.algos.vbi` docstrings in numpy/scipy style; the module and
  `solve` docstrings now document VBI's full-observation assumption (the
  per-point optimization conditions on the complete information set and does not
  integrate over unobserved variables).

### Added

- `vbi.bellman_step`: one exact value backup on the `BellmanPeriod` protocol —
  the per-iteration update of value-function iteration on the interface the
  torch stack speaks, with explicit discount factor, multi-reward summation, and
  deterministic (empty-shock) handling. Returns
  `(dr_from_data, value_array, policy_array)`. Optimizes one or more controls
  jointly (`scipy.optimize.minimize` over the stacked control vector) and
  reprojects each policy onto its own information set (design §5): drops grid
  axes outside a control's iset (Mechanism A) and reindexes a derived pre-state
  like `m = a·R + y` onto its own coordinate (Mechanism B). Legacy `vbi.solve`
  is unchanged (the deliberate discount-folded-into-continuation path).
- `vbi.solve_bellman`: value-function iteration driving `bellman_step` to a
  fixed point — each backup takes the previous iterate's value grid as its
  continuation (via the new `vbi.value_array_to_function`) and warm-starts the
  optimizer from the previous policy. Stops on the sup-norm value change
  (`converged`, `n_iter`, `residual` reported on `value_array.attrs`);
  non-convergence warns, or raises under `raise_on_nonconvergence`.
  Deterministic scope: internal shock discretization (`disc_params`) is not yet
  implemented.
- **Constraints** user-guide page documenting the ways to constrain an
  optimization problem: bound declaration on `Control`, the open-bounds
  policy-network transforms, the bilateral Fischer-Burmeister complementarity
  loss, how the mechanisms compose, and VBI's box-constraint handling, with a
  table of where each mechanism is available (#191).
- **Maliar method** user-guide page explaining the all-in-one expectation
  operator, the Euler and Bellman residual losses (and the slope-versus-level
  identification difference between them), `maliar_training_loop`, and how
  bounded controls connect to the constraints page (#215). The Algorithms guide
  now links to it and its duplicate "Solving a block directly" heading is
  retitled to "Overview".
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
- Added an **Algorithms** user-guide page documenting the direct (non-recurring)
  solve workflow — training a `BlockPolicyNet` against reward-based losses
  (`StaticRewardLoss`, `EstimatedDiscountedLifetimeRewardLoss`) on benchmark
  models (D-2, U-2), including multiple-control solves — with a runnable
  `plot_direct_block_solve.py` gallery example
- Expanded the Algorithms API reference with the `skagent.solver` and
  `skagent.loss` modules and `skagent.ann.train_block_nn`
- `skagent.algos.vbi.tensor_decision_rule`, which wraps a numpy-space VBI
  decision rule so it accepts and returns torch tensors (float32 on the grid
  device, detached) for interop with the torch solving stack. Suitable as a
  fixed / ground-truth / warm-start policy, not as a trainable FOC/Euler policy.

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
