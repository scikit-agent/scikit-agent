# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `skagent.relevance` gains the four single-decision incentive criteria of
  Everitt et al. (AAAI-21): `admits_voi`, `admits_ri`, `admits_voc` and
  `admits_ici`, with the `is_requisite` test and `minimal_reduction` they rest
  on. Out-of-domain and multi-decision queries raise rather than answer.
- `SCIM.with_edge` and `SCIM.without_edges`, the transforms those criteria are
  posed over, and `SCIM.utilities`, the utilities an agent owns whether or not
  its decision reaches them.
- `skagent.models.safety`, a package for influence diagrams from the AI-safety
  literature, opening with `incentives.py`: the grade-prediction and
  content-recommendation diagrams of Everitt et al. (Figs. 3a, 3b, 4a, 4b), each
  paired with the redesign that drops an incentive, plus `print_incentive_table`
  and `draw_shocks` for reading them.
- `skagent.utils.plot_block_diagram`, which draws a block's model diagram onto a
  matplotlib figure rather than into a notebook.
- Two gallery examples reading incentives off those diagrams:
  `examples/models/plot_incentives_1_grade_prediction.py`, where value of
  information and the response incentive separate, and
  `plot_incentives_2_content_recommendation.py`, where the two control criteria
  do. Each shows only the pair of criteria it develops, then checks those
  readings numerically against the mechanisms.

### Changed

- Every gallery page now opens with a short, page-specific summary, so the
  gallery's hover text distinguishes the examples instead of repeating shared
  framing, and each page that draws a model diagram uses it as its thumbnail.

### Changed

- Block dynamics no longer call `inspect.signature` once per variable per pass.
  `skagent.utils.param_names` memoizes a function's parameter names, and
  `takes_arguments` reads a decision rule's arity off its code object. Answers
  are unchanged; a block transition is about 3x faster, and value function
  iteration on the D-4 benchmark about 1.5x.

## [0.1.0] - 2026-08-12

First release.

### Fixed

- `tree_killer_block`'s chance mechanisms called torch's `.float()`, so the
  model could not be simulated on the numpy path its shocks declare. Its payoffs
  also left every decision optimal independently of the others: the patio's
  value now rises when the tree dies, and the magnitudes give the game strategic
  tension. Its shocks are now declared in constructor-tuple form, so
  `construct_shocks` can seed them.
- A continuation rebuilt by `vfi.value_array_to_function` read only the axes of
  the value grid, so an arrival state that was not gridded was silently
  discarded. It now raises `ValueError` when handed one. This was not academic:
  with D-3's survival state `liv` off the grid, the continuation could not
  depend on `liv'`, the survival probability cancelled out of the backup, and
  value iteration converged to the no-mortality MPC.

- `vfi.bellman_step` marks a grid point's optimum **unidentified** when two
  multi-start optima tie in value but disagree in the control, and the
  information-set projection now takes its invariance check and its surviving
  slice over identified points only. Previously an absorbing state with zero
  reward and continuation -- D-3's dead `liv = 0` slice, where every control is
  optimal and the optimizer returns its seed -- made the projection raise on a
  policy spread that carried no information.

- `vfi.bellman_step` seeds each per-point optimization from a _set_ of
  candidates (warm start, midpoint of finite bounds, and `x0` clamped into the
  bounds) and keeps the best optimum, instead of picking one seed by a priority
  rule. Under the old rule `x0` was reachable only when a bound was open, so a
  box like the natural borrowing limit's `[0, m + H]` was seeded at its midpoint
  -- far above the optimum and outside the optimizer's basin -- and in
  `solve_bellman` a collapsed iterate then re-seeded the next backup at its own
  bound. Candidates beyond the first are optimized only when their seed already
  matches or beats the incumbent optimum, which for a unimodal objective costs
  one extra function evaluation rather than an extra optimization.

- `d3_block` (Blanchard mortality) was unusable by the `vfi` solver and did not
  match its own analytical policy. Two fixes: the reward `liv * crra_utility(c)`
  now coerces `liv` with `as_tensor` (a bare `numpy * tensor` raised `TypeError`
  on the grid-backup path, same class as the `_clamp_min` fix); and utility is
  now computed from the _arrival_ `liv` (declared before the survival update
  `liv = liv * live`), the Blanchard "consume then face mortality" timing, so
  the solver recovers `c = kappa_s * (m + H)` exactly instead of drifting off by
  `O(1 - s)`. Perfect-foresight (`live = 1`) simulations are unchanged.

- `d1_block`'s consumption control declared an upper bound but no lower bound,
  so the `vfi` solver optimized over `[-1e12, W]` and its line search reached
  the `log(c < 0)` region. The optimizer then aborted and returned its own seed,
  which presented as a converged flat objective: `c = W` at every period,
  reported `converged=True` with a zero residual, and insensitive to grid
  refinement. `c` now has a `1e-4` floor, as `d4_block` and `u2_block` already
  do.

- `u2_block`'s cash-on-hand dynamic guarded a division with `torch.clamp`, which
  rejects the numpy/scalar inputs the VFI solver passes, so the block could not
  be solved by `vfi`. A `_clamp_min` helper now clamps on both torch tensors and
  numpy/Python scalars, leaving the tensor path unchanged.

- `u3_block`'s cash-on-hand dynamic carried the same `torch.clamp` guard and so
  was likewise unsolvable by `vfi`; it now uses `_clamp_min`. The tensor path is
  unchanged. Latent because no test had solved U-3 on the numpy path.

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

- D-3 (Blanchard mortality) VFI benchmark:
  `test_d3_single_backup_analytic_continuation` recovers `c = kappa_s * (m + H)`
  from a single `bellman_step`, integrating the hidden 2-node `Bernoulli`
  survival shock. `test_d3_iterated_converges_to_analytic` reaches the same
  policy by value iteration, with the survival state `liv` on the state grid.

- D-1 (finite-horizon log utility) VFI benchmark:
  `test_d1_finite_horizon_converges_to_analytic` recovers the non-stationary
  rule `c_t = (1 - beta)/(1 - beta^(T-t)) * W` to ~1% by ordinary value
  iteration, with the time counter `t` on the state grid. No finite-horizon code
  path is needed: `t` is an arrival state and the horizon is the reward's
  `(t < T)` cutoff, so backward induction is a fixed point in the extended state
  space, reached in O(T) iterations. The `t` axis must extend one slice past the
  last nonzero reward, so that the continuation's linear extrapolation off the
  top of the axis is flat at zero rather than a reflection of the last consuming
  period.

- U-1 (Hall random walk) VFI benchmark:
  `test_u1_continuous_shock_recovers_pih_closed_form` recovers
  `c = (r/R)(m + H)` from a single `bellman_step` under an analytic PIH
  continuation. The first benchmark with a continuous shock, and so the only
  benchmark-level exercise of `disc_params`: `eta ~ Normal` becomes a
  Gauss-Hermite node axis, and `m` then varies along both it and the asset axis.

- U-3 (buffer stock, two income shocks) VFI benchmarks. Both `psi` and `theta`
  feed the pre-state `m`, so both become node axes and `m` varies along all
  three grid axes while pinning down neither shock individually -- the case
  where the gather-and-fit consistency check is load-bearing.
  `test_u3_two_prestate_shocks_degenerate_limit` recovers the PIH closed form in
  the limit where U-3 reduces to U-2 (`sigma_theta = 0`, `CRRA = 1`), the only
  limit in which U-3 has one. `test_u3_two_prestate_shocks_properties` asserts,
  at U-3's own calibration, the properties that do not depend on the supplied
  continuation being the model's own.

- `skagent.information`: classifies each shock, per control, by whether the
  control's information set accounts for it -- `observed` (every route to the
  objective is intercepted, so a solver may condition on it), `hidden`
  (integrate inside the maximization), or `mixed` (partly informed _and_
  separately relevant, which needs filtering, so refuse). A d-separation test,
  not a syntactic one: a shock in no information set is still accounted for if
  it only reaches the objective through a pre-decision variable that is.
  `d_connected` is a Bayes-Ball sweep answering every candidate node in one
  traversal.
- `ModelAnalyzer.influence_graph(dynamic=True)` and `Block.shock_roles()`. The
  `dynamic` option makes a single-period diagram faithful to one period of a
  recurring problem: each reassigned variable's arrival value becomes its own
  `<name>*` node, and each deciding agent gets a continuation-value utility
  node. Without the first, conditioning on a variable's information-set entry
  conditions on next period's value; without the second, a shock reaching the
  objective only through the next period looks irrelevant.

- `skagent.relevance`: strategic-relevance analysis via the Koller & Milch
  s-reachability criterion. `is_s_reachable` and a `RelevanceGraph` wrapper
  (`relies_on`, `is_acyclic`, `sccs`, `condensation`, `draw`) over an
  influence-diagram `networkx.DiGraph`. Validated against Koller & Milch (2001)
  Fig. 3 (a)-(e).
- `ModelAnalyzer.influence_graph()`: returns the influence-diagram (SCIM) view
  consumed by `skagent.relevance` -- chance/decision/utility nodes with causal
  edges, parameter nodes dropped.
- `Block.relevance_graph()` and `Block.relies_on()`: strategic-relevance API on
  any block (`calibration` defaults to empty, since relevance is structural).
- `skagent.models.macid`: multi-agent influence diagram illustration models,
  starting with the Tree Killer (`tree_killer_block`), the Koller & Milch (2001)
  Fig. 1 MAID. Chance nodes are encoded as structural CPDs (deterministic
  mechanism plus an exogenous shock) and binary decisions relaxed to `[0, 1]`
  controls; cross-checked against PyCID's `tree_doctor` example.
- `skagent.algos.best_response`: solves a block's decisions one at a time in the
  order its relevance graph implies, each maximizing its own agent's payoff
  conditional on what it observes. `TabularBestResponseSolver` (`solve`,
  `best_response`, `conditional_payoffs`, `mixed_rule`) and `TabulatedRule`, a
  decision rule tabulated over information cells. A cyclic relevance graph
  raises rather than returning rules that are not best responses.
- `Block.calc_reward` takes an optional `agent`, computing only that agent's
  reward variables; their sum is the agent's payoff.
- `examples/models/plot_tree_killer_relevance.py`: computes the Tree Killer's
  relevance graph, reads a solution order off it, solves the game in that order,
  and checks each reliance claim numerically.
- `tests/test_benchmark_bound_consistency.py`: regression test asserting each
  unconstrained closed-form benchmark's analytical policy is feasible under the
  block's own control bounds on states that reach the borrowing region
  (`a' < 0`).
- Block guide: declaration order and symbol aliasing (a reward reads its inputs
  as of its own declaration point, so reward and transition can see different
  values for one symbol), plus three authoring rules for solvable blocks --
  declare both control bounds when the reward is undefined outside the feasible
  set, extend a terminal axis one slice past the last nonzero reward, and keep
  dynamics agnostic about torch vs numpy input.

### Changed

- New `skagent.influence` module owns the influence-diagram substrate: `SCIM` is
  now a class carrying the conditioning-context and objective vocabulary, a
  memoized Bayes-Ball d-separation engine, and the graph transforms, instead of
  a five-field namedtuple. `is_s_reachable(scim, d1, d2)`,
  `RelevanceGraph.from_scim(scim)` and `shock_roles(scim, shocks)` replace their
  positional-argument forms.

- `vfi` projects a policy onto an information-set variable that varies along
  **several** grid axes by gathering every `(coordinate, control)` pair,
  sorting, and fitting one 1-D rule, replacing the previous single-axis relabel
  plus monotonicity assert. Sorting supplies well-posedness, so monotonicity is
  no longer required; the guard is instead that samples sharing a coordinate
  must agree in the control, which checks the block's own claim that the control
  depends on nothing outside its information set. Restricted to a
  single-variable information set -- with other variables present, each of their
  slices would gather a different coordinate -- and raising otherwise.

- `ModelAnalyzer` classifies a dependency as `lag` by **declaration position**
  rather than by membership in the arrival-state set: a dependency reads its
  symbol's pre-assignment value unless that symbol is assigned earlier in the
  order. Whether that is a lag then depends on what supplies the pre-assignment
  value -- the previous period (`lag`), the calibration (`param`, even if the
  block reassigns the symbol later), or a within-period shock (`shock`). The
  membership test could not distinguish a variable read before its own update
  from one read after, so a dependency on a post-update value was reported as
  lagged.

- `ModelAnalyzer` assigns a control that declares no `agent` to the block's sole
  reward-owning agent when there is exactly one. Such a control was previously
  `"global"` while the reward belonged to a named agent, so the two never met
  and the control appeared to own no reward.
- Development and CI use [uv](https://docs.astral.sh/uv/) instead of pip: GitHub
  Actions installs via `astral-sh/setup-uv`, Read the Docs via asdf, both run
  `uv sync`; source/contributor docs use `uv sync` / `uv run`. The public PyPI
  install remains `pip install scikit-agent` (#166).

- `GymEnv._bounds_at` treats a single-point feasible set (`lo == hi`, which the
  natural borrowing limit produces at `m = -H`) as valid, returning that point;
  it now raises only on a genuinely inverted bound (`hi < lo`).

- `compute_gradients_for_tensors` returns a zero tensor (instead of `None`) for
  a variable with no computational path to the target, and raises `ValueError`
  when a `wrt` tensor does not require gradients; the `BellmanPeriod` gradient
  methods (`grad_reward_function`, `grad_transition_function`,
  `grad_pre_state_function`) inherit the tensor-only contract (#129)
- Declared `networkx >=3.3` as a dependency (previously transitive); the floor
  provides `is_d_separator`, used by `skagent.relevance`
- `ModelAnalyzer` now builds an annotated `networkx.DiGraph` (`self.G`) as its
  source of truth and derives `node_meta` / `edges` from it; `to_dict()` output
  is unchanged (byte-for-byte regression tested), so `ModelVisualizer` is
  unaffected
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
- `skagent.algos.vfi.ar_from_data` now produces decision rules that follow the
  library's calling convention — positional arguments in `control.iset` order
  (`dr(*iset_values)`) instead of the previous keyword form (`dr(m=…)`) — so a
  VFI-fitted rule is a drop-in for `BellmanPeriod`, `loss`, and `solver`.
  `vfi.solve` transposes each fitted policy to `control.iset` order to guarantee
  the positional argument order regardless of how the caller ordered the grid.
- Renamed `vfi.solve`'s `calibration` argument to `scope`. VFI uses it as the
  general evaluation scope (merged with each grid point to form `pre_states`),
  which legacy usage populates with fixed parameters _and_ fixed exogenous
  values such as a shock realization — broader than the parameters-only
  `calibration` used elsewhere in the library.
- Rewrote `skagent.algos.vbi` docstrings in numpy/scipy style; the module and
  `solve` docstrings now document VBI's full-observation assumption (the
  per-point optimization conditions on the complete information set and does not
  integrate over unobserved variables).

### Added

- `vfi.bellman_step`: one exact value backup on the `BellmanPeriod` protocol —
  the per-iteration update of value-function iteration on the interface the
  torch stack speaks, with explicit discount factor, multi-reward summation, and
  deterministic (empty-shock) handling. Returns
  `(dr_from_data, value_array, policy_array)`. Optimizes one or more controls
  jointly (`scipy.optimize.minimize` over the stacked control vector) and
  reprojects each policy onto its own information set (design §5): drops grid
  axes outside a control's iset (Mechanism A) and reindexes a derived pre-state
  like `m = a·R + y` onto its own coordinate (Mechanism B). Legacy `vfi.solve`
  is unchanged (the deliberate discount-folded-into-continuation path).
- `vfi.solve_bellman`: value-function iteration driving `bellman_step` to a
  fixed point — each backup takes the previous iterate's value grid as its
  continuation (via the new `vfi.value_array_to_function`) and warm-starts the
  optimizer from the previous policy. Stops on the sup-norm value change
  (`converged`, `n_iter`, `residual` reported on `value_array.attrs`);
  non-convergence warns, or raises under `raise_on_nonconvergence`.
- Discretized shock expectations in VFI (`disc_params`): `bellman_step`
  integrates _hidden_ shocks (in no control's information set) inside the
  per-point `max` via `Distribution.discretize` + `expected`, and
  `vfi.value_array_to_function` integrates _observed_-shock grid axes out of the
  arrival value (`W(s) = E_obs[V(s, obs)]`); `solve_bellman` threads
  `disc_params` through both. This lifts VFI's old full-observation restriction,
  so problems whose optimum is an expectation over an unobserved shock (and the
  U-2 log-utility permanent-income benchmark) are now solvable.
- **Constraints** user-guide page documenting the ways to constrain an
  optimization problem: bound declaration on `Control`, the open-bounds
  policy-network transforms, the bilateral Fischer-Burmeister complementarity
  loss, how the mechanisms compose, and VFI's box-constraint handling, with a
  table of where each mechanism is available (#191).
- **Maliar method** user-guide page explaining the all-in-one expectation
  operator, the Euler and Bellman residual losses (and the slope-versus-level
  identification difference between them), `maliar_training_loop`, and how
  bounded controls connect to the constraints page (#215). The Algorithms guide
  now links to it.
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
- `skagent.models.benchmarks.d2_optimal_c`, the D-2 closed-form consumption
  function `c = κ(m + H)` keyed on cash-on-hand
- Gallery example `examples/algorithms/plot_sb3_ppo.py` demonstrating PPO on the
  D-4 benchmark (binding borrowing constraint, no closed form; validated against
  a VFI reference)
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
- `skagent.algos.vfi.tensor_decision_rule`, which wraps a numpy-space VFI
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

- The D-2 benchmark's consumption control had no lower bound, so an exact solver
  could drive `c` negative (where CRRA utility is unbounded); added the `c >= 0`
  floor the sibling blocks already carry.
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

[Unreleased]: https://github.com/scikit-agent/scikit-agent/compare/v0.1.0...main
[0.1.0]: https://github.com/scikit-agent/scikit-agent/releases/tag/v0.1.0
