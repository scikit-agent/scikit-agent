# i208 Phase 2 — Design: `vbi.bellman_step` + `vbi.solve_bellman`

Re-base the exact (per-grid-point `scipy.optimize`) backward-induction solver
onto the `BellmanPeriod` protocol, so VBI speaks the same model interface as the
rest of the torch stack (`bellman.py`, `loss.py`, `ann.py`, `solver.py`). Legacy
`vbi.solve(block, continuation, …)` is left untouched as the deliberate
"β-folded-into-continuation" path.

A `BellmanPeriod` represents one _recurring_ period of an
infinite/finite-horizon problem, so the exact backup must be **iterated**: the
continuation at iteration _n_ is iteration _(n−1)_'s value function. The design
therefore splits into:

- **`bellman_step`** — one exact value-backup over the grid (the per-iteration
  update). Reusable on its own; under a zero/terminal continuation it _is_ the
  single-step solver the conftest cases assert against.
- **`solve_bellman`** — wraps `bellman_step` in a value-iteration loop that
  rebuilds the continuation from the previous iteration's value grid and stops
  on convergence or a max-iteration limit.

Status: **in progress** — PRs 1–4 (§9 steps 1–4) are implemented and tested:
`bellman_step` core, Mechanism-B reindex, multi-control vectorization, and
non-trivial `continuation_vf`. PRs 5–7 remain. See §9 for the per-PR breakdown
and progress.

---

## 1. Goals / non-goals

**Goals (the five Phase-2 gaps):**

1. **Explicit β.** Backup is `r + β·cv` with
   `β = bp.resolve_discount_factor(post)`.
2. **Multi-reward.** Period reward is `Σ_r reward[r]` over
   `bp.get_reward_syms(agent)`.
3. **Deterministic-safe.** All mechanics go through `bp` methods, which are
   empty-shock-safe; never touch `discretized_shock_dstn` (which crashes on
   `shocks={}`).
4. **Multiple controls.** One `scipy.minimize` over a stacked control vector
   with per-control bounds, mirroring `BlockPolicyNet._setup_bound` semantics.
5. **Grid → per-control iset projection.** Grid ranges over the value function's
   domain; each control's rule is reduced/reindexed to its own `iset`.

Plus the structural point this revision adds: **iterate the backup to a fixed
point** (value-function iteration), since `BellmanPeriod` is a recurring period.

Shock handling is upgraded relative to legacy `solve`: because
`distributions.py` can discretize a shock into nodes+weights (§4), the per-point
backup integrates **hidden** shocks inside the `max` and the iteration
integrates **observed** shocks into the arrival value — so VBI's old
full-observation restriction is lifted (hidden-shock optima like `case_2` are
now in scope).

**Non-goals:**

- Still an **exact grid** solver — the ground-truth / warm-start tool, not a
  replacement for `solver.solve_multiple_controls` (neural). Cost scales with
  grid size × shock nodes × control-vector optimization per point.
- Discretization quality is the user's `disc_params` choice; a coarse node count
  is an accuracy knob, not a correctness guarantee.

---

## 2. `bellman_step` — one exact backup

```python
def bellman_step(
    bp: BellmanPeriod,
    continuation_vf,  # callable(states, shocks, parameters) -> value
    state_grid: Grid,  # {var: 1-D sequence}; see §5 for which vars
    *,
    agent: str | None = None,
    scope: dict = {},  # fixed non-shock exogenous values (parameters)
    disc_params: dict = {},  # per-shock discretization args (§4)
    x0: float = 1.0,  # fallback seed; see per-point x0 policy below
    x0_policy: dict | None = None,  # {control_sym: DataArray} warm-start seeds (§3)
) -> tuple[dict[str, Callable], xr.DataArray, xr.DataArray]:
    ...
    return dr_from_data, value_array, policy_array
```

**Per-point `x0` (optimizer robustness).** The starting vector is chosen per
grid point, per control, in this priority order:

1. **Warm-start** — if `x0_policy` is given, `x0_j = x0_policy[c_j].sel(point)`
   (the previous iterate's optimum at this point; supplied by `solve_bellman`).
2. **Midpoint** — else, if both bounds are finite, `x0_j = (lb_j + ub_j) / 2`.
3. **Fallback** — else (an open bound), the scalar `x0` default (`1.0`).

This replaces legacy's arbitrary constant `1.0`, which can start on or outside a
bound; midpoint/warm-start keep multi-control L-BFGS-B inside the feasible box
and near the optimum.

Shocks are no longer passed as fixed realizations: hidden shocks are integrated
out via `disc_params` (§4), and observed shocks are gridded over their
discretization nodes. `scope` now holds only fixed non-shock exogenous values.

`continuation_vf` uses the `bp.compute_value` convention
`(states, shocks, parameters)`, so a `BlockValueNet.get_value_function()`, the
`value_array_to_function` interpolant of §4, or a plain closure all drop in.
Terminal continuation is `lambda s, sh, p: 0.0`.

Return shape (confirmed with user):

- `dr_from_data` — `{control_sym: callable}`, positional args in `control.iset`
  order, numpy/xarray space (wrap with `tensor_decision_rule` for the torch
  stack). Built as in Phase-1 `solve`: `ar_from_data(policy.transpose(*iset))`.
- `value_array` — gridded optimized **decision** value `V(s,e)` over the state
  grid.
- `policy_array` — gridded optimal controls (`dict[str, DataArray]` for
  multi-control — see O1).

### Per-point algorithm

For each point of the state grid (Cartesian product, as in legacy `solve`):

```
# A grid point fixes the arrival states and any OBSERVED shock node:
state_vals = dict(zip(state_grid.keys(), point))         # arrival states
obs        = {observed-shock node values at this point}  # from §4 discretization
states     = {arrival-state subset of state_vals}
params     = bp.calibration | scope
disc_hidden = discretized_shock_dstn(hidden_shocks, disc_params)   # §4; or None

# bounds, per control, evaluated at this point (pre-state available)
for control c_j:
    pre_j  = bp.compute_pre_state(c_j, states, shocks=obs, parameters=params)
    lb_j   = c_j.lower_bound(*[ (pre_j|params|obs|state_vals)[v] for v in sig(lower) ]) or -1e12
    ub_j   = c_j.upper_bound(...) or +1e12
bounds = [(lb_j, ub_j) for each control]
# per-control seed: warm-start > midpoint-of-finite-bounds > x0 fallback
x0_vec = [ x0_policy[c_j].sel(point) if x0_policy is not None
           else (lb_j + ub_j) / 2 if isfinite(lb_j) and isfinite(ub_j)
           else x0
           for each control c_j ]

def value_at(a, e):                                      # one hidden-shock node e
    controls = {c_j: a[j] for j}
    sh       = obs | e
    rewards  = bp.reward_function(states, controls, shocks=sh, parameters=params, agent=agent)
    r        = sum(rewards[s] for s in bp.get_reward_syms(agent))
    post     = bp.post_function(states, controls, shocks=sh, parameters=params)
    beta     = bp.resolve_discount_factor(post)
    s_next   = bp.transition_function(states, controls, shocks=sh, parameters=params)
    return r + beta * continuation_vf(s_next, sh, params)

def negated_value(a):                                    # a = stacked control vec
    if disc_hidden is None:                              # deterministic / no hidden shock
        return -value_at(a, {})
    return -expected(lambda e: value_at(a, e), disc_hidden)   # E_hidden inside the max

res = scipy.optimize.minimize(negated_value, x0=x0_vec, bounds=bounds)
value_array[point]       = -res.fun                          # V(s, obs)
policy_array[c_j][point] = res.x[j]
```

Notes / refactors vs legacy `solve`:

- **Multi-control (gap 4).** `negated_value` already takes a vector; the change
  is building the `controls` dict / `bounds` list by iterating controls and
  dropping the `>1 → raise`. `minimize(..., bounds=...)` (L-BFGS-B) handles the
  box.
- **β (gap 1).** Resolved _inside_ `value_at` from `post`, so β can be
  state/shock-dependent — and for D-3 the `s·β` discount emerges from the hidden
  `live` expectation (`E[liv']=s·liv`), not a hand-coded factor. Verified
  `block.transition` returns `vals = pre.copy() | …`, so calibration vars
  (`DiscFac`) reach `post`.
- **Multi-reward (gap 2):** sum over `get_reward_syms(agent)`.
- **Deterministic-safe (gap 3):** `disc_hidden is None` when `bp.get_shocks()`
  is empty → direct evaluation, no `discretized_shock_dstn` call (which would
  crash on empty shocks).
- **`scope` vs `params` (resolves legacy's single `scope` bag).**
  `parameters = bp.calibration | scope`; `scope` holds fixed non-shock exogenous
  values only. Shocks are handled by discretization, not a fixed-realization
  bag. (O2.)
- Optimizer-failure handling carried over from legacy (store result, log on
  `not res.success`), minus the stray `print`s.

---

## 3. `solve_bellman` — value iteration to a fixed point

```python
def solve_bellman(
    bp: BellmanPeriod,
    state_grid: Grid,
    *,
    continuation_vf=None,  # initial guess; None -> terminal (zero)
    agent: str | None = None,
    scope: dict = {},
    disc_params: dict = {},  # shock discretization (§4)
    tol: float = 1e-6,  # sup-norm on value change
    max_iter: int = 100,
    x0: float = 1.0,
    raise_on_nonconvergence: bool = False,  # O5
) -> tuple[dict[str, Callable], xr.DataArray, xr.DataArray]:
    cont = continuation_vf or (lambda s, sh, p: 0.0)
    value_prev = None
    x0_policy = None  # warm-start: prev iterate's optimum
    converged = False
    for it in range(max_iter):
        dr, value_array, policy_array = bellman_step(
            bp,
            cont,
            state_grid,
            agent=agent,
            scope=scope,
            disc_params=disc_params,
            x0=x0,
            x0_policy=x0_policy,
        )
        if value_prev is not None:
            resid = float(np.abs(value_array - value_prev).max())
            if resid < tol:
                converged = True
                break
        value_prev = value_array
        x0_policy = policy_array  # seed next backup from this optimum
        cont = value_array_to_function(value_array, bp, disc_params)  # W_n from V_n
    value_array.attrs.update(n_iter=it + 1, converged=converged, residual=resid)
    if not converged:  # O5: warn (or raise), never silent
        if raise_on_nonconvergence:
            raise RuntimeError(
                f"solve_bellman did not converge in {max_iter} iters "
                f"(residual={resid})."
            )
        warnings.warn(
            f"solve_bellman did not converge in {max_iter} iters "
            f"(residual={resid}); returning last iterate."
        )
    return dr, value_array, policy_array
```

- **Iteration 1** uses the terminal (zero) continuation → its result is exactly
  the single-step solution the conftest cases assert (so those tests can call
  `bellman_step` directly, or `solve_bellman(max_iter=1)`).
- **Stationary fixed point.** For an infinite-horizon problem the loop converges
  geometrically (modulus β) to the stationary V; for finite horizon, set
  `max_iter = T` and read each period's policy.
- **Convergence info** is attached to `value_array.attrs` (`n_iter`,
  `converged`, `residual`), keeping the 3-tuple return. A non-converged run
  returns the last iterate with `converged=False` **and emits a
  `warnings.warn`** — it does not raise (scipy `OptimizeResult.success`
  convention; the partial iterate stays usable for inspection/warm-start). An
  optional `raise_on_nonconvergence=False` lets a caller opt into strictness.
  (O5.)
- Optional **damping** (`V ← (1−λ)V_old + λV_new`) is a possible knob if a model
  oscillates; omit from v1 unless a test needs it.

---

## 4. Shock expectations via internal discretization

`distributions.py` already turns a continuous shock into a grid: every
`Distribution` exposes `.discretize(**kwargs) -> DiscreteDistribution`
(`.points`, `.weights`), `block.discretized_shock_dstn(shocks, disc_params)`
combines independent shocks into one joint discrete distribution, and
`distributions.expected(func, dist)` does the weighted sum. (This is exactly
what legacy `get_arrival_value_function` uses.) So **the solver discretizes
shocks internally — they are never state-grid axes**, and there is no "axes must
align with `disc_params` nodes" problem. This supersedes the old O6.

`disc_params` (e.g. `{"theta": {"N": 7}}`) is threaded into both `bellman_step`
and `solve_bellman`. Empty-shock-safe by guard: `combine_indep_dstns()` with no
distributions raises, so when `bp.get_shocks()` is empty the expectation
collapses to a single direct evaluation.

### Observed vs hidden shocks

A shock that appears in some control's `iset` is **observed** (the agent
conditions on its realization); one that does not is **hidden** (realized around
the decision but not seen when choosing). The two integrate at different points:

- **Hidden shocks** are integrated _inside_ the `max` of the per-point backup:
  `V(s, obs) = max_c E_hidden[ Σ_r u_r + β·V'(s') ]`, computed by wrapping the
  §2 objective in `expected(obj, disc_hidden)`. This lifts the old
  full-observation non-goal — `case_2` (`theta` hidden, optimum `c=E[theta]=0`)
  and **D-3** (`live ~ Bernoulli`, giving the `s·β` discount as `E[liv']`)
  become solvable. **U-1** (`eta ~ Normal`) uses Gauss–Hermite nodes from
  `.discretize`.
- **Observed shocks** are gridded over the shock's _discretization nodes_ and
  enter the control's `iset` (Mechanism A/B of §5). The decision value
  `V(s, obs)` is tabulated per node; the arrival value integrates them out.

### `value_array_to_function` — continuation from a value grid

Rebuilds the arrival value `W(s) = E_obs[V(s, obs)]` as a callable for the next
iteration's continuation:

(code block) python

def value_array_to_function(value_array, bp, disc_params={}) -> Callable: # 1.
Integrate out observed-shock node axes: W(arrival) = Σ p(obs) V(arrival, obs), #
weights from the same discretized distribution used to build the nodes. # Empty
/ no-observed-shock -> identity. # 2. Interpolate W over the arrival-state axes
(pointwise interp, ar_from_data trick). # Returns wf(states, shocks, parameters)
-> value over arrival states.

(end code block)

- **Deterministic (D-2, D-4, U-2 at `sigma_psi=0`):** no shocks, both the backup
  expectation and step (1) are identity; `value_array` over arrival states _is_
  `W`. Iteration is exact.
- **Stochastic (D-3, U-1, case_2, …):** hidden-shock expectation handled in the
  backup; observed-shock node axes integrated here. All weights come from
  `.discretize` — nothing has to be hand-aligned.

---

## 5. Two coordinate systems & the iset projection (gap 5)

(Unchanged from the prior design; the crux of `bellman_step`'s policy output.)

Two variable spaces:

- **Arrival states `s`** — consumed by `bp.transition_function`,
  `bp.post_function`, `continuation_vf` (`bp.arrival_states`).
- **Control iset `m_j`** — what `dr[c_j]` is a function of: an arrival state, an
  observed shock, _or a derived pre-decision variable_ (`m = a·R + y`).

Legacy `solve` gridded directly over the iset (using `screen=True`).
`bp.transition_function` exposes no `screen` and expects arrival states, so
`bellman_step` grids over **arrival states + observed shocks** (the value
function's domain) and projects each control's policy down to its iset:

### Mechanism A — iset vars are grid axes ("isel non-iset dims")

Drop the non-iset grid axes, assuming optimum invariance across them (the
definition of "outside the iset"). Implement via `isel` index 0 on each dropped
dim, guarded by an **invariance assert** (max spread over dropped axes < tol) so
a non-invariant case fails loudly. Then `transpose(*iset)` + `ar_from_data`.

- `case_0`: grid `[a]`, iset `[a]` → no reduction.
- `case_1`: grid `[a, theta]`, iset `[a, theta]` → no reduction (`theta`
  observed).
- `case_10`: grid `[a]`; `c.iset=[a]`; `d.iset=[]` → reduce `a`, constant rule.
- `case_11`: grid `[a, theta]`, iset `[a, theta]` → no reduction; real
  continuation.

### Mechanism B — iset var is a derived pre-state, reindex

At each grid point we already compute the iset value via `bp.compute_pre_state`;
build `policy_array[c_j]` on the **computed iset coordinate**, then
`ar_from_data`.

- `case_3`: grid `[m]`-derived; `psi` in `scope`.
- `d2`/`d4`/`u2`: grid `[a]`; `c.iset=[m]`, `m = a·R + y` (affine, monotone) →
  emit a `DataArray` indexed by `m`; `interp` over the regular `m` coordinate is
  exact for the linear benchmark policy (and for D-4's piecewise-linear one once
  the grid resolves the kink).

**Monotonicity assert.** The reindex-then-`interp` is only valid if the
grid-axis → iset-coordinate map is monotone (so the computed `m` coordinate is
sorted and `interp` is well-posed). Assert that the computed iset coordinate is
strictly monotone along its source grid axis (sign-consistent first
differences); a non-monotone map fails loudly rather than producing a silently
wrong interpolation. (Mirror of Mechanism A's invariance assert.)

> **O3 resolved — scope of B for v1.** Implement B only for the
> 1-axis-per-iset-var monotone map (covers conftest + D-2/D-4/U-2), guarded by
> the monotonicity assert above; raise a clear `NotImplementedError` for general
> multi-D scattered reindexing.

---

## 6. Helpers to add / refactor in `vbi.py`

- `bellman_step(...)` — single exact backup (§2).
- `solve_bellman(...)` — value-iteration wrapper (§3).
- `value_array_to_function(value_array, bp, disc_params={})` — continuation
  builder (§4).
- `_control_bounds(bp, control_sym, point_bag)` — `(lb, ub)` at a point, shared
  with legacy `solve`'s bound logic (legacy may optionally adopt it).
- `_project_to_iset(policy_array, iset, grid_axes, computed_isets)` — Mechanisms
  A/B
  - invariance assert.
- Reuse Phase-1 `ar_from_data`, `tensor_decision_rule`, `grid_to_data_array`.

No changes to `bellman.py`, `block.py`, `ann.py`, `loss.py`.

---

## 7. Test plan (`tests/test_vbi.py`)

**`test_bellman_step` (single backup, terminal continuation → `optimal_dr`):**
reuse conftest cases — `continuation = lambda s, sh, p: 0.0`.

| Test               | Case              | Exercises                                        |
| ------------------ | ----------------- | ------------------------------------------------ |
| interior optimum   | `case_0`          | base path, β present, cv=0                       |
| shock-dependent    | `case_1`          | observed shock in iset (Mech A)                  |
| consume-cash       | `case_3`          | Mech B reindex `m` (psi in `scope`)              |
| upper/lower binds  | `case_5`,`case_6` | double bounds                                    |
| single bounds      | `case_7`,`case_8` | open-bound defaults                              |
| empty iset         | `case_9`          | constant rule                                    |
| **multi-control**  | `case_10`         | stacked control vector, Mech A reduce `d`        |
| **non-trivial cv** | `case_11`         | real `continuation_vf`, β·cv, arrival transition |

**`test_solve_bellman` (iteration to fixed point) — the Phase-2 payoff.**
Benchmark coverage is split by whether the model needs the `E_e`
shock-expectation step (§4), which decides how much machinery each test pulls
in.

_Tier 1 — deterministic (no `E_e` step; the core benchmark coverage):_

- **D-2** (`benchmarks.d2_block`, no shocks): iterate from V₀=0 to convergence;
  recover `c = κ·(m + H)`, `κ = (R − (βR)^{1/σ})/R`, vs `d2_analytical_policy`
  within tol at interior grid points. Asserts `value_array.attrs["converged"]`.
  β from `DiscFac`.
- **D-4** (`benchmarks.d4_block`, no shocks; binding constraint `c ≤ m`, **no
  closed form**): converge and compare to the package's own exact oracle
  `d4_vfi_reference_policy` (itself a hand-rolled VFI on a cash-on-hand grid —
  `solve_bellman` generalizes exactly this). Validates the convergence loop
  _and_ active-bound handling against an independent exact solver.
  **Highest-value add.**
- **U-2** (`benchmarks.u2_block`, default `sigma_psi=0` ⇒ `MeanOneLogNormal`
  discretizes to a single node at `psi=1` ⇒ a degenerate hidden-shock
  expectation that is trivially exact): recover `c = (1−β)(m + 1/r)` vs
  `u2_analytical_policy`. Exercises log utility + normalized dynamics.
  **Deferred to PR6 (§9 step 6, shock discretization).** Although the shock is
  degenerate at `sigma_psi=0`, `psi` is a _declared_ shock that is hidden from
  `c`'s information set (`iset = ["m"]`), so today `bellman_step` raises
  `NotImplementedError` ("Shock 'psi' is hidden … and has no fixed value")
  rather than treating it as deterministic — for _any_ `sigma_psi`, not just the
  degenerate one. Both the U-2 **single-step** and **convergence** tests
  therefore land with PR6 (once the backup integrates the hidden `psi`, or a
  caller supplies `psi=1` via `scope`); PR5 covers Tier-1 determinism with D-2
  (closed form) and D-4 (oracle) only.

  Note the two axes are independent: PR6's discretization handles `sigma_psi>0`
  (multiple nodes) exactly as it handles the `sigma_psi=0` single node — the
  degeneracy is _not_ a machinery requirement. `sigma_psi=0` is required only by
  the **oracle**: `u2_analytical_policy`'s closed form `c = (1−β)(m + 1/r)` is
  the deterministic PIH rule, exact at `sigma_psi=0` (per the benchmark
  docstring). So PR6 can test U-2 two ways — the degenerate node **vs the closed
  form**, and a `sigma_psi>0` run that genuinely exercises the hidden-shock
  `expected()` (validated against an independent VFI or as property-only checks,
  the U-1/U-3 lane, since the closed form is only asserted exact at
  `sigma_psi=0`).

_Tier 2 — stochastic (uses the internal-discretization expectation of §4):_

- **D-3** (`benchmarks.d3_block`): **stochastic** —
  `live ~ Bernoulli(SurvivalProb)` is a hidden shock; the effective `s·β`
  discount _is_ `E[liv']=s·liv`, recovered by `Bernoulli.discretize` (2 nodes).
  Recover `c = κ_s·(m + H)` vs `d3_analytical_policy`. No `disc_params` tuning
  needed (discrete shock).
- **U-1** (`eta ~ Normal` income, `βR=1`): martingale/PIH;
  `Normal.discretize(N=…)` supplies the nodes. Tests the continuous-shock path.
  `disc_params={"eta":{"N":7}}`.
- **case_2** (conftest, hidden `theta`, optimum `c=E[theta]=0`): a minimal unit
  test of the hidden-shock expectation, independent of the benchmarks.
- **U-3** (no closed form): limiting-MPC property checks only; stretch goal.

_Finite-horizon flavor (separate from fixed-point iteration):_

- **D-1** (`benchmarks.d1_block`, deterministic, time-varying):
  `solve_bellman(max_iter=T)` reading the per-period policy
  `c_t=(1−β)/(1−β^{T−t})·W`. Exercises the integer `t` axis and a non-stationary
  rule; medium effort, optional for v1.

**convergence sanity:** `solve_bellman(max_iter=1)` reproduces `bellman_step`
with terminal continuation on a conftest case (loop wiring check).

**Protocol round-trips** (parallel to Phase-1 `test_vbi_protocol`):
`tensor_decision_rule(dr["c"])` flows through `bp.compute_controls` and
`BellmanEquationLoss`; `value_array`/`policy_array` are well-formed.

> **O4 resolved:** single-step correctness via `bellman_step`; infinite-horizon
> closed forms via `solve_bellman` to convergence. Both tested. **O7 — benchmark
> scope:** the discretization mechanism (§4) makes Tier 2 the _same_ code path
> as Tier 1, not a separate future feature — so D-2/D-4/U-2 and D-3/U-1/case_2
> can all land together. D-1 (finite horizon) and U-3 (no closed form,
> property-only) remain optional stretch. Confirm how far to go.

---

## 8. Open questions

- **O1 resolved** — multi-control `policy_array` is `dict[str, DataArray]`
  (lean), matching the module-wide convention (`dr_from_data`, solver's
  `dict_of_decision_rules`) and accommodating per-control isets on different
  coordinate axes (Mechanism B), which a single `Dataset` cannot cleanly hold.
- **O2** — `scope`/`params` split, shocks via discretization not a fixed bag
  (§2). ✔ resolved.
- **O3 resolved** — Mechanism B limited to 1-D monotone iset→grid maps in v1,
  guarded by the monotonicity assert (§5); general multi-D scattered reindexing
  raises `NotImplementedError`.
- **O5 resolved** — non-converged `solve_bellman` returns the last iterate with
  `converged=False` in `value_array.attrs` and emits a `warnings.warn`; it does
  **not** raise (matches scipy's `OptimizeResult.success` convention and keeps
  the partial iterate usable for inspection/warm-start). An optional
  `raise_on_nonconvergence=False` knob lets pipeline callers opt into
  strictness.
- **~~O6~~ resolved** — shocks are discretized _internally_ via `.discretize` /
  `discretized_shock_dstn` / `expected` (§4); they are never state-grid axes, so
  the old "axes must align with `disc_params` nodes" concern is moot. Hidden
  shocks integrate inside the backup's `max`; observed shocks become iset node
  axes integrated into the arrival value. Empty-shock-safe by guard.
- **Invariance-assert tol** (Mech A) — v1 ships a flat absolute `1e-3` (test
  ATOL scale). Hardening it (parameterize + relative-or-absolute combo for
  differently-scaled models) is deferred to **Phase 3** (tracked in
  `i66_todo.md`).
- **Optimizer robustness — partially resolved; multi-start now on the agenda.**
  Two cheap seeds land in v1: (a) per-point **midpoint `x0`** when both bounds
  are finite (else the scalar `x0` fallback), replacing the arbitrary constant
  `1.0` (§2); (b) **warm-start** — `solve_bellman` seeds each backup's
  `x0_policy` from the previous iterate's `policy_array`, cheap since the
  gridded policy shares the state-grid axes across iterations (§3).
  Multi-start/restarts were deferred "until a test forces them" — **two Tier-1
  benchmark cases now force them:**

  - **D-2 single backup (`test_d2_single_backup_analytic_continuation`).** Once
    the consumption floor is restored (`lower_bound = 0`), `c`'s bounds are both
    finite, so the midpoint seed is `(0 + (m + H)) / 2 ≈ 17.7` — far above the
    true optimum (`c ≈ 1.2`) and outside L-BFGS-B's basin, so it stalls at
    `c ≈ 6.97` even though the backup objective is **unimodal**. The midpoint of
    a box whose upper bound is the human-wealth-inflated natural borrowing limit
    `m + H` is a systematically bad seed. **Workaround in place:** the test
    passes a flat `x0_policy` warm-start (reproducing the robust pre-floor
    seeding). Multi-start would remove the workaround.
  - **D-2 / U-2 value iteration.** The iterated `solve_bellman` rides the
    consumption upper bound at every grid point (D-2: `c = m + H`, the
    consume-all-human-wealth flat fixed point; U-2: the `0.1·m + 2` cap), while
    a single backup under a good continuation is interior. The bad midpoint seed
    biases each per-point backup toward the top of the box, reinforcing the
    collapse; multi-start (trying a low seed alongside the midpoint/warm-start
    and keeping the best `res.fun`) is the natural mitigation to evaluate here.
    **Note:** this failure also has a grid-coverage/extrapolation dimension —
    the `a`-grid must span the borrowing region (`a ≥ −H`) and the continuation
    must not linearly extrapolate into the `V → −∞` wall at the natural limit
    (naive widening _diverges_); see the D-2/U-2 findings. Multi-start is
    necessary but may not be sufficient on its own. The near-term principled
    mitigation is the **slack artificial borrowing limit at the grid edge**
    (Deaton/Carroll — successors are kept in the trusted grid domain, so the
    grid floor acts as a slack state constraint verified by an invariance
    assert); the long-term one that sidesteps the maximization and extrapolation
    entirely is **EGM (§10)**.

  **Agenda for a multi-start pass:** try a small seed set per control — the
  midpoint, the modest scalar `x0` fallback (clamped into the box), and the
  warm-start when present — and keep the optimum with the best `res.fun`. This
  does not regress narrow-double-bound cases (`case_5`/`case_6`), where the
  midpoint is already a good seed, since it is still among the candidates.

- **TODO (docs) — in-period dynamics order & arrival-state aliasing.** A
  non-obvious modeling semantic surfaced building `case_11` and should be
  documented for end users (block-authoring guide / `block.py` + `bellman.py`
  docstrings), not just the design: **block `dynamics` run in declaration order
  within a period, so the same symbol can mean its _arrival_ value early and its
  _recomputed next-period_ value later.** In `case_11` (`u = -(a-b)^2` declared
  _before_ `b = c`), the reward reads the arrival `b` while
  `transition_function` returns `b' = c` — so `reward_function` and
  `transition_function` see different values for `b`, and a control can reach
  the objective only through the continuation. Authors who don't know this can
  write a block whose reward silently uses the wrong (arrival vs. recomputed)
  value. Document: how `get_arrival_states` infers arrival states from
  declaration order; that reward symbols read the value as of their declaration
  point; and a worked `case_11`-style example. (Captured in the agent memory
  `block_model_semantics.md`; promote to user-facing docs.)

---

## 9. Implementation order — one PR per step

Ship as **seven small PRs**, each adding one capability plus the tests that pin
it (no large omnibus PR). The steps below are dependency-sorted, so merging them
1 → 7 always satisfies prerequisites; the **Depends on** column records the
_minimum_ prerequisites so steps can be resequenced or parallelized where the
graph allows.

| PR            | Adds                                                                                                                                                     | Tests it lands                                                                                                                                                                       | Depends on     |
| ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------- |
| **1 ✅ DONE** | `bellman_step` core: single-control, grid-equals-iset (transpose projection), β + multi-reward + det-safe, per-point `x0` (warm-start/midpoint/fallback) | `case_0/1/5/6/7/8/9` + return-contract, warm-start, and guard tests (`test_vbi_bellman_step`, 11 passing)                                                                            | — (foundation) |
| **2 ✅ DONE** | Mechanism B reindex + monotonicity assert (§5)                                                                                                           | `case_3`, D-2 single backup under analytic continuation                                                                                                                              | PR1            |
| **3 ✅ DONE** | Multi-control vectorization; `policy_array` → `dict[str, DataArray]` (O1)                                                                                | `case_10`                                                                                                                                                                            | PR1            |
| **4 ✅ DONE** | Non-trivial `continuation_vf` (β·cv, arrival transition)                                                                                                 | `case_11`                                                                                                                                                                            | PR1            |
| **5**         | `value_array_to_function` + `solve_bellman` loop (warm-start `x0_policy`, non-conv warn — O5)                                                            | Tier 1: D-2 closed form, **D-4 vs `d4_vfi_reference_policy`**. (U-2 deferred to PR6 — its `psi` shock, though degenerate at `sigma_psi=0`, is hidden and unhandled until §9 step 6.) | PR2, PR4       |
| **6**         | Discretized shock expectation (§4): hidden-shock `expected` in the backup + observed-shock node integration in `value_array_to_function`                 | `case_2` (hidden unit test), then Tier 2: D-3 discrete, U-1 continuous, **U-2 single-step + convergence** (degenerate `psi` node)                                                    | PR5            |
| **7**         | Protocol round-trip + convergence-sanity                                                                                                                 | round-trip through `compute_controls`/`BellmanEquationLoss`; `solve_bellman(max_iter=1)` ≡ `bellman_step`. (D-1 finite-horizon, U-3 property-only = optional stretch.)               | PR5            |

**Dependency notes.**

- **PR1 is the foundation**; PRs 2/3/4 are independent extensions of it (each
  touches `bellman_step` in a separate concern), so they may land in any order
  or in parallel after PR1.
- **PR5 is gated on PR2 _and_ PR4**, not just PR1: its Tier-1 benchmarks
  (D-2/D-4/U-2) all use the `m = a·R + y` Mechanism-B reindex (PR2), and the
  iteration loop rebuilds the continuation from each iterate (the machinery PR4
  validates). **PR3 (multi-control) is _not_ a prerequisite for PR5** — Tier 1
  is single-control — so PR3 can be sequenced before or after PR5 freely.
- **PRs 6 and 7 both build on PR5** (the iteration loop /
  `value_array_to_function`) and are independent of each other.

**As-built note (PR1).** PR1 ships the **grid-equals-iset** contract (the policy
projection is just a `transpose(*iset)`, as in Phase-1 `solve`). The
**Mechanism-A _reduction_** (dropping non-iset grid axes under an invariance
assert) is first _needed_ by a control whose iset is narrower than the grid —
which initially occurs in `case_10`/`case_11` — so it lands with **PR3**, not
PR1. PR1 guards the boundary: a grid wider than the iset, a derived-pre-state
iset (Mechanism B), multiple controls, or a non-empty `disc_params` each raise a
clear `NotImplementedError` pointing at the owning PR. Hidden shocks are
supplied as fixed realizations via `scope` in PR1 (proper integration is PR6).
`policy_array` is already returned as `dict[str, DataArray]` (O1), so PR3 widens
its contents without changing the shape.

---

## 10. Future stage (Phase 3): Endogenous Grid Method (EGM)

**Status: not scheduled in PRs 1–7.** EGM is a faster, more robust _alternative_
solver for the differentiable single-continuous-control (consumption-like)
subclass — Carroll's "method of endogenous gridpoints" (2006). It is the
long-term answer to the extrapolation / ride-the-bound pathology of §8: because
it grids over _end-of-period_ assets and constructs the cash-on-hand grid where
the solution actually lives, it does **no within-loop maximization** (so no seed
/ multi-start problem) and **no continuation extrapolation** (so no `V → −∞`
divergence at the natural borrowing limit). The general `bellman_step` maximizer
(§2) remains the reference implementation and the cross-check oracle; EGM must
reproduce it on D-2/D-4/U-2 within tolerance.

### 10.1 When EGM applies (guarded specialization, not a replacement)

EGM requires, and the solver must assert:

1. **One continuous control** whose optimum is characterized by an interior
   **Euler FOC** (not an active bound), i.e. `u'(c) = β R E[V_m(m')]` via the
   envelope condition `V_m(m) = u'(c(m))`.
2. **Invertible marginal utility** `u'⁻¹` (closed form for CRRA/log; a monotone
   1-D root solve otherwise).
3. A **monotone budget map** from end-of-period assets `a'` to next cash-on-hand
   `m'` (so the endogenous `m = a' + c` grid is sorted and `interp` is
   well-posed — the same monotonicity condition Mechanism B already asserts,
   §5).
4. A **single occasionally-binding lower constraint** (the borrowing limit),
   handled as one kink.

Anything outside this class (multiple interacting continuous controls,
non-Euler/non-differentiable rewards, several binding constraints) falls back to
`bellman_step`; the extensions (DC-EGM for discrete-continuous, nested/multi-dim
EGM) are research-grade and explicitly out of scope. The solver raises a clear
`NotImplementedError` when a block violates 1–4.

### 10.2 One EGM backup (policy-function form)

Given the previous iterate's consumption rule `c_n(·)`:

```
# exogenous grid over END-OF-PERIOD assets a' (post-decision state)
for each a'_i in aprime_grid:
    # integrate the shock(s) with the §4 discretization (empty-safe)
    def integrand(shock):
        m1   = bp.<budget>(a'_i, shock, params)        # next cash-on-hand m'
        c1   = c_n(m1)                                  # next-period consumption
        return u_prime(c1) * dm1_da(a'_i, shock)        # both via autodiff (§10.3)
    w_prime = beta * R * expected(integrand, disc)      # post-decision marginal value
    c_i     = u_prime_inv(w_prime)                      # invert the Euler FOC
    m_i     = a'_i + c_i                                # ENDOGENOUS cash-on-hand
# c_i on the endogenous m_i grid IS the updated rule; interp c over m_i (sorted)
# constraint kink: for m < m_i[0] the limit binds -> c = m - a_min (c = m if a_min=0)
```

Iterate to a fixed point on the consumption rule (sup-norm), mirroring
`solve_bellman`'s loop and 3-tuple return contract (`dr`, a value grid rebuilt
by integrating `u`, and `policy_array`), so EGM is a drop-in faster path for the
applicable models.

### 10.3 Fit to the block protocol (autodiff, minimal new surface)

The torch stack makes the derivatives free, so EGM needs almost no per-block
hand-coding:

- **`u_prime`** — `torch.autograd` of the block's reward w.r.t. the control (no
  hand-differentiation; works for any differentiable reward).
- **`dm1_da`** — autodiff of the block's `m`-dynamics w.r.t. `a'` (generalizes
  the affine `∂m'/∂a' = R` used above).
- **`u_prime_inv`** — closed form dispatched per utility where available (CRRA:
  `c = w'^{-1/σ}`), else a bracketed 1-D root on the monotone `u'`.
- Reuses §4 shock discretization (`discretized_shock_dstn` / `expected`) and the
  Mechanism-B monotone reindex (§5) verbatim.

The one likely protocol addition is a small hook to identify the "Euler control"
and its budget/marginal-utility symbols (or to infer them from the single
`Control` + reward), tracked with the Phase-3 items in `i66_todo.md`.

### 10.4 API & dependencies

- New `vbi.solve_egm(bp, aprime_grid, *, disc_params={}, tol=…, max_iter=…)`
  parallel to `solve_bellman`, same return contract.
- **Depends on PR6** (§4 discretization) for `E_shock`; the deterministic
  benchmarks (D-2/D-4/U-2 at `sigma_psi=0`) work with the empty-shock guard even
  before PR6.
- **Validation:** must match `solve_bellman` / the closed forms on D-2
  (`κ(m+H)`), D-4 (vs `d4_vfi_reference_policy`), and U-2 (`(1−β)(m+1/r)`), and
  should show the grid-independence and no-divergence that the maximizer lacks
  near the natural borrowing limit.
