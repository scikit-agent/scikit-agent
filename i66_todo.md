# i66 TODO — VBI tests & solver gaps

## Done

- Cleaned up `src/skagent/algos/vbi.py`:
  - Removed stale `# old! (should be negative)` comment on `negated_value`.
  - Fixed upper-bound default `1e-12` → `1e12` ("a very high number").
  - Removed leftover debug `print`s (`print(feq)`, `print(bounds)`,
    `print(policy_data)`).
  - Changed open lower-bound default `-1e-6` → `-1e12` (effectively −∞,
    symmetric with upper).
- Added comprehensive `test_vbi_conftest` class in `tests/test_vbi.py` (8 tests,
  all passing alongside the 3 legacy tests). Each conftest case is solved with
  VBI under a terminal (zero) continuation and checked against its analytic
  `optimal_dr` to within `1e-3` at interior grid points:

  | Case     | Exercises                                         | Optimum     |
  | -------- | ------------------------------------------------- | ----------- |
  | `case_0` | unconstrained interior optimum                    | `c = 0`     |
  | `case_1` | shock-dependent policy (`theta` in info set)      | `c = theta` |
  | `case_3` | consume cash-on-hand                              | `c = m`     |
  | `case_5` | double-bounded, upper binds                       | `c = a`     |
  | `case_6` | double-bounded, lower binds                       | `c = a`     |
  | `case_7` | lower-bound only → tests open upper-bound default | `c = 1`     |
  | `case_8` | upper-bound only → tests open lower-bound default | `c = a`     |
  | `case_9` | empty information set (constant rule)             | `c = 3`     |

  Cases 7 and 8 specifically validate the open-bound defaults (`±1e12`).

## Context — what `main` already provides

Since this branch was first written, `main` landed a full torch-based solving
stack that VBI predates and must now integrate with:

- **`bellman.py` — `BellmanPeriod`.** Wraps a `Block` with an explicit
  `discount_variable`; `resolve_discount_factor(post)` extracts β from the
  post-transition bag. torch-native `transition_function` / `reward_function` /
  `post_function` / `compute_controls(df, …)` / `compute_value(vf, …)`, plus
  gradient variants. Multi-reward summation and empty-shock (deterministic)
  handling are already solved here.
- **`ann.py` — `BlockValueNet` / `BlockPolicyNet` / `BlockPolicyValueNet`** (all
  `BellmanPeriodMixin`, one net per control). Expose `get_value_function()` /
  `get_decision_rule()` / `get_decision_function()`.
- **`loss.py` — `BellmanEquationLoss`** (`V(s) − [u + β·E V(s′)]` + optional FOC
  weight), `EulerEquationLoss`, `StaticRewardLoss`,
  `EstimatedDiscountedLifetimeRewardLoss`. All take a `BellmanPeriod`; all are
  callables `loss(df, input_grid) -> tensor`.
- **`solver.py` —
  `solve_multiple_controls(control_order, bellman_period, givens, calibration, loss=…)`**
  already drives multi-control _neural_ solving.

**Consequence:** the torch-based VFI already exists (`BellmanEquationLoss` +
value/policy nets + `solver`). VBI's unique remaining value is that it is the
only **exact grid** solver (per-grid-point `scipy.optimize`, no
function-approximation error). So the goal is _not_ to rebuild VBI in torch — it
is to bring the exact grid solver onto the `BellmanPeriod` protocol the rest of
the stack speaks, then wire it into the nets as a warm-start / ground-truth
tool.

## Friction points (verified against current `src/`)

- `vbi.solve` still rides the **old `DBlock` continuation API**
  (`get_state_rule_value_function_from_continuation` @ `block.py:567`,
  `get_decision_value_function`, `get_arrival_value_function`). These still
  exist, so nothing is broken, but it is a parallel universe to `BellmanPeriod`.
- VBI decision rules use the
  **`ar_from_data(**args)`** signature; the rest of the stack uses `f(states_t,
  shocks_t,
  parameters)`. The 11 VBI tests assert the old form (`dr["c"](m=1.5)`, `arr_vf({"y":
  10})`), so the protocol change is breaking and must be shimmed or migrated.
- Original structural gaps, still real: no explicit β (folded into
  `continuation`, `block.py` backup is `r + cv`), single reward
  (`calc_reward(...).values()[0]`), single control (`vbi.solve` raises for >1),
  deterministic-block crash (`discretized_shock_dstn` on empty shocks).

## Work plan

Order is **1 → 2 → 3**: protocol first (unblocks interop), standalone exact
solver second, bridge/warm-start last.

**Assumption:** add a new opt-in `solve_bellman(bp, …)` entry point and leave
the legacy `solve(block, continuation, …)` untouched. The legacy path is the
deliberate "user controls their own discount factor by folding it into the
continuation" feature, now documented as the explicit alternative to β.

### Phase 1 — Protocol adapter (no solver-internals change) ✅ DONE

Insight that simplified the plan: the library is two-tiered — a low-level
**decision rule** called positionally in `control.iset` order
(`block.transition` does `dr[sym](*[vals[v] for v in iset])`; `BlockPolicyNet`
matches), and a higher-level **decision function**
`df(states, shocks, parameters)` built on top of it via `compute_pre_state`.
`ar_from_data` was already a _rule_ generator; it only used the wrong calling
convention (`ar(**kwargs)` instead of positional). So Phase 1 was a convention
fix, not a new `df` wrapper — the `df` falls out of the existing
`compute_pre_state` + rule composition, exactly as the nets get theirs.

Implemented:

- **`ar_from_data(da)`** rewritten to the library convention: positional args in
  `da.dims` order, numpy/xarray space, scalar→scalar and array→pointwise
  (shared-dim indexers, not outer product). Empty iset → constant rule.
- **`tensor_decision_rule(np_rule)`** (new) wraps any numpy rule for the torch
  stack: tensor in/out, float32 on the grid device, **detached** (numpy interp
  severs the graph). Valid as a fixed / ground-truth / warm-start policy, not as
  a trainable FOC/Euler policy.
- **`solve`** builds each rule from `policy_data.transpose(*control.iset)`, so
  it owns the contract that the rule's positional args follow `control.iset`
  regardless of the caller's grid order. (No `align_to_iset` reduction — see
  Phase 2.)
- Migrated all VBI tests to the positional convention; added `test_vbi_protocol`
  round-tripping a tensorized VBI dr through `BellmanPeriod.compute_controls`
  and `BellmanEquationLoss.__call__`. 13 passing.

**Semantics noted while doing this (older code, divergent from the rest of the
library):**

- **Full observation.** `solve`'s per-point optimization wraps the candidate
  action as a _constant_ rule and never integrates over anything — it assumes
  the iset is sufficient (everything relevant is observed). The only expectation
  machinery (`discretized_shock_dstn` + `expected`) lives in
  `get_arrival_value_function`, unused by the optimization. This is why the VBI
  test set skips `case_2` (hidden shock, optimum `c = E[theta] = 0`), and why
  `case_3` only passes because `psi` is value-irrelevant under terminal
  continuation. Hidden-shock / partial-observation problems are out of scope for
  the legacy `solve`.
- **`calibration` is a scope bag, not parameters.**
  `pre_states = calibration.copy(); pre_states.update(state_point)` — so
  `solve`'s `calibration` argument is the general _scope_ under which
  dynamics/reward/ continuation are evaluated, and legacy usage puts fixed
  exogenous values (e.g. a shock realization `psi`) there, not just
  single-valued parameters. This is broader than `calibration` elsewhere in the
  library. Read it as "scope" — hence the `solve` argument is renamed
  `calibration` → `scope`.

### Phase 2 — Re-base the exact solver on `BellmanPeriod` (standalone completion)

Add `vbi.solve_bellman(bp: BellmanPeriod, continuation_vf, state_grid, …)` that
uses `bp` for model mechanics instead of the `DBlock` continuation methods:

- **Explicit β:** backup becomes `r + β·cv`, β from
  `bp.resolve_discount_factor(post)` (closes gap #1). Keep legacy `solve` as the
  β-folded-into-continuation path.
- **Multi-reward:** sum over `bp.get_reward_syms(agent)` (gap #2).
- **Deterministic blocks:** take the shock expectation through `bp`
  (empty-shock-safe) instead of VBI's direct `discretized_shock_dstn` (gap #3).
- **Multiple controls:** vectorize the `scipy.minimize` call over a control
  vector with per-control bounds (gap #5); `BlockPolicyNet`'s bound handling is
  the reference. Stays _exact_ — distinct from `solver.solve_multiple_controls`,
  which is neural.
- **Grid → per-control iset projection:** Phase 1 tightened `solve` so the state
  grid equals the control's iset (extra scope vars go in `scope`), which works
  only because terminal continuation makes those vars value-irrelevant. With a
  real continuation and/or multiple controls, the state grid is legitimately
  wider than any single control's iset (the value function ranges over all
  arrival states; per-control isets differ), so `solve_bellman` must project the
  gridded policy down to each control's iset. This is the reduction that Phase 1
  removed (`isel` non-iset dims, assuming the optimum is invariant across them —
  the definition of "outside the iset").
- Then add **benchmark policy-matching tests** (e.g. D-3 VFI → `c = κ·m` with
  `κ = (R − (βR)^{1/σ})/R`), now expressible because the discounted recursion
  exists.

### Phase 3 — Bridge / warm-start

- Add `vbi_value_to_net(value_da, bp) -> BlockValueNet` and
  `vbi_policy_to_net(policy_da, bp) -> BlockPolicyNet` that supervised-fit a net
  to the exact grid solution.
- Uses: warm-start `solver.solve_multiple_controls` (initialize nets near the
  exact solution); serve as ground truth in tests comparing neural-VFI output
  against the exact grid.
- xarray stays VBI-internal; tensors only cross at the Phase-1 callable
  boundary.
